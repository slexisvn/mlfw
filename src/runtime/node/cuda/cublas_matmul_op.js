import { AutogradNode } from '../../../autograd/node.js';
import { AutogradMeta } from '../../../tensor/core/autograd_meta.js';
import { GradMode } from '../../../autograd/grad_mode.js';
import { wireInputEdges } from '../../../autograd/accumulator.js';
import { wrapResult, getGpuContiguousFn } from '../../../dispatcher/jit_dispatch.js';
import { contiguous } from '../../../tensor/ops/ops.js';
import { DeviceType } from '../../../tensor/types/device.js';
import { isEagerDeferred, deviceBufferForInput, deviceBufferForOutput, uploadIfStale } from './resident.js';
import { acquire, release, copyDeviceToHost } from './memory.js';
import { cublasGemmDevice, cublasGemmBatchedDevice } from './cublas.js';
import { matmul as opsMatmul } from '../../../tensor/ops/ops.js';

function devIn(arr) { return isEagerDeferred() ? deviceBufferForInput(arr) : uploadIfStale(arr); }
function devOut(arr, pending) {
  if (isEagerDeferred()) return deviceBufferForOutput(arr);
  const dptr = acquire(arr.byteLength); pending.push([arr, dptr]); return dptr;
}
function flushOut(pending) { for (const [arr, dptr] of pending) { copyDeviceToHost(arr, dptr); release(dptr, arr.byteLength); } }

function operandData(T) {
  let lay = operandLayout(T);
  if (lay) return { data: T._impl.storage.rawData, lay };
  const fn = getGpuContiguousFn();
  if (isEagerDeferred() && fn) {
    const data = fn(T._impl.storage.rawData, T.shape, T.strides, T._impl.storageOffset, T.dtype);
    const shape = T.shape, rank = shape.length;
    const rows = shape[rank - 2], cols = shape[rank - 1];
    let batch = 1;
    for (let i = 0; i < rank - 2; i++) batch *= shape[i];
    return { data, lay: { trans: false, batchStride: rows * cols, batch, rows, cols } };
  }
  const Tc = contiguous(T);
  return { data: Tc._impl.storage.rawData, lay: operandLayout(Tc) };
}

function operandLayout(T) {
  if (T.storageOffset !== 0) return null;
  const shape = T.shape, strides = T.strides, rank = shape.length;
  if (rank < 2) return null;
  const rows = shape[rank - 2], cols = shape[rank - 1];
  const sRow = strides[rank - 2], sCol = strides[rank - 1];
  let trans;
  if (sRow === cols && sCol === 1) trans = false;
  else if (sRow === 1 && sCol === rows) trans = true;
  else return null;
  let expected = rows * cols;
  for (let i = rank - 3; i >= 0; i--) { if (strides[i] !== expected) return null; expected *= shape[i]; }
  let batch = 1;
  for (let i = 0; i < rank - 2; i++) batch *= shape[i];
  return { trans, batchStride: rows * cols, batch, rows, cols };
}

class MatmulBackward extends AutogradNode {
  constructor(A, B) { super(2); this.A = A; this.B = B; }
  apply(gradOutputs) {
    const dC = gradOutputs[0];
    const A = this.A, B = this.B, ra = A.shape.length, rb = B.shape.length;
    const dA = opsMatmul(dC, B.transpose(rb - 2, rb - 1));
    const dB = opsMatmul(A.transpose(ra - 2, ra - 1), dC);
    return [dA, dB];
  }
}

function attach(node, output) {
  const meta = new AutogradMeta();
  meta.setGradFn(node, 0);
  meta.requiresGrad = true;
  output._impl.setAutogradMeta(meta);
  output._impl._updateKeySet();
}

export function gpuMatmul(A, B) {
  if (!A.device || A.device.type !== DeviceType.GPU || B.device.type !== DeviceType.GPU) return null;
  if (A.dtype !== 'f32' || B.dtype !== 'f32') return null;
  const ra = A.shape.length, rb = B.shape.length;
  if (ra < 2 || rb < 2 || ra !== rb) return null;
  const M = A.shape[ra - 2], K = A.shape[ra - 1], N = B.shape[rb - 1];
  if (B.shape[rb - 2] !== K) return null;

  const a = operandData(A), b = operandData(B);
  const layA = a.lay, layB = b.lay;
  if (!layA || !layB || layA.batch !== layB.batch) return null;

  const batch = layA.batch, pending = [];
  const dA = devIn(a.data);
  const dB = devIn(b.data);
  const outArr = new Float32Array(batch * M * N);
  const dC = devOut(outArr, pending);
  if (batch > 1) {
    cublasGemmBatchedDevice(batch, M, N, K, dA, layA.batchStride, layA.trans, dB, layB.batchStride, layB.trans, dC, M * N);
  } else {
    cublasGemmDevice(M, N, K, dA, layA.trans, dB, layB.trans, dC);
  }
  flushOut(pending);

  const outShape = [...A.shape.slice(0, ra - 2), M, N];
  const out = wrapResult(outArr, outShape, A.dtype, A.device);

  if (GradMode.isEnabled() && ((A._impl.autogradMeta && A.requiresGrad) || (B._impl.autogradMeta && B.requiresGrad))) {
    const node = new MatmulBackward(A, B);
    wireInputEdges(node, [A, B]);
    attach(node, out);
  }
  return out;
}
