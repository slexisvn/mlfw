import koffi from 'koffi';
import { cu } from './ffi.js';
import type { CudaHandle, DevicePtr } from './ffi.js';
import { getDevice } from './device.js';
import { setDevice } from './runtime_api.js';
import { acquire, release } from './memory.js';
import { loadCudaLib, CUDNN_SPEC } from './lib_resolver.js';
import { isEagerCapturing } from '../../../dispatcher/eager_mode.js';

interface CudnnApi {
  create(h: (CudaHandle | null)[]): number;
  destroy(h: CudaHandle | null): number;
  setStream(h: CudaHandle | null, stream: CudaHandle | null): number;
  getErrorString(status: number): string;
  destroyTensorDesc(d: CudaHandle | null): number;
  destroyRNNDesc(d: CudaHandle | null): number;
  destroyRNNDataDesc(d: CudaHandle | null): number;
  destroyDropoutDesc(d: CudaHandle | null): number;
  createRNNDesc(d: (CudaHandle | null)[]): number;
  setRNNDesc(d: CudaHandle | null, algo: number, cellMode: number, biasMode: number, dirMode: number, inputMode: number, dataType: number, mathPrec: number, mathType: number, inputSize: number, hiddenSize: number, projSize: number, numLayers: number, dropoutDesc: CudaHandle | null, auxFlags: number): number;
  createRNNDataDesc(d: (CudaHandle | null)[]): number;
  setRNNDataDesc(d: CudaHandle | null, dataType: number, layout: number, maxSeqLength: number, batchSize: number, vectorSize: number, seqLengthArray: ArrayBufferView, paddingFill: null): number;
  createTensorDesc(d: (CudaHandle | null)[]): number;
  setTensorNdDesc(d: CudaHandle | null, dataType: number, nbDims: number, dimA: ArrayBufferView, strideA: ArrayBufferView): number;
  createDropoutDesc(d: (CudaHandle | null)[]): number;
  setDropoutDesc(d: CudaHandle | null, h: CudaHandle | null, dropout: number, states: DevicePtr, stateSize: bigint, seed: bigint): number;
  dropoutGetStatesSize(h: CudaHandle | null, sz: bigint[]): number;
  getRNNWeightSpaceSize(h: CudaHandle | null, d: CudaHandle | null, sz: bigint[]): number;
  getRNNTempSpaceSizes(h: CudaHandle | null, d: CudaHandle | null, fwdMode: number, xDesc: CudaHandle | null, work: bigint[], reserve: bigint[]): number;
  getRNNWeightParams(h: CudaHandle | null, d: CudaHandle | null, pseudoLayer: number, wss: bigint, weightSpace: DevicePtr, linLayerID: number, mDesc: CudaHandle | null, mAddr: bigint[], bDesc: CudaHandle | null, bAddr: bigint[]): number;
  rnnForward(h: CudaHandle | null, d: CudaHandle | null, fwdMode: number, devSeqLengths: DevicePtr, xDesc: CudaHandle | null, x: DevicePtr, yDesc: CudaHandle | null, y: DevicePtr, hDesc: CudaHandle | null, hx: DevicePtr, hy: DevicePtr, cDesc: CudaHandle | null, cx: DevicePtr, cy: DevicePtr, wss: bigint, weightSpace: DevicePtr, workSize: bigint, workSpace: DevicePtr, reserveSize: bigint, reserveSpace: DevicePtr): number;
  rnnBackwardData(h: CudaHandle | null, d: CudaHandle | null, devSeqLengths: DevicePtr, yDesc: CudaHandle | null, y: DevicePtr, dy: DevicePtr, xDesc: CudaHandle | null, dx: DevicePtr, hDesc: CudaHandle | null, hx: DevicePtr, dhy: DevicePtr, dhx: DevicePtr, cDesc: CudaHandle | null, cx: DevicePtr, dcy: DevicePtr, dcx: DevicePtr, wss: bigint, weightSpace: DevicePtr, workSize: bigint, workSpace: DevicePtr, reserveSize: bigint, reserveSpace: DevicePtr): number;
  rnnBackwardWeights(h: CudaHandle | null, d: CudaHandle | null, addGrad: number, devSeqLengths: DevicePtr, xDesc: CudaHandle | null, x: DevicePtr, hDesc: CudaHandle | null, hx: DevicePtr, yDesc: CudaHandle | null, y: DevicePtr, wss: bigint, dweightSpace: DevicePtr, workSize: bigint, workSpace: DevicePtr, reserveSize: bigint, reserveSpace: DevicePtr): number;
}

export type RNNLayerDevs = { x2hW: DevicePtr; x2hB: DevicePtr; h2hW: DevicePtr; h2hB: DevicePtr };

export type RNNPlanOpts = {
  inputSize: number;
  hiddenSize: number;
  seqLen: number;
  batch: number;
  numLayers: number;
  cellMode?: number;
  gates?: number;
};

type RNNPlan = {
  rd: CudaHandle | null;
  xDesc: CudaHandle | null;
  yDesc: CudaHandle | null;
  hDesc: CudaHandle | null;
  devSeq: DevicePtr;
  wss: bigint;
  wssN: number;
  workSize: bigint;
  workN: number;
  reserveSize: bigint;
  reserveN: number;
  inputSize: number;
  hiddenSize: number;
  seqLen: number;
  batch: number;
  numLayers: number;
  gates: number;
};

export type RNNForwardCtx = {
  p: RNNPlan;
  weightSpace: DevicePtr;
  workSpace: DevicePtr;
  reserveSpace: DevicePtr;
  training: boolean;
  released?: boolean;
};

function d2d(dst: DevicePtr, src: DevicePtr, bytes: number): void {
  if (isEagerCapturing()) cu.memcpyDtoDAsync(dst, src, bytes, getDevice().stream);
  else cu.memcpyDtoD(dst, src, bytes);
}
function memZero(dst: DevicePtr, bytes: number): void {
  if (isEagerCapturing()) cu.memsetD8Async(dst, 0, bytes, getDevice().stream);
  else cu.memsetD8(dst, 0, bytes);
}

let c: CudnnApi, _inited: boolean | null = null;
function ensure(): boolean {
  if (_inited !== null) return _inited;
  try {
    const dnn = koffi.load(loadCudaLib(CUDNN_SPEC));
    c = {
      create: dnn.func('int cudnnCreate(_Out_ void **h)'),
      destroy: dnn.func('int cudnnDestroy(void *h)'),
      setStream: dnn.func('int cudnnSetStream(void *h, void *stream)'),
      getErrorString: dnn.func('str cudnnGetErrorString(int status)'),
      destroyTensorDesc: dnn.func('int cudnnDestroyTensorDescriptor(void *d)'),
      destroyRNNDesc: dnn.func('int cudnnDestroyRNNDescriptor(void *d)'),
      destroyRNNDataDesc: dnn.func('int cudnnDestroyRNNDataDescriptor(void *d)'),
      destroyDropoutDesc: dnn.func('int cudnnDestroyDropoutDescriptor(void *d)'),
      createRNNDesc: dnn.func('int cudnnCreateRNNDescriptor(_Out_ void **d)'),
      setRNNDesc: dnn.func('int cudnnSetRNNDescriptor_v8(void *d, int algo, int cellMode, int biasMode, int dirMode, int inputMode, int dataType, int mathPrec, int mathType, int inputSize, int hiddenSize, int projSize, int numLayers, void *dropoutDesc, uint auxFlags)'),
      createRNNDataDesc: dnn.func('int cudnnCreateRNNDataDescriptor(_Out_ void **d)'),
      setRNNDataDesc: dnn.func('int cudnnSetRNNDataDescriptor(void *d, int dataType, int layout, int maxSeqLength, int batchSize, int vectorSize, void *seqLengthArray, void *paddingFill)'),
      createTensorDesc: dnn.func('int cudnnCreateTensorDescriptor(_Out_ void **d)'),
      setTensorNdDesc: dnn.func('int cudnnSetTensorNdDescriptor(void *d, int dataType, int nbDims, void *dimA, void *strideA)'),
      createDropoutDesc: dnn.func('int cudnnCreateDropoutDescriptor(_Out_ void **d)'),
      setDropoutDesc: dnn.func('int cudnnSetDropoutDescriptor(void *d, void *h, float dropout, uint64 states, size_t stateSize, uint64 seed)'),
      dropoutGetStatesSize: dnn.func('int cudnnDropoutGetStatesSize(void *h, _Out_ size_t *sz)'),
      getRNNWeightSpaceSize: dnn.func('int cudnnGetRNNWeightSpaceSize(void *h, void *d, _Out_ size_t *sz)'),
      getRNNTempSpaceSizes: dnn.func('int cudnnGetRNNTempSpaceSizes(void *h, void *d, int fwdMode, void *xDesc, _Out_ size_t *work, _Out_ size_t *reserve)'),
      getRNNWeightParams: dnn.func('int cudnnGetRNNWeightParams(void *h, void *d, int pseudoLayer, size_t wss, uint64 weightSpace, int linLayerID, void *mDesc, _Out_ uint64 *mAddr, void *bDesc, _Out_ uint64 *bAddr)'),
      rnnForward: dnn.func('int cudnnRNNForward(void *h, void *d, int fwdMode, uint64 devSeqLengths, void *xDesc, uint64 x, void *yDesc, uint64 y, void *hDesc, uint64 hx, uint64 hy, void *cDesc, uint64 cx, uint64 cy, size_t wss, uint64 weightSpace, size_t workSize, uint64 workSpace, size_t reserveSize, uint64 reserveSpace)'),
      rnnBackwardData: dnn.func('int cudnnRNNBackwardData_v8(void *h, void *d, uint64 devSeqLengths, void *yDesc, uint64 y, uint64 dy, void *xDesc, uint64 dx, void *hDesc, uint64 hx, uint64 dhy, uint64 dhx, void *cDesc, uint64 cx, uint64 dcy, uint64 dcx, size_t wss, uint64 weightSpace, size_t workSize, uint64 workSpace, size_t reserveSize, uint64 reserveSpace)'),
      rnnBackwardWeights: dnn.func('int cudnnRNNBackwardWeights_v8(void *h, void *d, int addGrad, uint64 devSeqLengths, void *xDesc, uint64 x, void *hDesc, uint64 hx, void *yDesc, uint64 y, size_t wss, uint64 dweightSpace, size_t workSize, uint64 workSpace, size_t reserveSize, uint64 reserveSpace)'),
    };
    _inited = true;
  } catch (e) {
    _inited = false;
  }
  return _inited;
}
export function cudnnAvailable(): boolean { return ensure(); }

const FLOAT = 0, ALGO_STANDARD = 0, LSTM = 2, GRU = 3, DOUBLE_BIAS = 2, UNIDIR = 0, LINEAR_INPUT = 0, ALLOW_CONVERSION = 2;
export const CELL_LSTM = LSTM, CELL_GRU = GRU;
const LAYOUT_SEQ_MAJOR_UNPACKED = 0;
const FWD_TRAINING = 1, FWD_INFERENCE = 0, WGRAD_ADD = 0;

function ck(label: string, status: number): void {
  if (status !== 0) throw new Error('cuDNN ' + label + ' failed: ' + status + ' (' + c.getErrorString(status) + ')');
}
const o2 = (): (CudaHandle | null)[] => [null];

let _handle: CudaHandle | null = null, _dropout: CudaHandle | null = null, _dropoutStates: DevicePtr | null = null, _dropoutStatesBytes = 0, _weightDescs: { m: CudaHandle | null; b: CudaHandle | null } | null = null;
function handle(): CudaHandle | null {
  if (!ensure()) throw new Error('cuDNN not available');
  const dev = getDevice();
  setDevice();
  if (!_handle) {
    const h = o2(); ck('create', c.create(h)); _handle = h[0];
    ck('setStream', c.setStream(_handle, dev.stream));
  }
  return _handle;
}
function dropoutDesc(): CudaHandle | null {
  if (!_dropout) {
    const h = handle();
    const d = o2(); ck('createDropout', c.createDropoutDesc(d));
    const sz = [0n]; ck('dropStatesSize', c.dropoutGetStatesSize(h, sz));
    _dropoutStatesBytes = Math.max(Number(sz[0]), 1);
    _dropoutStates = acquire(_dropoutStatesBytes);
    ck('setDropout', c.setDropoutDesc(d[0], h, 0.0, _dropoutStates, sz[0], 0n));
    _dropout = d[0];
  }
  return _dropout;
}
function weightDescs(): { m: CudaHandle | null; b: CudaHandle | null } {
  if (!_weightDescs) {
    const m = o2(), b = o2();
    ck('cwm', c.createTensorDesc(m));
    ck('cwb', c.createTensorDesc(b));
    _weightDescs = { m: m[0], b: b[0] };
  }
  return _weightDescs;
}

export function destroyCudnn(): void {
  if (_inited !== true) return;
  for (const p of _descCache.values()) {
    c.destroyRNNDesc(p.rd);
    c.destroyRNNDataDesc(p.xDesc);
    c.destroyRNNDataDesc(p.yDesc);
    c.destroyTensorDesc(p.hDesc);
    release(p.devSeq, p.batch * 4);
  }
  _descCache.clear();
  if (_weightDescs) {
    c.destroyTensorDesc(_weightDescs.m);
    c.destroyTensorDesc(_weightDescs.b);
    _weightDescs = null;
  }
  if (_dropout) { c.destroyDropoutDesc(_dropout); _dropout = null; }
  if (_dropoutStates !== null) { release(_dropoutStates, _dropoutStatesBytes); _dropoutStates = null; _dropoutStatesBytes = 0; }
  if (_handle) { c.destroy(_handle); _handle = null; }
}
function tensorDesc(dims: readonly number[], strides: readonly number[]): CudaHandle | null {
  const d = o2(); ck('createTensor', c.createTensorDesc(d));
  ck('setTensorNd', c.setTensorNdDesc(d[0], FLOAT, dims.length, new Int32Array(dims), new Int32Array(strides)));
  return d[0];
}
function dataDesc(seqLen: number, batch: number, vec: number, seqArr: Int32Array): CudaHandle | null {
  const d = o2(); ck('createRNNData', c.createRNNDataDesc(d));
  ck('setRNNData', c.setRNNDataDesc(d[0], FLOAT, LAYOUT_SEQ_MAJOR_UNPACKED, seqLen, batch, vec, seqArr, null));
  return d[0];
}

const _descCache = new Map<string, RNNPlan>();
function plan({ inputSize, hiddenSize, seqLen, batch, numLayers, cellMode = LSTM, gates = 4 }: RNNPlanOpts): RNNPlan {
  const key = `${cellMode}_${inputSize}_${hiddenSize}_${seqLen}_${batch}_${numLayers}`;
  let p = _descCache.get(key);
  if (p) return p;
  const h = handle();
  const rnnDesc: (CudaHandle | null)[] = o2(); ck('createRNN', c.createRNNDesc(rnnDesc));
  ck('setRNN', c.setRNNDesc(rnnDesc[0], ALGO_STANDARD, cellMode, DOUBLE_BIAS, UNIDIR, LINEAR_INPUT, FLOAT, FLOAT, ALLOW_CONVERSION, inputSize, hiddenSize, hiddenSize, numLayers, dropoutDesc(), 0));
  const rd = rnnDesc[0];
  const seqArr = new Int32Array(batch).fill(seqLen);
  const xDesc = dataDesc(seqLen, batch, inputSize, seqArr);
  const yDesc = dataDesc(seqLen, batch, hiddenSize, seqArr);
  const hDesc = tensorDesc([numLayers, batch, hiddenSize], [batch * hiddenSize, hiddenSize, 1]);
  const wssP = [0n]; ck('wss', c.getRNNWeightSpaceSize(h, rd, wssP));
  const workP = [0n], resvP = [0n];
  ck('temp', c.getRNNTempSpaceSizes(h, rd, FWD_TRAINING, xDesc, workP, resvP));
  const devSeq = acquire(batch * 4);
  cu.memcpyHtoD(devSeq, seqArr, batch * 4);
  p = { rd, xDesc, yDesc, hDesc, devSeq, wss: wssP[0], wssN: Number(wssP[0]),
    workSize: workP[0], workN: Math.max(Number(workP[0]), 1), reserveSize: resvP[0], reserveN: Math.max(Number(resvP[0]), 1),
    inputSize, hiddenSize, seqLen, batch, numLayers, gates };
  _descCache.set(key, p);
  return p;
}

function packWeights(p: RNNPlan, weightSpace: DevicePtr, layerDevs: readonly RNNLayerDevs[], download: boolean): void {
  const h = handle();
  const { m: mDesc, b: bDesc } = weightDescs();
  const { numLayers, hiddenSize, inputSize, wss, gates } = p;
  for (let l = 0; l < numLayers; l++) {
    const inF = l === 0 ? inputSize : hiddenSize;
    const ld = layerDevs[l];
    for (let id = 0; id < 2 * gates; id++) {
      const isInput = id < gates, gate = id % gates;
      const mAddr = [0n], bAddr = [0n];
      ck('wparam', c.getRNNWeightParams(h, p.rd, l, wss, weightSpace, id, mDesc, mAddr, bDesc, bAddr));
      const wLen = isInput ? hiddenSize * inF : hiddenSize * hiddenSize;
      const wBytes = wLen * 4, bBytes = hiddenSize * 4;
      const wSrc = BigInt(isInput ? ld.x2hW : ld.h2hW) + BigInt(gate * wBytes);
      const bSrc = BigInt(isInput ? ld.x2hB : ld.h2hB) + BigInt(gate * bBytes);
      if (download) {
        d2d(wSrc, mAddr[0], wBytes);
        d2d(bSrc, bAddr[0], bBytes);
      } else {
        d2d(mAddr[0], wSrc, wBytes);
        d2d(bAddr[0], bSrc, bBytes);
      }
    }
  }
}

export function cudnnRNNForward(
  xDev: DevicePtr,
  layerDevs: readonly RNNLayerDevs[],
  opts: RNNPlanOpts,
  hxDev: DevicePtr,
  cxDev: DevicePtr,
  yDev: DevicePtr,
  hyDev: DevicePtr,
  cyDev: DevicePtr,
  training = true,
): RNNForwardCtx {
  const h = handle();
  const p = plan({ ...opts, numLayers: layerDevs.length });
  const weightSpace = acquire(p.wssN);
  packWeights(p, weightSpace, layerDevs, false);
  const workSpace = acquire(p.workN);
  const reserveSpace = training ? acquire(p.reserveN) : 0n;
  ck('forward', c.rnnForward(h, p.rd, training ? FWD_TRAINING : FWD_INFERENCE, p.devSeq, p.xDesc, xDev, p.yDesc, yDev,
    p.hDesc, hxDev || 0n, hyDev, p.hDesc, cxDev || 0n, cyDev, p.wss, weightSpace, p.workSize, workSpace,
    training ? p.reserveSize : 0n, reserveSpace));
  return { p, weightSpace, workSpace, reserveSpace, training };
}

export function releaseRNNForward(ctx: RNNForwardCtx | null | undefined): void {
  if (!ctx || ctx.released) return;
  ctx.released = true;
  release(ctx.weightSpace, ctx.p.wssN);
  release(ctx.workSpace, ctx.p.workN);
  if (ctx.reserveSpace && ctx.reserveSpace !== 0n) release(ctx.reserveSpace, ctx.p.reserveN);
}

export function cudnnRNNBackward(
  ctx: RNNForwardCtx,
  xDev: DevicePtr,
  yDev: DevicePtr,
  hxDev: DevicePtr,
  cxDev: DevicePtr,
  dyDev: DevicePtr,
  dhyDev: DevicePtr,
  dcyDev: DevicePtr,
  dxDev: DevicePtr,
  dhxDev: DevicePtr,
  dcxDev: DevicePtr,
  gradLayerDevs: readonly RNNLayerDevs[],
): void {
  const { p, weightSpace, workSpace, reserveSpace } = ctx;
  const h = handle();
  const dweight = acquire(p.wssN);
  memZero(dweight, p.wssN);
  ck('backwardData', c.rnnBackwardData(h, p.rd, p.devSeq, p.yDesc, yDev, dyDev, p.xDesc, dxDev,
    p.hDesc, hxDev || 0n, dhyDev || 0n, dhxDev || 0n, p.hDesc, cxDev || 0n, dcyDev || 0n, dcxDev || 0n,
    p.wss, weightSpace, p.workSize, workSpace, p.reserveSize, reserveSpace));
  ck('backwardWeights', c.rnnBackwardWeights(h, p.rd, WGRAD_ADD, p.devSeq, p.xDesc, xDev, p.hDesc, hxDev || 0n, p.yDesc, yDev,
    p.wss, dweight, p.workSize, workSpace, p.reserveSize, reserveSpace));
  packWeights(p, dweight, gradLayerDevs, true);
  release(dweight, p.wssN);
  releaseRNNForward(ctx);
}
