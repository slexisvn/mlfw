import { AutogradNode } from '../../../autograd/node.js';
import { AutogradMeta } from '../../../tensor/core/autograd_meta.js';
import { GradMode } from '../../../autograd/grad_mode.js';
import { GradAccumulator } from '../../../autograd/accumulator.js';
import { wrapResult, gpuContiguousArray } from '../../../dispatcher/jit_dispatch.js';
import { isEagerDeferred, deviceBufferForInput, deviceBufferForOutput, uploadIfStale } from './resident.js';
import { acquire, release, copyDeviceToHost } from './memory.js';
import { cudnnLSTMForward, cudnnLSTMBackward, releaseLSTMForward } from './cudnn.js';

const carr = (t) => gpuContiguousArray(t);
const prod = (s) => s.reduce((a, b) => a * b, 1);
function devIn(arr) { return isEagerDeferred() ? deviceBufferForInput(arr) : uploadIfStale(arr); }
function devOut(arr, pending) {
  if (isEagerDeferred()) return deviceBufferForOutput(arr);
  const dptr = acquire(arr.byteLength); pending.push([arr, dptr]); return dptr;
}
function flushOut(pending) {
  for (const [arr, dptr] of pending) { copyDeviceToHost(arr, dptr); release(dptr, arr.byteLength); }
}

class CudnnLSTMBackward extends AutogradNode {
  constructor(fwd, info, numInputs) { super(numInputs); this.fwd = fwd; this.info = info; }
  apply(gradOutputs) {
    const [dy, dhy, dcy] = gradOutputs;
    const f = this.info;
    const { seqLen, batch, inputSize, hiddenSize } = f.opts;
    const numLayers = f.numLayers, stateN = numLayers * batch * hiddenSize, pending = [];

    const xDev = devIn(f.xArr);
    const yDev = devIn(f.yArr);
    const hxDev = f.hasInit ? devIn(f.hxArr) : 0n;
    const cxDev = f.hasInit ? devIn(f.cxArr) : 0n;
    const dyDev = devIn(carr(dy));
    const dhyDev = dhy ? devIn(carr(dhy)) : 0n;
    const dcyDev = dcy ? devIn(carr(dcy)) : 0n;

    const dxArr = new Float32Array(seqLen * batch * inputSize);
    const dxDev = devOut(dxArr, pending);
    let dhxArr = null, dcxArr = null, dhxDev = 0n, dcxDev = 0n;
    if (f.hasInit) {
      dhxArr = new Float32Array(stateN); dcxArr = new Float32Array(stateN);
      dhxDev = devOut(dhxArr, pending); dcxDev = devOut(dcxArr, pending);
    }
    const gradArrs = f.weightShapes.map((s) => ({
      x2hW: new Float32Array(prod(s.x2hW)), x2hB: new Float32Array(prod(s.x2hB)),
      h2hW: new Float32Array(prod(s.h2hW)), h2hB: new Float32Array(prod(s.h2hB)),
    }));
    const gradDevs = gradArrs.map((g) => ({
      x2hW: devOut(g.x2hW, pending), x2hB: devOut(g.x2hB, pending),
      h2hW: devOut(g.h2hW, pending), h2hB: devOut(g.h2hB, pending),
    }));

    cudnnLSTMBackward(this.fwd, xDev, yDev, hxDev, cxDev, dyDev, dhyDev, dcyDev, dxDev, dhxDev, dcxDev, gradDevs);
    flushOut(pending);

    const w = (arr, shape) => wrapResult(arr, shape, f.dtype, f.device);
    const grads = [w(dxArr, f.inputShape)];
    for (let l = 0; l < gradArrs.length; l++) {
      const g = gradArrs[l], s = f.weightShapes[l];
      grads.push(w(g.x2hW, s.x2hW), w(g.x2hB, s.x2hB), w(g.h2hW, s.h2hW), w(g.h2hB, s.h2hB));
    }
    if (f.hasInit) grads.push(w(dhxArr, f.stateShape), w(dcxArr, f.stateShape));
    return grads;
  }
}

function attach(node, output, outputNr) {
  const meta = new AutogradMeta();
  meta.setGradFn(node, outputNr);
  meta.requiresGrad = true;
  output._impl.setAutogradMeta(meta);
  output._impl._updateKeySet();
}

export function cudnnLSTMOp(input, cells, opts, hx = null, cx = null) {
  const { inputSize, hiddenSize, seqLen, batch } = opts;
  const numLayers = cells.length, pending = [];

  const xArr = carr(input);
  const xDev = devIn(xArr);
  const layerDevs = cells.map((c) => ({
    x2hW: devIn(carr(c.x2h.weight)), x2hB: devIn(carr(c.x2h.bias)),
    h2hW: devIn(carr(c.h2h.weight)), h2hB: devIn(carr(c.h2h.bias)),
  }));
  const hxArr = hx ? carr(hx) : null, cxArr = cx ? carr(cx) : null;
  const hxDev = hx ? devIn(hxArr) : 0n, cxDev = cx ? devIn(cxArr) : 0n;

  const stateN = numLayers * batch * hiddenSize, stateShape = [numLayers, batch, hiddenSize];
  const yArr = new Float32Array(seqLen * batch * hiddenSize);
  const hyArr = new Float32Array(stateN), cyArr = new Float32Array(stateN);
  const yDev = devOut(yArr, pending), hyDev = devOut(hyArr, pending), cyDev = devOut(cyArr, pending);

  const inputs = [input];
  for (const c of cells) inputs.push(c.x2h.weight, c.x2h.bias, c.h2h.weight, c.h2h.bias);
  if (hx) inputs.push(hx, cx);
  let any = false;
  for (const t of inputs) { if (t._impl.autogradMeta && t.requiresGrad) { any = true; break; } }
  const training = GradMode.isEnabled() && any;

  const fwd = cudnnLSTMForward(xDev, layerDevs, opts, hxDev, cxDev, yDev, hyDev, cyDev, training);
  flushOut(pending);

  const out = wrapResult(yArr, [seqLen, batch, hiddenSize], input.dtype, input.device);
  const hyT = wrapResult(hyArr, stateShape, input.dtype, input.device);
  const cyT = wrapResult(cyArr, stateShape, input.dtype, input.device);

  if (training) {
    const info = {
      xArr, yArr, hxArr, cxArr, opts, numLayers, hasInit: !!hx,
      dtype: input.dtype, device: input.device, inputShape: [...input.shape], stateShape,
      weightShapes: cells.map((c) => ({ x2hW: [...c.x2h.weight.shape], x2hB: [...c.x2h.bias.shape], h2hW: [...c.h2h.weight.shape], h2hB: [...c.h2h.bias.shape] })),
    };
    const node = new CudnnLSTMBackward(fwd, info, inputs.length);
    for (let i = 0; i < inputs.length; i++) {
      node.saveInputMetadata(i, [...inputs[i].shape], inputs[i].dtype);
      const m = inputs[i]._impl.autogradMeta;
      if (m && m.requiresGrad) {
        if (m.gradFn) node.setNextEdge(i, m.gradFn, m.outputNr || 0);
        else { let acc = m.getGradAccumulator(); if (!acc) { acc = new GradAccumulator(inputs[i]); m.setGradAccumulator(acc); } node.setNextEdge(i, acc, 0); }
      } else node.setNextEdge(i, null, 0);
    }
    attach(node, out, 0); attach(node, hyT, 1); attach(node, cyT, 2);
  } else {
    releaseLSTMForward(fwd);
  }
  return [out, hyT, cyT];
}
