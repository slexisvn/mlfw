import { AutogradNode } from '../../../autograd/node.js';
import { AutogradMeta } from '../../../tensor/core/autograd_meta.js';
import { GradMode } from '../../../autograd/grad_mode.js';
import { wireInputEdges } from '../../../autograd/accumulator.js';
import { wrapResult, gpuContiguousArray } from '../../../dispatcher/jit_dispatch.js';
import type { NumericTypedArray } from '../../../tensor/types/dtype.js';
import { deviceBufferForInput, deviceBufferForOutput } from './resident.js';
import { cudnnRNNForward, cudnnRNNBackward, releaseRNNForward, CELL_LSTM, CELL_GRU } from './cudnn.js';
import type { RNNForwardCtx, RNNPlanOpts } from './cudnn.js';
import type { Tensor } from '../../../tensor/core/tensor.js';
import type { DType } from '../../../tensor/types/dtype.js';
import type { Device } from '../../../tensor/types/device.js';
import type { DevicePtr } from './ffi.js';

type RNNCell = {
  x2h: { weight: Tensor; bias: Tensor };
  h2h: { weight: Tensor; bias: Tensor };
};

type WeightShapes = { x2hW: number[]; x2hB: number[]; h2hW: number[]; h2hB: number[] };

type RNNBackwardInfo = {
  xArr: NumericTypedArray;
  yArr: Float32Array;
  hxArr: NumericTypedArray | null;
  cxArr: NumericTypedArray | null;
  opts: RNNPlanOpts;
  numLayers: number;
  hasInit: boolean;
  hasCell: boolean;
  dtype: DType;
  device: Device;
  inputShape: number[];
  stateShape: number[];
  weightShapes: WeightShapes[];
};

const carr = (t: Tensor): NumericTypedArray => gpuContiguousArray(t);
const prod = (s: readonly number[]): number => s.reduce((a, b) => a * b, 1);
const devIn = deviceBufferForInput;
const devOut = deviceBufferForOutput;

const _fwdReclaim = new FinalizationRegistry<RNNForwardCtx>((ctx) => releaseRNNForward(ctx));

const KIND: Record<string, { cellMode: number; gates: number; hasCell: boolean }> = {
  lstm: { cellMode: CELL_LSTM, gates: 4, hasCell: true },
  gru: { cellMode: CELL_GRU, gates: 3, hasCell: false },
};

class CudnnRNNBackward extends AutogradNode {
  fwd: RNNForwardCtx;
  info: RNNBackwardInfo;

  constructor(fwd: RNNForwardCtx, info: RNNBackwardInfo, numInputs: number) { super(numInputs); this.fwd = fwd; this.info = info; }
  apply(gradOutputs: readonly (Tensor | null)[]): Tensor[] {
    const [dy, dhy, dcy] = gradOutputs;
    const f = this.info;
    const { seqLen, batch, inputSize, hiddenSize } = f.opts;
    const numLayers = f.numLayers, stateN = numLayers * batch * hiddenSize;

    const xDev = devIn(f.xArr);
    const yDev = devIn(f.yArr);
    const hxDev = f.hasInit ? devIn(f.hxArr!) : 0n;
    const cxDev = f.hasCell && f.hasInit ? devIn(f.cxArr!) : 0n;
    const dyDev = devIn(carr(dy!));
    const dhyDev = dhy ? devIn(carr(dhy)) : 0n;
    const dcyDev = f.hasCell && dcy ? devIn(carr(dcy)) : 0n;

    const dxArr = new Float32Array(seqLen * batch * inputSize);
    const dxDev = devOut(dxArr);
    let dhxArr: Float32Array | null = null, dcxArr: Float32Array | null = null, dhxDev: DevicePtr = 0n, dcxDev: DevicePtr = 0n;
    if (f.hasInit) {
      dhxArr = new Float32Array(stateN); dhxDev = devOut(dhxArr);
      if (f.hasCell) { dcxArr = new Float32Array(stateN); dcxDev = devOut(dcxArr); }
    }
    const gradArrs = f.weightShapes.map((s) => ({
      x2hW: new Float32Array(prod(s.x2hW)), x2hB: new Float32Array(prod(s.x2hB)),
      h2hW: new Float32Array(prod(s.h2hW)), h2hB: new Float32Array(prod(s.h2hB)),
    }));
    const gradDevs = gradArrs.map((g) => ({
      x2hW: devOut(g.x2hW), x2hB: devOut(g.x2hB),
      h2hW: devOut(g.h2hW), h2hB: devOut(g.h2hB),
    }));

    cudnnRNNBackward(this.fwd, xDev, yDev, hxDev, cxDev, dyDev, dhyDev, dcyDev, dxDev, dhxDev, dcxDev, gradDevs);
    _fwdReclaim.unregister(this);

    const w = (arr: NumericTypedArray, shape: readonly number[]): Tensor => wrapResult(arr, shape, f.dtype, f.device);
    const grads = [w(dxArr, f.inputShape)];
    for (let l = 0; l < gradArrs.length; l++) {
      const g = gradArrs[l], s = f.weightShapes[l];
      grads.push(w(g.x2hW, s.x2hW), w(g.x2hB, s.x2hB), w(g.h2hW, s.h2hW), w(g.h2hB, s.h2hB));
    }
    if (f.hasInit) {
      grads.push(w(dhxArr!, f.stateShape));
      if (f.hasCell) grads.push(w(dcxArr!, f.stateShape));
    }
    return grads;
  }
}

function attach(node: AutogradNode, output: Tensor, outputNr: number): void {
  const meta = new AutogradMeta();
  meta.setGradFn(node, outputNr);
  meta.requiresGrad = true;
  output._impl.setAutogradMeta(meta);
  output._impl._updateKeySet();
}

function cudnnRNNOp(
  kindName: string,
  input: Tensor,
  cells: readonly RNNCell[],
  opts: RNNPlanOpts,
  hx: Tensor | null = null,
  cx: Tensor | null = null,
): (Tensor | null)[] {
  const kind = KIND[kindName];
  const { hiddenSize, seqLen, batch } = opts;
  const numLayers = cells.length;
  const planOpts = { ...opts, numLayers, cellMode: kind.cellMode, gates: kind.gates };

  const xArr = carr(input);
  const xDev = devIn(xArr);
  const layerDevs = cells.map((c) => ({
    x2hW: devIn(carr(c.x2h.weight)), x2hB: devIn(carr(c.x2h.bias)),
    h2hW: devIn(carr(c.h2h.weight)), h2hB: devIn(carr(c.h2h.bias)),
  }));
  const hxArr = hx ? carr(hx) : null, cxArr = kind.hasCell && cx ? carr(cx) : null;
  const hxDev = hx ? devIn(hxArr!) : 0n, cxDev = cxArr ? devIn(cxArr) : 0n;

  const stateN = numLayers * batch * hiddenSize, stateShape = [numLayers, batch, hiddenSize];
  const yArr = new Float32Array(seqLen * batch * hiddenSize);
  const hyArr = new Float32Array(stateN);
  const cyArr = kind.hasCell ? new Float32Array(stateN) : null;
  const yDev = devOut(yArr), hyDev = devOut(hyArr);
  const cyDev = cyArr ? devOut(cyArr) : 0n;

  const inputs = [input];
  for (const c of cells) inputs.push(c.x2h.weight, c.x2h.bias, c.h2h.weight, c.h2h.bias);
  if (hx) { inputs.push(hx); if (kind.hasCell) inputs.push(cx!); }
  let any = false;
  for (const t of inputs) { if (t._impl.autogradMeta && t.requiresGrad) { any = true; break; } }
  const training = GradMode.isEnabled() && any;

  const fwd = cudnnRNNForward(xDev, layerDevs, planOpts, hxDev, cxDev, yDev, hyDev, cyDev, training);

  const out = wrapResult(yArr, [seqLen, batch, hiddenSize], input.dtype, input.device);
  const hyT = wrapResult(hyArr, stateShape, input.dtype, input.device);
  const cyT = cyArr ? wrapResult(cyArr, stateShape, input.dtype, input.device) : null;

  if (training) {
    const info = {
      xArr, yArr, hxArr, cxArr, opts: planOpts, numLayers, hasInit: !!hx, hasCell: kind.hasCell,
      dtype: input.dtype, device: input.device, inputShape: [...input.shape], stateShape,
      weightShapes: cells.map((c) => ({ x2hW: [...c.x2h.weight.shape], x2hB: [...c.x2h.bias.shape], h2hW: [...c.h2h.weight.shape], h2hB: [...c.h2h.bias.shape] })),
    };
    const node = new CudnnRNNBackward(fwd, info, inputs.length);
    _fwdReclaim.register(node, fwd, node);
    wireInputEdges(node, inputs);
    attach(node, out, 0); attach(node, hyT, 1);
    if (cyT) attach(node, cyT, 2);
  } else {
    releaseRNNForward(fwd);
  }
  return kind.hasCell ? [out, hyT, cyT] : [out, hyT];
}

export function cudnnLSTMOp(input: Tensor, cells: readonly RNNCell[], opts: RNNPlanOpts, hx: Tensor | null = null, cx: Tensor | null = null): (Tensor | null)[] {
  return cudnnRNNOp('lstm', input, cells, opts, hx, cx);
}

export function cudnnGRUOp(input: Tensor, cells: readonly RNNCell[], opts: RNNPlanOpts, h0: Tensor | null = null): (Tensor | null)[] {
  return cudnnRNNOp('gru', input, cells, opts, h0, null);
}
