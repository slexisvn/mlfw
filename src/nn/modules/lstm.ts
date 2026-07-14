import { Module } from '../module.js';
import { Linear } from './linear.js';
import { zeros } from '../../tensor/factory/creation_ops.js';
import { add, mul, sigmoid, tanh, stack } from '../../tensor/ops/ops.js';
import { select, split } from '../../tensor/ops/ops.js';
import { scan } from '../../tracing/scan.js';
import { getActiveTracer } from '../../tracing/tracer.js';
import { getCudnnLSTM, getWebgpuRNN } from '../../dispatcher/jit_dispatch.js';
import { DeviceType } from '../../tensor/types/device.js';
import type { NNTensor, TensorPair } from '../types.js';

export class LSTMCell extends Module {
  inputSize: number;
  hiddenSize: number;
  x2h: Linear;
  h2h: Linear;

  constructor(inputSize: number, hiddenSize: number, bias = true) {
    super();
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.x2h = new Linear(inputSize, 4 * hiddenSize, bias);
    this.h2h = new Linear(hiddenSize, 4 * hiddenSize, bias);
  }

  forward(input: NNTensor, state: TensorPair | null = null): TensorPair {
    const h = state !== null ? state[0] : zeros([input.shape[0], this.hiddenSize], { device: input.device }) as NNTensor;
    const c = state !== null ? state[1] : zeros([input.shape[0], this.hiddenSize], { device: input.device }) as NNTensor;
    const gates = add(this.x2h.forward(input), this.h2h.forward(h)) as NNTensor;
    const [i, f, g, o] = split(gates, this.hiddenSize, -1);
    const inputGate = sigmoid(i);
    const forgetGate = sigmoid(f);
    const cellGate = tanh(g);
    const outputGate = sigmoid(o);
    const cNext = add(mul(forgetGate, c), mul(inputGate, cellGate));
    const hNext = mul(outputGate, tanh(cNext));
    return [hNext as NNTensor, cNext as NNTensor];
  }
}

export class LSTM extends Module {
  inputSize: number;
  hiddenSize: number;
  numLayers: number;
  batchFirst: boolean;
  cells: LSTMCell[];

  constructor(inputSize: number, hiddenSize: number, numLayers = 1, batchFirst = false, bias = true) {
    super();
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.numLayers = numLayers;
    this.batchFirst = batchFirst;
    this.cells = [];
    for (let l = 0; l < numLayers; l++) {
      const cell = new LSTMCell(l === 0 ? inputSize : hiddenSize, hiddenSize, bias);
      this.cells.push(cell);
      this.registerModule('cell_' + l, cell);
    }
  }

  forward(input: NNTensor, state: TensorPair | null = null): [NNTensor, TensorPair] {
    const cudnn = getCudnnLSTM();
    if (cudnn && input.device.type === DeviceType.GPU && !getActiveTracer()) {
      const xs = this.batchFirst ? input.transpose(0, 1) : input;
      const opts = { inputSize: this.inputSize, hiddenSize: this.hiddenSize, seqLen: xs.shape[0], batch: xs.shape[1] };
      const [out, hy, cy] = cudnn(xs, this.cells, opts, state ? state[0] : null, state ? state[1] : null) as [NNTensor, NNTensor, NNTensor];
      return [this.batchFirst ? out.transpose(0, 1) : out, [hy, cy]];
    }
    const webgpuRnn = getWebgpuRNN();
    if (webgpuRnn && input.device.type === DeviceType.WEBGPU && !getActiveTracer()) {
      const xs = this.batchFirst ? input.transpose(0, 1) : input;
      const opts = { inputSize: this.inputSize, hiddenSize: this.hiddenSize, seqLen: xs.shape[0], batch: xs.shape[1] };
      const fused = webgpuRnn(xs, this.cells, opts, state ? state[0] : null, state ? state[1] : null) as [NNTensor, NNTensor, NNTensor] | null;
      if (fused) {
        const [out, hy, cy] = fused;
        return [this.batchFirst ? out.transpose(0, 1) : out, [hy, cy]];
      }
    }
    const x = this.batchFirst ? input.transpose(0, 1) : input;
    const batch = x.shape[1];
    let layerIn = x;
    const finalHs: NNTensor[] = [];
    const finalCs: NNTensor[] = [];
    for (let l = 0; l < this.numLayers; l++) {
      const h0 = state !== null ? select(state[0], 0, l) as NNTensor : zeros([batch, this.hiddenSize], { device: x.device }) as NNTensor;
      const c0 = state !== null ? select(state[1], 0, l) as NNTensor : zeros([batch, this.hiddenSize], { device: x.device }) as NNTensor;
      const cell = this.cells[l];
      const [[hN, cN], ys] = scan((carry, xt) => {
        const [h2, c2] = cell.forward(xt as NNTensor, carry as TensorPair);
        return [[h2, c2], h2];
      }, [h0, c0], layerIn) as [TensorPair, NNTensor];
      finalHs.push(hN);
      finalCs.push(cN);
      layerIn = ys;
    }
    let output = layerIn;
    if (this.batchFirst) output = output.transpose(0, 1);
    return [output, [stack(finalHs, 0) as NNTensor, stack(finalCs, 0) as NNTensor]];
  }
}
