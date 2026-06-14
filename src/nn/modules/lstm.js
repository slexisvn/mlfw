import { Module } from '../module.js';
import { Linear } from './linear.js';
import { zeros } from '../../tensor/factory/creation_ops.js';
import { add, mul, sigmoid, tanh, stack } from '../../tensor/ops/ops.js';
import { select, split } from '../../tensor/view/view_ops.js';

export class LSTMCell extends Module {
  constructor(inputSize, hiddenSize, bias = true) {
    super();
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.x2h = new Linear(inputSize, 4 * hiddenSize, bias);
    this.h2h = new Linear(hiddenSize, 4 * hiddenSize, bias);
  }

  forward(input, state = null) {
    const h = state !== null ? state[0] : zeros([input.shape[0], this.hiddenSize]);
    const c = state !== null ? state[1] : zeros([input.shape[0], this.hiddenSize]);
    const gates = add(this.x2h.forward(input), this.h2h.forward(h));
    const [i, f, g, o] = split(gates, this.hiddenSize, -1);
    const inputGate = sigmoid(i);
    const forgetGate = sigmoid(f);
    const cellGate = tanh(g);
    const outputGate = sigmoid(o);
    const cNext = add(mul(forgetGate, c), mul(inputGate, cellGate));
    const hNext = mul(outputGate, tanh(cNext));
    return [hNext, cNext];
  }
}

export class LSTM extends Module {
  constructor(inputSize, hiddenSize, numLayers = 1, batchFirst = false, bias = true) {
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

  forward(input, state = null) {
    const x = this.batchFirst ? input.transpose(0, 1) : input;
    const seqLen = x.shape[0];
    const batch = x.shape[1];
    const hs = [];
    const cs = [];
    for (let l = 0; l < this.numLayers; l++) {
      hs.push(state !== null ? select(state[0], 0, l) : zeros([batch, this.hiddenSize]));
      cs.push(state !== null ? select(state[1], 0, l) : zeros([batch, this.hiddenSize]));
    }
    const outputs = [];
    for (let t = 0; t < seqLen; t++) {
      let inp = select(x, 0, t);
      for (let l = 0; l < this.numLayers; l++) {
        const [hNext, cNext] = this.cells[l].forward(inp, [hs[l], cs[l]]);
        hs[l] = hNext;
        cs[l] = cNext;
        inp = hs[l];
      }
      outputs.push(inp);
    }
    let output = stack(outputs, 0);
    if (this.batchFirst) output = output.transpose(0, 1);
    return [output, [stack(hs, 0), stack(cs, 0)]];
  }
}
