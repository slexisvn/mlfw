import { Module } from '../module.js';
import { Linear } from './linear.js';
import { zeros } from '../../tensor/factory/creation_ops.js';
import { add, sub, mul, sigmoid, tanh, stack } from '../../tensor/ops/ops.js';
import { select, split } from '../../tensor/view/view_ops.js';
import { scan } from '../../tracing/scan.js';

export class GRUCell extends Module {
  constructor(inputSize, hiddenSize, bias = true) {
    super();
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.x2h = new Linear(inputSize, 3 * hiddenSize, bias);
    this.h2h = new Linear(hiddenSize, 3 * hiddenSize, bias);
  }

  forward(input, hidden = null) {
    const h = hidden !== null ? hidden : zeros([input.shape[0], this.hiddenSize]);
    const gx = this.x2h.forward(input);
    const gh = this.h2h.forward(h);
    const [xr, xz, xn] = split(gx, this.hiddenSize, -1);
    const [hr, hz, hn] = split(gh, this.hiddenSize, -1);
    const r = sigmoid(add(xr, hr));
    const z = sigmoid(add(xz, hz));
    const n = tanh(add(xn, mul(r, hn)));
    return add(n, mul(z, sub(h, n)));
  }
}

export class GRU extends Module {
  constructor(inputSize, hiddenSize, numLayers = 1, batchFirst = false, bias = true) {
    super();
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.numLayers = numLayers;
    this.batchFirst = batchFirst;
    this.cells = [];
    for (let l = 0; l < numLayers; l++) {
      const cell = new GRUCell(l === 0 ? inputSize : hiddenSize, hiddenSize, bias);
      this.cells.push(cell);
      this.registerModule('cell_' + l, cell);
    }
  }

  forward(input, h0 = null) {
    const x = this.batchFirst ? input.transpose(0, 1) : input;
    const batch = x.shape[1];
    let layerIn = x;
    const finalHs = [];
    for (let l = 0; l < this.numLayers; l++) {
      const hInit = h0 !== null ? select(h0, 0, l) : zeros([batch, this.hiddenSize]);
      const cell = this.cells[l];
      const [hN, ys] = scan((h, xt) => {
        const h2 = cell.forward(xt, h);
        return [h2, h2];
      }, hInit, layerIn);
      finalHs.push(hN);
      layerIn = ys;
    }
    let output = layerIn;
    if (this.batchFirst) output = output.transpose(0, 1);
    return [output, stack(finalHs, 0)];
  }
}
