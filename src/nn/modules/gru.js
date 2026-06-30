import { Module } from '../module.js';
import { Linear } from './linear.js';
import { zeros } from '../../tensor/factory/creation_ops.js';
import { add, sub, mul, sigmoid, tanh, stack } from '../../tensor/ops/ops.js';
import { select, split } from '../../tensor/view/view_ops.js';
import { scan } from '../../tracing/scan.js';
import { getActiveTracer } from '../../tracing/tracer.js';
import { getCudnnGRU } from '../../dispatcher/jit_dispatch.js';
import { DeviceType } from '../../tensor/types/device.js';

export class GRUCell extends Module {
  constructor(inputSize, hiddenSize, bias = true) {
    super();
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.x2h = new Linear(inputSize, 3 * hiddenSize, bias);
    this.h2h = new Linear(hiddenSize, 3 * hiddenSize, bias);
  }

  forward(input, hidden = null) {
    const h = hidden !== null ? hidden : zeros([input.shape[0], this.hiddenSize], { device: input.device });
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
    const cudnn = getCudnnGRU();
    if (cudnn && input.device.type === DeviceType.GPU && !getActiveTracer()) {
      const xs = this.batchFirst ? input.transpose(0, 1) : input;
      const opts = { inputSize: this.inputSize, hiddenSize: this.hiddenSize, seqLen: xs.shape[0], batch: xs.shape[1] };
      const [out, hy] = cudnn(xs, this.cells, opts, h0);
      return [this.batchFirst ? out.transpose(0, 1) : out, hy];
    }
    const x = this.batchFirst ? input.transpose(0, 1) : input;
    const batch = x.shape[1];
    let layerIn = x;
    const finalHs = [];
    for (let l = 0; l < this.numLayers; l++) {
      const hInit = h0 !== null ? select(h0, 0, l) : zeros([batch, this.hiddenSize], { device: x.device });
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
