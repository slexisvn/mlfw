import { describe, it, expect } from 'vitest';
import { tensor } from '../../src/index.js';
import * as nn from '../../src/index.js';

const flat = (v) => Array.from(v && typeof v.contiguous === 'function' ? v.contiguous().data : v.data);
const sig = (x) => 1 / (1 + Math.exp(-x));

function numGrad(lossFn, param, idx) {
  const d = param._impl.storage.data;
  const orig = d[idx];
  const eps = 1e-3;
  d[idx] = orig + eps; const lp = flat(lossFn())[0];
  d[idx] = orig - eps; const lm = flat(lossFn())[0];
  d[idx] = orig;
  return (lp - lm) / (2 * eps);
}

describe('LSTMCell matches the PyTorch LSTM formulation', () => {
  it('one step equals a hand-computed reference', () => {
    const cell = new nn.LSTMCell(2, 3);
    const Wx = flat(cell.x2h.weight), bx = flat(cell.x2h.bias);
    const Wh = flat(cell.h2h.weight), bh = flat(cell.h2h.bias);
    const x = [0.7, -0.4], h = [0.1, 0.2, 0.3], c = [-0.2, 0.05, 0.4], H = 3, I = 2;
    const pre = [];
    for (let k = 0; k < 4 * H; k++) {
      let s = bx[k] + bh[k];
      for (let j = 0; j < I; j++) s += x[j] * Wx[k * I + j];
      for (let j = 0; j < H; j++) s += h[j] * Wh[k * H + j];
      pre.push(s);
    }
    const refH = [];
    for (let k = 0; k < H; k++) {
      const i = sig(pre[k]);
      const f = sig(pre[H + k]);
      const g = Math.tanh(pre[2 * H + k]);
      const o = sig(pre[3 * H + k]);
      const cN = f * c[k] + i * g;
      refH.push(o * Math.tanh(cN));
    }
    const [hN] = cell.forward(tensor([x]), [tensor([h]), tensor([c])]);
    const out = flat(hN);
    for (let k = 0; k < H; k++) expect(out[k]).toBeCloseTo(refH[k], 5);
  });
});

describe('LSTM sequence shapes', () => {
  it('batch_first input -> [B,T,H] output and [layers,B,H] hidden + cell', () => {
    const lstm = new nn.LSTM(4, 8, 2, true);
    const inp = tensor(Array.from({ length: 3 }, () => Array.from({ length: 5 }, () => Array.from({ length: 4 }, () => 0.1))));
    const [o, state] = lstm.forward(inp);
    const [hn, cn] = state;
    expect(o.shape).toEqual([3, 5, 8]);
    expect(hn.shape).toEqual([2, 3, 8]);
    expect(cn.shape).toEqual([2, 3, 8]);
  });

  it('seq-first input -> [T,B,H] output', () => {
    const lstm = new nn.LSTM(3, 6, 1, false);
    const inp = tensor(Array.from({ length: 4 }, () => Array.from({ length: 2 }, () => Array.from({ length: 3 }, () => 0.2))));
    const [o, state] = lstm.forward(inp);
    expect(o.shape).toEqual([4, 2, 6]);
    expect(state[0].shape).toEqual([1, 2, 6]);
  });
});

describe('LSTM backpropagation', () => {
  it('autograd gradient matches numerical', () => {
    const lstm = new nn.LSTM(2, 3, 1, true);
    const x = tensor([[[0.5, -0.3], [0.2, 0.8]]]);
    const lossFn = () => { const [o] = lstm.forward(x); return o.sum(); };
    const p = [...lstm.parameters()][0];
    lossFn().backward();
    const auto = flat(p.grad)[0];
    expect(auto).toBeCloseTo(numGrad(lossFn, p, 0), 2);
  });
});

describe('LSTM + attention text classifier trains end-to-end', () => {
  it('loss decreases and embeddings update', () => {
    class TextClf extends nn.Module {
      constructor(v, em, h, c) {
        super();
        this.embed = new nn.Embedding(v, em);
        this.lstm = new nn.LSTM(em, h, 1, true);
        this.attn = new nn.Linear(h, 1);
        this.fc = new nn.Linear(h, c);
      }
      forward(ids) {
        const e = this.embed.forward(ids);
        const [o] = this.lstm.forward(e);
        const s = this.attn.forward(o).softmax(1);
        return this.fc.forward(o.mul(s).sum(1));
      }
    }
    const clf = new TextClf(50, 8, 16, 3);
    const ids = tensor([[3, 15, 42, 7, 0], [1, 9, 2, 5, 8]]);
    const target = tensor([0, 2]);
    const ce = new nn.CrossEntropyLoss();
    const w0 = flat(clf.embed.weight)[12];
    let first = 0, last = 0;
    for (let step = 0; step < 25; step++) {
      clf.zeroGrad();
      const loss = ce.forward(clf.forward(ids), target);
      const lv = flat(loss)[0];
      if (step === 0) first = lv;
      if (step === 24) last = lv;
      loss.backward();
      for (const p of clf.parameters()) {
        if (!p.grad) continue;
        const pd = p._impl.storage.data, gd = flat(p.grad);
        for (let i = 0; i < pd.length; i++) pd[i] -= 0.1 * gd[i];
      }
    }
    expect(last).toBeLessThan(first);
    expect(flat(clf.embed.weight)[12]).not.toBe(w0);
  });
});
