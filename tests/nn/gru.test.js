import { describe, it, expect } from 'vitest';
import { tensor, stack } from '../../src/index.js';
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

describe('GRUCell matches the PyTorch GRU formulation', () => {
  it('one step equals a hand-computed reference', () => {
    const cell = new nn.GRUCell(2, 3);
    const Wx = flat(cell.x2h.weight), bx = flat(cell.x2h.bias);
    const Wh = flat(cell.h2h.weight), bh = flat(cell.h2h.bias);
    const x = [1.0, -0.5], h = [0.1, 0.2, 0.3], H = 3, I = 2;
    const gx = [], gh = [];
    for (let i = 0; i < 3 * H; i++) { let s = bx[i]; for (let j = 0; j < I; j++) s += x[j] * Wx[i * I + j]; gx.push(s); }
    for (let i = 0; i < 3 * H; i++) { let s = bh[i]; for (let j = 0; j < H; j++) s += h[j] * Wh[i * H + j]; gh.push(s); }
    const ref = [];
    for (let k = 0; k < H; k++) {
      const r = sig(gx[k] + gh[k]);
      const z = sig(gx[H + k] + gh[H + k]);
      const n = Math.tanh(gx[2 * H + k] + r * gh[2 * H + k]);
      ref.push((1 - z) * n + z * h[k]);
    }
    const out = flat(cell.forward(tensor([x]), tensor([h])));
    for (let k = 0; k < H; k++) expect(out[k]).toBeCloseTo(ref[k], 5);
  });
});

describe('GRU sequence shapes', () => {
  it('batch_first input -> [B,T,H] output and [layers,B,H] hidden', () => {
    const gru = new nn.GRU(4, 8, 2, true);
    const inp = tensor(Array.from({ length: 3 }, () => Array.from({ length: 5 }, () => Array.from({ length: 4 }, () => 0.1))));
    const [o, hn] = gru.forward(inp);
    expect(o.shape).toEqual([3, 5, 8]);
    expect(hn.shape).toEqual([2, 3, 8]);
  });

  it('seq-first input -> [T,B,H] output', () => {
    const gru = new nn.GRU(3, 6, 1, false);
    const inp = tensor(Array.from({ length: 4 }, () => Array.from({ length: 2 }, () => Array.from({ length: 3 }, () => 0.2))));
    const [o, hn] = gru.forward(inp);
    expect(o.shape).toEqual([4, 2, 6]);
    expect(hn.shape).toEqual([1, 2, 6]);
  });
});

describe('GRU backpropagation', () => {
  it('autograd gradient matches numerical', () => {
    const gru = new nn.GRU(2, 3, 1, true);
    const x = tensor([[[0.5, -0.3], [0.2, 0.8]]]);
    const lossFn = () => { const [o] = gru.forward(x); return o.sum(); };
    const p = [...gru.parameters()][0];
    lossFn().backward();
    const auto = flat(p.grad)[0];
    expect(auto).toBeCloseTo(numGrad(lossFn, p, 0), 2);
  });
});

describe('softmax / log_softmax / stack gradients', () => {
  it('softmax gradient matches numerical', () => {
    const x = tensor([[1.0, 2.0, 0.5]], { requiresGrad: true });
    const w = tensor([[0.3, -0.7, 1.1]]);
    const lf = () => x.softmax(1).mul(w).sum();
    lf().backward();
    const auto = flat(x.grad);
    for (let i = 0; i < 3; i++) expect(auto[i]).toBeCloseTo(numGrad(lf, x, i), 2);
  });

  it('log_softmax gradient matches numerical', () => {
    const x = tensor([[0.5, -1.0, 2.0]], { requiresGrad: true });
    const w = tensor([[1.0, 0.5, -0.3]]);
    const lf = () => x.log_softmax(1).mul(w).sum();
    lf().backward();
    const auto = flat(x.grad);
    for (let i = 0; i < 3; i++) expect(auto[i]).toBeCloseTo(numGrad(lf, x, i), 2);
  });

  it('stack routes gradient to each input slice', () => {
    const a = tensor([[1.0, 2.0]], { requiresGrad: true });
    const b = tensor([[3.0, 4.0]], { requiresGrad: true });
    const w = tensor([[[0.5, -1.0]], [[2.0, 0.3]]]);
    stack([a, b], 0).mul(w).sum().backward();
    expect(flat(a.grad)).toEqual([0.5, -1.0]);
    expect(flat(b.grad)[0]).toBeCloseTo(2.0, 5);
    expect(flat(b.grad)[1]).toBeCloseTo(0.3, 5);
  });
});

describe('Embedding is trainable (output requires grad)', () => {
  it('lookup propagates gradient to the weight rows used', () => {
    const emb = new nn.Embedding(20, 4);
    const ids = tensor([[1, 5, 3], [2, 0, 7]]);
    const out = emb.forward(ids);
    expect(out.shape).toEqual([2, 3, 4]);
    expect(out.requiresGrad).toBe(true);
    out.sum().backward();
    expect(emb.weight.grad).toBeTruthy();
    expect(flat(emb.weight.grad).some((v) => v !== 0)).toBe(true);
  });
});

describe('GRU + attention text classifier trains end-to-end', () => {
  it('loss decreases and embeddings update', () => {
    class TextClf extends nn.Module {
      constructor(v, em, h, c) {
        super();
        this.embed = new nn.Embedding(v, em);
        this.gru = new nn.GRU(em, h, 1, true);
        this.attn = new nn.Linear(h, 1);
        this.fc = new nn.Linear(h, c);
      }
      forward(ids) {
        const e = this.embed.forward(ids);
        const [o] = this.gru.forward(e);
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
