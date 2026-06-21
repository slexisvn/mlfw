import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { BackwardGraphBuilder } from '../../../src/compiler/ad/backward_builder.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../../src/backend/target.js';
import '../../../src/compiler/ad/index.js';

const F32 = ScalarType.F32;
const t = (s) => new TensorType(s, F32);
const numel = (s) => s.reduce((a, b) => a * b, 1);

function gradOfSumLoss(mkFwd, inArrs, outShape) {
  const cf = compileGraph(mkFwd(), CPUTarget());
  const fk = cf.listKernels()[0];
  const fwdOut = (arrs) => {
    const o = new Float32Array(numel(outShape));
    cf.run(fk, ...arrs, o);
    return o;
  };
  const loss = (arrs) => { let s = 0; for (const e of fwdOut(arrs)) s += e; return s; };

  const fwdB = mkFwd();
  const r = new BackwardGraphBuilder().build(fwdB);
  const bwd = r.backwardFunc;
  const argId = new Map(fwdB.args.map((a, i) => [a.id, i]));
  const savedSrc = r.savedValues.map(sv => (argId.has(sv.id) ? argId.get(sv.id) : -1));
  const bnOut = fwdOut(inArrs);
  const cb = compileGraph(bwd, CPUTarget());
  const bk = cb.listKernels()[0];
  const bins = [new Float32Array(numel(outShape))];
  bins[0].fill(1);
  for (const s of savedSrc) bins.push(s >= 0 ? inArrs[s].slice() : bnOut.slice());
  const bouts = bwd.outputTypes.map(o => new Float32Array(Math.max(numel(o.shape), 1)));
  cb.run(bk, ...bins, ...bouts);

  const numGrad = (idx) => {
    const base = inArrs[idx];
    const g = new Float32Array(base.length);
    const eps = 1e-3;
    for (let i = 0; i < base.length; i++) {
      const saved = base[i];
      const a = inArrs.map(x => x.slice());
      a[idx][i] = saved + eps; const lp = loss(a);
      a[idx][i] = saved - eps; const lm = loss(a);
      g[i] = (lp - lm) / (2 * eps);
    }
    return g;
  };
  return { analytic: bouts, gradInputIndices: r.gradInputIndices, numGrad };
}

describe('stop_gradient halts the gradient through its path', () => {
  it('f(x) = stop_gradient(x) * x has grad x (not 2x)', () => {
    const fwd = buildFunction('sg', [t([4])], [t([4])], (b, [x]) => {
      b.returnOp([b.mul(b.stopGradient(x).getResult(0), x).getResult(0)]);
    });
    const r = new BackwardGraphBuilder().build(fwd);
    const bwd = r.backwardFunc;
    const cb = compileGraph(bwd, CPUTarget());
    const bk = cb.listKernels()[0];
    const X = [2, 3, 4, 5];
    const bins = bwd.inputTypes.map(it => new Float32Array(numel(it.shape)));
    bins[0].set([1, 1, 1, 1]);
    for (let i = 1; i < bins.length; i++) bins[i].set(X);
    const bouts = bwd.outputTypes.map(o => new Float32Array(Math.max(numel(o.shape), 1)));
    cb.run(bk, ...bins, ...bouts);
    expect([...bouts[0]]).toEqual(X);
  });
});

describe('batch_norm VJP matches finite differences', () => {
  it('grads for input/gamma/beta/mean/variance are correct', () => {
    const N = 2, C = 3, H = 2, W = 2;
    const xs = [N, C, H, W], cs = [C];
    const mkFwd = () => buildFunction('bn', [t(xs), t(cs), t(cs), t(cs), t(cs)], [t(xs)], (b, [x, g, be, m, v]) => {
      b.returnOp([b.batchnorm(x, g, be, m, v, 1, 1e-5).getResult(0)]);
    });
    const inArrs = [
      Float32Array.from({ length: numel(xs) }, (_, i) => Math.sin(i * 0.7)),
      Float32Array.from({ length: C }, (_, i) => 1 + 0.3 * i),
      Float32Array.from({ length: C }, (_, i) => 0.1 * i),
      Float32Array.from({ length: C }, (_, i) => 0.2 * i - 0.1),
      Float32Array.from({ length: C }, (_, i) => 0.5 + 0.1 * i),
    ];
    const { analytic, gradInputIndices, numGrad } = gradOfSumLoss(mkFwd, inArrs, xs);
    expect(gradInputIndices.length).toBe(5);
    for (let k = 0; k < gradInputIndices.length; k++) {
      const an = analytic[k], nu = numGrad(gradInputIndices[k]);
      let e = 0;
      for (let i = 0; i < an.length; i++) e = Math.max(e, Math.abs(an[i] - nu[i]));
      expect(e).toBeLessThan(2e-2);
    }
  });
});

describe('reverse op forward + VJP', () => {
  it('flips along the given axes and the gradient reverses', () => {
    const fwd = buildFunction('rev', [t([3, 2])], [t([3, 2])], (b, [x]) => {
      b.returnOp([b.reverse(x, [0]).getResult(0)]);
    });
    const r = new BackwardGraphBuilder().build(fwd);
    const cf = compileGraph(fwd, CPUTarget());
    const o = new Float32Array(6);
    cf.run(cf.listKernels()[0], new Float32Array([0, 1, 2, 3, 4, 5]), o);
    expect([...o]).toEqual([4, 5, 2, 3, 0, 1]);

    const cb = compileGraph(r.backwardFunc, CPUTarget());
    const bins = r.backwardFunc.inputTypes.map(it => new Float32Array(numel(it.shape)));
    bins[0].set([10, 11, 20, 21, 30, 31]);
    const bouts = r.backwardFunc.outputTypes.map(o => new Float32Array(Math.max(numel(o.shape), 1)));
    cb.run(cb.listKernels()[0], ...bins, ...bouts);
    expect([...bouts[0]]).toEqual([30, 31, 20, 21, 10, 11]);
  });
});

describe('scan VJP (compiled RNN backward, incl. captured weights)', () => {
  it('grads for xs, init carry, and free-variable weights match finite differences', () => {
    const T = 4, D = 3;
    const mk = () => buildFunction('rnn', [t([T, D]), t([D]), t([D]), t([D])], [t([D]), t([T, D])], (b, [xs, init, Wx, Wc]) => {
      const s = b.scanOp([xs], [init], (bb, xt, cy) => {
        const term = bb.add(bb.mul(cy[0], Wc).getResult(0), bb.mul(xt[0], Wx).getResult(0)).getResult(0);
        const nc = bb.tanh(term).getResult(0);
        return [[nc], [nc]];
      });
      b.returnOp([s.getResult(0), s.getResult(1)]);
    });
    const inArrs = [
      Float32Array.from({ length: T * D }, (_, i) => 0.3 * Math.sin(i * 0.7)),
      Float32Array.from({ length: D }, (_, i) => 0.1 * i),
      Float32Array.from({ length: D }, (_, i) => 0.5 + 0.1 * i),
      Float32Array.from({ length: D }, (_, i) => 0.3 - 0.05 * i),
    ];
    const cf = compileGraph(mk(), CPUTarget());
    const fk = cf.listKernels()[0];
    const loss = (a) => {
      const oc = new Float32Array(D), oy = new Float32Array(T * D);
      cf.run(fk, ...a, oc, oy);
      let s = 0; for (const e of oc) s += e; for (const e of oy) s += e; return s;
    };

    const fb = mk();
    const r = new BackwardGraphBuilder().build(fb);
    const bwd = r.backwardFunc;
    expect(r.gradInputIndices).toEqual([0, 1, 2, 3]);
    const argId = new Map(fb.args.map((a, i) => [a.id, i]));
    const savedSrc = r.savedValues.map(sv => (argId.has(sv.id) ? argId.get(sv.id) : -1));
    const cb = compileGraph(bwd, CPUTarget());
    const bins = [new Float32Array(D), new Float32Array(T * D)];
    bins[0].fill(1); bins[1].fill(1);
    for (const s of savedSrc) bins.push(s >= 0 ? inArrs[s].slice() : new Float32Array(1));
    const bouts = bwd.outputTypes.map(o => new Float32Array(Math.max(numel(o.shape), 1)));
    cb.run(cb.listKernels()[0], ...bins, ...bouts);

    const eps = 1e-3;
    const numGrad = (idx) => {
      const base = inArrs[idx];
      const g = new Float32Array(base.length);
      for (let i = 0; i < base.length; i++) {
        const sv = base[i];
        const a = inArrs.map(x => x.slice());
        a[idx][i] = sv + eps; const lp = loss(a);
        a[idx][i] = sv - eps; const lm = loss(a);
        g[i] = (lp - lm) / (2 * eps);
      }
      return g;
    };
    for (let k = 0; k < r.gradInputIndices.length; k++) {
      const an = bouts[k], nu = numGrad(r.gradInputIndices[k]);
      let e = 0, sc = 0;
      for (let i = 0; i < an.length; i++) { e = Math.max(e, Math.abs(an[i] - nu[i])); sc = Math.max(sc, Math.abs(nu[i])); }
      expect(e / (sc + 1e-9)).toBeLessThan(1e-2);
    }
  });
});

describe('scaled_dot_product_attention forward + VJP', () => {
  const B = 1, H = 2, L = 3, D = 4, sc = 1 / Math.sqrt(D), shp = [B, H, L, D];
  const numelS = numel(shp);
  const mk = () => buildFunction('attn', [t(shp), t(shp), t(shp)], [t(shp)], (b, [q, k, v]) => {
    b.returnOp([b.scaledDotProductAttention(q, k, v, sc).getResult(0)]);
  });
  const rnd = (o) => Float32Array.from({ length: numelS }, (_, i) => Math.sin(i * 0.53 + o));

  it('forward matches a direct softmax(QKᵀ·scale)V reference', () => {
    const Q = rnd(1), K = rnd(7), V = rnd(13);
    const cf = compileGraph(mk(), CPUTarget());
    const o = new Float32Array(numelS);
    cf.run(cf.listKernels()[0], Q, K, V, o);
    const ref = new Float32Array(numelS);
    for (let bh = 0; bh < B * H; bh++) {
      const off = bh * L * D;
      for (let i = 0; i < L; i++) {
        const s = new Float32Array(L); let mx = -1e9;
        for (let j = 0; j < L; j++) { let d = 0; for (let dd = 0; dd < D; dd++) d += Q[off + i * D + dd] * K[off + j * D + dd]; s[j] = d * sc; if (s[j] > mx) mx = s[j]; }
        let sm = 0; for (let j = 0; j < L; j++) { s[j] = Math.exp(s[j] - mx); sm += s[j]; }
        for (let j = 0; j < L; j++) s[j] /= sm;
        for (let dd = 0; dd < D; dd++) { let oo = 0; for (let j = 0; j < L; j++) oo += s[j] * V[off + j * D + dd]; ref[off + i * D + dd] = oo; }
      }
    }
    let e = 0; for (let i = 0; i < numelS; i++) e = Math.max(e, Math.abs(o[i] - ref[i]));
    expect(e).toBeLessThan(1e-5);
  });

  it('VJP for Q, K, V matches finite differences', () => {
    const inArrs = [rnd(1), rnd(7), rnd(13)];
    const { analytic, gradInputIndices, numGrad } = gradOfSumLoss(mk, inArrs, shp);
    for (let k = 0; k < gradInputIndices.length; k++) {
      const an = analytic[k], nu = numGrad(gradInputIndices[k]);
      let e = 0, s = 0;
      for (let i = 0; i < an.length; i++) { e = Math.max(e, Math.abs(an[i] - nu[i])); s = Math.max(s, Math.abs(nu[i])); }
      expect(e / (s + 1e-9)).toBeLessThan(1e-2);
    }
  });
});

describe('pool2d VJP matches finite differences (non-overlapping)', () => {
  for (const ptype of ['avg', 'max']) {
    it(`${ptype} pool grad is correct`, () => {
      const N = 2, C = 2, H = 4, W = 4, k = 2, OH = H / k, OW = W / k;
      const xs = [N, C, H, W], os = [N, C, OH, OW];
      const mk = () => buildFunction('pl', [t(xs)], [t(os)], (b, [x]) => {
        b.returnOp([b.pool2d(x, ptype, [k, k], [k, k], [[0, 0], [0, 0]]).getResult(0)]);
      });
      const inArrs = [Float32Array.from({ length: numel(xs) }, (_, i) => Math.sin(i * 0.37) * 2)];
      const { analytic, gradInputIndices, numGrad } = gradOfSumLoss(mk, inArrs, os);
      const an = analytic[0], nu = numGrad(gradInputIndices[0]);
      let e = 0, sc = 0;
      for (let i = 0; i < an.length; i++) { e = Math.max(e, Math.abs(an[i] - nu[i])); sc = Math.max(sc, Math.abs(nu[i])); }
      expect(e / (sc + 1e-9)).toBeLessThan(1e-2);
    });
  }
});

describe('cond/if VJP routes gradient to the taken branch', () => {
  const D = 3;
  const mk = () => buildFunction('cond', [t([D]), t([D]), t([D]), t([])], [t([D])], (b, [x, W1, W2, s]) => {
    const zero = b.broadcast(b.scalarConstant(0, ScalarType.F32).getResult(0), [], []).getResult(0);
    const pred = b.compare(s, zero, 'gt').getResult(0);
    const ifop = b.ifOp(pred, [t([D])],
      (tb) => { tb.yieldOp([tb.mul(x, W1).getResult(0)]); },
      (eb) => { eb.yieldOp([eb.mul(x, W2).getResult(0)]); });
    b.returnOp([ifop.getResult(0)]);
  });

  function runGrads(sVal) {
    const arrs = [
      Float32Array.from({ length: D }, (_, i) => 0.5 + 0.2 * i),
      Float32Array.from({ length: D }, (_, i) => 1 + 0.1 * i),
      Float32Array.from({ length: D }, (_, i) => 2 - 0.1 * i),
      new Float32Array([sVal]),
    ];
    const cf = compileGraph(mk(), CPUTarget());
    const fk = cf.listKernels()[0];
    const loss = (a) => { const o = new Float32Array(D); cf.run(fk, ...a, o); let s = 0; for (const e of o) s += e; return s; };
    const fb = mk();
    const r = new BackwardGraphBuilder().build(fb);
    const argId = new Map(fb.args.map((a, i) => [a.id, i]));
    const savedSrc = r.savedValues.map(sv => (argId.has(sv.id) ? argId.get(sv.id) : -1));
    const cb = compileGraph(r.backwardFunc, CPUTarget());
    const bins = [new Float32Array(D)]; bins[0].fill(1);
    for (const s of savedSrc) bins.push(s >= 0 ? arrs[s].slice() : new Float32Array(1));
    const bouts = r.backwardFunc.outputTypes.map(o => new Float32Array(Math.max(numel(o.shape), 1)));
    cb.run(cb.listKernels()[0], ...bins, ...bouts);
    const numGrad = (idx) => {
      const base = arrs[idx], g = new Float32Array(base.length), eps = 1e-3;
      for (let i = 0; i < base.length; i++) {
        const sv = base[i]; const a = arrs.map(x => x.slice());
        a[idx][i] = sv + eps; const lp = loss(a); a[idx][i] = sv - eps; const lm = loss(a);
        g[i] = (lp - lm) / (2 * eps);
      }
      return g;
    };
    return { analytic: bouts, gradInputIndices: r.gradInputIndices, numGrad };
  }

  it('pred true: grads flow to W1, W2 grad is zero', () => {
    const { analytic, gradInputIndices, numGrad } = runGrads(1);
    for (let k = 0; k < gradInputIndices.length; k++) {
      const idx = gradInputIndices[k];
      if (idx === 3) continue;
      const an = analytic[k], nu = numGrad(idx);
      let e = 0, sc = 0;
      for (let i = 0; i < an.length; i++) { e = Math.max(e, Math.abs(an[i] - nu[i])); sc = Math.max(sc, Math.abs(nu[i])); }
      expect(e / (sc + 1e-9)).toBeLessThan(1e-2);
      if (idx === 2) for (const v of an) expect(Math.abs(v)).toBeLessThan(1e-6);
    }
  });
});

describe('conv2d VJP matches finite differences (stride 1)', () => {
  it('grads for input and weight are correct', () => {
    const N = 2, Cin = 2, Cout = 3, H = 5, W = 5, KH = 3, KW = 3, pad = 1;
    const xs = [N, Cin, H, W], ws = [Cout, Cin, KH, KW];
    const OH = H + 2 * pad - KH + 1, OW = W + 2 * pad - KW + 1, os = [N, Cout, OH, OW];
    const mkFwd = () => buildFunction('cv', [t(xs), t(ws)], [t(os)], (b, [x, w]) => {
      b.returnOp([b.conv(x, w, [1, 1], [[pad, pad], [pad, pad]]).getResult(0)]);
    });
    const inArrs = [
      Float32Array.from({ length: numel(xs) }, (_, i) => Math.sin(i * 0.3)),
      Float32Array.from({ length: numel(ws) }, (_, i) => Math.cos(i * 0.4)),
    ];
    const { analytic, gradInputIndices, numGrad } = gradOfSumLoss(mkFwd, inArrs, os);
    expect(gradInputIndices.length).toBe(2);
    for (let k = 0; k < gradInputIndices.length; k++) {
      const an = analytic[k], nu = numGrad(gradInputIndices[k]);
      let e = 0, sc = 0;
      for (let i = 0; i < an.length; i++) { e = Math.max(e, Math.abs(an[i] - nu[i])); sc = Math.max(sc, Math.abs(nu[i])); }
      expect(e / (sc + 1e-9)).toBeLessThan(1e-2);
    }
  });
});
