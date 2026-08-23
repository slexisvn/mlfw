import { describe, it, expect } from 'vitest';
import { compile } from '../../../../src/index.js';
import { CPUTarget } from '../../../../src/backend/target.js';
import * as nn from '../../../../src/nn/index.js';
import { randn } from '../../../../src/tensor/factory/creation_ops.js';
import { manualSeed, unseed } from '../../../../src/util/random.js';
import { countLoops, countTempBuffers, stripLiterals } from '../../../_utils/kernel_source.js';
import { flat } from '../../../_utils/tensor_data.js';

class Head extends nn.Module {
  constructor(dim) {
    super();
    this.q = new nn.Linear(dim, dim);
    this.k = new nn.Linear(dim, dim);
    this.v = new nn.Linear(dim, dim);
    this.scale = 1 / Math.sqrt(dim);
  }
  forward(x) {
    const q = this.q.forward(x);
    const k = this.k.forward(x);
    const v = this.v.forward(x);
    const scores = q.matmul(k.transpose(-2, -1)).mul(this.scale);
    return nn.F.softmax(scores, -1).matmul(v);
  }
}

function build(model, x, opts = {}) {
  return compile(model, [x], { target: CPUTarget(), scheduling: { enabled: true }, ...opts });
}

function maxAbsDiff(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}

function maxRelDiff(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]) / (1 + Math.abs(b[i]));
    if (d > m) m = d;
  }
  return m;
}

/** Buffers written across a whole row of the reduced tensor, i.e. broadcast materializations. */
function fullSizeTemporaries(src, elements) {
  const s = stripLiterals(src);
  return [...s.matchAll(/new\s+Float32Array\((\d+)\)/g)]
    .map((m) => Number(m[1]))
    .filter((n) => n >= elements);
}

describe('reduction fusion lowering', () => {
  it('softmax lowers to one nest per pass, with no broadcast materialization', () => {
    manualSeed(7);
    const x = randn([4, 6, 8]);
    const src = build({ forward: (t) => nn.F.softmax(t, -1) }, x).source();
    unseed();

    // max pass, sum pass, normalise pass -- plus two row-sized init nests.
    expect(countLoops(src)).toBe(3 * 3 + 2 * 2);
    // Only the exp intermediate is score-sized: the row max/sum stay [4, 6].
    expect(fullSizeTemporaries(src, 4 * 6 * 8)).toEqual([4 * 6 * 8]);
    expect(countTempBuffers(src)).toBe(3);
    // The reduction accumulator stays a register across the fused prologue store,
    // which is what keeps the sum bit-identical to the unfused lowering.
    expect(src).toMatch(/_acc_\d+ = \(_acc_\d+ \+ cse_\d+\)/);
  });

  it('attention head keeps only the score tensor as a full-size temporary', () => {
    manualSeed(11);
    const model = new Head(16);
    const x = randn([2, 8, 16]);
    const src = build(model, x).source();
    unseed();

    const scoreElements = 2 * 8 * 8;
    expect(fullSizeTemporaries(src, scoreElements).filter((n) => n === scoreElements)).toEqual([scoreElements]);
  });

  it('elementwise-only fusion still lowers to a single inlined loop', () => {
    manualSeed(13);
    const model = new nn.Sequential(new nn.Linear(16, 32), new nn.ReLU());
    const x = randn([4, 16]);
    const src = build(model, x).source();
    unseed();

    expect(src).toMatch(/Math\.max\(.*\+/s);
    expect(countTempBuffers(src)).toBe(1);
  });

  it('reports what lowering did with each fusion region, not just the grouping', () => {
    manualSeed(17);
    const events = [];
    compile({ forward: (t) => nn.F.softmax(t, -1) }, [randn([2, 4, 5])], {
      target: CPUTarget(),
      trace: { level: 3, sink: (e) => { if (e.type === 'explain') events.push(e); } },
    });
    unseed();

    const grouping = events.filter((e) => e.category === 'fusion');
    const lowering = events.filter((e) => e.category === 'fusion.lowering');
    expect(grouping.length).toBeGreaterThan(0);
    expect(lowering.length).toBe(grouping.length);
    for (const e of grouping) expect(e.decision).toBe('grouped');
    for (const e of lowering) {
      expect(e.decision).toBe('reduction-nest');
      expect(e.reason).toBeTruthy();
    }
  });
});

describe('reduction fusion preserves results', () => {
  const REDUCTIONS = [
    { name: 'softmax', fn: (t) => nn.F.softmax(t, -1) },
    { name: 'log_softmax', fn: (t) => nn.F.log_softmax(t, -1) },
    { name: 'sum-normalised', fn: (t) => t.div(t.sum(-1, true)) },
    { name: 'mean-centred', fn: (t) => t.sub(t.mean(-1, true)) },
    { name: 'max-shifted', fn: (t) => t.sub(t.max(-1, true)) },
    { name: 'min-shifted', fn: (t) => t.sub(t.min(-1, true)) },
    { name: 'prod-normalised', fn: (t) => t.div(t.prod(-1, true)) },
    { name: 'leading-axis', fn: (t) => t.sub(t.sum(0, true)) },
    { name: 'two-axes', fn: (t) => t.sub(t.sum([1, 2], true)) },
    { name: 'all-axes', fn: (t) => t.div(t.sum()) },
    { name: 'yields both the reduced and the scaled tensor', fn: (t) => t.mul(2).sub(t.mul(2).max(-1, true)) },
  ];

  for (const c of REDUCTIONS) {
    it(`${c.name}: bit-identical with fusion and scheduling on or off`, () => {
      manualSeed(c.name.length * 13 + 5);
      const x = randn([2, 4, 5]).abs().add(0.5);
      const model = { forward: c.fn };
      const eager = flat(c.fn(x));
      const variants = [];
      for (const fusion of [true, false]) {
        for (const scheduling of [true, false]) {
          variants.push(flat(build(model, x, { fusion: { enabled: fusion }, scheduling: { enabled: scheduling } })(x)));
        }
      }
      unseed();
      for (const v of variants) expect(maxAbsDiff(variants[0], v)).toBe(0);
      expect(maxRelDiff(variants[0], eager)).toBeLessThan(1e-6);
    });
  }

  const SHAPES = [[2, 8, 16], [1, 1, 8], [3, 7, 16], [4, 16, 32]];
  for (const shape of SHAPES) {
    it(`attention head ${JSON.stringify(shape)}: bit-identical and within f32 of eager`, () => {
      manualSeed(shape[0] * 977 + shape[1] * 31 + shape[2]);
      const model = new Head(shape[2]);
      const x = randn(shape);
      const eager = flat(model.forward(x));
      const variants = [];
      for (const fusion of [true, false]) {
        for (const scheduling of [true, false]) {
          variants.push(flat(build(model, x, { fusion: { enabled: fusion }, scheduling: { enabled: scheduling } })(x)));
        }
      }
      unseed();
      for (const v of variants) expect(maxAbsDiff(variants[0], v)).toBe(0);
      expect(maxRelDiff(variants[0], eager)).toBeLessThan(1e-6);
    });
  }
});
