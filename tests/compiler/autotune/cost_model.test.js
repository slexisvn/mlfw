import { describe, it, expect } from 'vitest';
import { LearnedCostModel, AnalyticalCostModel } from '../../../src/compiler/autotune/cost_model.js';
import { CPUTarget } from '../../../src/backend/target.js';

const feat = (...v) => [v];

describe('LearnedCostModel (MLP) captures non-linear interactions', () => {
  it('fits XOR, whose optimal linear regression provably has MSE 0.25', () => {
    const data = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]];
    const m = new LearnedCostModel(null, { seed: 1 });
    for (const [a, b, y] of data) m.addSample(feat(a, b), y);
    m.train();

    let mse = 0;
    for (const [a, b, y] of data) {
      const p = m.predict(feat(a, b));
      mse += (p - y) ** 2;
    }
    mse /= data.length;
    expect(mse).toBeLessThan(0.05);
  });

  it('predicts 0 before training and reports trained status', () => {
    const m = new LearnedCostModel();
    expect(m.trained).toBe(false);
    expect(m.predict(feat(1, 2))).toBe(0);
    m.addSample(feat(1, 2), 0.5);
    m.addSample(feat(2, 1), 1.5);
    m.train();
    expect(m.trained).toBe(true);
  });

  it('round-trips through serialize/deserialize with identical predictions', () => {
    const m = new LearnedCostModel(null, { seed: 3 });
    for (let i = 0; i < 24; i++) {
      const a = i % 5;
      const b = (i * 3) % 7;
      m.addSample(feat(a, b), a * b - a + 0.5 * b);
    }
    m.train();

    const m2 = LearnedCostModel.deserialize(m.serialize());
    expect(m2.trained).toBe(true);
    for (let i = 0; i < 6; i++) {
      const f = feat(i, i + 1);
      expect(m2.predict(f)).toBeCloseTo(m.predict(f), 9);
    }
  });

  it('training is deterministic for a fixed seed', () => {
    const build = () => {
      const m = new LearnedCostModel(null, { seed: 9 });
      for (let i = 0; i < 16; i++) m.addSample(feat(i % 4, (i * 2) % 5), (i % 4) * ((i * 2) % 5));
      m.train();
      return m.predict(feat(3, 2));
    };
    expect(build()).toBe(build());
  });
});

describe('AnalyticalCostModel still scores schedules', () => {
  it('produces a finite score with a breakdown', () => {
    const cm = new AnalyticalCostModel(CPUTarget());
    const est = cm.estimateFromFeatures({
      threadBlockSize: 0, gridSize: 0, numParallelLoops: 1, numLoops: 2,
      numVectorizedLoops: 1, innermostExtent: 8, strideOneAccesses: 4, nonStrideOneAccesses: 1,
      numSerialLoops: 1, arithmeticIntensity: 0.5, numMathOps: 4, numExternCalls: 0,
    });
    expect(Number.isFinite(est.score)).toBe(true);
    expect(est.breakdown).toHaveProperty('parallelism');
  });
});
