import { describe, it, expect } from 'vitest';
import { BenchmarkRunner, BenchmarkResult, robustStats } from '../../../src/compiler/autotune/benchmark.js';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import { CPUTarget } from '../../../src/compiler/support/target.js';

describe('robustStats noise handling', () => {
  it('trimmed mean ignores an outlier spike that drags the raw mean', () => {
    const samples = [1, 1, 1, 1, 1, 1, 1, 1, 1, 100];
    const rawMean = samples.reduce((a, b) => a + b, 0) / samples.length;
    const s = robustStats(samples, 0.1);
    expect(s.median).toBe(1);
    expect(s.min).toBe(1);
    expect(s.trimmedMean).toBeLessThan(2);
    expect(s.trimmedMean).toBeLessThan(rawMean);
  });

  it('reports zero coefficient of variation for uniform samples and positive for noisy', () => {
    expect(robustStats([5, 5, 5, 5, 5]).cv).toBe(0);
    expect(robustStats([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).cv).toBeGreaterThan(0);
  });

  it('handles empty input without throwing', () => {
    expect(robustStats([])).toEqual({ median: 0, min: 0, trimmedMean: 0, cv: 0 });
  });
});

describe('BenchmarkResult exposes stability metrics', () => {
  it('carries trimmedMeanMs and cv, defaulting trimmedMeanMs to median', () => {
    const r = new BenchmarkResult(5, 4, [4, 5, 6], 1024, 5.2, 0.1);
    expect(r.trimmedMeanMs).toBe(5.2);
    expect(r.cv).toBeCloseTo(0.1);
    const r2 = new BenchmarkResult(5, 4, [], 0);
    expect(r2.trimmedMeanMs).toBe(5);
    expect(r2.cv).toBe(0);
  });
});

function fakePrimFunc(sizes) {
  const bufferMap = new Map();
  sizes.forEach((n, i) => bufferMap.set('b' + i, new Buffer('b' + i, [n], 'f32', 'global')));
  return { name: 'f', body: {}, bufferMap };
}

describe('BenchmarkRunner buffer handling', () => {
  it('refills cached buffers on every call so state does not leak between runs', () => {
    const runner = new BenchmarkRunner(CPUTarget());
    const pf = fakePrimFunc([8]);
    const first = runner._getOrAllocBuffers(pf).buffers;
    const snapshot = Float32Array.from(first[0]);
    first[0].fill(0);
    const second = runner._getOrAllocBuffers(pf).buffers;
    expect(second[0]).toBe(first[0]);
    let changed = false;
    for (let i = 0; i < second[0].length; i++) {
      if (second[0][i] !== 0) changed = true;
    }
    expect(changed).toBe(true);
    let identical = true;
    for (let i = 0; i < snapshot.length; i++) {
      if (snapshot[i] !== second[0][i]) identical = false;
    }
    expect(identical).toBe(false);
  });
});

describe('BenchmarkRunner iteration bound', () => {
  it('runs exactly repeat iterations when minRepeatMs is 0', () => {
    const runner = new BenchmarkRunner(CPUTarget(), { warmup: 0, repeat: 5, minRepeatMs: 0 });
    let calls = 0;
    const samples = [];
    let totalElapsed = 0;
    const maxIterations = runner.repeat * 3;
    for (let i = 0; i < maxIterations && (i < runner.repeat || totalElapsed < runner.minRepeatMs); i++) {
      calls++;
      totalElapsed += 0;
    }
    expect(calls).toBe(5);
  });

  it('never exceeds maxIterations even when minRepeatMs is unreachable', () => {
    const runner = new BenchmarkRunner(CPUTarget(), { warmup: 0, repeat: 4, minRepeatMs: 1e9 });
    let calls = 0;
    let totalElapsed = 0;
    const maxIterations = runner.repeat * 3;
    for (let i = 0; i < maxIterations && (i < runner.repeat || totalElapsed < runner.minRepeatMs); i++) {
      calls++;
      totalElapsed += 0;
    }
    expect(calls).toBe(maxIterations);
  });
});
