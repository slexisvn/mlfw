import { describe, it, expect } from 'vitest';
import { BenchmarkRunner } from '../../../src/compiler/autotune/benchmark.js';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import { CPUTarget } from '../../../src/backend/target.js';

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
