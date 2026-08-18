import { describe, it, expect } from 'vitest';
import {
  BASELINE, DEFAULT_MIN_GAIN, optimizationCandidates, selectWinner, candidateByName,
  gateCacheKey, graphSignature,
} from '../../../src/compiler/pipeline/opt_gate.js';
import { CPUTarget, CUDATarget, WasmTarget, archSupportsTensorCore, archMajor } from '../../../src/backend/target.js';

describe('tensor-core support is derived from the compute capability', () => {
  it('reads the major version out of an sm_XY string', () => {
    expect(archMajor('sm_86')).toBe(8);
    expect(archMajor('sm_70')).toBe(7);
    expect(archMajor('sm_61')).toBe(6);
    expect(archMajor(null)).toBeNull();
    expect(archMajor('turing')).toBeNull();
  });

  it('Volta and newer have tensor cores, Pascal and older do not', () => {
    expect(archSupportsTensorCore('sm_86')).toBe(true);
    expect(archSupportsTensorCore('sm_75')).toBe(true);
    expect(archSupportsTensorCore('sm_70')).toBe(true);
    expect(archSupportsTensorCore('sm_61')).toBe(false);
    expect(archSupportsTensorCore('sm_52')).toBe(false);
    expect(archSupportsTensorCore(null)).toBe(false);
  });

  it('a CUDA target built for a real Ampere device reports tensor cores', () => {
    expect(CUDATarget({ arch: 'sm_86' }).supportsTensorCore).toBe(true);
    expect(CUDATarget({ arch: 'sm_60' }).supportsTensorCore).toBe(false);
  });

  it('an explicit override still wins over the derived value', () => {
    expect(CUDATarget({ arch: 'sm_86', supportsTensorCore: false }).supportsTensorCore).toBe(false);
    expect(CUDATarget({ arch: 'sm_60', supportsTensorCore: true }).supportsTensorCore).toBe(true);
  });

  it('a target with no arch stays conservative', () => {
    expect(CUDATarget().supportsTensorCore).toBe(false);
  });
});

describe('candidate enumeration only proposes what the target can run', () => {
  it('offers layout on a CPU target that supports blocked layouts', () => {
    const names = optimizationCandidates(CPUTarget()).map(c => c.name);
    expect(names).toEqual(['layout']);
  });

  it('offers nothing on a target with neither blocked layout nor tensor cores', () => {
    expect(optimizationCandidates(WasmTarget())).toEqual([]);
  });

  it('offers tensorize only when the GPU actually has tensor cores', () => {
    expect(optimizationCandidates(CUDATarget()).map(c => c.name)).toEqual([]);
    expect(optimizationCandidates(CUDATarget({ arch: 'sm_86' })).map(c => c.name)).toContain('tensorize');
  });

  it('offers the combined candidate when both levers are available', () => {
    const names = optimizationCandidates(CUDATarget({ arch: 'sm_86', supportsBlockedLayout: true })).map(c => c.name);
    expect(names).toEqual(['layout', 'tensorize', 'layout+tensorize']);
  });

  it('returns no candidates for a null target', () => {
    expect(optimizationCandidates(null)).toEqual([]);
  });

  it('candidate overrides name the flags they turn on', () => {
    const cands = optimizationCandidates(CUDATarget({ arch: 'sm_86', supportsBlockedLayout: true }));
    expect(candidateByName(cands, 'layout').optimization).toEqual({ layout: true });
    expect(candidateByName(cands, 'tensorize').optimization).toEqual({ tensorize: true });
    expect(candidateByName(cands, 'layout+tensorize').optimization).toEqual({ layout: true, tensorize: true });
    expect(candidateByName(cands, BASELINE)).toBeNull();
  });
});

describe('the gate never picks a configuration that is not a measured win', () => {
  const m = (name, ms, correct = true) => ({ name, ms, correct });

  it('keeps the baseline when no candidate is faster', () => {
    const d = selectWinner([m(BASELINE, 10), m('layout', 12), m('tensorize', 11)]);
    expect(d.winner).toBe(BASELINE);
    expect(d.gain).toBe(1);
  });

  it('keeps the baseline when the win is inside the noise margin', () => {
    const d = selectWinner([m(BASELINE, 10), m('layout', 9.8)]);
    expect(d.winner).toBe(BASELINE);
  });

  it('takes a candidate that clears the margin', () => {
    const d = selectWinner([m(BASELINE, 10), m('layout', 5)]);
    expect(d.winner).toBe('layout');
    expect(d.gain).toBeCloseTo(2, 6);
    expect(d.winnerMs).toBe(5);
  });

  it('ignores a faster candidate that produced wrong numbers', () => {
    const d = selectWinner([m(BASELINE, 10), m('tensorize', 1, false), m('layout', 8)]);
    expect(d.winner).toBe('layout');
    expect(d.winnerMs).toBe(8);
  });

  it('falls back to the baseline when the only fast candidate is wrong', () => {
    const d = selectWinner([m(BASELINE, 10), m('tensorize', 1, false)]);
    expect(d.winner).toBe(BASELINE);
    expect(d.gain).toBe(1);
  });

  it('picks the fastest among several correct wins', () => {
    const d = selectWinner([m(BASELINE, 10), m('layout', 7), m('tensorize', 4), m('layout+tensorize', 5)]);
    expect(d.winner).toBe('tensorize');
  });

  it('ignores candidates that failed to time', () => {
    const d = selectWinner([m(BASELINE, 10), m('layout', 0), m('tensorize', -1)]);
    expect(d.winner).toBe(BASELINE);
  });

  it('honours a custom margin', () => {
    expect(selectWinner([m(BASELINE, 10), m('layout', 9)], 1.05).winner).toBe('layout');
    expect(selectWinner([m(BASELINE, 10), m('layout', 9)], 1.5).winner).toBe(BASELINE);
  });

  it('refuses to decide without a correct baseline', () => {
    expect(() => selectWinner([m('layout', 5)])).toThrow(/baseline/);
    expect(() => selectWinner([m(BASELINE, 10, false), m('layout', 5)])).toThrow(/baseline/);
  });

  it('the default margin is a real margin, not zero', () => {
    expect(DEFAULT_MIN_GAIN).toBeGreaterThan(1);
  });
});

describe('gate decisions are keyed so they can be reused', () => {
  it('the signature separates different graphs and different input shapes', () => {
    const a = graphSignature(['matmul', 'relu'], [[1, 4]]);
    const b = graphSignature(['matmul', 'gelu'], [[1, 4]]);
    const c = graphSignature(['matmul', 'relu'], [[8, 4]]);
    expect(a).not.toBe(b);
    expect(a).not.toBe(c);
    expect(graphSignature(['matmul', 'relu'], [[1, 4]])).toBe(a);
  });

  it('the cache key separates targets and candidate sets', () => {
    const sig = graphSignature(['matmul'], [[1, 4]]);
    const cands = optimizationCandidates(CUDATarget({ arch: 'sm_86', supportsBlockedLayout: true }));
    expect(gateCacheKey(sig, 'cuda', cands)).not.toBe(gateCacheKey(sig, 'cpu', cands));
    expect(gateCacheKey(sig, 'cuda', cands)).not.toBe(gateCacheKey(sig, 'cuda', cands.slice(0, 1)));
    expect(gateCacheKey(sig, 'cuda', cands)).toBe(gateCacheKey(sig, 'cuda', cands));
  });
});
