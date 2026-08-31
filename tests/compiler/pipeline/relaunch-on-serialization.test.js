import { describe, it, expect } from 'vitest';
import { tensor } from '../../../src/index.js';
import * as nn from '../../../src/nn/index.js';
import { compile } from '../../../src/tracing/compile.js';
import { CPUTarget, CUDATarget } from '../../../src/compiler/support/target.js';
import { TraceLevel } from '../../../src/compiler/support/trace.js';

function rng(s) { let x = s >>> 0; return () => { x = (x * 1664525 + 1013904223) >>> 0; return x / 4294967296; }; }
function data(r, sh) {
  const n = sh.reduce((a, b) => a * b, 1); const f = [];
  for (let i = 0; i < n; i++) f.push(r() * 2 - 1);
  const nest = (fl, s) => s.length === 1 ? fl.slice(0, s[0]) : Array.from({ length: s[0] }, (_, i) => nest(fl.slice(i * fl.length / s[0], (i + 1) * fl.length / s[0]), s.slice(1)));
  return nest(f, sh);
}

function compileWithTrace(fwd, input, target) {
  const events = [];
  const compiled = compile({ forward: fwd }, [input], {
    target,
    scheduling: { enabled: true },
    trace: { level: TraceLevel.DEBUG, sink: (e) => events.push(e) },
  });
  const module = compiled.result().module;
  const kernels = module.listKernels().map((name) => {
    const md = module.getKernelMetadata(name);
    const threads = md.blockDim
      ? md.blockDim.reduce((a, b) => a * b, 1) * md.gridDim.reduce((a, b) => a * b, 1)
      : null;
    return { name, threads, serialized: !!md.launchDiagnosis, diagnosis: md.launchDiagnosis };
  });
  return {
    kernels,
    serializeWarnings: events.filter((e) => e.type === 'warning' && e.phase === 'codegen'),
    relaunchDecisions: events.filter((e) => e.type === 'explain' && e.category === 'relaunch'),
    relaunchRefusals: events.filter((e) => e.type === 'warning' && e.phase === 'relaunch'),
  };
}

function normConvGraph() {
  const gn = new nn.GroupNorm(4, 8); gn.eval();
  const cv = new nn.Conv2d(8, 8, 3, 1, 1); cv.eval();
  return { fwd: (a) => cv.forward(gn.forward(a)), input: tensor(data(rng(1), [1, 8, 8, 8])) };
}

function rnnGraph() {
  const lstm = new nn.LSTM(8, 32, 1, false, true); lstm.eval();
  return { fwd: (a) => lstm.forward(a)[0], input: tensor(data(rng(2), [5, 2, 8])) };
}

describe('a kernel that cannot run at its launch geometry is re-split, not silently serialized', () => {
  it('splits the fused kernel and recompiles', () => {
    const { fwd, input } = normConvGraph();
    const r = compileWithTrace(fwd, input, CUDATarget());

    expect(r.serializeWarnings.length).toBeGreaterThan(0);
    expect(r.relaunchDecisions.length).toBe(1);
    expect(r.relaunchDecisions[0].decision).toContain('split into separate kernels');
    expect(r.kernels.length).toBeGreaterThan(1);
  });

  it('leaves the majority of the work parallel instead of one single-thread kernel', () => {
    const { fwd, input } = normConvGraph();
    const r = compileWithTrace(fwd, input, CUDATarget());

    const parallel = r.kernels.filter((k) => k.threads > 1);
    expect(parallel.length).toBeGreaterThan(1);
    expect(Math.max(...r.kernels.map((k) => k.threads))).toBeGreaterThan(64);
  });

  it('retries at most once and reports what it could not fix', () => {
    const { fwd, input } = normConvGraph();
    const r = compileWithTrace(fwd, input, CUDATarget());

    expect(r.relaunchDecisions.length).toBeLessThanOrEqual(1);
    const stillSerialized = r.kernels.filter((k) => k.serialized);
    if (stillSerialized.length > 0) {
      expect(r.relaunchRefusals.length).toBe(1);
      expect(r.relaunchRefusals[0].message).toContain('keeping serialized kernel');
      for (const k of stillSerialized) expect(k.diagnosis.reason).toBeTruthy();
    }
  });
});

describe('sequential control flow is never re-split', () => {
  it('keeps an RNN scan in one kernel and says why', () => {
    const { fwd, input } = rnnGraph();
    const r = compileWithTrace(fwd, input, CUDATarget());

    expect(r.relaunchDecisions.length).toBe(0);
    if (r.serializeWarnings.length > 0) {
      expect(r.relaunchRefusals.length).toBeGreaterThan(0);
      expect(r.relaunchRefusals.some((e) => e.message.includes('sequential recurrence'))).toBe(true);
    }
  });
});

describe('targets without a launch geometry are untouched', () => {
  it('never serializes or relaunches on CPU', () => {
    const { fwd, input } = normConvGraph();
    const r = compileWithTrace(fwd, input, CPUTarget());

    expect(r.serializeWarnings).toEqual([]);
    expect(r.relaunchDecisions).toEqual([]);
    expect(r.relaunchRefusals).toEqual([]);
    for (const k of r.kernels) expect(k.serialized).toBe(false);
  });
});
