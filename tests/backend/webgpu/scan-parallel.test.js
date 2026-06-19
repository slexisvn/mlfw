import { describe, it, expect } from 'vitest';
import * as nn from '../../../src/index.js';
import { trace } from '../../../src/index.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { WebGPUTarget } from '../../../src/backend/target.js';

function lstmSource(E, H, T, B, kind = 'lstm') {
  const Mod = kind === 'gru' ? nn.GRU : nn.LSTM;
  const m = new Mod(E, H, 1, false);
  const inp = nn.tensor(Array.from({ length: T }, () =>
    Array.from({ length: B }, () => Array.from({ length: E }, (_, k) => 0.1 * (k + 1)))));
  const mod = trace((...a) => m.forward(...a)[0], [inp], { name: kind });
  const func = mod.functions().next().value;
  const result = compileGraph(func, WebGPUTarget(), { scheduling: { enabled: true } });
  const kernels = result.listKernels();
  return { kernels, src: result.getSource(kernels[0]), plan: result.module.executionPlan, result };
}

function workgroupSize(src) {
  const m = src.match(/@workgroup_size\(([^)]*)\)/);
  return m ? m[1].split(',').map(s => parseInt(s.trim(), 10)) : null;
}

describe('scan compiles RNNs to ONE rolled WebGPU kernel (de-unroll)', () => {
  it('LSTM forward is a single kernel, not per-timestep unrolled', () => {
    const { kernels } = lstmSource(4, 4, 6, 2);
    expect(kernels.length).toBe(1);
  });

  it('GRU forward is a single kernel', () => {
    const { kernels } = lstmSource(8, 8, 5, 2, 'gru');
    expect(kernels.length).toBe(1);
  });
});

describe('parallel persistent-RNN kernel (single-workgroup)', () => {
  it('parallelizes over batch×gates: workgroup_size = B*4H, time loop stays serial', () => {
    const B = 2, H = 4;
    const { src } = lstmSource(4, H, 3, B);
    const wg = workgroupSize(src);
    expect(wg[0]).toBe(B * 4 * H);
    expect(wg[0]).toBeGreaterThan(1);
    expect(src).toMatch(/for \(var t_\d+: i32 = 0; t_\d+ < 3;/);
    expect(src).not.toMatch(/let t_\d+: i32 = i32\(_lid/);
  });

  it('keeps the loop-carried state (h/c) in workgroup memory with barriers', () => {
    const { src } = lstmSource(4, 4, 3, 2);
    expect(src).toMatch(/var<workgroup>/);
    expect(src).toMatch(/workgroupBarrier\(\);/);
  });

  it('GRU parallelizes the same way', () => {
    const B = 2, H = 8;
    const { src } = lstmSource(8, H, 4, B, 'gru');
    const wg = workgroupSize(src);
    expect(wg[0]).toBe(B * 3 * H);
    expect(src).toMatch(/for \(var t_\d+: i32 = 0; t_\d+ </);
  });

  it('thread-strides past 256: B*4H=1024 runs parallel on a capped workgroup with a reused wg pool', () => {
    const { src } = lstmSource(16, 32, 4, 8);
    const wg = workgroupSize(src);
    expect(wg[0]).toBeGreaterThan(1);
    expect(wg[0]).toBeLessThanOrEqual(256);
    expect(src).toMatch(/var<workgroup> _wg_f32: array<f32, \d+>;/);
    expect(src).toMatch(/for \(var t_\d+: i32 = 0; t_\d+ </);
  });

  it('switches to per-step host-loop dispatch when the scan exceeds the smem budget', () => {
    const { plan, result } = lstmSource(32, 64, 4, 8);
    expect(plan).toBeTruthy();
    expect(plan.scanLoop).toBeTruthy();
    expect(plan.scanLoop.T).toBe(4);
    expect(plan.scanLoop.carry.length).toBe(2);
    expect(plan.scanLoop.carry.every(c => c.a !== c.b)).toBe(true);
    expect(plan.scanLoop.loopEnd).toBeGreaterThan(plan.scanLoop.loopStart);
    expect(result.listKernels().length).toBeGreaterThan(1);
    const parallel = result.listKernels().filter(n => workgroupSize(result.getSource(n))[0] > 1);
    expect(parallel.length).toBeGreaterThan(0);
  });
});

describe('thread-distributed local intermediates are scalar-replaced (no huge private arrays)', () => {
  it('fused elementwise kernels use scalar registers, not full-size per-thread arrays', () => {
    const { plan, result } = lstmSource(64, 96, 4, 64);
    expect(plan.scanLoop).toBeTruthy();
    for (const name of result.listKernels()) {
      const src = result.getSource(name);
      const bigPrivate = [...src.matchAll(/var (_lt\d+): array<\w+, (\d+)>/g)].filter(m => Number(m[2]) > 256);
      expect(bigPrivate.length, `${name} emits oversized private array(s): ${bigPrivate.map(m => m[0]).join(', ')}`).toBe(0);
    }
    const anyScalar = result.listKernels().some(n => /var _s\d+: f32;/.test(result.getSource(n)));
    expect(anyScalar).toBe(true);
  });
});
