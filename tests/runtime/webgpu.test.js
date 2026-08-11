import { describe, it, expect } from 'vitest';
import { prewarmPipelines } from '../../src/runtime/webgpu.js';

function mockKernel(name) {
  return { name, source: '@compute @workgroup_size(1,1,1) fn ' + name + '(){}', metadata: { bindings: [] } };
}

describe('prewarmPipelines (async pipeline pre-creation for fast first-compile)', () => {
  it('creates every pipeline concurrently via createComputePipelineAsync', async () => {
    let started = 0, active = 0, maxConcurrent = 0;
    const device = {
      createShaderModule: () => ({}),
      createBindGroupLayout: () => ({}),
      createPipelineLayout: () => ({}),
      createComputePipelineAsync: () => {
        started++; active++; maxConcurrent = Math.max(maxConcurrent, active);
        return new Promise(resolve => setTimeout(() => { active--; resolve({}); }, 10));
      },
    };
    const kernels = [mockKernel('k0'), mockKernel('k1'), mockKernel('k2'), mockKernel('k3')];
    await prewarmPipelines(device, kernels);
    expect(started).toBe(4);
    expect(maxConcurrent).toBeGreaterThan(1);
  });

  it('compiles each unique kernel once (dedups repeats within a plan)', async () => {
    let started = 0;
    const device = {
      createShaderModule: () => ({}), createBindGroupLayout: () => ({}), createPipelineLayout: () => ({}),
      createComputePipelineAsync: () => { started++; return Promise.resolve({}); },
    };
    const shared = mockKernel('body');
    await prewarmPipelines(device, [shared, shared, shared]);
    expect(started).toBe(1);
  });

  it('falls back without error when createComputePipelineAsync is unavailable', async () => {
    const device = { createShaderModule: () => ({}), createBindGroupLayout: () => ({}), createPipelineLayout: () => ({}) };
    await expect(prewarmPipelines(device, [mockKernel('a')])).resolves.toBeUndefined();
  });

  it('swallows a failed async compile so prewarm never rejects (lazy sync path takes over later)', async () => {
    const device = {
      createShaderModule: () => ({}), createBindGroupLayout: () => ({}), createPipelineLayout: () => ({}),
      createComputePipelineAsync: () => Promise.reject(new Error('FXC boom')),
    };
    await expect(prewarmPipelines(device, [mockKernel('bad')])).resolves.toBeUndefined();
  });
});
