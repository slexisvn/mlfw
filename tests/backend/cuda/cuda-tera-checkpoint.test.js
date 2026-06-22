import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { existsSync, unlinkSync } from 'fs';
import { TeraRuntime } from '../../../src/cli/runtime.js';
import { preloadCudaRuntime } from '../../../src/runtime/backend_registry.js';

const CKPT = '_cuda_tera_ckpt.test.ckpt';
const flat = (t) => t.toArray().flat(Infinity);

describe('Tera: load a checkpoint onto CUDA and infer (.to(device="gpu"))', () => {
  beforeAll(async () => { await preloadCudaRuntime(); });
  afterAll(() => { if (existsSync(CKPT)) unlinkSync(CKPT); });

  it('save -> load -> .to(device="gpu") -> forward matches the host run', async () => {
    const rt = new TeraRuntime({ output: () => {} });
    const src = [
      'x = tensor([[1.0, 2.0, 3.0, 4.0], [0.5, -0.5, 0.5, -0.5], [2.0, 0.0, -1.0, 1.0]])',
      'net = Linear(4, 3)',
      'ycpu = net(x)',
      `save(net, "${CKPT}")`,
      'restored = Linear(4, 3)',
      `load(restored, "${CKPT}")`,
      'restored.eval()',
      'restored.to(device="gpu")',
      'ygpu = restored(x.to(device="gpu"))',
    ].join('\n');
    await rt.execute(src);

    const cpu = flat(rt.getVariable('ycpu'));
    const gpu = flat(rt.getVariable('ygpu'));
    expect(String(rt.getVariable('ygpu').device)).not.toBe('cpu');
    expect(gpu.length).toBe(cpu.length);
    let maxErr = 0;
    for (let i = 0; i < cpu.length; i++) maxErr = Math.max(maxErr, Math.abs(cpu[i] - gpu[i]) / (1 + Math.abs(cpu[i])));
    expect(maxErr, `cpu=${JSON.stringify(cpu)} gpu=${JSON.stringify(gpu)}`).toBeLessThan(1e-5);
  });
});
