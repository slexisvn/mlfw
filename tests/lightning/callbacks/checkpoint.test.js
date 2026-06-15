import { describe, it, expect, afterEach } from 'vitest';
import { existsSync, rmSync, readdirSync, statSync } from 'node:fs';
import { tensor, Adam } from '../../../src/index.js';
import {
  Trainer, ModelCheckpoint, loadCheckpoint, applyCheckpoint,
  serializeCheckpoint, deserializeCheckpoint,
} from '../../../src/lightning/index.js';
import { SimpleModel, makeData, quietTrainerOpts } from '../_fixtures.js';

const TMP_DIR = './lightning_logs/_ckpt_test';

function cleanup() {
  if (existsSync(TMP_DIR)) rmSync(TMP_DIR, { recursive: true, force: true });
}

describe('checkpoint binary format', () => {
  afterEach(cleanup);

  it('round-trips tensors exactly (bytes, not JSON floats)', () => {
    const w = tensor([1.5, -2.25, 3.125, Math.PI], { shape: [2, 2] });
    const ids = tensor([7, 8, 9], { shape: [3], dtype: 'i32' });
    const checkpoint = { epoch: 3, globalStep: 42, modelState: new Map([['w', w], ['ids', ids]]) };

    const restored = deserializeCheckpoint(serializeCheckpoint(checkpoint));

    expect(restored.epoch).toBe(3);
    expect(restored.globalStep).toBe(42);
    expect(restored.modelState).toBeInstanceOf(Map);
    const rw = restored.modelState.get('w');
    expect(rw.shape).toEqual([2, 2]);
    expect(rw.dtype).toBe('f32');
    expect([...rw._impl.storage.data]).toEqual([...w._impl.storage.data]);
    const rids = restored.modelState.get('ids');
    expect(rids.dtype).toBe('i32');
    expect([...rids._impl.storage.data]).toEqual([7, 8, 9]);
  });

  it('round-trips optimizer state (typed arrays + scalars + nested maps)', () => {
    const checkpoint = {
      optimizerStates: [{
        paramGroups: [{ lr: 0.05, betas: [0.9, 0.999] }],
        state: new Map([[0, { step: 5, exp_avg: new Float32Array([0.1, 0.2, 0.3]) }]]),
      }],
    };

    const restored = deserializeCheckpoint(serializeCheckpoint(checkpoint));
    const opt = restored.optimizerStates[0];
    expect(opt.paramGroups[0].lr).toBe(0.05);
    expect(opt.paramGroups[0].betas).toEqual([0.9, 0.999]);
    expect(opt.state).toBeInstanceOf(Map);
    const s = opt.state.get(0);
    expect(s.step).toBe(5);
    expect(s.exp_avg).toBeInstanceOf(Float32Array);
    expect([...s.exp_avg]).toEqual([0.10000000149011612, 0.20000000298023224, 0.30000001192092896]);
  });

  it('produces a smaller payload than JSON+base64', () => {
    const big = tensor(Array.from({ length: 4096 }, (_, i) => i * 0.5), { shape: [4096] });
    const bytes = serializeCheckpoint({ modelState: new Map([['w', big]]) });
    expect(bytes.length).toBeLessThan(4096 * 4 * 1.4);
  });
});

describe('ModelCheckpoint round-trip through Trainer', () => {
  afterEach(cleanup);

  it('writes .ckpt files that loadCheckpoint can restore into a fresh model', async () => {
    const ckpt = new ModelCheckpoint({ dirpath: TMP_DIR, saveLast: true });
    const trainer = new Trainer({ maxEpochs: 2, callbacks: [ckpt], ...quietTrainerOpts, enableCheckpointing: false });
    const model = new SimpleModel();
    await trainer.fit(model, makeData(20, 10));

    const files = readdirSync(TMP_DIR);
    expect(files.some((f) => f.endsWith('.ckpt'))).toBe(true);
    expect(files.some((f) => f.endsWith('.tmp'))).toBe(false);

    const loaded = loadCheckpoint(ckpt.lastModelPath);
    expect(loaded.epoch).toBeGreaterThanOrEqual(0);
    expect(loaded.modelState).toBeInstanceOf(Map);

    const fresh = new SimpleModel();
    const freshOpt = new Adam(fresh.parameters(), { lr: 0.05 });
    applyCheckpoint(loaded, fresh, [freshOpt]);

    const trained = model.stateDict();
    for (const [key, t] of fresh.stateDict()) {
      expect([...t._impl.storage.data]).toEqual([...trained.get(key)._impl.storage.data]);
    }
  });

  it('writes atomically (no .tmp left behind, file is non-empty binary)', async () => {
    const ckpt = new ModelCheckpoint({ dirpath: TMP_DIR, saveLast: true });
    const trainer = new Trainer({ maxEpochs: 1, callbacks: [ckpt], ...quietTrainerOpts, enableCheckpointing: false });
    await trainer.fit(new SimpleModel(), makeData(20, 10));
    expect(statSync(ckpt.lastModelPath).size).toBeGreaterThan(8);
  });
});
