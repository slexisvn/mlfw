import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtempSync, rmSync, existsSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import {
  tensor, Linear, ReLU, MSELoss, Adam, TensorDataset, DataLoader,
  compile, CPUTarget, sub, mul, sum, manual_seed, unseed,
} from '../../src/index.js';
import { LightningModule, Trainer, ModelCheckpoint, loadCheckpoint, applyCheckpoint } from '../../src/lightning/index.js';
import { flat } from '../_utils/tensor_data.js';

let dir;
beforeEach(() => {
  manual_seed(11);
  dir = mkdtempSync(join(tmpdir(), 'mlfw-journey-'));
});
afterEach(() => {
  unseed();
  rmSync(dir, { recursive: true, force: true });
});

class Regressor extends LightningModule {
  constructor() {
    super();
    this.l1 = new Linear(2, 16);
    this.act = new ReLU();
    this.l2 = new Linear(16, 1);
    this.loss = new MSELoss();
  }
  forward(x) { return this.l2.forward(this.act.forward(this.l1.forward(x))); }
  trainingStep(batch) {
    const [x, y] = batch;
    return this.loss.forward(this.forward(x), y);
  }
  configureOptimizers() { return new Adam(this.parameters(), { lr: 0.03 }); }
}

const N = 64;
const xs = [];
const ys = [];
for (let i = 0; i < N; i++) {
  const a = ((i * 7) % 11) / 11;
  const b = ((i * 3) % 13) / 13;
  xs.push(a, b);
  ys.push(2 * a - b);
}
const X = tensor(xs, { shape: [N, 2] });
const Y = tensor(ys, { shape: [N, 1] });
const loader = () => new DataLoader(new TensorDataset(tensor(xs, { shape: [N, 2] }), tensor(ys, { shape: [N, 1] })), { batchSize: 16 });
const mse = (m) => { const d = sub(m.forward(X), Y); return sum(mul(d, d)).item() / N; };
const quiet = { logger: false, enableProgress: false };

describe('journey: train -> checkpoint -> load -> infer', () => {
  it('a reloaded model reproduces the trained model exactly, eager and compiled', async () => {
    const model = new Regressor();
    const before = mse(model);

    const ckpt = new ModelCheckpoint({ dirpath: join(dir, 'ckpt'), saveLast: true });
    await new Trainer({ maxEpochs: 30, callbacks: [ckpt], enableCheckpointing: true, ...quiet }).fit(model, loader());

    const after = mse(model);
    expect(after, `training did not converge: ${before} -> ${after}`).toBeLessThan(before * 0.2);

    const path = ckpt.lastModelPath;
    expect(path, 'ModelCheckpoint did not record a checkpoint path').toBeTruthy();
    expect(existsSync(path)).toBe(true);

    const restored = new Regressor();
    restored.parameters();
    expect(mse(restored)).not.toBeCloseTo(after, 6);

    applyCheckpoint(loadCheckpoint(path), restored);

    expect(flat(restored.forward(X))).toEqual(flat(model.forward(X)));
    expect(mse(restored)).toBe(after);

    const compiled = compile(restored, [X], { target: CPUTarget() });
    const out = await compiled(X);
    const eager = flat(model.forward(X));
    for (let i = 0; i < eager.length; i++) expect(out.data[i]).toBeCloseTo(eager[i], 4);
  });

  it('optimizer state round-trips so resumed training keeps improving', async () => {
    const model = new Regressor();
    const ckpt = new ModelCheckpoint({ dirpath: join(dir, 'resume'), saveLast: true });
    await new Trainer({ maxEpochs: 10, callbacks: [ckpt], enableCheckpointing: true, ...quiet }).fit(model, loader());
    const mid = mse(model);

    const resumed = new Regressor();
    resumed.parameters();
    const opt = resumed.configureOptimizers();
    const payload = applyCheckpoint(loadCheckpoint(ckpt.lastModelPath), resumed, [opt]);

    expect(payload.optimizerStates, 'checkpoint carried no optimizer state').toBeTruthy();
    expect(mse(resumed)).toBe(mid);

    await new Trainer({ maxEpochs: 20, ...quiet }).fit(resumed, loader());
    expect(mse(resumed), `resumed training regressed: ${mid} -> ${mse(resumed)}`).toBeLessThan(mid);
  });
});
