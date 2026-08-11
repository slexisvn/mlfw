import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { tensor, Linear, ReLU, MSELoss, Adam, TensorDataset, DataLoader, sum, sub, mul, manual_seed, unseed } from '../../src/index.js';
import { LightningModule, Trainer } from '../../src/lightning/index.js';

beforeEach(() => manual_seed(2));
afterEach(() => unseed());

class Net extends LightningModule {
  constructor() {
    super();
    this.l1 = new Linear(2, 8);
    this.r = new ReLU();
    this.l2 = new Linear(8, 1);
    this.lf = new MSELoss();
  }
  forward(x) { return this.l2.forward(this.r.forward(this.l1.forward(x))); }
  trainingStep(batch) {
    const [x, y] = batch;
    const loss = this.lf.forward(this.forward(x), y);
    this.log('train_loss', loss);
    return loss;
  }
  configureOptimizers() { return new Adam(this.parameters(), { lr: 0.03 }); }
}

const N = 64;
const xs = [], ys = [];
for (let i = 0; i < N; i++) { const a = ((i * 7) % 11) / 11, b = ((i * 3) % 13) / 13; xs.push(a, b); ys.push(a + b); }
const X = tensor(xs, { shape: [N, 2] });
const Y = tensor(ys, { shape: [N, 1] });
const loader = () => new DataLoader(new TensorDataset(tensor(xs, { shape: [N, 2] }), tensor(ys, { shape: [N, 1] })), { batchSize: 16 });
const mse = (m) => { const d = sub(m.forward(X), Y); return sum(mul(d, d)).item() / N; };
const quiet = { logger: false, enableCheckpointing: false, enableProgress: false };

describe('compiled training via Trainer(compile=true)', () => {
  it('separate mode converges and matches eager from the same init', async () => {
    const eager = new Net();
    const compiled = new Net();
    compiled.parameters();
    compiled.loadStateDict(eager.stateDict());
    const start = mse(eager);

    await new Trainer({ maxEpochs: 40, ...quiet }).fit(eager, loader());
    await new Trainer({ maxEpochs: 40, compile: true, ...quiet }).fit(compiled, loader());

    const eagerLoss = mse(eager);
    const compiledLoss = mse(compiled);
    expect(compiledLoss).toBeLessThan(start * 0.2);
    expect(compiledLoss).toBeLessThan(0.02);
    expect(Math.abs(compiledLoss - eagerLoss)).toBeLessThan(5e-3);
  });

  it('joint mode converges', async () => {
    const compiled = new Net();
    const start = mse(compiled);
    await new Trainer({ maxEpochs: 40, compile: true, compileMode: 'joint', ...quiet }).fit(compiled, loader());
    expect(mse(compiled)).toBeLessThan(start * 0.2);
  });
});
