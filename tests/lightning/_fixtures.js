import {
  tensor, Linear, MSELoss, Adam,
  TensorDataset, DataLoader,
} from '../../src/index.js';
import { LightningModule, TrainerState } from '../../src/lightning/index.js';

export class SimpleModel extends LightningModule {
  constructor() {
    super();
    this.linear = new Linear(2, 1);
    this.lossFn = new MSELoss();
  }

  forward(x) {
    return this.linear.forward(x);
  }

  trainingStep(batch) {
    const [x, y] = batch;
    const loss = this.lossFn.forward(this.forward(x), y);
    this.log('train_loss', loss, { progBar: true });
    return loss;
  }

  validationStep(batch) {
    const [x, y] = batch;
    const loss = this.lossFn.forward(this.forward(x), y);
    this.log('val_loss', loss);
  }

  predictStep(batch) {
    const [x] = batch;
    return this.forward(x);
  }

  configureOptimizers() {
    return new Adam(this.parameters(), { lr: 0.05 });
  }
}

export function makeData(n = 40, batchSize = 10) {
  const xData = [];
  const yData = [];
  for (let i = 0; i < n; i++) {
    const x1 = ((i * 7) % 11) / 11;
    const x2 = ((i * 3) % 13) / 13;
    xData.push(x1, x2);
    yData.push(x1 + x2);
  }
  const x = tensor(xData, { shape: [n, 2] });
  const y = tensor(yData, { shape: [n, 1] });
  return new DataLoader(new TensorDataset(x, y), { batchSize });
}

export const quietTrainerOpts = {
  logger: false,
  enableCheckpointing: false,
  enableProgress: false,
};

export function fakeTrainerWithMetric(name, value) {
  const state = new TrainerState();
  if (value !== undefined) state.epochMetrics.update(name, value);
  return { state, shouldStop: false };
}
