import { describe, it, expect, vi } from 'vitest';
import {
  tensor, add, matmul, sub, mul, mean, sum,
  Linear, Parameter, MSELoss, CrossEntropyLoss,
  Adam, SGD, StepLR, ReduceLROnPlateau,
  TensorDataset, DataLoader,
} from '../../src/index.js';
import {
  LightningModule, Trainer, Callback,
  EarlyStopping, ModelCheckpoint, ProgressCallback,
  LearningRateMonitor, Timer, GradientAccumulationScheduler,
  ConsoleLogger, CSVLogger,
  MeanMetric, SumMetric, Accuracy, Precision, Recall, F1Score,
  ConfusionMatrix, MetricCollection,
  TrainerState, MetricAccumulator, Stage,
} from '../../src/lightning/index.js';

class SimpleModel extends LightningModule {
  constructor() {
    super();
    this.linear = new Linear(2, 1);
    this.lossFn = new MSELoss();
  }

  forward(x) {
    return this.linear.forward(x);
  }

  trainingStep(batch, batchIdx) {
    const [x, y] = batch;
    const pred = this.forward(x);
    const loss = this.lossFn.forward(pred, y);
    this.log('train_loss', loss, { progBar: true, onStep: true, onEpoch: true });
    return loss;
  }

  validationStep(batch, batchIdx) {
    const [x, y] = batch;
    const pred = this.forward(x);
    const loss = this.lossFn.forward(pred, y);
    this.log('val_loss', loss, { onEpoch: true });
  }

  configureOptimizers() {
    return new Adam(this.parameters(), { lr: 0.01 });
  }
}

function makeData(n = 20) {
  const xData = [];
  const yData = [];
  for (let i = 0; i < n; i++) {
    const x1 = Math.random();
    const x2 = Math.random();
    xData.push(x1, x2);
    yData.push(x1 + x2);
  }
  const x = tensor(xData, { shape: [n, 2] });
  const y = tensor(yData, { shape: [n, 1] });
  return new DataLoader(new TensorDataset(x, y), { batchSize: 10 });
}

describe('TrainerState', () => {
  it('initializes with defaults', () => {
    const state = new TrainerState();
    expect(state.stage).toBe(Stage.IDLE);
    expect(state.epoch).toBe(0);
    expect(state.globalStep).toBe(0);
    expect(state.shouldStop).toBe(false);
  });

  it('MetricAccumulator computes mean', () => {
    const acc = new MetricAccumulator();
    acc.update('loss', 1.0);
    acc.update('loss', 3.0);
    expect(acc.compute('loss')).toBe(2.0);
  });

  it('MetricAccumulator computes sum', () => {
    const acc = new MetricAccumulator();
    acc.update('count', 5, 'sum');
    acc.update('count', 3, 'sum');
    expect(acc.compute('count')).toBe(8);
  });

  it('MetricAccumulator computes min/max', () => {
    const acc = new MetricAccumulator();
    acc.update('val', 5, 'min');
    acc.update('val', 2, 'min');
    acc.update('val', 8, 'min');
    expect(acc.compute('val')).toBe(2);

    const acc2 = new MetricAccumulator();
    acc2.update('val', 5, 'max');
    acc2.update('val', 2, 'max');
    acc2.update('val', 8, 'max');
    expect(acc2.compute('val')).toBe(8);
  });
});

describe('LightningModule', () => {
  it('extends Module', () => {
    const model = new SimpleModel();
    expect(model.training).toBe(true);
    const params = [...model.parameters()];
    expect(params.length).toBeGreaterThan(0);
  });

  it('logs to buffer', () => {
    const model = new SimpleModel();
    model.log('test_metric', 0.5, { onStep: true, onEpoch: false });
    expect(model._logBuffer.has('test_metric')).toBe(true);
    expect(model._logBuffer.get('test_metric').value).toBe(0.5);
  });
});

describe('Trainer.fit', () => {
  it('trains and loss decreases', async () => {
    const model = new SimpleModel();
    const trainLoader = makeData(40);
    const trainer = new Trainer({
      maxEpochs: 20,
      logger: false,
      enableCheckpointing: false,
      enableProgress: false,
    });

    let firstLoss = null;
    let lastLoss = null;
    const lossTracker = new (class extends Callback {
      onTrainBatchEnd(trainer, model, outputs) {
        const v = typeof outputs === 'number' ? outputs : outputs.item();
        if (firstLoss === null) firstLoss = v;
        lastLoss = v;
      }
    })();
    trainer.callbackConnector.add(lossTracker);

    await trainer.fit(model, trainLoader);
    expect(lastLoss).toBeLessThan(firstLoss);
  });

  it('respects maxSteps', async () => {
    const model = new SimpleModel();
    const trainLoader = makeData(40);
    const trainer = new Trainer({
      maxEpochs: 100,
      maxSteps: 5,
      logger: false,
      enableCheckpointing: false,
      enableProgress: false,
    });

    await trainer.fit(model, trainLoader);
    expect(trainer.globalStep).toBeLessThanOrEqual(6);
  });

  it('fastDevRun limits batches', async () => {
    const model = new SimpleModel();
    const trainLoader = makeData(40);
    const trainer = new Trainer({
      fastDevRun: 2,
      logger: false,
      enableCheckpointing: false,
      enableProgress: false,
    });

    await trainer.fit(model, trainLoader);
    expect(trainer.globalStep).toBeLessThanOrEqual(2);
  });
});

describe('Trainer.validate', () => {
  it('runs validation loop', async () => {
    const model = new SimpleModel();
    const valLoader = makeData(20);
    const trainer = new Trainer({
      logger: false,
      enableCheckpointing: false,
      enableProgress: false,
    });

    model.validationStep = function(batch, batchIdx) {
      const [x, y] = batch;
      const pred = this.forward(x);
      const loss = this.lossFn.forward(pred, y);
      this.log('val_loss', loss, { onEpoch: true });
    };

    const metrics = await trainer.validate(model, valLoader);
    expect(metrics).toBeDefined();
  });
});

describe('Trainer with validation', () => {
  it('runs val after each epoch', async () => {
    const model = new SimpleModel();
    const trainLoader = makeData(20);
    const valLoader = makeData(10);
    let valCount = 0;
    const counter = new (class extends Callback {
      onValidationStart() { valCount++; }
    })();

    const trainer = new Trainer({
      maxEpochs: 3,
      logger: false,
      enableCheckpointing: false,
      enableProgress: false,
      callbacks: [counter],
    });

    await trainer.fit(model, trainLoader, valLoader);
    expect(valCount).toBe(3);
  });
});

describe('configureOptimizers formats', () => {
  it('handles {optimizer, lrScheduler}', async () => {
    class ModelWithScheduler extends SimpleModel {
      configureOptimizers() {
        const optimizer = new Adam(this.parameters(), { lr: 0.01 });
        return {
          optimizer,
          lrScheduler: {
            scheduler: new StepLR(optimizer, 5),
            interval: 'epoch',
          },
        };
      }
    }
    const model = new ModelWithScheduler();
    const trainLoader = makeData(20);
    const trainer = new Trainer({
      maxEpochs: 2,
      logger: false,
      enableCheckpointing: false,
      enableProgress: false,
    });
    await trainer.fit(model, trainLoader);
    expect(trainer.globalStep).toBeGreaterThan(0);
  });
});

describe('Manual optimization', () => {
  it('allows manual backward and step', async () => {
    class ManualModel extends LightningModule {
      constructor() {
        super();
        this.linear = new Linear(2, 1);
        this.lossFn = new MSELoss();
        this.automaticOptimization = false;
      }

      forward(x) { return this.linear.forward(x); }

      trainingStep(batch, batchIdx) {
        const [x, y] = batch;
        const pred = this.forward(x);
        const loss = this.lossFn.forward(pred, y);
        this.optimizers[0].zeroGrad();
        this.manualBackward(loss);
        this.optimizers[0].step();
        this.log('loss', loss, { onStep: true });
        return { loss };
      }

      configureOptimizers() {
        return new SGD(this.parameters(), { lr: 0.01 });
      }
    }

    const model = new ManualModel();
    const trainLoader = makeData(20);
    const trainer = new Trainer({
      maxEpochs: 3,
      logger: false,
      enableCheckpointing: false,
      enableProgress: false,
    });
    await trainer.fit(model, trainLoader);
    expect(trainer.globalStep).toBeGreaterThan(0);
  });
});

describe('EarlyStopping', () => {
  it('stops training when patience exceeded', async () => {
    const es = new EarlyStopping({ monitor: 'val_loss', patience: 2, mode: 'min' });

    class ESModel extends SimpleModel {
      validationStep(batch, batchIdx) {
        this.log('val_loss', 1.0, { onEpoch: true });
      }
    }

    const model = new ESModel();
    const trainLoader = makeData(20);
    const valLoader = makeData(10);
    const trainer = new Trainer({
      maxEpochs: 100,
      logger: false,
      enableCheckpointing: false,
      enableProgress: false,
      callbacks: [es],
    });

    await trainer.fit(model, trainLoader, valLoader);
    expect(trainer.state.epoch).toBeLessThan(100);
  });
});

describe('Metrics', () => {
  it('MeanMetric computes average', () => {
    const m = new MeanMetric();
    m.update(1);
    m.update(3);
    m.update(5);
    expect(m.compute()).toBe(3);
  });

  it('SumMetric computes total', () => {
    const m = new SumMetric();
    m.update(2);
    m.update(4);
    expect(m.compute()).toBe(6);
  });

  it('Accuracy multiclass', () => {
    const acc = new Accuracy({ task: 'multiclass', numClasses: 3 });
    const preds = tensor([0.9, 0.05, 0.05, 0.1, 0.8, 0.1, 0.1, 0.1, 0.8], { shape: [3, 3] });
    const target = tensor([0, 1, 2], { shape: [3] });
    acc.update(preds, target);
    expect(acc.compute()).toBe(1.0);
  });

  it('Accuracy binary', () => {
    const acc = new Accuracy({ task: 'binary' });
    const preds = tensor([0.9, 0.1, 0.8, 0.3]);
    const target = tensor([1, 0, 1, 0]);
    acc.update(preds, target);
    expect(acc.compute()).toBe(1.0);
  });

  it('ConfusionMatrix', () => {
    const cm = new ConfusionMatrix({ numClasses: 2 });
    const preds = tensor([0.9, 0.1, 0.2, 0.8, 0.8, 0.2], { shape: [3, 2] });
    const target = tensor([0, 1, 0]);
    cm.update(preds, target);
    const matrix = cm.compute();
    expect(matrix[0][0]).toBe(2);
    expect(matrix[1][1]).toBe(1);
  });

  it('MetricCollection', () => {
    const collection = new MetricCollection({
      accuracy: new Accuracy({ task: 'multiclass', numClasses: 2 }),
      mean: new MeanMetric(),
    });
    const preds = tensor([0.9, 0.1, 0.2, 0.8], { shape: [2, 2] });
    const target = tensor([0, 1]);
    collection.get('accuracy').update(preds, target);
    collection.get('mean').update(0.5);
    const results = collection.compute();
    expect(results.accuracy).toBe(1.0);
    expect(results.mean).toBe(0.5);
  });

  it('F1Score multiclass', () => {
    const f1 = new F1Score({ task: 'multiclass', numClasses: 2, average: 'macro' });
    const preds = tensor([0.9, 0.1, 0.2, 0.8, 0.8, 0.2, 0.1, 0.9], { shape: [4, 2] });
    const target = tensor([0, 1, 0, 1]);
    f1.update(preds, target);
    expect(f1.compute()).toBe(1.0);
  });
});

describe('Timer callback', () => {
  it('records training time', async () => {
    const timer = new Timer();
    const model = new SimpleModel();
    const trainLoader = makeData(20);
    const trainer = new Trainer({
      maxEpochs: 2,
      logger: false,
      enableCheckpointing: false,
      enableProgress: false,
      callbacks: [timer],
    });
    await trainer.fit(model, trainLoader);
    expect(timer.totalTrainingTime).toBeGreaterThan(0);
    expect(timer.epochDurations.length).toBe(2);
  });
});

describe('ConsoleLogger', () => {
  it('logs metrics to console', () => {
    const spy = vi.spyOn(console, 'log').mockImplementation(() => {});
    const logger = new ConsoleLogger();
    logger.logMetrics({ loss: 0.5, acc: 0.9 }, 10);
    expect(spy).toHaveBeenCalled();
    spy.mockRestore();
  });
});

describe('GradientAccumulationScheduler', () => {
  it('updates accumulation at scheduled epochs', () => {
    const scheduler = new GradientAccumulationScheduler({
      scheduling: { 0: 4, 2: 2, 4: 1 },
    });
    expect(scheduler.getCurrentAccumulation(0)).toBe(4);
    expect(scheduler.getCurrentAccumulation(1)).toBe(4);
    expect(scheduler.getCurrentAccumulation(2)).toBe(2);
    expect(scheduler.getCurrentAccumulation(3)).toBe(2);
    expect(scheduler.getCurrentAccumulation(4)).toBe(1);
  });
});

describe('LearningRateMonitor', () => {
  it('records lr history', async () => {
    const lrm = new LearningRateMonitor();
    const model = new SimpleModel();
    const trainLoader = makeData(20);
    const trainer = new Trainer({
      maxEpochs: 2,
      logger: false,
      enableCheckpointing: false,
      enableProgress: false,
      callbacks: [lrm],
    });
    await trainer.fit(model, trainLoader);
    expect(lrm.lrHistory['lr']).toBeDefined();
    expect(lrm.lrHistory['lr'].length).toBeGreaterThan(0);
  });
});
