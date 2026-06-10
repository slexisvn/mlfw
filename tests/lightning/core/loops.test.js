import { describe, it, expect } from 'vitest';
import { Linear, MSELoss, SGD, Adam, StepLR, ReduceLROnPlateau } from '../../../src/index.js';
import { LightningModule, Trainer, Callback } from '../../../src/lightning/index.js';
import { SimpleModel, makeData, quietTrainerOpts } from '../_fixtures.js';

describe('Trainer.fit (training loop)', () => {
  it('decreases the training loss over epochs', async () => {
    const model = new SimpleModel();
    const trainer = new Trainer({ maxEpochs: 25, ...quietTrainerOpts });

    let firstLoss = null;
    let lastLoss = null;
    trainer.callbackConnector.add(new (class extends Callback {
      onTrainBatchEnd(_t, _m, outputs) {
        const v = typeof outputs === 'number' ? outputs : outputs.item();
        if (firstLoss === null) firstLoss = v;
        lastLoss = v;
      }
    })());

    await trainer.fit(model, makeData());
    expect(lastLoss).toBeLessThan(firstLoss);
  });

  it('stops at maxSteps regardless of remaining epochs', async () => {
    const model = new SimpleModel();
    const trainer = new Trainer({ maxEpochs: 100, maxSteps: 5, ...quietTrainerOpts });
    await trainer.fit(model, makeData());
    expect(trainer.globalStep).toBeLessThanOrEqual(6);
    expect(trainer.globalStep).toBeGreaterThanOrEqual(5);
  });

  it('fastDevRun=N limits training to N batches in a single epoch', async () => {
    const model = new SimpleModel();
    const trainer = new Trainer({ fastDevRun: 2, ...quietTrainerOpts });
    let batches = 0;
    let epochs = 0;
    trainer.callbackConnector.add(new (class extends Callback {
      onTrainBatchEnd() { batches++; }
      onTrainEpochStart() { epochs++; }
    })());
    await trainer.fit(model, makeData());
    expect(batches).toBe(2);
    expect(epochs).toBe(1);
  });

  it('increments globalStep once per processed batch', async () => {
    const model = new SimpleModel();
    const trainer = new Trainer({ maxEpochs: 2, ...quietTrainerOpts });
    await trainer.fit(model, makeData(40, 10));
    expect(trainer.globalStep).toBe(8);
  });

  it('dispatches lifecycle hooks in the expected order', async () => {
    const events = [];
    const recorder = new (class extends Callback {
      onFitStart() { events.push('fitStart'); }
      onTrainStart() { events.push('trainStart'); }
      onTrainEpochStart() { events.push('epochStart'); }
      onTrainEpochEnd() { events.push('epochEnd'); }
      onTrainEnd() { events.push('trainEnd'); }
      onFitEnd() { events.push('fitEnd'); }
    })();
    const trainer = new Trainer({ maxEpochs: 1, callbacks: [recorder], ...quietTrainerOpts });
    await trainer.fit(new SimpleModel(), makeData(10, 10));
    expect(events).toEqual(['fitStart', 'trainStart', 'epochStart', 'epochEnd', 'trainEnd', 'fitEnd']);
  });

  it('supports manual optimization (model drives backward/step)', async () => {
    const steps = [];
    class ManualModel extends LightningModule {
      constructor() {
        super();
        this.linear = new Linear(2, 1);
        this.lossFn = new MSELoss();
        this.automaticOptimization = false;
      }
      forward(x) { return this.linear.forward(x); }
      trainingStep(batch) {
        const [x, y] = batch;
        const loss = this.lossFn.forward(this.forward(x), y);
        this.optimizers[0].zeroGrad();
        this.manualBackward(loss);
        this.optimizers[0].step();
        steps.push(loss.item());
        return { loss };
      }
      configureOptimizers() { return new SGD(this.parameters(), { lr: 0.05 }); }
    }
    const trainer = new Trainer({ maxEpochs: 15, ...quietTrainerOpts });
    await trainer.fit(new ManualModel(), makeData());
    expect(steps.length).toBeGreaterThan(0);
    expect(steps[steps.length - 1]).toBeLessThan(steps[0]);
  });
});

describe('Validation loop', () => {
  it('runs validation once per epoch when a val loader is given', async () => {
    const model = new SimpleModel();
    let valStarts = 0;
    const counter = new (class extends Callback {
      onValidationStart() { valStarts++; }
    })();
    const trainer = new Trainer({ maxEpochs: 3, callbacks: [counter], ...quietTrainerOpts });
    await trainer.fit(model, makeData(20), makeData(10));
    expect(valStarts).toBe(3);
  });

  it('honors checkValEveryNEpoch', async () => {
    const model = new SimpleModel();
    let valStarts = 0;
    const counter = new (class extends Callback {
      onValidationStart() { valStarts++; }
    })();
    const trainer = new Trainer({ maxEpochs: 4, checkValEveryNEpoch: 2, callbacks: [counter], ...quietTrainerOpts });
    await trainer.fit(model, makeData(20), makeData(10));
    expect(valStarts).toBe(2);
  });
});

describe('LR scheduler integration', () => {
  it('steps an epoch StepLR scheduler after each epoch', async () => {
    let optRef;
    class M extends SimpleModel {
      configureOptimizers() {
        optRef = new Adam(this.parameters(), { lr: 0.1 });
        return { optimizer: optRef, lrScheduler: { scheduler: new StepLR(optRef, 1, 0.1), interval: 'epoch' } };
      }
    }
    const trainer = new Trainer({ maxEpochs: 2, ...quietTrainerOpts });
    await trainer.fit(new M(), makeData(20, 10));
    expect(optRef.paramGroups[0].lr).toBeCloseTo(0.001, 6);
  });

  it('steps a plateau scheduler with the monitored metric', async () => {
    let optRef;
    class M extends SimpleModel {
      validationStep() { this.log('val_loss', 5.0); }
      configureOptimizers() {
        optRef = new Adam(this.parameters(), { lr: 0.1 });
        return {
          optimizer: optRef,
          lrScheduler: {
            scheduler: new ReduceLROnPlateau(optRef, { patience: 0, factor: 0.5 }),
            interval: 'epoch',
            monitor: 'val_loss',
          },
        };
      }
    }
    const trainer = new Trainer({ maxEpochs: 3, ...quietTrainerOpts });
    await trainer.fit(new M(), makeData(20, 10), makeData(10, 10));
    expect(optRef.paramGroups[0].lr).toBeLessThan(0.1);
  });
});

describe('Trainer.validate and predict', () => {
  it('validate returns the aggregated epoch metrics', async () => {
    const model = new SimpleModel();
    const trainer = new Trainer(quietTrainerOpts);
    const metrics = await trainer.validate(model, makeData(20, 10));
    expect(metrics).toHaveProperty('val_loss');
    expect(typeof metrics.val_loss).toBe('number');
  });

  it('predict collects one output per batch', async () => {
    const model = new SimpleModel();
    const trainer = new Trainer(quietTrainerOpts);
    const preds = await trainer.predict(model, makeData(30, 10));
    expect(preds).toHaveLength(3);
  });
});
