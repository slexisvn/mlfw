import { describe, it, expect } from 'vitest';
import { tensor, Parameter } from '../../src/index.js';
import { SGD } from '../../src/optim/sgd.js';
import { StepLR, CosineAnnealingLR, ReduceLROnPlateau } from '../../src/optim/lr_scheduler.js';

function makeOptimizer(lr = 0.1) {
  const p = new Parameter(tensor([1.0]));
  return new SGD([p], { lr });
}

describe('StepLR', () => {
  it('decays LR by gamma every stepSize epochs', () => {
    const opt = makeOptimizer(0.1);
    const scheduler = new StepLR(opt, 3, 0.5);

    expect(opt.paramGroups[0].lr).toBeCloseTo(0.1);
    scheduler.step();
    expect(opt.paramGroups[0].lr).toBeCloseTo(0.1);
    scheduler.step();
    expect(opt.paramGroups[0].lr).toBeCloseTo(0.1);
    scheduler.step();
    expect(opt.paramGroups[0].lr).toBeCloseTo(0.05);
    scheduler.step();
    expect(opt.paramGroups[0].lr).toBeCloseTo(0.05);
    scheduler.step();
    expect(opt.paramGroups[0].lr).toBeCloseTo(0.05);
    scheduler.step();
    expect(opt.paramGroups[0].lr).toBeCloseTo(0.025);
  });

  it('getLastLR returns current LRs', () => {
    const opt = makeOptimizer(0.1);
    const scheduler = new StepLR(opt, 5, 0.1);
    expect(scheduler.getLastLR()).toEqual([0.1]);
    for (let i = 0; i < 5; i++) scheduler.step();
    expect(scheduler.getLastLR()[0]).toBeCloseTo(0.01);
  });
});

describe('CosineAnnealingLR', () => {
  it('starts at baseLR and ends at etaMin', () => {
    const opt = makeOptimizer(0.1);
    const scheduler = new CosineAnnealingLR(opt, 10, 0.001);

    expect(opt.paramGroups[0].lr).toBeCloseTo(0.1);
    for (let i = 0; i < 10; i++) scheduler.step();
    expect(opt.paramGroups[0].lr).toBeCloseTo(0.001, 4);
  });

  it('at tMax/2 is approximately midpoint', () => {
    const opt = makeOptimizer(1.0);
    const scheduler = new CosineAnnealingLR(opt, 100, 0.0);

    for (let i = 0; i < 50; i++) scheduler.step();
    expect(opt.paramGroups[0].lr).toBeCloseTo(0.5, 1);
  });
});

describe('ReduceLROnPlateau', () => {
  it('reduces LR after patience bad epochs', () => {
    const opt = makeOptimizer(0.1);
    const scheduler = new ReduceLROnPlateau(opt, { patience: 3, factor: 0.5 });

    scheduler.step(1.0);
    scheduler.step(1.0);
    scheduler.step(1.0);
    scheduler.step(1.0);
    expect(opt.paramGroups[0].lr).toBeCloseTo(0.1);

    scheduler.step(1.0);
    expect(opt.paramGroups[0].lr).toBeCloseTo(0.05);
  });

  it('does not reduce when metric improves', () => {
    const opt = makeOptimizer(0.1);
    const scheduler = new ReduceLROnPlateau(opt, { patience: 2, mode: 'min' });

    scheduler.step(1.0);
    scheduler.step(0.9);
    scheduler.step(0.8);
    scheduler.step(0.7);
    expect(opt.paramGroups[0].lr).toBeCloseTo(0.1);
  });

  it('respects cooldown', () => {
    const opt = makeOptimizer(0.1);
    const scheduler = new ReduceLROnPlateau(opt, { patience: 1, factor: 0.5, cooldown: 2 });

    scheduler.step(1.0);
    scheduler.step(1.0);
    scheduler.step(1.0);
    expect(opt.paramGroups[0].lr).toBeCloseTo(0.05);

    scheduler.step(1.0);
    scheduler.step(1.0);
    expect(opt.paramGroups[0].lr).toBeCloseTo(0.05);

    scheduler.step(1.0);
    scheduler.step(1.0);
    scheduler.step(1.0);
    expect(opt.paramGroups[0].lr).toBeCloseTo(0.025);
  });

  it('respects minLR floor', () => {
    const opt = makeOptimizer(0.1);
    const scheduler = new ReduceLROnPlateau(opt, { patience: 0, factor: 0.01, minLR: 0.05 });

    scheduler.step(1.0);
    scheduler.step(1.0);
    expect(opt.paramGroups[0].lr).toBeCloseTo(0.05);

    scheduler.step(1.0);
    scheduler.step(1.0);
    expect(opt.paramGroups[0].lr).toBeCloseTo(0.05);
  });

  it('works in max mode', () => {
    const opt = makeOptimizer(0.1);
    const scheduler = new ReduceLROnPlateau(opt, { patience: 1, factor: 0.5, mode: 'max' });

    scheduler.step(0.5);
    scheduler.step(0.6);
    scheduler.step(0.6);
    expect(opt.paramGroups[0].lr).toBeCloseTo(0.1);

    scheduler.step(0.6);
    expect(opt.paramGroups[0].lr).toBeCloseTo(0.05);
  });

  it('throws when metric is missing', () => {
    const opt = makeOptimizer(0.1);
    const scheduler = new ReduceLROnPlateau(opt);
    expect(() => scheduler.step()).toThrow('requires a metric');
  });
});
