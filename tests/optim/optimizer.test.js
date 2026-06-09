import { describe, it, expect } from 'vitest';
import { tensor, Parameter, Linear, Sequential, ReLU } from '../../src/index.js';
import { Optimizer } from '../../src/optim/optimizer.js';

function makeParams() {
  const p1 = new Parameter(tensor([1, 2, 3]));
  const p2 = new Parameter(tensor([4, 5]));
  return [p1, p2];
}

class DummyOptimizer extends Optimizer {
  constructor(params, defaults = {}) {
    super(params, { lr: 0.1, ...defaults });
  }
  step() {
    for (const group of this._paramGroups) {
      for (const p of group.params) {
        if (p.grad === null) continue;
        const w = p._impl.storage.data;
        const g = p.grad._impl.storage.data;
        for (let i = 0; i < w.length; i++) w[i] -= group.lr * g[i];
        p._impl.bumpVersion();
      }
    }
  }
}

describe('Optimizer parameter groups', () => {
  it('wraps a flat parameter array into one group', () => {
    const params = makeParams();
    const opt = new DummyOptimizer(params);
    expect(opt.paramGroups).toHaveLength(1);
    expect(opt.paramGroups[0].params).toHaveLength(2);
    expect(opt.paramGroups[0].lr).toBe(0.1);
  });

  it('accepts a generator of parameters', () => {
    const model = new Sequential(new Linear(3, 2), new ReLU(), new Linear(2, 1));
    const opt = new DummyOptimizer(model.parameters());
    const totalParams = opt.paramGroups[0].params.length;
    expect(totalParams).toBeGreaterThan(0);
  });

  it('accepts explicit parameter groups with overrides', () => {
    const [p1, p2] = makeParams();
    const opt = new DummyOptimizer([
      { params: [p1], lr: 0.01 },
      { params: [p2], lr: 0.5 },
    ]);
    expect(opt.paramGroups).toHaveLength(2);
    expect(opt.paramGroups[0].lr).toBe(0.01);
    expect(opt.paramGroups[1].lr).toBe(0.5);
  });

  it('fills missing keys from defaults', () => {
    const [p1, p2] = makeParams();
    const opt = new DummyOptimizer([
      { params: [p1] },
      { params: [p2], lr: 0.5 },
    ]);
    expect(opt.paramGroups[0].lr).toBe(0.1);
    expect(opt.paramGroups[1].lr).toBe(0.5);
  });

  it('throws when a parameter appears in multiple groups', () => {
    const [p1] = makeParams();
    expect(() => new DummyOptimizer([
      { params: [p1] },
      { params: [p1] },
    ])).toThrow('more than one');
  });

  it('throws on empty parameter list', () => {
    expect(() => new DummyOptimizer([])).toThrow('empty');
  });
});

describe('Optimizer zeroGrad', () => {
  it('sets grad to null when setToNone is true', () => {
    const p = new Parameter(tensor([1, 2, 3]));
    p.grad = tensor([0.1, 0.2, 0.3]);
    const opt = new DummyOptimizer([p]);
    opt.zeroGrad(true);
    expect(p.grad).toBeNull();
  });

  it('zeros grad data when setToNone is false', () => {
    const p = new Parameter(tensor([1, 2, 3]));
    p.grad = tensor([0.1, 0.2, 0.3]);
    const opt = new DummyOptimizer([p]);
    opt.zeroGrad(false);
    expect(p.grad).not.toBeNull();
    expect([...p.grad._impl.storage.data]).toEqual([0, 0, 0]);
  });

  it('skips parameters with null grad', () => {
    const p = new Parameter(tensor([1, 2]));
    const opt = new DummyOptimizer([p]);
    expect(() => opt.zeroGrad()).not.toThrow();
  });
});

describe('Optimizer stateDict / loadStateDict', () => {
  it('round-trips state correctly', () => {
    const [p1, p2] = makeParams();
    const opt = new DummyOptimizer([
      { params: [p1], lr: 0.01 },
      { params: [p2], lr: 0.05 },
    ]);
    const id1 = opt._getParamId(p1);
    opt._state.set(id1, { step: 5, buf: new Float32Array([1, 2, 3]) });

    const dict = opt.stateDict();
    expect(dict.paramGroups).toHaveLength(2);
    expect(dict.paramGroups[0].lr).toBe(0.01);
    expect(dict.state.get(id1).step).toBe(5);

    dict.state.get(id1).buf[0] = 999;
    expect(opt._state.get(id1).buf[0]).toBe(1);

    const opt2 = new DummyOptimizer([
      { params: [p1], lr: 0.99 },
      { params: [p2], lr: 0.99 },
    ]);
    opt2.loadStateDict(dict);
    expect(opt2.paramGroups[0].lr).toBe(0.01);
    expect(opt2.paramGroups[1].lr).toBe(0.05);

    const restored = opt2._state.get(opt2._getParamId(p1));
    expect(restored.step).toBe(5);
    expect([...restored.buf]).toEqual([999, 2, 3]);
  });
});

describe('Optimizer abstract step', () => {
  it('base class throws on step()', () => {
    const params = makeParams();
    const opt = new Optimizer(params, { lr: 0.1 });
    expect(() => opt.step()).toThrow('not implemented');
  });
});
