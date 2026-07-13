import { describe, it, expect } from 'vitest';
import { tensor, Linear, Sequential, ModuleList, ModuleDict } from '../../src/index.js';
import { full } from '../../src/tensor/factory/creation_ops.js';
import { add, mul } from '../../src/tensor/ops/ops.js';
import { constant_ } from '../../src/nn/init.js';
import { Module } from '../../src/nn/module.js';

class DoubleModule extends Module {
  forward(input) {
    return mul(input, full(input.shape, 2, { dtype: input.dtype }));
  }
}

class AddOneModule extends Module {
  forward(input) {
    return add(input, full(input.shape, 1, { dtype: input.dtype }));
  }
}

describe('Sequential chains forward calls', () => {
  it('applies modules in order', () => {
    const seq = new Sequential(new DoubleModule(), new AddOneModule());
    const input = tensor([3]);
    const out = seq.forward(input);
    expect(out.item()).toBeCloseTo(7);
  });

  it('chains three Linear layers', () => {
    const l1 = new Linear(2, 2, false);
    const l2 = new Linear(2, 2, false);
    const l3 = new Linear(2, 1, false);
    constant_(l1.weight, 1);
    constant_(l2.weight, 1);
    constant_(l3.weight, 1);

    const seq = new Sequential(l1, l2, l3);
    const input = tensor([[1, 1]]);
    const out = seq.forward(input);
    expect(out.item()).toBeCloseTo(8);
  });
});

describe('Sequential collects all child parameters', () => {
  it('returns parameters from all children', () => {
    const seq = new Sequential(new Linear(2, 3), new Linear(3, 1));
    const params = [...seq.parameters()];
    expect(params.length).toBe(4);
  });
});

describe('Sequential push adds module', () => {
  it('push extends the chain', () => {
    const seq = new Sequential(new DoubleModule());
    seq.push(new AddOneModule());
    const input = tensor([5]);
    const out = seq.forward(input);
    expect(out.item()).toBeCloseTo(11);
  });
});

describe('Sequential iterator', () => {
  it('iterates over child modules in order', () => {
    const d = new DoubleModule();
    const a = new AddOneModule();
    const seq = new Sequential(d, a);
    const children = [...seq];
    expect(children[0]).toBe(d);
    expect(children[1]).toBe(a);
  });
});

describe('ModuleList collects parameters from children', () => {
  it('exposes all child parameters', () => {
    const list = new ModuleList([new Linear(2, 3), new Linear(3, 1)]);
    const params = [...list.parameters()];
    expect(params.length).toBe(4);
  });

  it('forward throws', () => {
    const list = new ModuleList([new Linear(2, 3)]);
    expect(() => list.forward()).toThrow(/ModuleList/);
  });
});

describe('ModuleDict', () => {
  it('collects parameters from children', () => {
    const dict = new ModuleDict({
      encoder: new Linear(4, 3),
      decoder: new Linear(3, 4),
    });
    const params = [...dict.parameters()];
    expect(params.length).toBe(4);
  });

  it('forward throws', () => {
    const dict = new ModuleDict({ a: new Linear(2, 3) });
    expect(() => dict.forward()).toThrow(/ModuleDict/);
  });

  it('set and get work correctly', () => {
    const dict = new ModuleDict({});
    const lin = new Linear(2, 3);
    dict.set('layer', lin);
    expect(dict.get('layer')).toBe(lin);
    expect(dict.has('layer')).toBe(true);
  });
});
