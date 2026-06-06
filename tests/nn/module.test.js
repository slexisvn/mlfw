import { describe, it, expect } from 'vitest';
import { tensor, zeros, ones, sum, Linear, Sequential } from '../../src/index.js';
import { Module } from '../../src/nn/module.js';
import { Parameter } from '../../src/nn/parameter.js';
import { constant_ } from '../../src/nn/init.js';

describe('parameter recursion through nested modules', () => {
  it('collects parameters from all nested children', () => {
    const net = new Sequential(
      new Linear(4, 3),
      new Linear(3, 2),
    );
    const params = [...net.parameters()];
    expect(params.length).toBe(4);
  });

  it('non-recursive parameters returns only own params', () => {
    const net = new Sequential(
      new Linear(4, 3),
      new Linear(3, 2),
    );
    const ownParams = [...net.parameters(false)];
    expect(ownParams.length).toBe(0);
  });
});

describe('namedParameters prefixing', () => {
  it('produces dot-separated paths for nested modules', () => {
    const net = new Sequential(new Linear(2, 3));
    const names = [...net.namedParameters()].map(([name]) => name);
    expect(names).toContain('0.weight');
    expect(names).toContain('0.bias');
  });
});

describe('stateDict and loadStateDict round-trip', () => {
  it('loadStateDict copies data into existing parameters', () => {
    const net1 = new Linear(3, 2);
    constant_(net1.weight, 1);
    constant_(net1.bias, 2);

    const net2 = new Linear(3, 2);
    constant_(net2.weight, 0);
    constant_(net2.bias, 0);

    net2.loadStateDict(net1.stateDict());
    expect([...net2.weight.data].every(v => v === 1)).toBe(true);
    expect([...net2.bias.data].every(v => v === 2)).toBe(true);
  });

  it('stateDict contains both params and buffers', () => {
    const net = new Linear(3, 2);
    const dict = net.stateDict();
    expect(dict.has('weight')).toBe(true);
    expect(dict.has('bias')).toBe(true);
  });
});

describe('train and eval propagation', () => {
  it('eval sets training=false on all submodules', () => {
    const net = new Sequential(new Linear(2, 3), new Linear(3, 1));
    net.eval();
    for (const m of net.modules()) {
      expect(m.training).toBe(false);
    }
  });

  it('train(true) restores training mode on all submodules', () => {
    const net = new Sequential(new Linear(2, 3));
    net.eval();
    net.train(true);
    for (const m of net.modules()) {
      expect(m.training).toBe(true);
    }
  });
});

describe('zeroGrad', () => {
  it('zeros out gradients of all parameters that have grads', () => {
    const net = new Linear(3, 2);
    const x = tensor([[1, 2, 3]], { requiresGrad: false });
    const out = net.forward(x);
    sum(out).backward();

    const paramsWithGrad = [...net.parameters()].filter(p => p.grad !== null);
    expect(paramsWithGrad.length).toBeGreaterThan(0);

    for (const p of paramsWithGrad) {
      expect([...p.grad.data].some(v => v !== 0)).toBe(true);
    }

    net.zeroGrad();
    for (const p of paramsWithGrad) {
      expect([...p.grad.data].every(v => v === 0)).toBe(true);
    }
  });
});

describe('apply', () => {
  it('calls function on every module in tree', () => {
    const net = new Sequential(new Linear(2, 3), new Linear(3, 1));
    const visited = [];
    net.apply(m => visited.push(m.constructor.name));
    expect(visited).toContain('Linear');
    expect(visited).toContain('Sequential');
    expect(visited.length).toBe(3);
  });
});

describe('modules generator', () => {
  it('yields self and all descendants', () => {
    const net = new Sequential(new Linear(2, 3), new Linear(3, 1));
    const all = [...net.modules()];
    expect(all.length).toBe(3);
  });
});

describe('_autoDetect', () => {
  it('auto-registers Parameter properties', () => {
    class MyModule extends Module {
      constructor() {
        super();
        this.w = new Parameter(zeros([3, 3]));
      }
    }
    const m = new MyModule();
    const params = [...m.parameters()];
    expect(params.length).toBe(1);
    expect(params[0].shape).toEqual([3, 3]);
  });

  it('auto-registers child Module properties', () => {
    class Parent extends Module {
      constructor() {
        super();
        this.child = new Linear(2, 3);
      }
    }
    const p = new Parent();
    const children = [...p.children()];
    expect(children.length).toBe(1);
  });
});
