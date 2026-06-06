import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { DCEPass } from '../../../src/compiler/passes/simplify/dce.js';
import { PassResult } from '../../../src/compiler/passes/pass.js';

function run(func) {
  return new DCEPass().run(func);
}

function opNames(func) {
  const names = [];
  for (const op of func.ops()) names.push(op.opName);
  return names;
}

function findOps(func, opName) {
  const result = [];
  for (const op of func.ops()) {
    if (op.opName === opName) result.push(op);
  }
  return result;
}

describe('DCEPass removes unused ops', () => {
  it('erases op whose result is unused', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      b.neg(args[0]);
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });

    expect(opNames(func)).toContain('neg');
    expect(run(func)).toBe(PassResult.CHANGED);
    expect(opNames(func)).not.toContain('neg');
    expect(opNames(func)).toContain('add');
  });

  it('returns UNCHANGED when all ops are used', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
  });

  it('never removes return op', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      b.neg(args[0]);
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });

    run(func);
    expect(findOps(func, 'return').length).toBe(1);
  });
});

describe('DCEPass cascading elimination', () => {
  it('erasing a dead op makes its operand-defining op dead too', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const n = b.neg(args[0]);
      b.exp(n.getResult(0));
      b.returnOp([args[0]]);
    });

    expect(opNames(func)).toContain('neg');
    expect(opNames(func)).toContain('exp');
    expect(run(func)).toBe(PassResult.CHANGED);
    expect(opNames(func)).not.toContain('neg');
    expect(opNames(func)).not.toContain('exp');
  });

  it('cascade stops at ops with remaining uses', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t, t], (b, args) => {
      const n = b.neg(args[0]);
      b.exp(n.getResult(0));
      b.returnOp([n.getResult(0), args[1]]);
    });

    run(func);
    expect(opNames(func)).toContain('neg');
    expect(opNames(func)).not.toContain('exp');
  });

  it('deep chain: a → b → c all dead → all erased', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const a = b.neg(args[0]);
      const c = b.exp(a.getResult(0));
      b.neg(c.getResult(0));
      b.returnOp([args[1]]);
    });

    expect(run(func)).toBe(PassResult.CHANGED);
    expect(opNames(func)).toEqual(['return']);
  });
});

describe('DCEPass preserves side-effectful ops', () => {
  it('custom_call is never erased even with no uses', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.customCall('print', [args[0]], [t]);
      b.returnOp([args[0]]);
    });

    run(func);
    expect(findOps(func, 'custom_call').length).toBe(1);
  });
});

describe('DCEPass multiple dead subgraphs', () => {
  it('removes multiple independent dead chains in one pass', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      b.neg(args[0]);
      b.exp(args[1]);
      const live = b.add(args[0], args[1]);
      b.mul(args[0], args[1]);
      b.returnOp([live.getResult(0)]);
    });

    const before = opNames(func).length;
    expect(run(func)).toBe(PassResult.CHANGED);
    const after = opNames(func);
    expect(after).toContain('add');
    expect(after).toContain('return');
    expect(after).not.toContain('neg');
    expect(after).not.toContain('exp');
    expect(after).not.toContain('mul');
    expect(after.length).toBe(before - 3);
  });

  it('keeps shared producer alive when only one consumer is dead', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const n = b.neg(args[0]);
      b.exp(n.getResult(0));
      b.returnOp([n.getResult(0)]);
    });

    run(func);
    expect(opNames(func)).toContain('neg');
    expect(opNames(func)).not.toContain('exp');
  });
});

describe('DCEPass idempotency', () => {
  it('second run returns UNCHANGED after first pass cleans up', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.neg(args[0]);
      b.exp(args[0]);
      b.returnOp([args[0]]);
    });

    expect(run(func)).toBe(PassResult.CHANGED);
    expect(run(func)).toBe(PassResult.UNCHANGED);
  });
});

describe('DCEPass with matmul/conv', () => {
  it('unused dot is erased', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [lhs], (b, args) => {
      b.matmul(args[0], args[1]);
      b.returnOp([args[0]]);
    });

    expect(run(func)).toBe(PassResult.CHANGED);
    expect(findOps(func, 'dot').length).toBe(0);
  });

  it('used dot is preserved', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const out = new TensorType([4, 6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
  });

  it('dead conv with dead input chain erased completely', () => {
    const t = new TensorType([4], ScalarType.F32);
    const inp = new TensorType([1, 3, 8, 8], ScalarType.F32);
    const ker = new TensorType([16, 3, 3, 3], ScalarType.F32);
    const func = buildFunction('f', [t, inp, ker], [t], (b, args) => {
      const n = b.neg(args[0]);
      b.conv(args[1], args[2], [1, 1], [0, 0, 0, 0]);
      b.returnOp([n.getResult(0)]);
    });

    run(func);
    expect(findOps(func, 'conv').length).toBe(0);
    expect(findOps(func, 'neg').length).toBe(1);
  });
});

describe('DCEPass op count correctness', () => {
  it('only dead ops are removed — live count stays exact', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const a = b.add(args[0], args[1]);
      const n = b.neg(a.getResult(0));
      b.exp(args[0]);
      b.mul(args[0], args[1]);
      b.returnOp([n.getResult(0)]);
    });

    run(func);
    const names = opNames(func);
    expect(names).toEqual(['add', 'neg', 'return']);
  });
});
