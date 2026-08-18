import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { CanonicalizePass } from '../../../../src/compiler/passes/canonicalize/canonicalize.js';
import { registry } from '../../../../src/compiler/ir/graph/ops.js';
import { OpTrait } from '../../../../src/compiler/ir/graph/op_registry.js';
import { PassResult } from '../../../../src/compiler/passes/pass.js';

function run(func) {
  return new CanonicalizePass().run(func);
}

function retVal(func) {
  return func.getReturnOp().getOperand(0);
}

function f32(shape) {
  return new TensorType(shape, ScalarType.F32);
}

describe('IDEMPOTENT drives f(x, x) -> x', () => {
  it('maximum(x, x) collapses to x', () => {
    const t = f32([4, 8]);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.maximum(args[0], args[0]).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.CHANGED);
    expect(retVal(func)).toBe(func.args[0]);
    expect(func.findOp(op => op.opName === 'maximum')).toBeNull();
  });

  it('logical_and(x, x) collapses to x', () => {
    const t = new TensorType([4], ScalarType.BOOL);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.create('logical_and', [args[0], args[0]]).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.CHANGED);
    expect(retVal(func)).toBe(func.args[0]);
  });

  it('leaves f(x, y) alone when the operands differ', () => {
    const t = f32([4]);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      b.returnOp([b.maximum(args[0], args[1]).getResult(0)]);
    });

    run(func);
    expect(func.findOp(op => op.opName === 'maximum')).not.toBeNull();
  });

  it('does not fire for a non-idempotent op with identical operands', () => {
    const t = f32([4]);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.mul(args[0], args[0]).getResult(0)]);
    });

    run(func);
    expect(func.findOp(op => op.opName === 'mul')).not.toBeNull();
  });
});

describe('COMMUTATIVE drives constant-to-the-right without per-op wiring', () => {
  it('moves the constant to the rhs for every commutative op, not just add and mul', () => {
    const t = f32([4]);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const c = b.scalarConstant(3, ScalarType.F32);
      b.returnOp([b.maximum(c.getResult(0), args[0]).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.CHANGED);
    const max = func.findOp(op => op.opName === 'maximum');
    expect(max.getOperand(0)).toBe(func.args[0]);
    expect(max.getOperand(1).definingOp.opName).toBe('constant');
  });

  it('leaves a non-commutative op with a constant lhs untouched', () => {
    const t = f32([4]);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const c = b.scalarConstant(3, ScalarType.F32);
      b.returnOp([b.sub(c.getResult(0), args[0]).getResult(0)]);
    });

    run(func);
    const sub = func.findOp(op => op.opName === 'sub');
    expect(sub.getOperand(0).definingOp.opName).toBe('constant');
  });

  it('the pattern is derived from the trait, so every commutative op is covered', () => {
    const commutative = registry.allOps().filter(def => def.isCommutative);
    expect(commutative.length).toBeGreaterThan(1);
    for (const def of commutative) {
      expect(def.numOperands, `${def.name} declares COMMUTATIVE but is not binary`).toBe(2);
    }
  });
});

describe('ASSOCIATIVE drives constant reassociation', () => {
  it('(x + c1) + c2 folds the constants into a single add', () => {
    const t = f32([4]);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const inner = b.add(args[0], b.scalarConstant(2, ScalarType.F32).getResult(0));
      const outer = b.add(inner.getResult(0), b.scalarConstant(5, ScalarType.F32).getResult(0));
      b.returnOp([outer.getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.CHANGED);

    const adds = func.findOps(op => op.opName === 'add');
    expect(adds.length).toBe(1);
    expect(adds[0].getOperand(0)).toBe(func.args[0]);
    expect(adds[0].getOperand(1).definingOp.getAttr('value')).toBe(7);
  });

  it('(x * c1) * c2 folds through the same trait-derived pattern', () => {
    const t = f32([4]);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const inner = b.mul(args[0], b.scalarConstant(3, ScalarType.F32).getResult(0));
      const outer = b.mul(inner.getResult(0), b.scalarConstant(4, ScalarType.F32).getResult(0));
      b.returnOp([outer.getResult(0)]);
    });

    run(func);

    const muls = func.findOps(op => op.opName === 'mul');
    expect(muls.length).toBe(1);
    expect(muls[0].getOperand(1).definingOp.getAttr('value')).toBe(12);
  });

  it('does not reassociate when the intermediate is used elsewhere', () => {
    const t = f32([4]);
    const func = buildFunction('f', [t], [t, t], (b, args) => {
      const inner = b.add(args[0], b.scalarConstant(2, ScalarType.F32).getResult(0));
      const outer = b.add(inner.getResult(0), b.scalarConstant(5, ScalarType.F32).getResult(0));
      b.returnOp([outer.getResult(0), inner.getResult(0)]);
    });

    run(func);
    expect(func.findOps(op => op.opName === 'add').length).toBe(2);
  });

  it('does not reassociate a non-associative op', () => {
    const t = f32([4]);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const inner = b.sub(args[0], b.scalarConstant(2, ScalarType.F32).getResult(0));
      const outer = b.sub(inner.getResult(0), b.scalarConstant(5, ScalarType.F32).getResult(0));
      b.returnOp([outer.getResult(0)]);
    });

    run(func);
    expect(func.findOps(op => op.opName === 'sub').length).toBe(2);
  });
});

describe('the traits that drive these rewrites are actually declared', () => {
  it('add and mul carry COMMUTATIVE and ASSOCIATIVE', () => {
    for (const name of ['add', 'mul']) {
      const def = registry.get(name);
      expect(def.isCommutative, `${name} COMMUTATIVE`).toBe(true);
      expect(def.isAssociative, `${name} ASSOCIATIVE`).toBe(true);
    }
  });

  it('maximum, minimum and the boolean lattice ops carry IDEMPOTENT', () => {
    for (const name of ['maximum', 'minimum', 'logical_and', 'logical_or']) {
      expect(registry.get(name).hasTrait(OpTrait.IDEMPOTENT), `${name} IDEMPOTENT`).toBe(true);
    }
  });
});
