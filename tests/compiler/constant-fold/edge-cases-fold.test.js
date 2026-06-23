import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { ConstantFoldPass } from '../../../src/compiler/passes/simplify/constant_fold.js';
import { PassResult } from '../../../src/compiler/passes/pass.js';
import { registry } from '../../../src/compiler/ir/graph/ops.js';

function run(func) {
  return new ConstantFoldPass().run(func);
}

function retVal(func) {
  return func.getReturnOp().getOperand(0);
}

describe('diamond-shaped constant subgraph', () => {
  it('shared constant feeds both operands of add — both resolve independently', () => {
    const t = new TensorType([], ScalarType.F32);
    const func = buildFunction('f', [], [t], (b) => {
      const c = b.scalarConstant(5, ScalarType.F32);
      b.returnOp([b.add(c.getResult(0), c.getResult(0)).getResult(0)]);
    });

    run(func);

    expect(retVal(func).definingOp.opName).toBe('constant');
    expect(retVal(func).definingOp.getAttr('value')).toBe(10);
  });

  it('diamond through different ops: add(neg(c), mul(c, c))', () => {
    const t = new TensorType([], ScalarType.F32);
    const func = buildFunction('f', [], [t], (b) => {
      const c = b.scalarConstant(3, ScalarType.F32);
      const n = b.neg(c.getResult(0));
      const m = b.mul(c.getResult(0), c.getResult(0));
      b.returnOp([b.add(n.getResult(0), m.getResult(0)).getResult(0)]);
    });

    run(func);

    expect(retVal(func).definingOp.opName).toBe('constant');
    expect(retVal(func).definingOp.getAttr('value')).toBe(6);
  });
});

describe('mixed foldable and non-foldable ops in same function', () => {
  it('folds one branch, leaves the other, both return values correct', () => {
    const t = new TensorType([], ScalarType.F32);
    const func = buildFunction('f', [t], [t, t], (b, args) => {
      const a = b.scalarConstant(2, ScalarType.F32);
      const c = b.scalarConstant(3, ScalarType.F32);
      const constResult = b.add(a.getResult(0), c.getResult(0));
      const liveResult = b.mul(args[0], a.getResult(0));
      b.returnOp([constResult.getResult(0), liveResult.getResult(0)]);
    });

    run(func);

    const ret = func.getReturnOp();
    expect(ret.getOperand(0).definingOp.opName).toBe('constant');
    expect(ret.getOperand(0).definingOp.getAttr('value')).toBe(5);
    expect(ret.getOperand(1).definingOp.opName).toBe('mul');
    expect(ret.getOperand(1).definingOp.getOperand(0)).toBe(func.args[0]);
  });
});

describe('resolution through reshape and broadcast (passthrough folds)', () => {
  it('reshape of constant folds because reshape.fold passes value through', () => {
    const t = new TensorType([], ScalarType.F32);
    const tOut = new TensorType([1], ScalarType.F32);
    const func = buildFunction('f', [], [tOut], (b) => {
      const c = b.scalarConstant(7, ScalarType.F32);
      const reshaped = b.reshape(c.getResult(0), [1]);
      const two = b.scalarConstant(2, ScalarType.F32);
      const r2 = b.reshape(two.getResult(0), [1]);
      b.returnOp([b.add(reshaped.getResult(0), r2.getResult(0)).getResult(0)]);
    });

    run(func);

    expect(retVal(func).definingOp.opName).toBe('constant');
    expect(retVal(func).definingOp.getAttr('value')).toBe(9);
  });
});

describe('multiple independent foldable subexpressions', () => {
  it('three separate constant chains all fold in single pass', () => {
    const t = new TensorType([], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const a1 = b.scalarConstant(1, ScalarType.F32);
      const a2 = b.scalarConstant(2, ScalarType.F32);
      const s1 = b.add(a1.getResult(0), a2.getResult(0));

      const b1 = b.scalarConstant(3, ScalarType.F32);
      const b2 = b.scalarConstant(4, ScalarType.F32);
      const s2 = b.mul(b1.getResult(0), b2.getResult(0));

      const c1 = b.scalarConstant(10, ScalarType.F32);
      const c2 = b.scalarConstant(5, ScalarType.F32);
      const s3 = b.sub(c1.getResult(0), c2.getResult(0));

      const combined = b.add(s1.getResult(0), s2.getResult(0));
      const combined2 = b.add(combined.getResult(0), s3.getResult(0));
      b.returnOp([b.mul(args[0], combined2.getResult(0)).getResult(0)]);
    });

    run(func);

    const mul = retVal(func).definingOp;
    expect(mul.opName).toBe('mul');
    expect(mul.getOperand(0)).toBe(func.args[0]);
    expect(mul.getOperand(1).definingOp.opName).toBe('constant');
    expect(mul.getOperand(1).definingOp.getAttr('value')).toBe(20);
  });
});

describe('integer dtype folding', () => {
  it('folds I32 constants correctly (integer arithmetic, not float)', () => {
    const t = new TensorType([], ScalarType.I32);
    const func = buildFunction('f', [], [t], (b) => {
      const a = b.scalarConstant(7, ScalarType.I32);
      const c = b.scalarConstant(3, ScalarType.I32);
      b.returnOp([b.sub(a.getResult(0), c.getResult(0)).getResult(0)]);
    });

    run(func);

    const constOp = retVal(func).definingOp;
    expect(constOp.opName).toBe('constant');
    expect(constOp.getAttr('value')).toBe(4);
    expect(constOp.getResult(0).type.dtype).toBe(ScalarType.I32);
  });
});

describe('visited set prevents re-traversal', () => {
  it('same intermediate used in both operands does not cause repeated work or errors', () => {
    const t = new TensorType([], ScalarType.F32);
    const func = buildFunction('f', [], [t], (b) => {
      const a = b.scalarConstant(2, ScalarType.F32);
      const c = b.scalarConstant(3, ScalarType.F32);
      const sum = b.add(a.getResult(0), c.getResult(0));
      const doubled = b.mul(sum.getResult(0), sum.getResult(0));
      const again = b.add(doubled.getResult(0), sum.getResult(0));
      b.returnOp([again.getResult(0)]);
    });

    run(func);

    expect(retVal(func).definingOp.opName).toBe('constant');
    expect(retVal(func).definingOp.getAttr('value')).toBe(30);
  });
});

describe('integer dtype fold representability guard', () => {
  it('aborts fold when integer result exceeds safe integer range', () => {
    const t = new TensorType([], ScalarType.I64);
    const func = buildFunction('f', [], [t], (b) => {
      const a = b.scalarConstant(10000000000, ScalarType.I64);
      const c = b.scalarConstant(10000000000, ScalarType.I64);
      b.returnOp([b.mul(a.getResult(0), c.getResult(0)).getResult(0)]);
    });

    run(func);

    expect(retVal(func).definingOp.opName).toBe('mul');
  });

  it('aborts fold when integer division yields a non-integer result', () => {
    const t = new TensorType([], ScalarType.I32);
    const func = buildFunction('f', [], [t], (b) => {
      const a = b.scalarConstant(7, ScalarType.I32);
      const c = b.scalarConstant(2, ScalarType.I32);
      b.returnOp([b.div(a.getResult(0), c.getResult(0)).getResult(0)]);
    });

    run(func);

    expect(retVal(func).definingOp.opName).toBe('div');
  });

  it('still folds integer results within safe range', () => {
    const t = new TensorType([], ScalarType.I32);
    const func = buildFunction('f', [], [t], (b) => {
      const a = b.scalarConstant(8, ScalarType.I32);
      const c = b.scalarConstant(2, ScalarType.I32);
      b.returnOp([b.mul(a.getResult(0), c.getResult(0)).getResult(0)]);
    });

    run(func);

    expect(retVal(func).definingOp.opName).toBe('constant');
    expect(retVal(func).definingOp.getAttr('value')).toBe(16);
  });

  it('does not block folds for float dtypes producing non-finite values', () => {
    const t = new TensorType([], ScalarType.F32);
    const func = buildFunction('f', [], [t], (b) => {
      const a = b.scalarConstant(1, ScalarType.F32);
      const c = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.div(a.getResult(0), c.getResult(0)).getResult(0)]);
    });

    run(func);

    expect(retVal(func).definingOp.opName).toBe('constant');
    expect(retVal(func).definingOp.getAttr('value')).toBe(Infinity);
  });
});

describe('folds bail on tensor (array) constants', () => {
  it('arithmetic/exp folds return undefined for array operands instead of corrupting them', () => {
    expect(registry.get('add').fold([2, 3])).toBe(5);
    expect(registry.get('add').fold([new Float32Array([1, 2]), new Float32Array([3, 4])])).toBeUndefined();
    expect(registry.get('mul').fold([new Float32Array([1, 2]), 3])).toBeUndefined();
    expect(registry.get('div').fold([new Float32Array([1]), new Float32Array([2])])).toBeUndefined();
    expect(registry.get('neg').fold([new Float32Array([1, 2])])).toBeUndefined();
    expect(registry.get('exp').fold([new Float32Array([0, 1])])).toBeUndefined();
    expect(registry.get('exp').fold([0])).toBe(1);
  });
});
