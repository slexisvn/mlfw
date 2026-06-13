import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { ConstantFoldPass } from '../../../src/compiler/passes/simplify/constant_fold.js';
import { PassResult } from '../../../src/compiler/passes/pass.js';

function run(func) {
  return new ConstantFoldPass().run(func);
}

function retVal(func) {
  return func.getReturnOp().getOperand(0);
}

describe('recursive resolution through op chains', () => {
  it('resolves 3-deep chain: the final constant absorbs all intermediate ops', () => {
    const t = new TensorType([], ScalarType.F32);
    const func = buildFunction('f', [], [t], (b) => {
      const a = b.scalarConstant(2, ScalarType.F32);
      const c = b.scalarConstant(3, ScalarType.F32);
      const sum = b.add(a.getResult(0), c.getResult(0));
      const four = b.scalarConstant(4, ScalarType.F32);
      const prod = b.mul(sum.getResult(0), four.getResult(0));
      const one = b.scalarConstant(1, ScalarType.F32);
      b.returnOp([b.sub(prod.getResult(0), one.getResult(0)).getResult(0)]);
    });

    run(func);

    const result = retVal(func).definingOp;
    expect(result.opName).toBe('constant');
    expect(result.getAttr('value')).toBe(19);
  });

  it('recursive resolution walks through neg(add(const, const))', () => {
    const t = new TensorType([], ScalarType.F32);
    const func = buildFunction('f', [], [t], (b) => {
      const a = b.scalarConstant(3, ScalarType.F32);
      const c = b.scalarConstant(7, ScalarType.F32);
      const sum = b.add(a.getResult(0), c.getResult(0));
      b.returnOp([b.neg(sum.getResult(0)).getResult(0)]);
    });

    run(func);

    expect(retVal(func).definingOp.opName).toBe('constant');
    expect(retVal(func).definingOp.getAttr('value')).toBe(-10);
  });

  it('shared intermediate: both branches of add read from same mul result', () => {
    const t = new TensorType([], ScalarType.F32);
    const func = buildFunction('f', [], [t], (b) => {
      const a = b.scalarConstant(3, ScalarType.F32);
      const c = b.scalarConstant(4, ScalarType.F32);
      const prod = b.mul(a.getResult(0), c.getResult(0));
      b.returnOp([b.add(prod.getResult(0), prod.getResult(0)).getResult(0)]);
    });

    run(func);

    expect(retVal(func).definingOp.opName).toBe('constant');
    expect(retVal(func).definingOp.getAttr('value')).toBe(24);
  });
});

describe('partial constant subgraph', () => {
  it('folds constant sub-expression but leaves arg-dependent part intact', () => {
    const t = new TensorType([], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const a = b.scalarConstant(3, ScalarType.F32);
      const c = b.scalarConstant(4, ScalarType.F32);
      const constSum = b.add(a.getResult(0), c.getResult(0));
      b.returnOp([b.mul(args[0], constSum.getResult(0)).getResult(0)]);
    });

    run(func);

    const mul = retVal(func).definingOp;
    expect(mul.opName).toBe('mul');
    expect(mul.getOperand(0)).toBe(func.args[0]);
    expect(mul.getOperand(1).definingOp.opName).toBe('constant');
    expect(mul.getOperand(1).definingOp.getAttr('value')).toBe(7);
  });

  it('a single non-constant operand poisons the entire chain above it', () => {
    const t = new TensorType([], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const c = b.scalarConstant(5, ScalarType.F32);
      const sum = b.add(args[0], c.getResult(0));
      const two = b.scalarConstant(2, ScalarType.F32);
      b.returnOp([b.mul(sum.getResult(0), two.getResult(0)).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
    expect(retVal(func).definingOp.opName).toBe('mul');
  });

  it('two independent constant sub-trees both fold in the same pass', () => {
    const t = new TensorType([], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const a = b.scalarConstant(2, ScalarType.F32);
      const c = b.scalarConstant(3, ScalarType.F32);
      const left = b.add(a.getResult(0), c.getResult(0));
      const d = b.scalarConstant(10, ScalarType.F32);
      const e = b.scalarConstant(1, ScalarType.F32);
      const right = b.sub(d.getResult(0), e.getResult(0));
      const combined = b.mul(left.getResult(0), right.getResult(0));
      b.returnOp([b.add(args[0], combined.getResult(0)).getResult(0)]);
    });

    run(func);

    const add = retVal(func).definingOp;
    expect(add.opName).toBe('add');
    expect(add.getOperand(0)).toBe(func.args[0]);
    expect(add.getOperand(1).definingOp.opName).toBe('constant');
    expect(add.getOperand(1).definingOp.getAttr('value')).toBe(45);
  });
});

describe('non-foldable op blocking', () => {
  it('op without fold function blocks resolution even with constant inputs', () => {
    const t = new TensorType([], ScalarType.F32);
    const func = buildFunction('f', [], [t], (b) => {
      const a = b.scalarConstant(2, ScalarType.F32);
      const c = b.scalarConstant(3, ScalarType.F32);
      const maxOp = b.maximum(a.getResult(0), c.getResult(0));
      const d = b.scalarConstant(1, ScalarType.F32);
      b.returnOp([b.add(maxOp.getResult(0), d.getResult(0)).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
    expect(retVal(func).definingOp.opName).toBe('add');
  });

  it('constant ops themselves are not re-processed (0-operand skip)', () => {
    const t = new TensorType([], ScalarType.F32);
    const func = buildFunction('f', [], [t], (b) => {
      b.returnOp([b.scalarConstant(42, ScalarType.F32).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
  });

  it('block argument (definingOp=null) stops resolution immediately', () => {
    const t = new TensorType([], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
    expect(retVal(func).definingOp.opName).toBe('neg');
  });
});

describe('result type and replacement wiring', () => {
  it('folded constant inherits the erased op result type exactly', () => {
    const t = new TensorType([], ScalarType.F64);
    const func = buildFunction('f', [], [t], (b) => {
      const a = b.scalarConstant(3, ScalarType.F64);
      const c = b.scalarConstant(4, ScalarType.F64);
      b.returnOp([b.add(a.getResult(0), c.getResult(0)).getResult(0)]);
    });

    run(func);

    const constOp = retVal(func).definingOp;
    expect(constOp.getResult(0).type.dtype).toBe(ScalarType.F64);
  });

  it('erased op result is replaced for ALL consumers, not just the first', () => {
    const t = new TensorType([], ScalarType.F32);
    const func = buildFunction('f', [t], [t, t], (b, args) => {
      const a = b.scalarConstant(2, ScalarType.F32);
      const c = b.scalarConstant(3, ScalarType.F32);
      const sum = b.add(a.getResult(0), c.getResult(0));
      const r1 = b.mul(args[0], sum.getResult(0));
      const r2 = b.add(args[0], sum.getResult(0));
      b.returnOp([r1.getResult(0), r2.getResult(0)]);
    });

    run(func);

    const ret = func.getReturnOp();
    const mulOp = ret.getOperand(0).definingOp;
    const addOp = ret.getOperand(1).definingOp;
    expect(mulOp.getOperand(1).definingOp.opName).toBe('constant');
    expect(addOp.getOperand(1).definingOp.opName).toBe('constant');
    expect(mulOp.getOperand(1).definingOp.getAttr('value')).toBe(5);
    expect(addOp.getOperand(1).definingOp.getAttr('value')).toBe(5);
  });

  it('erased op is fully detached — parentBlock becomes null', () => {
    const t = new TensorType([], ScalarType.F32);
    const func = buildFunction('f', [], [t], (b) => {
      const a = b.scalarConstant(2, ScalarType.F32);
      const c = b.scalarConstant(3, ScalarType.F32);
      const addOp = b.add(a.getResult(0), c.getResult(0));
      b.returnOp([addOp.getResult(0)]);
      return { addOp };
    });

    const addOp = [...func.ops()].find(op => op.opName === 'add');
    run(func);

    expect(addOp.parentBlock).toBeNull();
  });
});

describe('single-pass behavior', () => {
  it('only folds ops whose ALL operands resolve — no iterative convergence', () => {
    const t = new TensorType([], ScalarType.F32);
    const func = buildFunction('f', [], [t], (b) => {
      const a = b.scalarConstant(2, ScalarType.F32);
      const c = b.scalarConstant(3, ScalarType.F32);
      const sum = b.add(a.getResult(0), c.getResult(0));
      const d = b.scalarConstant(4, ScalarType.F32);
      b.returnOp([b.mul(sum.getResult(0), d.getResult(0)).getResult(0)]);
    });

    const result = run(func);
    expect(result).toBe(PassResult.CHANGED);

    expect(retVal(func).definingOp.opName).toBe('constant');
    expect(retVal(func).definingOp.getAttr('value')).toBe(20);
  });

  it('second run returns UNCHANGED — everything foldable was already folded', () => {
    const t = new TensorType([], ScalarType.F32);
    const func = buildFunction('f', [], [t], (b) => {
      const a = b.scalarConstant(2, ScalarType.F32);
      const c = b.scalarConstant(3, ScalarType.F32);
      b.returnOp([b.add(a.getResult(0), c.getResult(0)).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.CHANGED);
    expect(run(func)).toBe(PassResult.UNCHANGED);
  });

  it('no dangling operand refs after partial fold', () => {
    const t = new TensorType([], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const a = b.scalarConstant(3, ScalarType.F32);
      const c = b.scalarConstant(4, ScalarType.F32);
      const constPart = b.add(a.getResult(0), c.getResult(0));
      b.returnOp([b.mul(args[0], constPart.getResult(0)).getResult(0)]);
    });

    run(func);

    for (const op of func.ops()) {
      for (let i = 0; i < op.numOperands; i++) {
        const v = op.getOperand(i);
        if (!v.isBlockArgument()) {
          expect(v.definingOp).not.toBeNull();
          expect(v.definingOp.parentBlock).not.toBeNull();
        }
      }
    }
  });
});

describe('transcendental fold through recursive resolution', () => {
  it('exp feeds into add — both resolve through the chain', () => {
    const t = new TensorType([], ScalarType.F32);
    const func = buildFunction('f', [], [t], (b) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      const expResult = b.exp(zero.getResult(0));
      const one = b.scalarConstant(1, ScalarType.F32);
      b.returnOp([b.add(expResult.getResult(0), one.getResult(0)).getResult(0)]);
    });

    run(func);

    expect(retVal(func).definingOp.opName).toBe('constant');
    expect(retVal(func).definingOp.getAttr('value')).toBeCloseTo(2, 10);
  });
});

describe('transpose of a constant folds to the transposed constant', () => {
  const F = ScalarType.F32;

  it('folds transpose([[1,2,3],[4,5,6]], perm [1,0]) to [1,4,2,5,3,6]', () => {
    const func = buildFunction('t', [], [new TensorType([3, 2], F)], (b) => {
      const wc = b.constant([1, 2, 3, 4, 5, 6], new TensorType([2, 3], F));
      const tr = b._buildOp('transpose', [wc.getResult(0)], [new TensorType([3, 2], F)], { permutation: [1, 0] });
      b.returnOp([tr.getResult(0)]);
    });

    run(func);
    const ret = retVal(func).definingOp;
    expect(ret.opName).toBe('constant');
    expect(Array.from(ret.getAttr('value'))).toEqual([1, 4, 2, 5, 3, 6]);
  });

  it('leaves transpose of a non-constant operand untouched', () => {
    const func = buildFunction('t', [new TensorType([2, 3], F)], [new TensorType([3, 2], F)], (b, a) => {
      const tr = b._buildOp('transpose', [a[0]], [new TensorType([3, 2], F)], { permutation: [1, 0] });
      b.returnOp([tr.getResult(0)]);
    });

    run(func);
    expect(retVal(func).definingOp.opName).toBe('transpose');
  });
});
