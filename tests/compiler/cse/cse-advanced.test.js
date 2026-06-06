import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { CSEPass } from '../../../src/compiler/passes/simplify/cse.js';
import { PassResult } from '../../../src/compiler/passes/pass.js';

function run(func) {
  return new CSEPass().run(func);
}

function retVal(func, i = 0) {
  return func.getReturnOp().getOperand(i);
}

describe('side-effect ops are never eliminated', () => {
  it('two identical custom_call ops both survive — sideEffects blocks CSE', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t, t], (b, args) => {
      const c1 = b.customCall('my_kernel', [args[0]], [t]);
      const c2 = b.customCall('my_kernel', [args[0]], [t]);
      b.returnOp([c1.getResult(0), c2.getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
    expect(retVal(func, 0).definingOp).not.toBe(retVal(func, 1).definingOp);
  });
});

describe('structuralEquals checks operand identity, not value equivalence', () => {
  it('two adds fed by different constants with same value: constants CSE, then adds CSE', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t, t], (b, args) => {
      const c1 = b.scalarConstant(3, ScalarType.F32);
      const a1 = b.add(args[0], c1.getResult(0));
      const c2 = b.scalarConstant(3, ScalarType.F32);
      const a2 = b.add(args[0], c2.getResult(0));
      b.returnOp([a1.getResult(0), a2.getResult(0)]);
    });

    run(func);

    expect(retVal(func, 0)).toBe(retVal(func, 1));
  });

  it('add(x, c1) vs add(x, c2) where c1≠c2 are NOT eliminated despite same op', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t, t], (b, args) => {
      const c1 = b.scalarConstant(3, ScalarType.F32);
      const c2 = b.scalarConstant(7, ScalarType.F32);
      const a1 = b.add(args[0], c1.getResult(0));
      const a2 = b.add(args[0], c2.getResult(0));
      b.returnOp([a1.getResult(0), a2.getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
  });
});

describe('diamond DAG: shared subexpression feeding symmetric branches', () => {
  it('two branches compute neg(shared_add) — both collapse to one neg', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t, t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      const n1 = b.neg(sum.getResult(0));
      const n2 = b.neg(sum.getResult(0));
      b.returnOp([n1.getResult(0), n2.getResult(0)]);
    });

    run(func);

    expect(retVal(func, 0)).toBe(retVal(func, 1));
  });

  it('diamond: mul(add(x,y), add(x,y)) — two adds collapse, mul sees same operand twice', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const s1 = b.add(args[0], args[1]);
      const s2 = b.add(args[0], args[1]);
      b.returnOp([b.mul(s1.getResult(0), s2.getResult(0)).getResult(0)]);
    });

    run(func);

    const mul = retVal(func).definingOp;
    expect(mul.opName).toBe('mul');
    expect(mul.getOperand(0)).toBe(mul.getOperand(1));
  });
});

describe('different op types on same operands are NOT eliminated', () => {
  it('neg(x) and exp(x) are structurally different — both survive', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t, t], (b, args) => {
      const n = b.neg(args[0]);
      const e = b.exp(args[0]);
      b.returnOp([n.getResult(0), e.getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
  });

  it('add(x, y) and sub(x, y) share operands but differ in opName', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t, t], (b, args) => {
      const a = b.add(args[0], args[1]);
      const s = b.sub(args[0], args[1]);
      b.returnOp([a.getResult(0), s.getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
  });
});

describe('CSE of ops whose operands come from different sources but resolve same', () => {
  it('neg(arg0) built twice reads same Value → identical structuralHash → eliminated', () => {
    const t = new TensorType([2, 3, 4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const n1 = b.neg(args[0]);
      const n2 = b.neg(args[0]);
      b.returnOp([b.add(n1.getResult(0), n2.getResult(0)).getResult(0)]);
    });

    run(func);

    const add = retVal(func).definingOp;
    expect(add.getOperand(0)).toBe(add.getOperand(1));
  });
});

describe('many duplicates with same hash — candidate list grows correctly', () => {
  it('5 identical exp(x) all collapse to one', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const exps = [];
      for (let i = 0; i < 5; i++) {
        exps.push(b.exp(args[0]).getResult(0));
      }
      let acc = exps[0];
      for (let i = 1; i < exps.length; i++) {
        acc = b.add(acc, exps[i]).getResult(0);
      }
      b.returnOp([acc]);
    });

    run(func);

    const liveExps = func.findOps(op => op.opName === 'exp')
      .filter(op => op.parentBlock);
    expect(liveExps).toHaveLength(1);
  });
});

describe('CSE with convert ops — dtype attribute must match', () => {
  it('two convert(x, I32) collapse', () => {
    const t = new TensorType([4], ScalarType.F32);
    const iT = new TensorType([4], ScalarType.I32);
    const func = buildFunction('f', [t], [iT, iT], (b, args) => {
      const c1 = b.convert(args[0], ScalarType.I32);
      const c2 = b.convert(args[0], ScalarType.I32);
      b.returnOp([c1.getResult(0), c2.getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.CHANGED);
    expect(retVal(func, 0)).toBe(retVal(func, 1));
  });

  it('convert(x, I32) and convert(x, F64) do NOT collapse — different target_dtype', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [
      new TensorType([4], ScalarType.I32),
      new TensorType([4], ScalarType.F64)
    ], (b, args) => {
      const c1 = b.convert(args[0], ScalarType.I32);
      const c2 = b.convert(args[0], ScalarType.F64);
      b.returnOp([c1.getResult(0), c2.getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
  });
});

describe('CSE does not merge ops from different subexpressions that happen to hash-collide', () => {
  it('add(x,y) and add(x,x) never match despite same opName — different operand[1]', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t, t], (b, args) => {
      const a1 = b.add(args[0], args[1]);
      const a2 = b.add(args[0], args[0]);
      b.returnOp([a1.getResult(0), a2.getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
  });
});

describe('CSE across complex subgraphs', () => {
  it('two identical sub-DAGs: add(neg(x), mul(y, y)) — entire subtree collapses', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t, t], (b, args) => {
      const n1 = b.neg(args[0]);
      const m1 = b.mul(args[1], args[1]);
      const r1 = b.add(n1.getResult(0), m1.getResult(0));

      const n2 = b.neg(args[0]);
      const m2 = b.mul(args[1], args[1]);
      const r2 = b.add(n2.getResult(0), m2.getResult(0));

      b.returnOp([r1.getResult(0), r2.getResult(0)]);
    });

    run(func);

    expect(retVal(func, 0)).toBe(retVal(func, 1));
  });

  it('partially overlapping sub-DAGs: shared neg, different second operand', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t, t], [t, t], (b, args) => {
      const n1 = b.neg(args[0]);
      const n2 = b.neg(args[0]);
      const r1 = b.add(n1.getResult(0), args[1]);
      const r2 = b.add(n2.getResult(0), args[2]);
      b.returnOp([r1.getResult(0), r2.getResult(0)]);
    });

    run(func);

    const add1 = retVal(func, 0).definingOp;
    const add2 = retVal(func, 1).definingOp;
    expect(add1.getOperand(0)).toBe(add2.getOperand(0));
    expect(add1.getOperand(1)).not.toBe(add2.getOperand(1));
    expect(retVal(func, 0)).not.toBe(retVal(func, 1));
  });
});
