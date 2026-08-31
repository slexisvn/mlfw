import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { CSEPass } from '../../../../src/compiler/passes/simplify/cse.js';
import { PassResult } from '../../../../src/compiler/passes/pass.js';
import { TraceLog, TraceLevel } from '../../../../src/compiler/support/trace.js';

function run(func) {
  return new CSEPass().run(func);
}

function retVal(func, i = 0) {
  return func.getReturnOp().getOperand(i);
}

describe('identical ops with same operands get eliminated', () => {
  it('two add(x, y) collapse — second users rewired to first result', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t, t], (b, args) => {
      const a1 = b.add(args[0], args[1]);
      const a2 = b.add(args[0], args[1]);
      b.returnOp([a1.getResult(0), a2.getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.CHANGED);

    expect(retVal(func, 0)).toBe(retVal(func, 1));
    expect(retVal(func, 0).definingOp.opName).toBe('add');
  });

  it('three identical neg(x) collapse to one — all users share same result', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const n1 = b.neg(args[0]);
      const n2 = b.neg(args[0]);
      const n3 = b.neg(args[0]);
      const sum = b.add(n1.getResult(0), n2.getResult(0));
      b.returnOp([b.add(sum.getResult(0), n3.getResult(0)).getResult(0)]);
    });

    run(func);

    const adds = func.findOps(op => op.opName === 'add');
    const outerAdd = retVal(func).definingOp;
    const innerAdd = outerAdd.getOperand(0).definingOp;

    expect(innerAdd.getOperand(0)).toBe(innerAdd.getOperand(1));
    expect(outerAdd.getOperand(1)).toBe(innerAdd.getOperand(0));
  });
});

describe('commutative ops are matched up to operand order', () => {
  it('add(x, y) and add(y, x) collapse - add carries the commutative trait', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t, t], (b, args) => {
      const a1 = b.add(args[0], args[1]);
      const a2 = b.add(args[1], args[0]);
      b.returnOp([a1.getResult(0), a2.getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.CHANGED);
    expect(retVal(func, 0)).toBe(retVal(func, 1));
    expect(func.findOps(op => op.opName === 'add').length).toBe(1);
  });

  it('sub(x, y) and sub(y, x) are NOT eliminated - sub is not commutative', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t, t], (b, args) => {
      const s1 = b.sub(args[0], args[1]);
      const s2 = b.sub(args[1], args[0]);
      b.returnOp([s1.getResult(0), s2.getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
    expect(retVal(func, 0)).not.toBe(retVal(func, 1));
  });
});

describe('different operands prevent elimination', () => {
  it('add(x, x) and add(x, y) are NOT eliminated — different operand identity', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t, t], (b, args) => {
      const a1 = b.add(args[0], args[0]);
      const a2 = b.add(args[0], args[1]);
      b.returnOp([a1.getResult(0), a2.getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
  });
});

describe('attribute differences prevent elimination', () => {
  it('two reshapes with different target shapes are NOT eliminated', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t], [
      new TensorType([32], ScalarType.F32),
      new TensorType([2, 16], ScalarType.F32)
    ], (b, args) => {
      const r1 = b.reshape(args[0], [32]);
      const r2 = b.reshape(args[0], [2, 16]);
      b.returnOp([r1.getResult(0), r2.getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
  });

  it('two reshapes with SAME shape ARE eliminated', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const outT = new TensorType([32], ScalarType.F32);
    const func = buildFunction('f', [t], [outT, outT], (b, args) => {
      const r1 = b.reshape(args[0], [32]);
      const r2 = b.reshape(args[0], [32]);
      b.returnOp([r1.getResult(0), r2.getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.CHANGED);
    expect(retVal(func, 0)).toBe(retVal(func, 1));
  });

  it('two transposes with different permutations are NOT eliminated', () => {
    const t = new TensorType([2, 3, 4], ScalarType.F32);
    const func = buildFunction('f', [t], [
      new TensorType([3, 2, 4], ScalarType.F32),
      new TensorType([4, 3, 2], ScalarType.F32)
    ], (b, args) => {
      const t1 = b.transpose(args[0], [1, 0, 2]);
      const t2 = b.transpose(args[0], [2, 1, 0]);
      b.returnOp([t1.getResult(0), t2.getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
  });
});

describe('transitive CSE through op chains', () => {
  it('duplicate subexpressions deep in the graph get eliminated', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t, t], (b, args) => {
      const sum1 = b.add(args[0], args[1]);
      const neg1 = b.neg(sum1.getResult(0));

      const sum2 = b.add(args[0], args[1]);
      const neg2 = b.neg(sum2.getResult(0));

      b.returnOp([neg1.getResult(0), neg2.getResult(0)]);
    });

    run(func);

    expect(retVal(func, 0)).toBe(retVal(func, 1));
  });

  it('CSE of inner op causes outer ops to share operands, enabling further CSE on second run', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t, t], (b, args) => {
      const sum1 = b.add(args[0], args[1]);
      const sum2 = b.add(args[0], args[1]);
      const neg1 = b.neg(sum1.getResult(0));
      const neg2 = b.neg(sum2.getResult(0));
      b.returnOp([neg1.getResult(0), neg2.getResult(0)]);
    });

    run(func);

    expect(retVal(func, 0)).toBe(retVal(func, 1));
  });
});

describe('constants with same value get eliminated', () => {
  it('two scalarConstant(5) collapse to one', () => {
    const t = new TensorType([], ScalarType.F32);
    const func = buildFunction('f', [t], [t, t], (b, args) => {
      const c1 = b.scalarConstant(5, ScalarType.F32);
      const c2 = b.scalarConstant(5, ScalarType.F32);
      const r1 = b.add(args[0], c1.getResult(0));
      const r2 = b.add(args[0], c2.getResult(0));
      b.returnOp([r1.getResult(0), r2.getResult(0)]);
    });

    run(func);

    const r1Add = retVal(func, 0).definingOp;
    const r2Add = retVal(func, 1).definingOp;
    if (r1Add !== r2Add) {
      expect(r1Add.getOperand(1)).toBe(r2Add.getOperand(1));
    }
  });

  it('constants with different values are NOT eliminated', () => {
    const t = new TensorType([], ScalarType.F32);
    const func = buildFunction('f', [t], [t, t], (b, args) => {
      const c1 = b.scalarConstant(5, ScalarType.F32);
      const c2 = b.scalarConstant(7, ScalarType.F32);
      const r1 = b.add(args[0], c1.getResult(0));
      const r2 = b.add(args[0], c2.getResult(0));
      b.returnOp([r1.getResult(0), r2.getResult(0)]);
    });

    const opsBefore = func.opsArray().length;
    run(func);

    expect(retVal(func, 0).definingOp.getOperand(1).definingOp.getAttr('value')).toBe(5);
    expect(retVal(func, 1).definingOp.getOperand(1).definingOp.getAttr('value')).toBe(7);
  });
});

describe('multi-result op CSE', () => {
  it('duplicate splits with same attrs get eliminated — all results rewired', () => {
    const t = new TensorType([6, 4], ScalarType.F32);
    const outT = new TensorType([3, 4], ScalarType.F32);
    const func = buildFunction('f', [t], [outT, outT], (b, args) => {
      const s1 = b.split(args[0], 0, [3, 3]);
      const s2 = b.split(args[0], 0, [3, 3]);
      b.returnOp([s1.getResult(0), s2.getResult(1)]);
    });

    run(func);

    const ret = func.getReturnOp();
    const op0 = ret.getOperand(0).definingOp;
    const op1 = ret.getOperand(1).definingOp;
    expect(op0).toBe(op1);
  });
});

describe('pass behavior', () => {
  it('returns UNCHANGED when no duplicates exist', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.neg(sum.getResult(0)).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
  });

  it('second run returns UNCHANGED after all duplicates eliminated', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t, t], (b, args) => {
      const a1 = b.add(args[0], args[1]);
      const a2 = b.add(args[0], args[1]);
      b.returnOp([a1.getResult(0), a2.getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.CHANGED);
    expect(run(func)).toBe(PassResult.UNCHANGED);
  });

  it('erased op does not corrupt surviving candidate references', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t, t, t], (b, args) => {
      const a1 = b.add(args[0], args[1]);
      const a2 = b.add(args[0], args[1]);
      const a3 = b.add(args[0], args[1]);
      b.returnOp([a1.getResult(0), a2.getResult(0), a3.getResult(0)]);
    });

    run(func);

    const ret = func.getReturnOp();
    expect(ret.getOperand(0)).toBe(ret.getOperand(1));
    expect(ret.getOperand(1)).toBe(ret.getOperand(2));
  });

  it('no dangling values after CSE', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t, t], (b, args) => {
      const s1 = b.add(args[0], args[1]);
      const s2 = b.add(args[0], args[1]);
      const n1 = b.neg(s1.getResult(0));
      const n2 = b.neg(s2.getResult(0));
      b.returnOp([n1.getResult(0), n2.getResult(0)]);
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

describe('redundancy is matched across enclosing scopes, not just within a block', () => {
  it('an op inside a region collapses onto an identical op in the enclosing block', () => {
    const t = new TensorType([4], ScalarType.F32);
    const boolT = new TensorType([], ScalarType.BOOL);
    let innerAdd = null;
    const func = buildFunction('f', [t, t, boolT], [t], (b, args) => {
      const outer = b.add(args[0], args[1]);
      const branch = b.ifOp(args[2], [t], (tb) => {
        innerAdd = tb.add(args[0], args[1]);
        tb.yieldOp([innerAdd.getResult(0)]);
      }, (eb) => {
        eb.yieldOp([args[0]]);
      });
      b.returnOp([b.add(outer.getResult(0), branch.getResult(0)).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.CHANGED);

    const addsInRegions = [...func.opsRecursive()].filter(op => op.opName === 'add');
    expect(addsInRegions.length).toBe(2);
    expect(innerAdd.parentBlock).toBe(null);
  });

  it('an op in one branch does NOT collapse onto an identical op in a sibling branch', () => {
    const t = new TensorType([4], ScalarType.F32);
    const boolT = new TensorType([], ScalarType.BOOL);
    const func = buildFunction('f', [t, t, boolT], [t], (b, args) => {
      const branch = b.ifOp(args[2], [t], (tb) => {
        tb.yieldOp([tb.mul(args[0], args[1]).getResult(0)]);
      }, (eb) => {
        eb.yieldOp([eb.mul(args[0], args[1]).getResult(0)]);
      });
      b.returnOp([branch.getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
    expect([...func.opsRecursive()].filter(op => op.opName === 'mul').length).toBe(2);
  });
});

function runTraced(func, level = TraceLevel.DEBUG) {
  const events = [];
  const pass = new CSEPass();
  pass.trace = new TraceLog({ level, sink: (event) => events.push(event) });
  const result = pass.run(func);
  return { result, events, of: (type) => events.filter((e) => e.type === type) };
}

function twoIdenticalAdds() {
  const t = new TensorType([4], ScalarType.F32);
  return buildFunction('f', [t, t], [t, t], (b, args) => {
    const a1 = b.add(args[0], args[1]);
    const a2 = b.add(args[0], args[1]);
    b.returnOp([a1.getResult(0), a2.getResult(0)]);
  });
}

describe('CSE says which ops it removed', () => {
  it('names the op behind every value it deduplicated', () => {
    const explains = runTraced(twoIdenticalAdds()).of('explain');

    expect(explains.length).toBe(1);
    expect(explains[0]).toMatchObject({ category: 'cse', subject: 'add', decision: 'deduplicated' });
  });

  it('explains once per removed op, counting the ones inside regions', () => {
    const t = new TensorType([4], ScalarType.F32);
    const boolT = new TensorType([], ScalarType.BOOL);
    const func = buildFunction('f', [t, t, boolT], [t], (b, args) => {
      const outer = b.add(args[0], args[1]);
      const branch = b.ifOp(args[2], [t], (tb) => {
        tb.yieldOp([tb.add(args[0], args[1]).getResult(0)]);
      }, (eb) => {
        eb.yieldOp([eb.add(args[0], args[1]).getResult(0)]);
      });
      b.returnOp([b.add(outer.getResult(0), branch.getResult(0)).getResult(0)]);
    });

    const { of } = runTraced(func);

    expect(of('explain').length).toBe(2);
    expect(of('pass_detail')[0].eliminated).toBe(2);
  });

  it('stays quiet when it found nothing to remove', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t, t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0), b.mul(args[0], args[1]).getResult(0)]);
    });

    const { result, of } = runTraced(func);

    expect(result).toBe(PassResult.UNCHANGED);
    expect(of('explain')).toEqual([]);
    expect(of('pass_detail')).toEqual([]);
  });

  it('keeps quiet below the level the explanation is emitted at', () => {
    const { result, of } = runTraced(twoIdenticalAdds(), TraceLevel.VERBOSE);

    expect(result).toBe(PassResult.CHANGED);
    expect(of('explain')).toEqual([]);
  });
});
