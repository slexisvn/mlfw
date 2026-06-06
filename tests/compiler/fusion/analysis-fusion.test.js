import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType, DYNAMIC } from '../../../src/compiler/ir/graph/types.js';
import {
  FusionKind, FusionLegality, classifyFusionKind, classifyOpPattern
} from '../../../src/compiler/passes/fusion/fusion_analysis.js';
import { FusionGroup } from '../../../src/compiler/passes/fusion/fusion_groups.js';

function ops(func) {
  const list = [];
  for (const op of func.ops()) {
    if (op.opName !== 'return') list.push(op);
  }
  return list;
}

describe('classifyOpPattern', () => {
  it('add is ELEMENTWISE, reduce is REDUCTION, broadcast is BROADCAST', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const s = new TensorType([], ScalarType.F32);
    const outT = new TensorType([4], ScalarType.F32);
    const outB = new TensorType([3, 4], ScalarType.F32);
    const func = buildFunction('f', [t, t, s, outT], [t, outT, outB], (b, args) => {
      const sum = b.add(args[0], args[1]);
      const red = b.reduce(args[0], args[2], [1], 'sum');
      const bc = b.broadcast(args[3], [3, 4], [1]);
      b.returnOp([sum.getResult(0), red.getResult(0), bc.getResult(0)]);
    });
    const allOps = ops(func);
    expect(classifyOpPattern(allOps[0])).toBe(FusionKind.ELEMENTWISE);
    expect(classifyOpPattern(allOps[1])).toBe(FusionKind.REDUCTION);
    expect(classifyOpPattern(allOps[2])).toBe(FusionKind.BROADCAST);
  });

  it('transpose classifies as OPAQUE (not lowerable inline)', () => {
    const t = new TensorType([3, 5], ScalarType.F32);
    const func = buildFunction('f', [t], [new TensorType([5, 3], ScalarType.F32)], (b, args) => {
      b.returnOp([b.transpose(args[0], [1, 0]).getResult(0)]);
    });
    expect(classifyOpPattern(ops(func)[0])).toBe(FusionKind.OPAQUE);
  });
});

describe('classifyFusionKind priority', () => {
  it('OPAQUE dominates everything else', () => {
    const t = new TensorType([3, 5], ScalarType.F32);
    const tT = new TensorType([5, 3], ScalarType.F32);
    const func = buildFunction('f', [t, tT], [tT], (b, args) => {
      const tr = b.transpose(args[0], [1, 0]);
      b.returnOp([b.add(tr.getResult(0), args[1]).getResult(0)]);
    });
    expect(classifyFusionKind(ops(func))).toBe(FusionKind.OPAQUE);
  });

  it('REDUCTION dominates ELEMENTWISE', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const s = new TensorType([], ScalarType.F32);
    const func = buildFunction('f', [t, t, s], [new TensorType([4], ScalarType.F32)], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.reduce(sum.getResult(0), args[2], [1], 'sum').getResult(0)]);
    });
    expect(classifyFusionKind(ops(func))).toBe(FusionKind.REDUCTION);
  });

  it('BROADCAST dominates ELEMENTWISE but not REDUCTION', () => {
    const inT = new TensorType([4], ScalarType.F32);
    const outT = new TensorType([3, 4], ScalarType.F32);
    const func = buildFunction('f', [inT, outT], [outT], (b, args) => {
      const bc = b.broadcast(args[0], [3, 4], [1]);
      b.returnOp([b.add(bc.getResult(0), args[1]).getResult(0)]);
    });
    expect(classifyFusionKind(ops(func))).toBe(FusionKind.BROADCAST);
  });

  it('pure elementwise list returns ELEMENTWISE', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.neg(sum.getResult(0)).getResult(0)]);
    });
    expect(classifyFusionKind(ops(func))).toBe(FusionKind.ELEMENTWISE);
  });
});

describe('FusionLegality.canFuse', () => {
  it('elementwise pair with same shape is legal', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.neg(sum.getResult(0)).getResult(0)]);
    });
    const [addOp, negOp] = ops(func);
    expect(new FusionLegality().canFuse(addOp, negOp).legal).toBe(true);
  });

  it('elementwise pair with different shapes is illegal', () => {
    const t1 = new TensorType([4], ScalarType.F32);
    const t2 = new TensorType([8], ScalarType.F32);
    const func = buildFunction('f', [t1, t2], [t1, t2], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0), b.neg(args[1]).getResult(0)]);
    });
    const [a, c] = ops(func);
    const result = new FusionLegality().canFuse(a, c);
    expect(result.legal).toBe(false);
    expect(result.reason).toContain('shape mismatch');
  });

  it('DYNAMIC dims are compatible with any static dim', () => {
    const t1 = new TensorType([DYNAMIC, 8], ScalarType.F32);
    const t2 = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t1, t2], [t2], (b, args) => {
      const a = b.neg(args[0]);
      b.returnOp([b.add(a.getResult(0), args[1]).getResult(0)]);
    });
    const [negOp, addOp] = ops(func);
    expect(new FusionLegality().canFuse(negOp, addOp).legal).toBe(true);
  });

  it('broadcast -> elementwise is legal (consumer iterates at broadcast shape)', () => {
    const inT = new TensorType([4], ScalarType.F32);
    const outT = new TensorType([3, 4], ScalarType.F32);
    const func = buildFunction('f', [inT, outT], [outT], (b, args) => {
      const bc = b.broadcast(args[0], [3, 4], [1]);
      b.returnOp([b.add(bc.getResult(0), args[1]).getResult(0)]);
    });
    const [bcOp, addOp] = ops(func);
    expect(new FusionLegality().canFuse(bcOp, addOp).legal).toBe(true);
  });

  it('elementwise -> reduction is legal with allowReductionFusion=true', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const s = new TensorType([], ScalarType.F32);
    const func = buildFunction('f', [t, t, s], [new TensorType([4], ScalarType.F32)], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.reduce(sum.getResult(0), args[2], [1], 'sum').getResult(0)]);
    });
    const [addOp, reduceOp] = ops(func);
    expect(new FusionLegality().canFuse(addOp, reduceOp).legal).toBe(true);
  });

  it('elementwise -> reduction is illegal with allowReductionFusion=false', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const s = new TensorType([], ScalarType.F32);
    const func = buildFunction('f', [t, t, s], [new TensorType([4], ScalarType.F32)], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.reduce(sum.getResult(0), args[2], [1], 'sum').getResult(0)]);
    });
    const [addOp, reduceOp] = ops(func);
    expect(new FusionLegality({ allowReductionFusion: false }).canFuse(addOp, reduceOp).legal).toBe(false);
  });

  it('reduction -> reduction is illegal (multiple reductions)', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const s = new TensorType([], ScalarType.F32);
    const func = buildFunction('f', [t, s], [new TensorType([], ScalarType.F32)], (b, args) => {
      const r1 = b.reduce(args[0], args[1], [1], 'sum');
      b.returnOp([b.reduce(r1.getResult(0), args[1], [0], 'sum').getResult(0)]);
    });
    const [r1, r2] = ops(func);
    const result = new FusionLegality().canFuse(r1, r2);
    expect(result.legal).toBe(false);
    expect(result.reason).toContain('multiple reductions');
  });

  it('same op with itself is illegal', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const [addOp] = ops(func);
    expect(new FusionLegality().canFuse(addOp, addOp).legal).toBe(false);
  });

  it('null ops are illegal', () => {
    expect(new FusionLegality().canFuse(null, null).legal).toBe(false);
  });

  it('reduction -> elementwise is legal (epilogue pattern)', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const s = new TensorType([], ScalarType.F32);
    const outT = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, s, outT], [outT], (b, args) => {
      const red = b.reduce(args[0], args[1], [1], 'sum');
      const sum = b.add(red.getResult(0), args[2]);
      b.returnOp([sum.getResult(0)]);
    });
    const [redOp, addOp] = ops(func);
    expect(new FusionLegality().canFuse(redOp, addOp).legal).toBe(true);
  });

  it('isOpLowerable caches results for same opName', () => {
    const legality = new FusionLegality();
    const r1 = legality.isOpLowerable('add');
    const r2 = legality.isOpLowerable('add');
    expect(r1).toBe(r2);
    expect(legality._lowerableCache.has('add')).toBe(true);
  });
});

describe('FusionLegality.canMergeGroups', () => {
  it('rejects merge when combined size exceeds maxFusionSize', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const a = b.add(args[0], args[1]);
      const c = b.neg(a.getResult(0));
      const d = b.neg(c.getResult(0));
      const e = b.neg(d.getResult(0));
      b.returnOp([e.getResult(0)]);
    });
    const allOps = ops(func);

    const g1 = new FusionGroup(0);
    g1.addOp(allOps[0]);
    g1.addOp(allOps[1]);
    const g2 = new FusionGroup(1);
    g2.addOp(allOps[2]);
    g2.addOp(allOps[3]);

    const result = new FusionLegality({ maxFusionSize: 3 }).canMergeGroups(g1, g2);
    expect(result.legal).toBe(false);
    expect(result.reason).toContain('max fusion size');
  });

  it('rejects merge when combined groups have > 1 reduction', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const s = new TensorType([], ScalarType.F32);
    const outT = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, s], [outT, outT], (b, args) => {
      const r1 = b.reduce(args[0], args[1], [1], 'sum');
      const r2 = b.reduce(args[0], args[1], [1], 'max');
      b.returnOp([r1.getResult(0), r2.getResult(0)]);
    });
    const allOps = ops(func);

    const g1 = new FusionGroup(0);
    g1.addOp(allOps[0]);
    const g2 = new FusionGroup(1);
    g2.addOp(allOps[1]);

    expect(new FusionLegality().canMergeGroups(g1, g2).legal).toBe(false);
  });

  it('allows merge of two elementwise groups under maxFusionSize', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const a = b.add(args[0], args[1]);
      const c = b.neg(a.getResult(0));
      b.returnOp([c.getResult(0)]);
    });
    const allOps = ops(func);

    const g1 = new FusionGroup(0);
    g1.addOp(allOps[0]);
    const g2 = new FusionGroup(1);
    g2.addOp(allOps[1]);

    expect(new FusionLegality().canMergeGroups(g1, g2).legal).toBe(true);
  });
});
