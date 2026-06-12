import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { FusionGroup, FusionGroupBuilder } from '../../../src/compiler/passes/fusion/fusion_groups.js';
import { FusionLegality, FusionKind } from '../../../src/compiler/passes/fusion/fusion_analysis.js';

function ops(func) {
  const list = [];
  for (const op of func.ops()) {
    if (op.opName !== 'return') list.push(op);
  }
  return list;
}

describe('FusionGroup.computeIO', () => {
  it('identifies external inputs and externally-used outputs', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t, t], [t, t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      const n = b.neg(sum.getResult(0));
      const standalone = b.mul(args[0], args[2]);
      b.returnOp([n.getResult(0), standalone.getResult(0)]);
    });
    const [addOp, negOp] = ops(func);
    const group = new FusionGroup(0);
    group.addOp(addOp);
    group.addOp(negOp);

    const inputs = group.getInputValues();
    const outputs = group.getOutputValues();

    expect(inputs.length).toBe(2);
    expect(inputs.every(v => v.isBlockArgument())).toBe(true);

    expect(outputs.length).toBe(1);
    expect(outputs[0].definingOp).toBe(negOp);
  });

  it('intermediate value consumed only inside group is not an output', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.neg(sum.getResult(0)).getResult(0)]);
    });
    const [addOp, negOp] = ops(func);
    const group = new FusionGroup(0);
    group.addOp(addOp);
    group.addOp(negOp);

    const outputs = group.getOutputValues();
    expect(outputs.find(v => v.definingOp === addOp)).toBeUndefined();
  });

  it('value used both inside and outside group IS an output', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t, t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      const n = b.neg(sum.getResult(0));
      b.returnOp([sum.getResult(0), n.getResult(0)]);
    });
    const [addOp, negOp] = ops(func);
    const group = new FusionGroup(0);
    group.addOp(addOp);
    group.addOp(negOp);

    const outputs = group.getOutputValues();
    expect(outputs.length).toBe(2);
    expect(outputs.find(v => v.definingOp === addOp)).toBeDefined();
    expect(outputs.find(v => v.definingOp === negOp)).toBeDefined();
  });

  it('merge invalidates cached IO and recomputes correctly', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.neg(sum.getResult(0)).getResult(0)]);
    });
    const [addOp, negOp] = ops(func);

    const g1 = new FusionGroup(0);
    g1.addOp(addOp);
    expect(g1.getOutputValues().length).toBe(1);
    expect(g1.getOutputValues()[0].definingOp).toBe(addOp);

    const g2 = new FusionGroup(1);
    g2.addOp(negOp);
    g1.merge(g2);

    expect(g1.getOutputValues().length).toBe(1);
    expect(g1.getOutputValues()[0].definingOp).toBe(negOp);
  });
});

describe('FusionGroupBuilder.buildProducerConsumerGroups', () => {
  it('merges producer-consumer elementwise chain into one group', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.neg(sum.getResult(0)).getResult(0)]);
    });

    const groups = new FusionGroupBuilder(new FusionLegality()).buildProducerConsumerGroups(func);
    expect(groups.length).toBe(1);
    expect(groups[0].size).toBe(2);
  });

  it('three-op chain folds into a single group', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      const n = b.neg(sum.getResult(0));
      b.returnOp([b.exp(n.getResult(0)).getResult(0)]);
    });

    const groups = new FusionGroupBuilder(new FusionLegality()).buildProducerConsumerGroups(func);
    expect(groups.length).toBe(1);
    expect(groups[0].size).toBe(3);
  });

  it('returns no groups when only independent single ops exist', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });

    const groups = new FusionGroupBuilder(new FusionLegality()).buildProducerConsumerGroups(func);
    expect(groups.length).toBe(0);
  });

  it('respects maxFusionSize — stops merging beyond limit', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const a = b.add(args[0], args[1]);
      const c = b.neg(a.getResult(0));
      const d = b.exp(c.getResult(0));
      b.returnOp([d.getResult(0)]);
    });

    const legality = new FusionLegality({ maxFusionSize: 2 });
    const groups = new FusionGroupBuilder(legality).buildProducerConsumerGroups(func);

    expect(groups.every(g => g.size <= 2)).toBe(true);
  });

  it('reductions are excluded — elementwise+reduce does not form a group', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const s = new TensorType([], ScalarType.F32);
    const func = buildFunction('f', [t, t, s], [new TensorType([4], ScalarType.F32)], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.reduce(sum.getResult(0), args[2], [1], 'sum').getResult(0)]);
    });

    const groups = new FusionGroupBuilder(new FusionLegality()).buildProducerConsumerGroups(func);
    expect(groups.length).toBe(0);
  });

  it('skips constant ops — constants are not fused', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const c = b.scalarConstant(1.0, ScalarType.F32);
      const bc = b.broadcast(c.getResult(0), [4], []);
      const sum = b.add(args[0], bc.getResult(0));
      b.returnOp([sum.getResult(0)]);
    });

    const groups = new FusionGroupBuilder(new FusionLegality()).buildProducerConsumerGroups(func);
    for (const g of groups) {
      for (const op of g.ops) {
        expect(op.opName).not.toBe('constant');
      }
    }
  });
});

describe('FusionGroupBuilder.buildHorizontalGroups', () => {
  it('groups independent same-shape ops sharing inputs', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t, t], (b, args) => {
      const a = b.add(args[0], args[1]);
      const c = b.mul(args[0], args[1]);
      b.returnOp([a.getResult(0), c.getResult(0)]);
    });

    const groups = new FusionGroupBuilder(new FusionLegality()).buildHorizontalGroups(func);
    expect(groups.length).toBe(1);
    expect(groups[0].kind).toBe(FusionKind.HORIZONTAL);
    expect(groups[0].size).toBe(2);
  });

  it('rejects horizontal grouping when shapes differ', () => {
    const t1 = new TensorType([4], ScalarType.F32);
    const t2 = new TensorType([8], ScalarType.F32);
    const func = buildFunction('f', [t1, t2], [t1, t2], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0), b.neg(args[1]).getResult(0)]);
    });

    const groups = new FusionGroupBuilder(new FusionLegality()).buildHorizontalGroups(func);
    expect(groups.length).toBe(0);
  });

  it('rejects horizontal grouping when ops are dependent', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const a = b.neg(args[0]);
      b.returnOp([b.neg(a.getResult(0)).getResult(0)]);
    });

    const groups = new FusionGroupBuilder(new FusionLegality()).buildHorizontalGroups(func);
    expect(groups.length).toBe(0);
  });

  it('rejects horizontal grouping when ops are TRANSITIVELY dependent (not just direct)', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const a = b.neg(args[0]);
      const m = b.neg(a.getResult(0));
      b.returnOp([b.neg(m.getResult(0)).getResult(0)]);
    });

    const groups = new FusionGroupBuilder(new FusionLegality()).buildHorizontalGroups(func);
    expect(groups.length).toBe(0);
  });

  it('rejects transitive dependency routed through a different-shape intermediate', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const a = b.relu(args[0]);
      const r = b.reshape(a.getResult(0), [2, 2]);
      const back = b.reshape(r.getResult(0), [4]);
      const a2 = b.relu(back.getResult(0));
      b.returnOp([b.add(a.getResult(0), a2.getResult(0)).getResult(0)]);
    });

    const groups = new FusionGroupBuilder(new FusionLegality()).buildHorizontalGroups(func);
    expect(groups.length).toBe(0);
  });
});

describe('FusionGroupBuilder scales linearly on large graphs (no O(n^2)/O(n^3) blowup)', () => {
  function makeWide(n) {
    const t = new TensorType([8, 8], ScalarType.F32);
    return buildFunction('f', [t, t], [t], (b, args) => {
      let acc = args[0];
      const outs = [];
      for (let i = 0; i < n; i++) {
        const r = b.relu(b.add(acc, args[1]).getResult(0)).getResult(0);
        outs.push(r);
        acc = r;
      }
      let s = outs[0];
      for (let i = 1; i < outs.length; i++) s = b.add(s, outs[i]).getResult(0);
      b.returnOp([s]);
    });
  }

  it('fuses ~800 same-shape ops well under a wall-clock bound that O(n^2) would blow', () => {
    const func = makeWide(800);
    const t0 = performance.now();
    new FusionGroupBuilder(new FusionLegality()).buildAllGroups(func);
    const elapsed = performance.now() - t0;
    expect(elapsed).toBeLessThan(5000);
  });
});

describe('FusionGroupBuilder.buildAllGroups', () => {
  it('producer-consumer groups exclude ops from horizontal grouping', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t, t], (b, args) => {
      const a = b.add(args[0], args[1]);
      const c = b.neg(a.getResult(0));
      const d = b.mul(args[0], args[1]);
      b.returnOp([c.getResult(0), d.getResult(0)]);
    });

    const groups = new FusionGroupBuilder(new FusionLegality()).buildAllGroups(func);

    const fusedOps = new Set();
    for (const g of groups) {
      for (const op of g.ops) {
        expect(fusedOps.has(op)).toBe(false);
        fusedOps.add(op);
      }
    }
  });
});

describe('FusionGroup.allOpsInlineFusable', () => {
  it('returns true for pure elementwise group', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const group = new FusionGroup(0);
    group.addOp(ops(func)[0]);
    expect(group.allOpsInlineFusable()).toBe(true);
  });
});
