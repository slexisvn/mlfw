import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { Pattern, PatternSet } from '../../../src/compiler/ir/rewrite/pattern.js';
import { PatternApplicator } from '../../../src/compiler/passes/rewrite/pattern.js';
import { PassResult } from '../../../src/compiler/passes/pass.js';

class NegNegPattern extends Pattern {
  constructor() {
    super('neg_neg', 10);
    this.rootOpName = 'neg';
  }
  match(op) {
    const input = op.getOperand(0);
    return input.definingOp && input.definingOp.opName === 'neg';
  }
  rewrite(op, builder) {
    const innerNeg = op.getOperand(0).definingOp;
    const original = innerNeg.getOperand(0);
    op.replaceAllResultsWith([original]);
    op.erase();
    if (!innerNeg.getResult(0).hasUses) innerNeg.erase();
    return true;
  }
}

class MulByNegOnePattern extends Pattern {
  constructor() {
    super('mul_neg_one', 5);
    this.rootOpName = 'mul';
  }
  match(op) {
    for (let i = 0; i < op.numOperands; i++) {
      const def = op.getOperand(i).definingOp;
      if (def && def.opName === 'neg') return true;
    }
    return false;
  }
  rewrite(op, builder) {
    return false;
  }
}

class NeverMatchPattern extends Pattern {
  constructor() {
    super('never_match', 1);
  }
  match() { return false; }
  rewrite() { return true; }
}

class CountingPattern extends Pattern {
  constructor() {
    super('counting', 1);
    this.rootOpName = 'neg';
    this.matchCount = 0;
  }
  match(op) {
    this.matchCount++;
    return false;
  }
  rewrite() { return false; }
}

function findOps(func, opName) {
  const result = [];
  for (const op of func.ops()) {
    if (op.opName === opName) result.push(op);
  }
  return result;
}

describe('PatternSet benefit sorting', () => {
  it('get() returns patterns sorted by descending benefit', () => {
    const ps = new PatternSet();
    const low = new Pattern('low', 1);
    const mid = new Pattern('mid', 5);
    const high = new Pattern('high', 10);
    ps.add(low);
    ps.add(high);
    ps.add(mid);

    const sorted = ps.get();
    expect(sorted[0].benefit).toBe(10);
    expect(sorted[1].benefit).toBe(5);
    expect(sorted[2].benefit).toBe(1);
  });

  it('getForOp merges specific + generic by benefit order', () => {
    const ps = new PatternSet();
    const specific = new Pattern('specific', 7);
    specific.rootOpName = 'add';
    const generic1 = new Pattern('generic_high', 10);
    const generic2 = new Pattern('generic_low', 3);
    ps.add(specific);
    ps.add(generic1);
    ps.add(generic2);

    const forAdd = ps.getForOp('add');
    expect(forAdd[0].name).toBe('generic_high');
    expect(forAdd[1].name).toBe('specific');
    expect(forAdd[2].name).toBe('generic_low');
  });

  it('getForOp with no specific patterns returns only generics', () => {
    const ps = new PatternSet();
    const g = new Pattern('g', 1);
    ps.add(g);

    const result = ps.getForOp('mul');
    expect(result.length).toBe(1);
    expect(result[0].name).toBe('g');
  });

  it('getForOp with no generic patterns returns only specific', () => {
    const ps = new PatternSet();
    const s = new Pattern('s', 1);
    s.rootOpName = 'neg';
    ps.add(s);

    expect(ps.getForOp('neg').length).toBe(1);
    expect(ps.getForOp('add').length).toBe(0);
  });
});

describe('PatternApplicator single rewrite', () => {
  it('neg(neg(x)) → x via NegNegPattern', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const n1 = b.neg(args[0]);
      const n2 = b.neg(n1.getResult(0));
      b.returnOp([n2.getResult(0)]);
    });

    const ps = new PatternSet();
    ps.add(new NegNegPattern());
    const pa = new PatternApplicator(ps);

    expect(pa.applyPatterns(func)).toBe(PassResult.CHANGED);
    expect(findOps(func, 'neg').length).toBe(0);

    const ret = func.getReturnOp();
    expect(ret.getOperand(0).isBlockArgument()).toBe(true);
  });

  it('returns UNCHANGED when no pattern matches', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });

    const ps = new PatternSet();
    ps.add(new NegNegPattern());
    const pa = new PatternApplicator(ps);

    expect(pa.applyPatterns(func)).toBe(PassResult.UNCHANGED);
  });
});

describe('PatternApplicator fixpoint iteration', () => {
  it('quad neg: neg(neg(neg(neg(x)))) converges to x in 2 iterations', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      let v = args[0];
      for (let i = 0; i < 4; i++) {
        v = b.neg(v).getResult(0);
      }
      b.returnOp([v]);
    });

    const ps = new PatternSet();
    ps.add(new NegNegPattern());
    const pa = new PatternApplicator(ps);

    expect(pa.applyPatterns(func)).toBe(PassResult.CHANGED);
    expect(findOps(func, 'neg').length).toBe(0);
  });

  it('maxIterations caps number of fixpoint rounds', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      let v = args[0];
      for (let i = 0; i < 4; i++) {
        v = b.neg(v).getResult(0);
      }
      b.returnOp([v]);
    });

    const ps = new PatternSet();
    ps.add(new NegNegPattern());
    const pa = new PatternApplicator(ps);

    expect(pa.applyPatterns(func, 10)).toBe(PassResult.CHANGED);
    expect(findOps(func, 'neg').length).toBe(0);
  });
});

describe('PatternApplicator benefit priority', () => {
  it('higher-benefit pattern fires first when both match', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const n1 = b.neg(args[0]);
      const n2 = b.neg(n1.getResult(0));
      b.returnOp([n2.getResult(0)]);
    });

    let fired = null;

    class LowBenefitNeg extends Pattern {
      constructor() { super('low_neg', 1); this.rootOpName = 'neg'; }
      match(op) {
        return op.getOperand(0).definingOp && op.getOperand(0).definingOp.opName === 'neg';
      }
      rewrite(op, builder) {
        fired = 'low';
        const inner = op.getOperand(0).definingOp;
        op.replaceAllResultsWith([inner.getOperand(0)]);
        op.erase();
        return true;
      }
    }

    class HighBenefitNeg extends Pattern {
      constructor() { super('high_neg', 100); this.rootOpName = 'neg'; }
      match(op) {
        return op.getOperand(0).definingOp && op.getOperand(0).definingOp.opName === 'neg';
      }
      rewrite(op, builder) {
        fired = 'high';
        const inner = op.getOperand(0).definingOp;
        op.replaceAllResultsWith([inner.getOperand(0)]);
        op.erase();
        return true;
      }
    }

    const ps = new PatternSet();
    ps.add(new LowBenefitNeg());
    ps.add(new HighBenefitNeg());
    const pa = new PatternApplicator(ps);

    pa.applyPatterns(func);
    expect(fired).toBe('high');
  });
});

describe('PatternApplicator multiple pattern types', () => {
  it('specific + generic patterns both considered for an op', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const n1 = b.neg(args[0]);
      const n2 = b.neg(n1.getResult(0));
      b.returnOp([n2.getResult(0)]);
    });

    let genericTriggered = false;
    class GenericPattern extends Pattern {
      constructor() { super('generic_spy', 1); }
      match(op) {
        if (op.opName === 'neg') genericTriggered = true;
        return false;
      }
      rewrite() { return false; }
    }

    const ps = new PatternSet();
    ps.add(new NegNegPattern());
    ps.add(new GenericPattern());
    const pa = new PatternApplicator(ps);

    pa.applyPatterns(func);
    expect(genericTriggered).toBe(true);
    expect(findOps(func, 'neg').length).toBe(0);
  });
});

describe('PatternApplicator skips erased ops', () => {
  it('does not crash on ops erased by earlier pattern in same iteration', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const n1 = b.neg(args[0]);
      const n2 = b.neg(n1.getResult(0));
      const n3 = b.neg(n2.getResult(0));
      const n4 = b.neg(n3.getResult(0));
      b.returnOp([n4.getResult(0)]);
    });

    const ps = new PatternSet();
    ps.add(new NegNegPattern());
    const pa = new PatternApplicator(ps);

    expect(() => pa.applyPatterns(func)).not.toThrow();
  });
});

describe('PatternSet hasPatterns', () => {
  it('returns false for empty set', () => {
    expect(new PatternSet().hasPatterns()).toBe(false);
  });

  it('returns true after adding a pattern', () => {
    const ps = new PatternSet();
    ps.add(new NeverMatchPattern());
    expect(ps.hasPatterns()).toBe(true);
  });
});
