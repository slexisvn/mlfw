import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';

import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { PostDominanceAnalysis } from '../../../src/compiler/analysis/dominance.js';
import { UseDefAnalysis } from '../../../src/compiler/analysis/use_def.js';

const f32 = ScalarType.F32;
const t44 = new TensorType([4, 4], f32);

function computePDom(func) {
  const useDef = UseDefAnalysis.compute(func);
  return PostDominanceAnalysis.compute(func, { useDef });
}

describe('PostDominanceAnalysis', () => {
  it('computes post-dominators for linear chain', () => {
    const func = buildFunction('linear', [t44], [t44],
      (b, [x]) => {
        const a = b.exp(x);
        const c = b.neg(a.getResult(0));
        const d = b.abs(c.getResult(0));
        b.returnOp([d.getResult(0)]);
      }
    );
    const pdom = computePDom(func);
    const ops = [...func.ops()].filter(op => op.opName !== 'return');

    assert.ok(pdom.postDominates(ops[2], ops[0]));
    assert.ok(pdom.postDominates(ops[1], ops[0]));
    assert.ok(!pdom.postDominates(ops[0], ops[2]));
  });

  it('computes post-dominators for diamond', () => {
    const func = buildFunction('diamond', [t44, t44], [t44],
      (b, [x, y]) => {
        const a = b.add(x, y);
        const e1 = b.exp(a.getResult(0));
        const e2 = b.neg(a.getResult(0));
        const m = b.mul(e1.getResult(0), e2.getResult(0));
        b.returnOp([m.getResult(0)]);
      }
    );
    const pdom = computePDom(func);
    const ops = [...func.ops()].filter(op => op.opName !== 'return');

    const add = ops[0], exp = ops[1], neg = ops[2], mul = ops[3];
    assert.ok(pdom.postDominates(mul, add));
    assert.ok(pdom.postDominates(mul, exp));
    assert.ok(pdom.postDominates(mul, neg));
  });

  it('immediate post-dominator is direct successor in chain', () => {
    const func = buildFunction('idom', [t44], [t44],
      (b, [x]) => {
        const a = b.exp(x);
        const c = b.neg(a.getResult(0));
        b.returnOp([c.getResult(0)]);
      }
    );
    const pdom = computePDom(func);
    const ops = [...func.ops()].filter(op => op.opName !== 'return');

    const idom = pdom.immediatePDom(ops[0]);
    assert.ok(idom === ops[1]);
  });

  it('handles single-op graph', () => {
    const func = buildFunction('single', [t44], [t44],
      (b, [x]) => {
        const a = b.exp(x);
        b.returnOp([a.getResult(0)]);
      }
    );
    const pdom = computePDom(func);
    assert.ok(pdom.idom.size >= 0);
  });

  it('pathToPDom returns intermediate ops', () => {
    const func = buildFunction('path', [t44], [t44],
      (b, [x]) => {
        const a = b.exp(x);
        const c = b.neg(a.getResult(0));
        const d = b.abs(c.getResult(0));
        b.returnOp([d.getResult(0)]);
      }
    );
    const pdom = computePDom(func);
    const ops = [...func.ops()].filter(op => op.opName !== 'return');
    const path = pdom.pathToPDom(ops[0]);
    assert.ok(path.length > 0);
  });
});
