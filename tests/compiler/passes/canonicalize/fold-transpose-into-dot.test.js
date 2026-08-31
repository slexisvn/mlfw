import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { CanonicalizePass } from '../../../../src/compiler/passes/canonicalize/canonicalize.js';
import { PassResult } from '../../../../src/compiler/passes/pass.js';
import { DCEPass } from '../../../../src/compiler/passes/simplify/dce.js';
import { compile } from '../../../../src/index.js';
import { CPUTarget } from '../../../../src/compiler/support/target.js';
import * as nn from '../../../../src/nn/index.js';
import { randn } from '../../../../src/tensor/factory/creation_ops.js';
import { manualSeed, unseed } from '../../../../src/util/random.js';
import { flat } from '../../../_utils/tensor_data.js';

const F32 = ScalarType.F32;

function run(func) {
  const result = new CanonicalizePass().run(func);
  new DCEPass().run(func);
  return result;
}

function dotOf(func) {
  return func.findOp((op) => op.opName === 'dot');
}

function attrs(op) {
  return {
    lhsB: [...(op.getAttr('lhs_batch') || [])],
    lhsC: [...op.getAttr('lhs_contracting')],
    rhsB: [...(op.getAttr('rhs_batch') || [])],
    rhsC: [...op.getAttr('rhs_contracting')],
  };
}

describe('fold_transpose_into_dot', () => {
  it('folds the batched transpose an attention score matmul produces', () => {
    const q = new TensorType([2, 8, 16], F32);
    const k = new TensorType([2, 8, 16], F32);
    const out = new TensorType([2, 8, 8], F32);
    const func = buildFunction('scores', [q, k], [out], (b, args) => {
      const kt = b.transpose(args[1], [0, 2, 1]).getResult(0);
      b.returnOp([b.dot(args[0], kt, [2], [1], [0], [0]).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.CHANGED);
    expect(func.findOp((op) => op.opName === 'transpose')).toBeNull();
    expect(attrs(dotOf(func))).toEqual({ lhsB: [0], lhsC: [2], rhsB: [0], rhsC: [2] });
    expect(dotOf(func).getResult(0).type.shape).toEqual([2, 8, 8]);
  });

  it('folds a rank-2 transpose', () => {
    const a = new TensorType([4, 6], F32);
    const w = new TensorType([5, 6], F32);
    const out = new TensorType([4, 5], F32);
    const func = buildFunction('linear', [a, w], [out], (b, args) => {
      const wt = b.transpose(args[1], [1, 0]).getResult(0);
      b.returnOp([b.dot(args[0], wt, [1], [0]).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.CHANGED);
    expect(func.findOp((op) => op.opName === 'transpose')).toBeNull();
    expect(attrs(dotOf(func))).toEqual({ lhsB: [], lhsC: [1], rhsB: [], rhsC: [1] });
  });

  it('maps dimensions through the permutation, not its inverse', () => {
    // perm [1, 2, 0] is not an involution: operand dim d reads source dim perm[d].
    const lhs = new TensorType([3, 4, 5], F32);
    const src = new TensorType([5, 3, 4], F32);
    const out = new TensorType([3, 4, 4], F32);
    const func = buildFunction('cycled', [lhs, src], [out], (b, args) => {
      const t = b.transpose(args[1], [1, 2, 0]).getResult(0);
      b.returnOp([b.dot(args[0], t, [2], [2], [0], [0]).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.CHANGED);
    expect(attrs(dotOf(func))).toEqual({ lhsB: [0], lhsC: [2], rhsB: [1], rhsC: [0] });
    expect(dotOf(func).getResult(0).type.shape).toEqual([3, 4, 4]);
  });

  it('declines to fold when the permutation reorders the dot free dimensions', () => {
    // Free dims 1 and 2 map to source dims 2 and 1: folding would transpose the output.
    const lhs = new TensorType([2, 3, 4], F32);
    const src = new TensorType([2, 6, 5], F32);
    const out = new TensorType([2, 3, 5, 6], F32);
    const func = buildFunction('reordered', [lhs, src], [out], (b, args) => {
      const t = b.transpose(args[1], [0, 2, 1]).getResult(0);
      b.returnOp([b.dot(args[0], t, [2], [3], [0], [0]).getResult(0)]);
    });

    run(func);

    expect(func.findOp((op) => op.opName === 'transpose')).not.toBeNull();
  });
});

describe('fold_transpose_into_dot end to end', () => {
  it('removes the score transpose and its buffer from an attention head', () => {
    class Head extends nn.Module {
      constructor(dim) {
        super();
        this.q = new nn.Linear(dim, dim);
        this.k = new nn.Linear(dim, dim);
        this.v = new nn.Linear(dim, dim);
        this.scale = 1 / Math.sqrt(dim);
      }
      forward(x) {
        const q = this.q.forward(x);
        const k = this.k.forward(x);
        const v = this.v.forward(x);
        const scores = q.matmul(k.transpose(-2, -1)).mul(this.scale);
        return nn.F.softmax(scores, -1).matmul(v);
      }
    }

    manualSeed(23);
    const model = new Head(16);
    const x = randn([2, 8, 16]);
    const cf = compile(model, [x], { target: CPUTarget(), scheduling: { enabled: true } });
    const src = cf.source();
    const eager = flat(model.forward(x));
    const compiled = flat(cf(x));
    unseed();

    // A [2, 16, 8] copy of k would need 128 floats and a plain copy loop.
    expect(src).not.toMatch(/=\s*buf_\d+\[\(\(\(i0_\d+ \* 128\) \+ \(i2_\d+ \* 16\)\) \+ i1_\d+\)\]/);
    const kTransposeSized = [...src.matchAll(/new Float32Array\((\d+)\)/g)].filter((m) => Number(m[1]) === 256);
    expect(kTransposeSized.length).toBe(3); // q, k, v only

    let maxRel = 0;
    for (let i = 0; i < compiled.length; i++) {
      maxRel = Math.max(maxRel, Math.abs(compiled[i] - eager[i]) / (1 + Math.abs(eager[i])));
    }
    expect(maxRel).toBeLessThan(1e-6);
  });
});
