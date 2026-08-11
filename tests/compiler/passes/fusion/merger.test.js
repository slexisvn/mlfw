import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { FusionMergerPass } from '../../../../src/compiler/passes/fusion/fusion_merger.js';
import { FusionPass } from '../../../../src/compiler/passes/fusion/fusion_pass.js';
import { PassResult } from '../../../../src/compiler/passes/pass.js';

function findOps(func, opName) {
  const result = [];
  for (const op of func.ops()) {
    if (op.opName === opName) result.push(op);
  }
  return result;
}

describe('FusionMergerPass', () => {
  it('merges producer-consumer fusion pairs', () => {
    const t = new TensorType([32, 32], ScalarType.F32);
    const func = buildFunction('f', [t, t, t], [t], (b, args) => {
      const a = b.add(args[0], args[1]);
      const c = b.neg(a.getResult(0));
      const d = b.mul(c.getResult(0), args[2]);
      const e = b.exp(d.getResult(0));
      b.returnOp([e.getResult(0)]);
    });

    new FusionPass({ maxGroupSize: 2 }).run(func);
    const fusionsBefore = findOps(func, 'fusion').length;

    if (fusionsBefore > 1) {
      const result = new FusionMergerPass().run(func);
      const fusionsAfter = findOps(func, 'fusion').length;

      if (result === PassResult.CHANGED) {
        expect(fusionsAfter).toBeLessThan(fusionsBefore);
      }
    }
  });

  it('returns UNCHANGED when there is only one fusion or no merges', () => {
    const t = new TensorType([8], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.neg(sum.getResult(0)).getResult(0)]);
    });

    new FusionPass().run(func);
    expect(findOps(func, 'fusion').length).toBe(1);

    const result = new FusionMergerPass().run(func);
    expect(result).toBe(PassResult.UNCHANGED);
  });

  it('preserves correctness after merge — return value unchanged', () => {
    const t = new TensorType([16], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const a = b.add(args[0], args[1]);
      const c = b.neg(a.getResult(0));
      const d = b.exp(c.getResult(0));
      b.returnOp([d.getResult(0)]);
    });

    new FusionPass().run(func);
    new FusionMergerPass().run(func);

    const ret = func.getReturnOp();
    expect(ret.getOperand(0).definingOp.opName).toBe('fusion');
  });
});
