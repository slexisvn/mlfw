import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { EpilogueFusionPass } from '../../../../src/compiler/passes/fusion/epilogue_fusion.js';
import { PassResult } from '../../../../src/compiler/passes/pass.js';

function run(func, config = {}) {
  return new EpilogueFusionPass(config).run(func);
}

function findOps(func, opName) {
  const result = [];
  for (const op of func.ops()) {
    if (op.opName === opName) result.push(op);
  }
  return result;
}

function assertNoUseBeforeDef(func) {
  const defined = new Set(func.entryBlock.arguments);
  for (const op of func.entryBlock.ops()) {
    for (let i = 0; i < op.numOperands; i++) {
      const operand = op.getOperand(i);
      if (operand.isBlockArgument()) continue;
      expect(defined.has(operand)).toBe(true);
    }
    for (let i = 0; i < op.numResults; i++) defined.add(op.getResult(i));
  }
}

describe('EpilogueFusionPass', () => {
  it('fuses dot + add(bias) into fused_dot_epilogue', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const out = new TensorType([4, 6], ScalarType.F32);
    const bias = new TensorType([6], ScalarType.F32);
    const biasBC = new TensorType([4, 6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs, bias], [out], (b, args) => {
      const dot = b.matmul(args[0], args[1]);
      const bc = b.broadcast(args[2], [4, 6], [1]);
      const biased = b.add(dot.getResult(0), bc.getResult(0));
      b.returnOp([biased.getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.CHANGED);

    const fused = findOps(func, 'fused_dot_epilogue');
    expect(fused.length).toBe(1);
    expect(findOps(func, 'dot').length).toBe(0);
    expect(findOps(func, 'add').length).toBe(0);

    const tags = fused[0].getAttr('epilogue_tags');
    expect(tags).toContain('bias');
  });

  it('fuses dot + relu into fused_dot_epilogue with relu tag', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const out = new TensorType([4, 6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out], (b, args) => {
      const dot = b.matmul(args[0], args[1]);
      const zero = b.scalarConstant(0, ScalarType.F32);
      const bc = b.broadcast(zero.getResult(0), [4, 6], []);
      const result = b.maximum(dot.getResult(0), bc.getResult(0));
      b.returnOp([result.getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.CHANGED);

    const fused = findOps(func, 'fused_dot_epilogue');
    expect(fused.length).toBe(1);
    const tags = fused[0].getAttr('epilogue_tags');
    expect(tags).toContain('relu');
  });

  it('returns UNCHANGED when no dot ops exist', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
  });

  it('returns UNCHANGED when dot has no epilogue users', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const out = new TensorType([4, 6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
  });

  it('fused op preserves dot attributes (contracting dims)', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const out = new TensorType([4, 6], ScalarType.F32);
    const bias = new TensorType([6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs, bias], [out], (b, args) => {
      const dot = b.matmul(args[0], args[1]);
      const bc = b.broadcast(args[2], [4, 6], [1]);
      const biased = b.add(dot.getResult(0), bc.getResult(0));
      b.returnOp([biased.getResult(0)]);
    });

    run(func);

    const fused = findOps(func, 'fused_dot_epilogue')[0];
    expect(fused.getAttr('lhs_contracting')).toEqual([1]);
    expect(fused.getAttr('rhs_contracting')).toEqual([0]);
    expect(fused.getAttr('num_dot_operands')).toBe(2);
  });

  it('inserts fused op after an extra operand defined later than the dot', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const out = new TensorType([4, 6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out], (b, args) => {
      const dot = b.matmul(args[0], args[1]);
      const relu = b.maximum(
        dot.getResult(0),
        b.broadcast(b.scalarConstant(0, ScalarType.F32).getResult(0), [4, 6], []).getResult(0)
      );
      const lateBias = b.matmul(args[0], args[1]);
      const biased = b.add(relu.getResult(0), lateBias.getResult(0));
      b.returnOp([biased.getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.CHANGED);
    assertNoUseBeforeDef(func);
  });

  it('does not leave a dangling ref when a chain op result escapes the chain', () => {
    const lhs = new TensorType([8, 16], ScalarType.F32);
    const rhs = new TensorType([16, 16], ScalarType.F32);
    const out = new TensorType([8, 16], ScalarType.F32);
    const func = buildFunction('escape', [lhs, rhs, rhs], [out], (b, args) => {
      const [x, w1, w2] = args;
      const dot1 = b.matmul(x, w1).getResult(0);
      const relu1 = b.maximum(
        dot1,
        b.broadcast(b.scalarConstant(0, ScalarType.F32).getResult(0), [8, 16], []).getResult(0)
      ).getResult(0);
      const r1 = b.add(relu1, x).getResult(0);
      const dot2 = b.matmul(r1, w2).getResult(0);
      const r2 = b.add(dot2, r1).getResult(0);
      b.returnOp([r2]);
    });

    run(func);
    assertNoUseBeforeDef(func);
  });

  it('respects target.enableEpilogueFusion = false', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const out = new TensorType([4, 6], ScalarType.F32);
    const bias = new TensorType([6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs, bias], [out], (b, args) => {
      const dot = b.matmul(args[0], args[1]);
      const bc = b.broadcast(args[2], [4, 6], [1]);
      b.returnOp([b.add(dot.getResult(0), bc.getResult(0)).getResult(0)]);
    });

    expect(run(func, { target: { enableEpilogueFusion: false } })).toBe(PassResult.UNCHANGED);
  });
});
