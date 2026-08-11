import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType, Layout } from '../../../../src/compiler/ir/graph/types.js';
import { LayoutPolicy, LayoutPreference } from '../../../../src/compiler/passes/layout/layout_policy.js';
import { CPUTarget, CUDATarget, WasmTarget } from '../../../../src/backend/target.js';

function ops(func) {
  const list = [];
  for (const op of func.ops()) {
    if (op.opName !== 'return') list.push(op);
  }
  return list;
}

describe('LayoutPolicy conv rule', () => {
  it('GPU conv prefers NHWC layout for input and output', () => {
    const inp = new TensorType([1, 3, 224, 224], ScalarType.F32);
    const ker = new TensorType([64, 3, 3, 3], ScalarType.F32);
    const out = new TensorType([1, 64, 222, 222], ScalarType.F32);
    const func = buildFunction('f', [inp, ker], [out], (b, args) => {
      b.returnOp([b.conv(args[0], args[1], [1, 1], [0, 0, 0, 0]).getResult(0)]);
    });

    const policy = new LayoutPolicy(CUDATarget());
    const pref = policy.getPreference(ops(func)[0]);
    const nhwc = new Layout([0, 2, 3, 1]);
    expect(pref.inputs[0].equals(nhwc)).toBe(true);
    expect(pref.inputs[1]).toBeNull();
    expect(pref.outputs[0].equals(nhwc)).toBe(true);
  });

  it('CPU conv also prefers NHWC for rank-4', () => {
    const inp = new TensorType([1, 3, 32, 32], ScalarType.F32);
    const ker = new TensorType([16, 3, 3, 3], ScalarType.F32);
    const out = new TensorType([1, 16, 30, 30], ScalarType.F32);
    const func = buildFunction('f', [inp, ker], [out], (b, args) => {
      b.returnOp([b.conv(args[0], args[1], [1, 1], [0, 0, 0, 0]).getResult(0)]);
    });

    const policy = new LayoutPolicy(CPUTarget());
    const pref = policy.getPreference(ops(func)[0]);
    expect(pref.inputs[0].equals(new Layout([0, 2, 3, 1]))).toBe(true);
  });

  it('preferredConvLayout overrides default NHWC', () => {
    const custom = new Layout([0, 3, 1, 2]);
    const target = CUDATarget({ preferredConvLayout: custom });
    const inp = new TensorType([1, 3, 32, 32], ScalarType.F32);
    const ker = new TensorType([16, 3, 3, 3], ScalarType.F32);
    const out = new TensorType([1, 16, 30, 30], ScalarType.F32);
    const func = buildFunction('f', [inp, ker], [out], (b, args) => {
      b.returnOp([b.conv(args[0], args[1], [1, 1], [0, 0, 0, 0]).getResult(0)]);
    });

    const pref = new LayoutPolicy(target).getPreference(ops(func)[0]);
    expect(pref.inputs[0].equals(custom)).toBe(true);
    expect(pref.outputs[0].equals(custom)).toBe(true);
  });
});

describe('LayoutPolicy conv rule rank guard', () => {
  function fakeConv(inputRank) {
    return {
      opName: 'conv',
      getOperand(i) {
        return { type: { rank: inputRank } };
      }
    };
  }

  it('GPU conv with non-rank-4 input returns null (no invalid NHWC permutation)', () => {
    const policy = new LayoutPolicy(CUDATarget());
    expect(policy.getPreference(fakeConv(3))).toBeNull();
    expect(policy.getPreference(fakeConv(5))).toBeNull();
  });

  it('CPU conv with non-rank-4 input returns null', () => {
    const policy = new LayoutPolicy(CPUTarget());
    expect(policy.getPreference(fakeConv(3))).toBeNull();
  });

  it('GPU conv with rank-4 still yields valid NHWC permutation', () => {
    const policy = new LayoutPolicy(CUDATarget());
    const pref = policy.getPreference(fakeConv(4));
    expect(pref.inputs[0].equals(new Layout([0, 2, 3, 1]))).toBe(true);
  });
});

describe('LayoutPolicy dot rule', () => {
  it('CPU dot: LHS row-major, RHS column-major for rank-2', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const out = new TensorType([4, 6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    const pref = new LayoutPolicy(CPUTarget()).getPreference(ops(func)[0]);
    expect(pref.inputs[0].equals(Layout.rowMajor(2))).toBe(true);
    expect(pref.inputs[1].equals(Layout.columnMajor(2))).toBe(true);
    expect(pref.outputs[0].equals(Layout.rowMajor(2))).toBe(true);
  });

  it('GPU dot: both LHS and RHS row-major', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const out = new TensorType([4, 6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    const pref = new LayoutPolicy(CUDATarget()).getPreference(ops(func)[0]);
    expect(pref.inputs[0].equals(Layout.rowMajor(2))).toBe(true);
    expect(pref.inputs[1].equals(Layout.rowMajor(2))).toBe(true);
  });

  it('CPU dot with rank-3 RHS: RHS stays row-major (column-major only for rank-2)', () => {
    const lhs = new TensorType([2, 4, 8], ScalarType.F32);
    const rhs = new TensorType([2, 8, 6], ScalarType.F32);
    const out = new TensorType([2, 4, 6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    const pref = new LayoutPolicy(CPUTarget()).getPreference(ops(func)[0]);
    expect(pref.inputs[1].equals(Layout.rowMajor(3))).toBe(true);
  });
});

describe('LayoutPolicy reduce rule', () => {
  it('reduce output uses row-major layout', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const s = new TensorType([], ScalarType.F32);
    const outT = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, s], [outT], (b, args) => {
      b.returnOp([b.reduce(args[0], args[1], [1], 'sum').getResult(0)]);
    });

    const pref = new LayoutPolicy(CPUTarget()).getPreference(ops(func)[0]);
    expect(pref.outputs[0].equals(Layout.rowMajor(1))).toBe(true);
    expect(pref.inputs[0]).toBeNull();
  });
});

describe('LayoutPolicy.getPreference', () => {
  it('returns null for ops without a registered rule', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });

    expect(new LayoutPolicy(CPUTarget()).getPreference(ops(func)[0])).toBeNull();
  });

  it('registerRule overrides default rule', () => {
    const policy = new LayoutPolicy(CPUTarget());
    const custom = new Layout([1, 0]);

    policy.registerRule('dot', () => new LayoutPreference([custom, custom], [custom]));

    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [new TensorType([4, 6], ScalarType.F32)], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    const pref = policy.getPreference(ops(func)[0]);
    expect(pref.inputs[0].equals(custom)).toBe(true);
    expect(pref.inputs[1].equals(custom)).toBe(true);
  });
});

describe('LayoutPolicy.estimateConversionCost', () => {
  it('same layout costs 0', () => {
    const policy = new LayoutPolicy(CPUTarget());
    const layout = Layout.rowMajor(2);
    expect(policy.estimateConversionCost(layout, layout, new TensorType([4, 8], ScalarType.F32))).toBe(0);
  });

  it('different layouts cost = numel * 2', () => {
    const policy = new LayoutPolicy(CPUTarget());
    const from = Layout.rowMajor(2);
    const to = Layout.columnMajor(2);
    const t = new TensorType([4, 8], ScalarType.F32);
    expect(policy.estimateConversionCost(from, to, t)).toBe(32 * 2);
  });

  it('DYNAMIC shape costs a fixed fallback (1024)', () => {
    const policy = new LayoutPolicy(CPUTarget());
    const from = Layout.rowMajor(2);
    const to = Layout.columnMajor(2);
    const t = new TensorType([-1, 8], ScalarType.F32);
    expect(policy.estimateConversionCost(from, to, t)).toBe(1024);
  });
});

describe('LayoutPolicy.estimateBenefit', () => {
  function ops(func) {
    const list = [];
    for (const op of func.ops()) {
      if (op.opName !== 'return') list.push(op);
    }
    return list;
  }

  it('dot/conv/matmul consumers get high benefit (numEl * 4 * useCount)', () => {
    const lhs = new TensorType([16, 32], ScalarType.F32);
    const rhs = new TensorType([32, 16], ScalarType.F32);
    const out = new TensorType([16, 16], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    const policy = new LayoutPolicy(CPUTarget());
    const dotOp = ops(func)[0];
    const benefit = policy.estimateBenefit(dotOp, rhs, 1);
    expect(benefit).toBe(16 * 32 * 4 * 1);
  });

  it('reduce consumers get medium benefit (numEl * 2 * useCount)', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const s = new TensorType([], ScalarType.F32);
    const outT = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, s], [outT], (b, args) => {
      b.returnOp([b.reduce(args[0], args[1], [1], 'sum').getResult(0)]);
    });

    const policy = new LayoutPolicy(CPUTarget());
    const reduceOp = ops(func)[0];
    const benefit = policy.estimateBenefit(reduceOp, t, 1);
    expect(benefit).toBe(32 * 2 * 1);
  });

  it('elementwise consumers get low benefit', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });

    const policy = new LayoutPolicy(CPUTarget());
    const addOp = ops(func)[0];
    const benefit = policy.estimateBenefit(addOp, t, 1);
    expect(benefit).toBe(Math.floor(256 * 0.5));
  });

  it('small tensor (fits in cache) gets zero benefit for elementwise', () => {
    const t = new TensorType([8], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });

    const policy = new LayoutPolicy(CPUTarget());
    const addOp = ops(func)[0];
    expect(policy.estimateBenefit(addOp, t, 1)).toBe(0);
  });

  it('small tensor still gets benefit for dot (compute-intensive)', () => {
    const lhs = new TensorType([4, 4], ScalarType.F32);
    const rhs = new TensorType([4, 4], ScalarType.F32);
    const out = new TensorType([4, 4], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    const policy = new LayoutPolicy(CPUTarget());
    const dotOp = ops(func)[0];
    expect(policy.estimateBenefit(dotOp, rhs, 1)).toBeGreaterThan(0);
  });

  it('useCount multiplies benefit', () => {
    const lhs = new TensorType([16, 32], ScalarType.F32);
    const rhs = new TensorType([32, 16], ScalarType.F32);
    const out = new TensorType([16, 16], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    const policy = new LayoutPolicy(CPUTarget());
    const dotOp = ops(func)[0];
    const b1 = policy.estimateBenefit(dotOp, rhs, 1);
    const b3 = policy.estimateBenefit(dotOp, rhs, 3);
    expect(b3).toBe(b1 * 3);
  });
});
