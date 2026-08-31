import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { FusionCostModel } from '../../../../src/compiler/passes/fusion/fusion_cost.js';
import { FusionGroup } from '../../../../src/compiler/passes/fusion/fusion_groups.js';
import { CPUTarget, CUDATarget } from '../../../../src/compiler/support/target.js';

function ops(func) {
  const list = [];
  for (const op of func.ops()) {
    if (op.opName !== 'return') list.push(op);
  }
  return list;
}

function makeGroup(opList) {
  const group = new FusionGroup(0);
  for (const op of opList) group.addOp(op);
  return group;
}

describe('FusionCostModel.estimateFLOPs', () => {
  it('elementwise op FLOPs = output elements', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    expect(new FusionCostModel().estimateFLOPs(ops(func)[0])).toBe(32);
  });

  it('reduction op FLOPs = input elements (reduces over input)', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const s = new TensorType([], ScalarType.F32);
    const func = buildFunction('f', [t, s], [new TensorType([4], ScalarType.F32)], (b, args) => {
      b.returnOp([b.reduce(args[0], args[1], [1], 'sum').getResult(0)]);
    });
    expect(new FusionCostModel().estimateFLOPs(ops(func)[0])).toBe(32);
  });
});

describe('FusionCostModel.estimateBytes', () => {
  it('sums all input + output tensor sizes in bytes', () => {
    const t = new TensorType([16], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    expect(new FusionCostModel().estimateBytes(ops(func)[0])).toBe(16 * 4 * 3);
  });

  it('reduction: input bytes > output bytes', () => {
    const t = new TensorType([8, 16], ScalarType.F32);
    const s = new TensorType([], ScalarType.F32);
    const outT = new TensorType([8], ScalarType.F32);
    const func = buildFunction('f', [t, s], [outT], (b, args) => {
      b.returnOp([b.reduce(args[0], args[1], [1], 'sum').getResult(0)]);
    });
    const cost = new FusionCostModel();
    const bytes = cost.estimateBytes(ops(func)[0]);
    expect(bytes).toBe(8 * 16 * 4 + 4 + 8 * 4);
  });
});

describe('FusionCostModel.estimateGroupCost', () => {
  it('memorySaved = unfusedBytes - fusedBytes (eliminates intermediate)', () => {
    const t = new TensorType([64], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.neg(sum.getResult(0)).getResult(0)]);
    });
    const group = makeGroup(ops(func));
    const result = new FusionCostModel().estimateGroupCost(group);

    expect(result.fusedBytes).toBe(64 * 4 * 3);
    expect(result.unfusedBytes).toBe(64 * 4 * 3 + 64 * 4 * 2);
    expect(result.memorySaved).toBe(result.unfusedBytes - result.fusedBytes);
  });

  it('launchSaved = (groupSize - 1) * launchOverheadUs', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const a = b.add(args[0], args[1]);
      const c = b.neg(a.getResult(0));
      b.returnOp([b.exp(c.getResult(0)).getResult(0)]);
    });
    const cost = new FusionCostModel({ launchOverheadUs: 10 });
    expect(cost.estimateGroupCost(makeGroup(ops(func))).launchSaved).toBe(20);
  });

  it('parallelismLoss is nonzero when mixing elementwise + reduction', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const s = new TensorType([], ScalarType.F32);
    const func = buildFunction('f', [t, t, s], [new TensorType([4], ScalarType.F32)], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.reduce(sum.getResult(0), args[2], [1], 'sum').getResult(0)]);
    });
    const result = new FusionCostModel().estimateGroupCost(makeGroup(ops(func)));
    expect(result.parallelismLoss).toBeGreaterThan(0);
  });

  it('libraryCallLoss counts ops the target has a library kernel for', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.neg(sum.getResult(0)).getResult(0)]);
    });
    const cost = new FusionCostModel({ hasLibraryOp: (name) => name === 'add' });
    expect(cost.estimateGroupCost(makeGroup(ops(func))).libraryCallLoss).toBe(1);
  });
});

describe('FusionCostModel.shouldFuse', () => {
  it('fuse=true for beneficial elementwise chain — saves memory + launches', () => {
    const t = new TensorType([64, 64], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.neg(sum.getResult(0)).getResult(0)]);
    });
    const decision = new FusionCostModel().shouldFuse(makeGroup(ops(func)));
    expect(decision.fuse).toBe(true);
    expect(decision.cost.memorySaved).toBeGreaterThan(0);
    expect(decision.cost.launchSaved).toBeGreaterThan(0);
  });

  it('fuse=false for single-op group', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });
    expect(new FusionCostModel().shouldFuse(makeGroup(ops(func))).fuse).toBe(false);
  });

  it('fuse=false when a library kernel would be lost', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.neg(sum.getResult(0)).getResult(0)]);
    });
    expect(new FusionCostModel({ hasLibraryOp: (name) => name === 'add' }).shouldFuse(makeGroup(ops(func))).fuse).toBe(false);
  });

  it('fuse=false when group size exceeds maxCodeSizeOps', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.neg(sum.getResult(0)).getResult(0)]);
    });
    expect(new FusionCostModel({ maxCodeSizeOps: 1 }).shouldFuse(makeGroup(ops(func))).fuse).toBe(false);
  });

  it('fuse=false when register pressure exceeds maxRegistersPerThread', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.neg(sum.getResult(0)).getResult(0)]);
    });
    expect(new FusionCostModel({ maxRegistersPerThread: 1 }).shouldFuse(makeGroup(ops(func))).fuse).toBe(false);
  });
});

describe('FusionCostModel per-thread budgets a target may not expose', () => {
  function diamond() {
    const t = new TensorType([256, 256], ScalarType.F32);
    return buildFunction('f', [t, t], [t], (b, args) => {
      const s = b.add(args[0], args[1]);
      const m = b.mul(s.getResult(0), args[0]);
      b.returnOp([b.add(m.getResult(0), s.getResult(0)).getResult(0)]);
    });
  }

  it('an intermediate reused inside the group is charged as shared memory', () => {
    const cost = new FusionCostModel().estimateGroupCost(makeGroup(ops(diamond())));
    expect(cost.sharedMemoryUsage).toBe(256 * 256 * 4);
  });

  it('fuse=false when that charge exceeds the shared memory a GPU states it has', () => {
    const decision = new FusionCostModel({ maxSharedMemory: CUDATarget().sharedMemoryBytes })
      .shouldFuse(makeGroup(ops(diamond())));
    expect(decision.fuse).toBe(false);
    expect(decision.reason).toMatch(/shared memory/);
  });

  it('fuse=true on a target that states no shared memory, which is not a 48 KiB budget', () => {
    expect(CPUTarget().sharedMemoryBytes).toBe(0);
    const decision = new FusionCostModel({ maxSharedMemory: CPUTarget().sharedMemoryBytes })
      .shouldFuse(makeGroup(ops(diamond())));
    expect(decision.fuse).toBe(true);
  });

  it('a target stating no per-thread register file gets no register limit, not 255', () => {
    expect(CPUTarget().registersPerThread).toBe(0);
    expect(new FusionCostModel({ maxRegistersPerThread: CPUTarget().registersPerThread }).maxRegistersPerThread).toBe(Infinity);
    expect(new FusionCostModel({ maxRegistersPerThread: CUDATarget().registersPerThread }).maxRegistersPerThread).toBe(255);
    expect(new FusionCostModel().maxRegistersPerThread).toBe(255);
  });

  it('an unstated shared memory budget still falls back to the GPU default', () => {
    expect(new FusionCostModel().shouldFuse(makeGroup(ops(diamond()))).fuse).toBe(false);
  });
});
