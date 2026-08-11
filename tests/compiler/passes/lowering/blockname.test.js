import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { lowerGraphToPrimFunc } from '../../../../src/compiler/passes/lowering/graph_to_tensor.js';
import { BlockNode } from '../../../../src/compiler/ir/tensor/nodes.js';
import { Schedule } from '../../../../src/compiler/schedule/schedule.js';
import { SchedulePolicy } from '../../../../src/compiler/schedule/rules.js';
import { CUDATarget } from '../../../../src/backend/target.js';

function lower(name, inTypes, outTypes, bodyFn) {
  const func = buildFunction(name, inTypes, outTypes, bodyFn);
  return lowerGraphToPrimFunc(func);
}

const SKIP_KEYS = new Set(['_parent', '_parentKey', '_parentIdx', '_module']);

function collectBlockNames(node) {
  const names = [];
  const visited = new Set();
  const stack = [node];
  while (stack.length > 0) {
    const n = stack.pop();
    if (!n || typeof n !== 'object' || visited.has(n)) continue;
    visited.add(n);
    if (n instanceof BlockNode) names.push(n.name);
    for (const key of Object.keys(n)) {
      if (SKIP_KEYS.has(key)) continue;
      const val = n[key];
      if (Array.isArray(val)) for (const item of val) stack.push(item);
      else if (val && typeof val === 'object') stack.push(val);
    }
  }
  return names;
}

function nodeCount(root) {
  let count = 0;
  const visited = new Set();
  const stack = [root];
  while (stack.length > 0) {
    const n = stack.pop();
    if (!n || typeof n !== 'object' || visited.has(n)) continue;
    visited.add(n);
    count++;
    for (const key of Object.keys(n)) {
      if (SKIP_KEYS.has(key)) continue;
      const val = n[key];
      if (Array.isArray(val)) for (const item of val) stack.push(item);
      else if (val && typeof val === 'object') stack.push(val);
    }
  }
  return count;
}

describe('block name uniqueness across repeated ops', () => {
  it('chained matmuls produce unique block names', () => {
    const t = new TensorType([8, 8], ScalarType.F32);
    const pf = lower('mlp4', Array(5).fill(t), [t], (b, args) => {
      let x = args[0];
      for (let i = 1; i < 5; i++) x = b.matmul(x, args[i]).getResult(0);
      b.returnOp([x]);
    });

    const names = collectBlockNames(pf.body);
    expect(new Set(names).size).toBe(names.length);
    expect(names.length).toBe(8);
  });

  it('block names preserve operation hint with numeric suffix', () => {
    const t = new TensorType([4, 4], ScalarType.F32);
    const pf = lower('mm2', Array(3).fill(t), [t], (b, args) => {
      const a = b.matmul(args[0], args[1]).getResult(0);
      b.returnOp([b.matmul(a, args[2]).getResult(0)]);
    });

    const names = collectBlockNames(pf.body);
    for (const name of names) {
      expect(name).toMatch(/^.+_\d+$/);
    }
    expect(names.filter(n => n.startsWith('matmul_init')).length).toBe(2);
    expect(names.filter(n => n.match(/^matmul_\d+$/)).length).toBe(2);
  });

  it('chained convolutions produce unique block names', () => {
    const x = new TensorType([1, 1, 8, 8], ScalarType.F32);
    const k = new TensorType([1, 1, 3, 3], ScalarType.F32);
    const mid = new TensorType([1, 1, 6, 6], ScalarType.F32);
    const out = new TensorType([1, 1, 4, 4], ScalarType.F32);

    const pf = lower('conv2', [x, k, k], [out], (b, args) => {
      const c1 = b.conv(args[0], args[1], [1, 1], [[0, 0], [0, 0]]).getResult(0);
      b.returnOp([b.conv(c1, args[2], [1, 1], [[0, 0], [0, 0]]).getResult(0)]);
    });

    const names = collectBlockNames(pf.body);
    expect(new Set(names).size).toBe(names.length);
  });

  it('mixed ops (conv + pool + reduce) all get unique names', () => {
    const x = new TensorType([1, 1, 8, 8], ScalarType.F32);
    const k = new TensorType([1, 1, 3, 3], ScalarType.F32);
    const init = new TensorType([], ScalarType.F32);
    const out = new TensorType([1], ScalarType.F32);

    const pf = lower('mixed', [x, k, init], [out], (b, args) => {
      const c = b.conv(args[0], args[1], [1, 1], [[0, 0], [0, 0]]).getResult(0);
      const p = b.pool2d(c, 'max', [2, 2], [2, 2], [[0, 0], [0, 0]]).getResult(0);
      b.returnOp([b.reduce(p, args[2], [0, 1, 2, 3], 'sum').getResult(0)]);
    });

    const names = collectBlockNames(pf.body);
    expect(new Set(names).size).toBe(names.length);
  });
});

describe('GPU scheduling with unique block names', () => {
  it('scheduled IR grows linearly with layer count, not exponentially', () => {
    const t = new TensorType([8, 8], ScalarType.F32);

    function scheduledSize(layers) {
      const pf = lower('mlp', Array(layers + 1).fill(t), [t], (b, args) => {
        let x = args[0];
        for (let i = 1; i <= layers; i++) x = b.matmul(x, args[i]).getResult(0);
        b.returnOp([x]);
      });
      const sch = new Schedule(pf);
      new SchedulePolicy(CUDATarget()).applyToAllBlocks(sch);
      return nodeCount(pf.body);
    }

    const size2 = scheduledSize(2);
    const size8 = scheduledSize(8);
    expect(size8 / size2).toBeLessThan(6);
  });

  it('applyToAllBlocks applies each block name exactly once', () => {
    const t = new TensorType([4, 4], ScalarType.F32);
    const pf = lower('mm3', Array(4).fill(t), [t], (b, args) => {
      let x = args[0];
      for (let i = 1; i < 4; i++) x = b.matmul(x, args[i]).getResult(0);
      b.returnOp([x]);
    });

    const sch = new Schedule(pf);
    const applied = new SchedulePolicy(CUDATarget()).applyToAllBlocks(sch);

    const appliedNames = [...applied.keys()];
    expect(new Set(appliedNames).size).toBe(appliedNames.length);
  });

  it('renamed matmul blocks still match GPU tiling rules', () => {
    const t = new TensorType([32, 32], ScalarType.F32);
    const pf = lower('mm', [t, t], [t], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    const sch = new Schedule(pf);
    const applied = new SchedulePolicy(CUDATarget()).applyToAllBlocks(sch);

    const matmulRules = [...applied.entries()].filter(([name]) => name.includes('matmul'));
    expect(matmulRules.length).toBeGreaterThan(0);
  });
});
