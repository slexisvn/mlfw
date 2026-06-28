import { describe, it, expect } from 'vitest';
import { computeWorkloadKey, buildBlockMap } from '../../../src/compiler/autotune/workload_key.js';
import { TuningDatabase } from '../../../src/compiler/autotune/tuning_db.js';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import {
  PrimFunc,
  BlockNode,
  BufferStoreNode,
  BufferLoadNode,
} from '../../../src/compiler/ir/tensor/nodes.js';
import { CPUTarget } from '../../../src/backend/target.js';

function buf(name, shape = [16]) {
  return new Buffer(name, shape, 'f32', 'global');
}

function makeBlock(name, loadBufferName, shape = [16]) {
  const out = buf('out_' + name, shape);
  const inp = buf(loadBufferName, shape);
  const load = new BufferLoadNode(inp, []);
  const store = new BufferStoreNode(out, [], load);
  const block = new BlockNode(name, [], [{ buffer: inp }], [{ buffer: out }], store);
  return block;
}

function makePrimFunc(block) {
  return new PrimFunc('f', [], block, new Map());
}

describe('computeWorkloadKey — structural (shape-based) key', () => {
  it('dedups loads from differently named buffers of the same shape (keeps tuning O(n) at scale)', () => {
    const target = CPUTarget();
    const pfA = makePrimFunc(makeBlock('blk', 'srcA'));
    const pfB = makePrimFunc(makeBlock('blk', 'srcB'));
    const keyA = computeWorkloadKey(pfA, 'blk', target);
    const keyB = computeWorkloadKey(pfB, 'blk', target);
    expect(keyA).toBe(keyB);
  });

  it('produces distinct keys for loads of differently shaped buffers', () => {
    const target = CPUTarget();
    const pfA = makePrimFunc(makeBlock('blk', 'src', [16]));
    const pfB = makePrimFunc(makeBlock('blk', 'src', [32]));
    expect(computeWorkloadKey(pfA, 'blk', target)).not.toBe(computeWorkloadKey(pfB, 'blk', target));
  });

  it('is stable for identical structure', () => {
    const target = CPUTarget();
    const pf1 = makePrimFunc(makeBlock('blk', 'src'));
    const pf2 = makePrimFunc(makeBlock('blk', 'src'));
    expect(computeWorkloadKey(pf1, 'blk', target)).toBe(computeWorkloadKey(pf2, 'blk', target));
  });
});

describe('TuningDatabase.computeWorkloadKey delegates to workload_key', () => {
  it('matches the structural key', () => {
    const target = CPUTarget();
    const pf = makePrimFunc(makeBlock('blk', 'src'));
    const blockMap = buildBlockMap(pf.body);
    const db = new TuningDatabase();
    const dbKey = db.computeWorkloadKey(pf, 'blk', target, blockMap);
    const directKey = computeWorkloadKey(pf, 'blk', target, blockMap);
    expect(dbKey).toBe(directKey);
  });
});
