import { describe, it, expect } from 'vitest';
import { getSketchesForBlock } from '../../../src/compiler/autotune/search_space.js';
import { buildBlockMap } from '../../../src/compiler/autotune/workload_key.js';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import {
  PrimFunc,
  BlockNode,
  BufferStoreNode,
  BufferLoadNode,
} from '../../../src/compiler/ir/tensor/nodes.js';
import { CPUTarget } from '../../../src/backend/target.js';

function reductionPrimFunc() {
  const out = new Buffer('out', [1], 'f32', 'global');
  const inp = new Buffer('in', [16], 'f32', 'global');
  const load = new BufferLoadNode(inp, []);
  const body = new BufferStoreNode(out, [], load);
  const init = new BufferStoreNode(out, [], load);
  const block = new BlockNode('reduce_blk', [], [{ buffer: inp }], [{ buffer: out }], body, init);
  return new PrimFunc('f', [], block, new Map());
}

describe('getSketchesForBlock requires blockMap to classify reductions', () => {
  it('selects a different sketch family when blockMap is provided vs omitted', () => {
    const target = CPUTarget();
    const pf = reductionPrimFunc();
    const blockMap = buildBlockMap(pf.body);

    const withMap = getSketchesForBlock(pf, 'reduce_blk', target, blockMap);
    const withoutMap = getSketchesForBlock(pf, 'reduce_blk', target);

    expect(withMap[0].name).not.toBe(withoutMap[0].name);
  });
});
