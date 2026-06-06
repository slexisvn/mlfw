import { describe, it, expect } from 'vitest';
import { InplaceAnalysis } from '../../../src/compiler/passes/memory/inplace_analysis.js';
import { BufferLiveness } from '../../../src/compiler/passes/memory/buffer_liveness.js';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import {
  PrimFunc, BlockNode, SeqNode, BufferStoreNode, BufferLoadNode,
  IntImmNode, MathOpNode, VariableNode, BlockRealizeNode
} from '../../../src/compiler/ir/tensor/nodes.js';

function makeVar(name) {
  return new VariableNode(name, 'int32');
}

function makeBind(name) {
  return new BlockRealizeNode(makeVar(name + '_v'), makeVar(name));
}

function makeBlock(name, reads, writes, bodyExpr = null) {
  const binds = [makeBind('i')];
  const body = bodyExpr || (writes.length > 0
    ? new BufferStoreNode(writes[0].buffer, [], reads.length > 0 ? new BufferLoadNode(reads[0].buffer, []) : new IntImmNode(0))
    : new IntImmNode(0));
  return new BlockNode(name, binds, reads, writes, body);
}

function makePrimFunc(body, paramBuffers = []) {
  const params = paramBuffers.map((_, i) => makeVar(`arg_${i}`));
  const bufferMap = new Map();
  for (let i = 0; i < paramBuffers.length; i++) {
    bufferMap.set(params[i], paramBuffers[i]);
  }
  return new PrimFunc('test', params, body, bufferMap);
}

describe('InplaceAnalysis', () => {
  it('identifies inplace candidate when src last use < dst first use', () => {
    const paramBuf = new Buffer('param', [16], 'f32', 'global');
    const srcBuf = new Buffer('src', [8], 'f32', 'global');
    const dstBuf = new Buffer('dst', [8], 'f32', 'global');
    const outBuf = new Buffer('out', [8], 'f32', 'global');

    const block0 = makeBlock('b0', [{ buffer: paramBuf }], [{ buffer: srcBuf }]);
    const block1 = makeBlock('b1', [{ buffer: srcBuf }], [{ buffer: dstBuf }]);
    const block2 = makeBlock('b2', [{ buffer: dstBuf }], [{ buffer: outBuf }]);
    const seq = new SeqNode([block0, block1, block2]);
    const pf = makePrimFunc(seq, [paramBuf, outBuf]);

    const liveness = BufferLiveness.analyze(pf);
    const candidates = InplaceAnalysis.analyze(pf, liveness);

    expect(candidates.length).toBeGreaterThanOrEqual(1);
    const found = candidates.find(c => c.srcBuffer === srcBuf && c.dstBuffer === dstBuf);
    expect(found).toBeDefined();
  });

  it('rejects inplace when shapes do not match', () => {
    const paramBuf = new Buffer('param', [16], 'f32', 'global');
    const srcBuf = new Buffer('src', [8], 'f32', 'global');
    const dstBuf = new Buffer('dst', [4], 'f32', 'global');
    const outBuf = new Buffer('out', [4], 'f32', 'global');

    const block0 = makeBlock('b0', [{ buffer: paramBuf }], [{ buffer: srcBuf }]);
    const block1 = makeBlock('b1', [{ buffer: srcBuf }], [{ buffer: dstBuf }]);
    const block2 = makeBlock('b2', [{ buffer: dstBuf }], [{ buffer: outBuf }]);
    const seq = new SeqNode([block0, block1, block2]);
    const pf = makePrimFunc(seq, [paramBuf, outBuf]);

    const liveness = BufferLiveness.analyze(pf);
    const candidates = InplaceAnalysis.analyze(pf, liveness);

    const found = candidates.find(c => c.srcBuffer === srcBuf && c.dstBuffer === dstBuf);
    expect(found).toBeUndefined();
  });

  it('rejects inplace when dtypes do not match', () => {
    const paramBuf = new Buffer('param', [8], 'f32', 'global');
    const srcBuf = new Buffer('src', [8], 'f32', 'global');
    const dstBuf = new Buffer('dst', [8], 'f16', 'global');
    const outBuf = new Buffer('out', [8], 'f16', 'global');

    const block0 = makeBlock('b0', [{ buffer: paramBuf }], [{ buffer: srcBuf }]);
    const block1 = makeBlock('b1', [{ buffer: srcBuf }], [{ buffer: dstBuf }]);
    const block2 = makeBlock('b2', [{ buffer: dstBuf }], [{ buffer: outBuf }]);
    const seq = new SeqNode([block0, block1, block2]);
    const pf = makePrimFunc(seq, [paramBuf, outBuf]);

    const liveness = BufferLiveness.analyze(pf);
    const candidates = InplaceAnalysis.analyze(pf, liveness);

    const found = candidates.find(c => c.srcBuffer === srcBuf && c.dstBuffer === dstBuf);
    expect(found).toBeUndefined();
  });

  it('rejects inplace when scopes do not match', () => {
    const paramBuf = new Buffer('param', [8], 'f32', 'global');
    const srcBuf = new Buffer('src', [8], 'f32', 'global');
    const dstBuf = new Buffer('dst', [8], 'f32', 'shared');
    const outBuf = new Buffer('out', [8], 'f32', 'shared');

    const block0 = makeBlock('b0', [{ buffer: paramBuf }], [{ buffer: srcBuf }]);
    const block1 = makeBlock('b1', [{ buffer: srcBuf }], [{ buffer: dstBuf }]);
    const block2 = makeBlock('b2', [{ buffer: dstBuf }], [{ buffer: outBuf }]);
    const seq = new SeqNode([block0, block1, block2]);
    const pf = makePrimFunc(seq, [paramBuf, outBuf]);

    const liveness = BufferLiveness.analyze(pf);
    const candidates = InplaceAnalysis.analyze(pf, liveness);

    const found = candidates.find(c => c.srcBuffer === srcBuf && c.dstBuffer === dstBuf);
    expect(found).toBeUndefined();
  });

  it('rejects inplace for param buffers', () => {
    const paramBuf = new Buffer('param', [8], 'f32', 'global');
    const dstBuf = new Buffer('dst', [8], 'f32', 'global');
    const outBuf = new Buffer('out', [8], 'f32', 'global');

    const block0 = makeBlock('b0', [{ buffer: paramBuf }], [{ buffer: dstBuf }]);
    const block1 = makeBlock('b1', [{ buffer: dstBuf }], [{ buffer: outBuf }]);
    const seq = new SeqNode([block0, block1]);
    const pf = makePrimFunc(seq, [paramBuf, outBuf]);

    const liveness = BufferLiveness.analyze(pf);
    const candidates = InplaceAnalysis.analyze(pf, liveness);

    const found = candidates.find(c => c.srcBuffer === paramBuf || c.dstBuffer === paramBuf);
    expect(found).toBeUndefined();
  });

  it('rejects inplace when src is still live at dst first use', () => {
    const paramBuf = new Buffer('param', [8], 'f32', 'global');
    const srcBuf = new Buffer('src', [8], 'f32', 'global');
    const dstBuf = new Buffer('dst', [8], 'f32', 'global');
    const outBuf = new Buffer('out', [8], 'f32', 'global');

    const block0 = makeBlock('b0', [{ buffer: paramBuf }], [{ buffer: srcBuf }]);
    const body1 = new BufferStoreNode(dstBuf, [], new MathOpNode('+', new BufferLoadNode(srcBuf, []), new IntImmNode(1)));
    const block1 = makeBlock('b1', [{ buffer: srcBuf }], [{ buffer: dstBuf }], body1);
    const body2 = new BufferStoreNode(outBuf, [], new MathOpNode('+', new BufferLoadNode(srcBuf, []), new BufferLoadNode(dstBuf, [])));
    const block2 = makeBlock('b2', [{ buffer: srcBuf }, { buffer: dstBuf }], [{ buffer: outBuf }], body2);
    const seq = new SeqNode([block0, block1, block2]);
    const pf = makePrimFunc(seq, [paramBuf, outBuf]);

    const liveness = BufferLiveness.analyze(pf);
    const candidates = InplaceAnalysis.analyze(pf, liveness);

    const found = candidates.find(c => c.srcBuffer === srcBuf && c.dstBuffer === dstBuf);
    expect(found).toBeUndefined();
  });

  it('returns empty array when no candidates exist', () => {
    const paramBuf = new Buffer('param', [8], 'f32', 'global');
    const outBuf = new Buffer('out', [8], 'f32', 'global');
    const block = makeBlock('b0', [{ buffer: paramBuf }], [{ buffer: outBuf }]);
    const pf = makePrimFunc(block, [paramBuf, outBuf]);

    const liveness = BufferLiveness.analyze(pf);
    const candidates = InplaceAnalysis.analyze(pf, liveness);

    expect(candidates.length).toBe(0);
  });
});
