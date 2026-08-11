import { describe, it, expect } from 'vitest';
import { InplaceAnalysis } from '../../../../src/compiler/passes/memory/inplace_analysis.js';
import { BufferLiveness } from '../../../../src/compiler/passes/memory/buffer_liveness.js';
import { Buffer } from '../../../../src/compiler/ir/tensor/buffer.js';
import {
  PrimFunc, BlockNode, SeqNode, BufferStoreNode, BufferLoadNode,
  IntImmNode, MathOpNode, VariableNode, BlockRealizeNode
} from '../../../../src/compiler/ir/tensor/nodes.js';

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

  it('does not alias one src to two dsts (WAW hazard)', () => {
    const paramBuf = new Buffer('param', [8], 'f32', 'global');
    const srcBuf = new Buffer('src', [8], 'f32', 'global');
    const dst1 = new Buffer('dst1', [8], 'f32', 'global');
    const dst2 = new Buffer('dst2', [8], 'f32', 'global');
    const outBuf = new Buffer('out', [8], 'f32', 'global');

    const block0 = new BlockNode('b0', [makeBind('i')], [{ buffer: paramBuf }], [{ buffer: srcBuf }],
      new BufferStoreNode(srcBuf, [], new BufferLoadNode(paramBuf, [])));
    const body1 = new SeqNode([
      new BufferStoreNode(dst1, [], new BufferLoadNode(srcBuf, [])),
      new BufferStoreNode(dst2, [], new BufferLoadNode(srcBuf, [])),
    ]);
    const block1 = new BlockNode('b1', [makeBind('i')], [{ buffer: srcBuf }], [{ buffer: dst1 }, { buffer: dst2 }], body1);
    const body2 = new BufferStoreNode(outBuf, [], new MathOpNode('+', new BufferLoadNode(dst1, []), new BufferLoadNode(dst2, [])));
    const block2 = new BlockNode('b2', [makeBind('i')], [{ buffer: dst1 }, { buffer: dst2 }], [{ buffer: outBuf }], body2);
    const seq = new SeqNode([block0, block1, block2]);
    const pf = makePrimFunc(seq, [paramBuf, outBuf]);

    const liveness = BufferLiveness.analyze(pf);
    const candidates = InplaceAnalysis.analyze(pf, liveness);

    const usingSrc = candidates.filter(c => c.srcBuffer === srcBuf);
    expect(usingSrc.length).toBe(1);
  });

  function inplaceCandidate(block1Body, extraReads = []) {
    const paramBuf = new Buffer('param', [8], 'f32', 'global');
    const srcBuf = new Buffer('src', [8], 'f32', 'global');
    const dstBuf = new Buffer('dst', [8], 'f32', 'global');
    const outBuf = new Buffer('out', [8], 'f32', 'global');
    const i = makeVar('i');
    const reads = [{ buffer: srcBuf }, ...extraReads];
    const block0 = makeBlock('b0', [{ buffer: paramBuf }], [{ buffer: srcBuf }]);
    const block1 = new BlockNode('b1', [makeBind('i')], reads, [{ buffer: dstBuf }], block1Body(srcBuf, dstBuf, i));
    const block2 = makeBlock('b2', [{ buffer: dstBuf }], [{ buffer: outBuf }]);
    const pf = makePrimFunc(new SeqNode([block0, block1, block2]), [paramBuf, outBuf]);
    const candidates = InplaceAnalysis.analyze(pf, BufferLiveness.analyze(pf));
    return { candidates, srcBuf, dstBuf };
  }

  it('accepts inplace for an elementwise transform that reads src at the dst write index', () => {
    const { candidates, srcBuf, dstBuf } = inplaceCandidate((src, dst, i) =>
      new BufferStoreNode(dst, [i], new MathOpNode('+',
        new MathOpNode('*', new BufferLoadNode(src, [i]), new IntImmNode(2)), new IntImmNode(1))));
    expect(candidates.find(c => c.srcBuffer === srcBuf && c.dstBuffer === dstBuf)).toBeDefined();
  });

  it('rejects inplace for a stencil that reads src at a shifted index', () => {
    const { candidates, srcBuf, dstBuf } = inplaceCandidate((src, dst, i) =>
      new BufferStoreNode(dst, [i], new MathOpNode('+',
        new BufferLoadNode(src, [i]),
        new BufferLoadNode(src, [new MathOpNode('+', i, new IntImmNode(1))]))));
    expect(candidates.find(c => c.srcBuffer === srcBuf && c.dstBuffer === dstBuf)).toBeUndefined();
  });

  it('rejects inplace for an index permutation (transpose/reverse-like)', () => {
    const { candidates, srcBuf, dstBuf } = inplaceCandidate((src, dst, i) =>
      new BufferStoreNode(dst, [i], new BufferLoadNode(src, [new MathOpNode('-', new IntImmNode(7), i)])));
    expect(candidates.find(c => c.srcBuffer === srcBuf && c.dstBuffer === dstBuf)).toBeUndefined();
  });

  it('rejects inplace for a gather (data-dependent src index)', () => {
    const idxBuf = new Buffer('idx', [8], 'i32', 'global');
    const { candidates, srcBuf, dstBuf } = inplaceCandidate((src, dst, i) =>
      new BufferStoreNode(dst, [i], new BufferLoadNode(src, [new BufferLoadNode(idxBuf, [i])])),
      [{ buffer: idxBuf }]);
    expect(candidates.find(c => c.srcBuffer === srcBuf && c.dstBuffer === dstBuf)).toBeUndefined();
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
