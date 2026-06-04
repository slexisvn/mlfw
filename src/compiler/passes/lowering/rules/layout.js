import {
  BufferStoreNode, BufferLoadNode, BlockNode
} from '../../../ir/tensor/nodes.js';
import {
  registerLoweringRule, makeLoopNest, wrapInLoops
} from '../lowering_registry.js';

export function register() {
  registerLoweringRule('layout_transform', (ctx, op, inputs, outputs) => {
    const inBuf = inputs[0];
    const outBuf = outputs[0];
    const srcOrder = op.getAttr('src_layout');
    const dstOrder = op.getAttr('dst_layout');

    const invSrc = new Array(srcOrder.length);
    for (let i = 0; i < srcOrder.length; i++) invSrc[srcOrder[i]] = i;
    const perm = new Array(dstOrder.length);
    for (let i = 0; i < dstOrder.length; i++) perm[i] = invSrc[dstOrder[i]];

    const { loopVars, loopBinds, indices: outIndices } = makeLoopNest(ctx, outBuf.shape);
    const inIndices = new Array(inBuf.shape.length);
    for (let i = 0; i < perm.length; i++) {
      inIndices[perm[i]] = outIndices[i];
    }
    const store = new BufferStoreNode(outBuf, outIndices, new BufferLoadNode(inBuf, inIndices));
    const block = new BlockNode('layout_transform_block', loopBinds, [{ buffer: inBuf }], [{ buffer: outBuf }], store);
    return wrapInLoops(block, loopVars, outBuf.shape);
  });
}
