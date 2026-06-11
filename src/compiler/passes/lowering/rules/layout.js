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

    const { loopVars, loopBinds, indices: outIndices, extentNodes } = makeLoopNest(ctx, outBuf.shape, outBuf);
    const store = new BufferStoreNode(outBuf, outIndices, new BufferLoadNode(inBuf, outIndices));
    const block = new BlockNode(ctx.blockName('layout_transform_block'), loopBinds, [{ buffer: inBuf }], [{ buffer: outBuf }], store);
    return wrapInLoops(block, loopVars, outBuf.shape, extentNodes);
  });
}
