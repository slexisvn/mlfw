import {
  BufferStoreNode, BufferLoadNode, BlockNode
} from '../../../ir/tensor/nodes.js';
import {
  registerLoweringRule, makeLoopNest, wrapInLoops
} from '../lowering_registry.js';
import type { LoweringContext } from '../lowering_registry.js';
import type { Operation } from '../../../ir/graph/operation.js';
import type { Buffer } from '../../../ir/tensor/buffer.js';

export function register(): void {
  registerLoweringRule('layout_transform', ((ctx: LoweringContext, op: Operation, inputs: Buffer[], outputs: Buffer[]) => {
    const inBuf = inputs[0];
    const outBuf = outputs[0];
    const srcOrder = op.getAttr<readonly number[]>('src_layout');
    const dstOrder = op.getAttr<readonly number[]>('dst_layout');

    const { loopVars, loopBinds, indices: outIndices, extentNodes } = makeLoopNest(ctx, outBuf.shape, outBuf);
    const store = new BufferStoreNode(outBuf, outIndices, new BufferLoadNode(inBuf, outIndices));
    const block = new BlockNode(ctx.blockName('layout_transform_block'), loopBinds, [{ buffer: inBuf }], [{ buffer: outBuf }], store);
    return wrapInLoops(block, loopVars, outBuf.shape, extentNodes);
  }) as never);
}
