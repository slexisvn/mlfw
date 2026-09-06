import {
  BufferStoreNode, BufferLoadNode, BlockNode
} from '../../../ir/tensor/nodes.js';
import {
  registerLoweringRule, makeLoopNest, wrapInLoops, layoutOf, storageIndices
} from '../lowering_registry.js';
import { TensorType } from '../../../ir/graph/types.js';
import type { LoweringContext } from '../lowering_registry.js';
import type { Operation } from '../../../ir/graph/operation.js';
import type { Buffer } from '../../../ir/tensor/buffer.js';

export function register(): void {
  registerLoweringRule('layout_transform', ((ctx: LoweringContext, op: Operation, inputs: Buffer[], outputs: Buffer[]) => {
    const inBuf = inputs[0];
    const outBuf = outputs[0];
    const outLayout = layoutOf(op.getResult(0));
    const inLayout = layoutOf(op.getOperand(0));
    const logical = (op.getResult(0).type as TensorType).shape;
    const extentSource = !outLayout || !outLayout.isBlocked() ? outBuf
      : (inLayout && inLayout.isBlocked() ? null : inBuf);
    const { loopVars, loopBinds, indices, extentNodes } = makeLoopNest(ctx, logical, extentSource);
    const store = new BufferStoreNode(
      outBuf,
      storageIndices(indices, outLayout),
      new BufferLoadNode(inBuf, storageIndices(indices, inLayout))
    );
    const block = new BlockNode(ctx.blockName('layout_transform_block'), loopBinds, [{ buffer: inBuf }], [{ buffer: outBuf }], store);
    return wrapInLoops(block, loopVars, logical, extentNodes);
  }) as never);
}
