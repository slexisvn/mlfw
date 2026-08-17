import {
  BufferStoreNode, BufferLoadNode, SeqNode, BlockNode,
  IfThenElseNode, WhileNode, ForNode, ForKind, IntImmNode, SyncThreadsNode
} from '../../../ir/tensor/nodes.js';
import { Buffer } from '../../../ir/tensor/buffer.js';
import { MemoryScope } from '../../../ir/tensor/tensor_types.js';
import {
  registerLoweringRule, getLoweringRule, lowerConstant, isConstantOp,
  makeLoopNest, wrapInLoops
} from '../lowering_registry.js';
import type { BufferOwner, LoweringContext } from '../lowering_registry.js';
import type { TirNode, VariableNode } from '../../../ir/tensor/nodes.js';
import type { Block, Region } from '../../../ir/graph/block.js';
import type { Value } from '../../../ir/graph/value.js';

type RegionLowering = { stmts: TirNode[]; yieldBuffers: Buffer[] };

function copyBuffer(ctx: LoweringContext, srcBuf: Buffer, dstBuf: Buffer): TirNode {
  const { loopVars, loopBinds, indices, extentNodes } = makeLoopNest(ctx, dstBuf.shape, dstBuf);
  const store = new BufferStoreNode(dstBuf, indices, new BufferLoadNode(srcBuf, indices));
  const block = new BlockNode(ctx.blockName('cf_copy'), loopBinds, [{ buffer: srcBuf }], [{ buffer: dstBuf }], store);
  return wrapInLoops(block, loopVars, dstBuf.shape, extentNodes);
}

function sliceCopyIn(ctx: LoweringContext, seqBuf: Buffer, stepBuf: Buffer, counterVar: VariableNode): TirNode {
  const { loopVars, loopBinds, indices, extentNodes } = makeLoopNest(ctx, stepBuf.shape, stepBuf);
  const store = new BufferStoreNode(stepBuf, indices, new BufferLoadNode(seqBuf, [counterVar, ...indices]));
  const block = new BlockNode(ctx.blockName('scan_in'), loopBinds, [{ buffer: seqBuf }], [{ buffer: stepBuf }], store);
  return wrapInLoops(block, loopVars, stepBuf.shape, extentNodes);
}

function sliceCopyOut(ctx: LoweringContext, stepBuf: Buffer, seqBuf: Buffer, counterVar: VariableNode): TirNode {
  const { loopVars, loopBinds, indices, extentNodes } = makeLoopNest(ctx, stepBuf.shape, stepBuf);
  const store = new BufferStoreNode(seqBuf, [counterVar, ...indices], new BufferLoadNode(stepBuf, indices));
  const block = new BlockNode(ctx.blockName('scan_out'), loopBinds, [{ buffer: stepBuf }], [{ buffer: seqBuf }], store);
  return wrapInLoops(block, loopVars, stepBuf.shape, extentNodes);
}

function lowerRegionBody(ctx: LoweringContext, region: Region, argBuffers: readonly Buffer[]): RegionLowering {
  const entryBlock = region.entryBlock as Block;
  const valueMap = new Map<Value, Buffer>();
  for (let i = 0; i < entryBlock.arguments.length; i++) {
    valueMap.set(entryBlock.arguments[i], argBuffers[i]);
    if (argBuffers[i]) ctx.bufferMap.set(entryBlock.arguments[i], argBuffers[i]);
  }
  const stmts: TirNode[] = [];
  for (const innerOp of entryBlock.ops()) {
    if (innerOp.opName === 'yield') {
      const results: Buffer[] = new Array(innerOp.numOperands);
      for (let i = 0; i < innerOp.numOperands; i++) {
        results[i] = valueMap.get(innerOp.getOperand(i)) || ctx.getOrAllocBuffer(innerOp.getOperand(i));
      }
      return { stmts, yieldBuffers: results };
    }
    if (isConstantOp(innerOp.opName)) {
      const stmt = lowerConstant(ctx, innerOp);
      if (stmt) stmts.push(stmt);
      continue;
    }
    const outerOperands: (Buffer | Value)[] = new Array(innerOp.numOperands);
    for (let i = 0; i < innerOp.numOperands; i++) {
      outerOperands[i] = valueMap.get(innerOp.getOperand(i)) || innerOp.getOperand(i);
    }
    const inputs: Buffer[] = new Array(outerOperands.length);
    for (let i = 0; i < outerOperands.length; i++) {
      inputs[i] = outerOperands[i] instanceof Buffer ? outerOperands[i] as Buffer : ctx.getOrAllocBuffer(outerOperands[i] as Value);
    }
    const outputs: Buffer[] = new Array(innerOp.numResults);
    for (let i = 0; i < innerOp.numResults; i++) {
      const proxy = { type: innerOp.getResult(i).type };
      outputs[i] = ctx.getOrAllocBuffer(proxy);
      valueMap.set(innerOp.getResult(i), outputs[i]);
      ctx.bufferMap.set(innerOp.getResult(i), outputs[i]);
    }
    const rule = getLoweringRule(innerOp.opName);
    if (!rule) throw new Error(`No lowering rule for op '${innerOp.opName}' inside region`);
    const stmt = rule(ctx, innerOp, inputs, outputs);
    if (stmt) stmts.push(stmt);
  }
  return { stmts, yieldBuffers: [] };
}

export { lowerRegionBody };

export function register(): void {
  registerLoweringRule('if', (ctx, op, inputs, outputs) => {
    const predBuf = inputs[0];
    const predLoad = new BufferLoadNode(predBuf, []);
    const thenRegion = op.regions[0];
    const elseRegion = op.regions[1];

    const resultBufs: Buffer[] = new Array(op.numResults);
    for (let i = 0; i < op.numResults; i++) {
      resultBufs[i] = outputs[i] || ctx.getOrAllocBuffer(op.getResult(i));
      ctx.bufferMap.set(op.getResult(i), resultBufs[i]);
    }

    const thenResult = lowerRegionBody(ctx, thenRegion, []);
    const thenStmts = thenResult.stmts.slice();
    for (let i = 0; i < resultBufs.length && i < thenResult.yieldBuffers.length; i++) {
      const src = thenResult.yieldBuffers[i];
      if (src && src !== resultBufs[i]) thenStmts.push(copyBuffer(ctx, src, resultBufs[i]));
    }
    const thenBody = thenStmts.length === 1 ? thenStmts[0] : new SeqNode(thenStmts);

    let elseBody: TirNode | null = null;
    if (elseRegion && elseRegion.entryBlock) {
      const elseResult = lowerRegionBody(ctx, elseRegion, []);
      const elseStmts = elseResult.stmts.slice();
      for (let i = 0; i < resultBufs.length && i < elseResult.yieldBuffers.length; i++) {
        const src = elseResult.yieldBuffers[i];
        if (src && src !== resultBufs[i]) elseStmts.push(copyBuffer(ctx, src, resultBufs[i]));
      }
      if (elseStmts.length > 0) {
        elseBody = elseStmts.length === 1 ? elseStmts[0] : new SeqNode(elseStmts);
      }
    }

    return new IfThenElseNode(predLoad, thenBody, elseBody);
  });

  registerLoweringRule('while', (ctx, op, inputs, outputs) => {
    const condRegion = op.regions[0];
    const bodyRegion = op.regions[1];

    const loopBufs: Buffer[] = new Array(inputs.length);
    const initStmts: TirNode[] = [];
    for (let i = 0; i < inputs.length; i++) {
      const stateBuf = outputs[i] || ctx.getOrAllocBuffer({ type: op.getResult(i).type });
      loopBufs[i] = stateBuf;
      initStmts.push(copyBuffer(ctx, inputs[i], stateBuf));
    }

    for (let i = 0; i < op.numResults; i++) {
      ctx.bufferMap.set(op.getResult(i), loopBufs[i]);
    }

    const condVar = new Buffer(`_wcond_${ctx.varCounter++}`, [], 'bool', MemoryScope.GLOBAL);

    const condResult = lowerRegionBody(ctx, condRegion, loopBufs);
    const condStmts = condResult.stmts.slice();
    const predBuf = condResult.yieldBuffers[0];
    condStmts.push(new BufferStoreNode(condVar, [], new BufferLoadNode(predBuf, [])));
    const condBody = condStmts.length === 1 ? condStmts[0] : new SeqNode(condStmts);

    const bodyResult = lowerRegionBody(ctx, bodyRegion, loopBufs);
    const bodyStmts = bodyResult.stmts.slice();
    for (let i = 0; i < loopBufs.length && i < bodyResult.yieldBuffers.length; i++) {
      const src = bodyResult.yieldBuffers[i];
      if (src && src !== loopBufs[i]) bodyStmts.push(copyBuffer(ctx, src, loopBufs[i]));
    }
    const loopBody = bodyStmts.length === 1 ? bodyStmts[0] : new SeqNode(bodyStmts);

    return new SeqNode([...initStmts, new WhileNode(condVar, condBody, loopBody)]);
  });

  registerLoweringRule('scan', (ctx, op, inputs, outputs) => {
    const numXs = op.getAttr<number>('num_xs') as number;
    const numCarry = op.getAttr<number>('num_carry') as number;
    const xsBufs = inputs.slice(0, numXs);
    const carryInitBufs = inputs.slice(numXs);

    const carryBufs: Buffer[] = new Array(numCarry);
    const initStmts: TirNode[] = [];
    for (let i = 0; i < numCarry; i++) {
      const stateBuf = outputs[i] || ctx.getOrAllocBuffer({ type: op.getResult(i).type });
      carryBufs[i] = stateBuf;
      initStmts.push(copyBuffer(ctx, carryInitBufs[i], stateBuf));
      ctx.bufferMap.set(op.getResult(i), stateBuf);
    }
    const numYs = outputs.length - numCarry;
    const ysBufs: Buffer[] = new Array(numYs);
    for (let i = 0; i < numYs; i++) {
      const yBuf = outputs[numCarry + i] || ctx.getOrAllocBuffer({ type: op.getResult(numCarry + i).type });
      ysBufs[i] = yBuf;
      ctx.bufferMap.set(op.getResult(numCarry + i), yBuf);
    }

    const counter = ctx.allocVar('t');
    const extentT = ctx.extentNode(xsBufs[0].shape[0], xsBufs[0], 0);

    const bodyStmts: TirNode[] = [];
    const xtBufs: Buffer[] = new Array(numXs);
    for (let i = 0; i < numXs; i++) {
      const xt = ctx.getOrAllocBuffer({ type: { shape: xsBufs[i].shape.slice(1), dtype: xsBufs[i].dtype } } as unknown as BufferOwner);
      xtBufs[i] = xt;
      bodyStmts.push(sliceCopyIn(ctx, xsBufs[i], xt, counter));
    }

    const bodyResult = lowerRegionBody(ctx, op.regions[0], [...xtBufs, ...carryBufs]);
    for (const s of bodyResult.stmts) bodyStmts.push(s);
    for (let i = 0; i < numCarry; i++) {
      const src = bodyResult.yieldBuffers[i];
      if (src && src !== carryBufs[i]) bodyStmts.push(copyBuffer(ctx, src, carryBufs[i]));
    }
    for (let i = 0; i < numYs; i++) {
      bodyStmts.push(sliceCopyOut(ctx, bodyResult.yieldBuffers[numCarry + i], ysBufs[i], counter));
    }
    bodyStmts.push(new SyncThreadsNode());

    const loopBody = bodyStmts.length === 1 ? bodyStmts[0] : new SeqNode(bodyStmts);
    return new SeqNode([...initStmts, new ForNode(counter, new IntImmNode(0), extentT, ForKind.RECURRENCE, loopBody)]);
  });
}
