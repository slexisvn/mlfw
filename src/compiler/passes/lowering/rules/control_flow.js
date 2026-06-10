import {
  BufferStoreNode, BufferLoadNode, SeqNode, BlockNode,
  IfThenElseNode, WhileNode
} from '../../../ir/tensor/nodes.js';
import { Buffer } from '../../../ir/tensor/buffer.js';
import { MemoryScope } from '../../../ir/tensor/tensor_types.js';
import {
  registerLoweringRule, getLoweringRule, lowerConstant, CONSTANT_OPS,
  makeLoopNest, wrapInLoops
} from '../lowering_registry.js';

function copyBuffer(ctx, srcBuf, dstBuf) {
  const { loopVars, loopBinds, indices } = makeLoopNest(ctx, dstBuf.shape);
  const store = new BufferStoreNode(dstBuf, indices, new BufferLoadNode(srcBuf, indices));
  const block = new BlockNode(ctx.blockName('cf_copy'), loopBinds, [{ buffer: srcBuf }], [{ buffer: dstBuf }], store);
  return wrapInLoops(block, loopVars, dstBuf.shape);
}

function lowerRegionBody(ctx, region, argBuffers) {
  const entryBlock = region.entryBlock;
  const valueMap = new Map();
  for (let i = 0; i < entryBlock.arguments.length; i++) {
    valueMap.set(entryBlock.arguments[i], argBuffers[i]);
  }
  const stmts = [];
  for (const innerOp of entryBlock.ops()) {
    if (innerOp.opName === 'yield') {
      const results = new Array(innerOp.numOperands);
      for (let i = 0; i < innerOp.numOperands; i++) {
        results[i] = valueMap.get(innerOp.getOperand(i)) || ctx.getOrAllocBuffer(innerOp.getOperand(i));
      }
      return { stmts, yieldBuffers: results };
    }
    if (CONSTANT_OPS.has(innerOp.opName)) {
      stmts.push(lowerConstant(ctx, innerOp));
      continue;
    }
    const outerOperands = new Array(innerOp.numOperands);
    for (let i = 0; i < innerOp.numOperands; i++) {
      outerOperands[i] = valueMap.get(innerOp.getOperand(i)) || innerOp.getOperand(i);
    }
    const inputs = new Array(outerOperands.length);
    for (let i = 0; i < outerOperands.length; i++) {
      inputs[i] = outerOperands[i] instanceof Buffer ? outerOperands[i] : ctx.getOrAllocBuffer(outerOperands[i]);
    }
    const outputs = new Array(innerOp.numResults);
    for (let i = 0; i < innerOp.numResults; i++) {
      const proxy = { type: innerOp.getResult(i).type };
      outputs[i] = ctx.getOrAllocBuffer(proxy);
      valueMap.set(innerOp.getResult(i), outputs[i]);
    }
    const rule = getLoweringRule(innerOp.opName);
    if (!rule) throw new Error(`No lowering rule for op '${innerOp.opName}' inside region`);
    const stmt = rule(ctx, innerOp, inputs, outputs);
    if (stmt) stmts.push(stmt);
  }
  return { stmts, yieldBuffers: [] };
}

export { lowerRegionBody };

export function register() {
  registerLoweringRule('if', (ctx, op, inputs, outputs) => {
    const predBuf = inputs[0];
    const predLoad = new BufferLoadNode(predBuf, []);
    const thenRegion = op.regions[0];
    const elseRegion = op.regions[1];

    const resultBufs = new Array(op.numResults);
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

    let elseBody = null;
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

    const loopBufs = new Array(inputs.length);
    for (let i = 0; i < inputs.length; i++) loopBufs[i] = inputs[i];

    for (let i = 0; i < op.numResults; i++) {
      const buf = loopBufs[i] || ctx.getOrAllocBuffer(op.getResult(i));
      ctx.bufferMap.set(op.getResult(i), buf);
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

    return new WhileNode(condVar, condBody, loopBody);
  });
}
