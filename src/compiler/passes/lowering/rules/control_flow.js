import {
  BufferStoreNode, BufferLoadNode, SeqNode,
  IfThenElseNode, WhileNode
} from '../../../ir/tensor/nodes.js';
import { Buffer } from '../../../ir/tensor/buffer.js';
import {
  registerLoweringRule, getLoweringRule, lowerConstant, CONSTANT_OPS
} from '../lowering_registry.js';

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
    const thenResult = lowerRegionBody(ctx, thenRegion, []);
    const thenBody = thenResult.stmts.length === 1 ? thenResult.stmts[0] : new SeqNode(thenResult.stmts);
    let elseBody = null;
    if (elseRegion && elseRegion.entryBlock) {
      const elseResult = lowerRegionBody(ctx, elseRegion, []);
      if (elseResult.stmts.length > 0) {
        elseBody = elseResult.stmts.length === 1 ? elseResult.stmts[0] : new SeqNode(elseResult.stmts);
      }
    }
    for (let i = 0; i < thenResult.yieldBuffers.length && i < outputs.length; i++) {
      const src = thenResult.yieldBuffers[i];
      if (src !== outputs[i]) {
        ctx.bufferMap.set(op.getResult(i), src);
      }
    }
    return new IfThenElseNode(predLoad, thenBody, elseBody);
  });

  registerLoweringRule('while', (ctx, op, inputs, outputs) => {
    const loopBufs = new Array(inputs.length);
    for (let i = 0; i < inputs.length; i++) loopBufs[i] = inputs[i];
    const condRegion = op.regions[0];
    const bodyRegion = op.regions[1];
    const condVar = ctx.allocVar('_wcond', 'bool');
    const condResult = lowerRegionBody(ctx, condRegion, loopBufs);
    const condBody = condResult.stmts.length === 1 ? condResult.stmts[0] : new SeqNode(condResult.stmts);
    const bodyResult = lowerRegionBody(ctx, bodyRegion, loopBufs);
    const loopBody = bodyResult.stmts.length === 1 ? bodyResult.stmts[0] : new SeqNode(bodyResult.stmts);
    for (let i = 0; i < outputs.length && i < bodyResult.yieldBuffers.length; i++) {
      if (bodyResult.yieldBuffers[i] !== outputs[i]) {
        ctx.bufferMap.set(op.getResult(i), bodyResult.yieldBuffers[i]);
      }
    }
    return new WhileNode(condVar, condBody, loopBody);
  });
}
