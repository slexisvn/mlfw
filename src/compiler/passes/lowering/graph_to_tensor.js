import { PrimFunc, SeqNode, BufferStoreNode, BufferLoadNode, BlockNode } from '../../ir/tensor/nodes.js';
import { topoSortOpSet } from '../../ir/graph/graph_algorithms.js';
import { registry } from '../../ir/graph/ops.js';

import { LoweringContext, registerLoweringRule, hasLoweringRule, getLoweringRule, lowerConstant, isConstantOp, makeLoopNest, wrapInLoops } from './lowering_registry.js';
import { isTerminatorOp, isBroadcastOp } from '../../ir/graph/op_traits.js';
import { register as registerElementwise } from './rules/elementwise.js';
import { register as registerShape } from './rules/shape.js';
import { register as registerReduction } from './rules/reduction.js';
import { register as registerLinalg } from './rules/linalg.js';
import { register as registerControlFlow } from './rules/control_flow.js';
import { register as registerLayout } from './rules/layout.js';
import { register as registerQuantization } from './rules/quantization.js';
import { register as registerFusion, canInlineFuse, lowerFusion, canLowerAsElementwiseFusion, lowerFusionAsIndividualOps, registerInlineFusionBuilder } from './rules/fusion.js';
import { register as registerPooling } from './rules/pooling.js';
import { register as registerResize } from './rules/resize.js';
import { register as registerAttention } from './rules/attention.js';
import { elementwiseOpNames } from './rules/elementwise.js';

const BROADCAST_VIEW_SAFE_EXTRA = ['compare', 'select', 'clamp', 'convert', 'copy_to_device', 'dot', 'fusion'];

for (const opName of [...elementwiseOpNames(), ...BROADCAST_VIEW_SAFE_EXTRA]) {
  if (registry.has(opName)) registry.registerOpAttr(opName, 'broadcastViewSafe', true);
}

function broadcastViewSafeForUser(value, user, visited = new Set()) {
  const def = registry.get(user.opName);
  if (!def || !def.getAttr('broadcastViewSafe')) return false;
  if (user.opName !== 'fusion') return true;
  const region = user.regions[0];
  if (!region) return false;
  const block = region.entryBlock;
  for (let o = 0; o < user.numOperands; o++) {
    if (user.getOperand(o) !== value) continue;
    const blockArg = block.arguments[o];
    for (const inner of blockArg.getUsers()) {
      const key = `${blockArg.id}:${inner.id}`;
      if (visited.has(key)) continue;
      visited.add(key);
      if (!broadcastViewSafeForUser(blockArg, inner, visited)) return false;
    }
  }
  return true;
}

registerElementwise();
registerShape();
registerReduction();
registerLinalg();
registerControlFlow();
registerLayout();
registerQuantization();
registerFusion();
registerPooling();
registerResize();
registerAttention();

export { LoweringContext, hasLoweringRule, registerLoweringRule, canInlineFuse, registerInlineFusionBuilder };

function topologicalOps(graphFunc) {
  return topoSortOpSet([...graphFunc.ops()], 'ignore');
}

export function lowerGraphToPrimFunc(graphFunc, target = null, context = null) {
  const ctx = new LoweringContext();
  const params = [];
  const bufferMap = new Map();

  for (const arg of graphFunc.args) {
    const v = ctx.allocVar('arg');
    params.push(v);
    bufferMap.set(v, ctx.getOrAllocBuffer(arg));
  }

  const retOp = graphFunc.getReturnOp();
  const inputBuffers = new Set();
  for (const arg of graphFunc.args) {
    inputBuffers.add(ctx.getOrAllocBuffer(arg));
  }

  const copyPairs = [];
  const usedReturnBuffers = new Set();
  for (let i = 0; i < retOp.numOperands; i++) {
    const v = ctx.allocVar('ret');
    params.push(v);
    const srcBuf = ctx.getOrAllocBuffer(retOp.getOperand(i));
    if (inputBuffers.has(srcBuf) || usedReturnBuffers.has(srcBuf)) {
      const outBuf = ctx.allocFreshBuffer(retOp.getOperand(i));
      bufferMap.set(v, outBuf);
      copyPairs.push({ src: srcBuf, dst: outBuf });
    } else {
      bufferMap.set(v, srcBuf);
      usedReturnBuffers.add(srcBuf);
    }
  }

  const returnedValues = new Set();
  for (let i = 0; i < retOp.numOperands; i++) {
    returnedValues.add(retOp.getOperand(i));
  }

  const stmts = [];

  for (const op of graphFunc.ops()) {
    if (isConstantOp(op.opName)) stmts.push(lowerConstant(ctx, op));
  }

  for (const op of topologicalOps(graphFunc)) {
    if (isTerminatorOp(op.opName)) continue;
    if (isConstantOp(op.opName)) continue;

    if (op.opName === 'fusion') {
      if (canLowerAsElementwiseFusion(op)) {
        stmts.push(lowerFusion(ctx, op));
      } else {
        lowerFusionAsIndividualOps(ctx, op, stmts);
      }
      continue;
    }

    if (isBroadcastOp(op.opName)
        && !returnedValues.has(op.getResult(0))
        && op.getOperand(0).getUsers().length === 1
        && op.getResult(0).getUsers().every((u) => broadcastViewSafeForUser(op.getResult(0), u))) {
      const srcBuf = ctx.getOrAllocBuffer(op.getOperand(0));
      const outShape = op.getResult(0).type.shape;
      const dims = op.getAttr('broadcast_dimensions');
      let broadcastDims;
      if (dims && dims.length > 0) {
        broadcastDims = dims;
      } else {
        const offset = outShape.length - srcBuf.shape.length;
        broadcastDims = Array.from({length: srcBuf.shape.length}, (_, i) => i + offset);
      }
      srcBuf.broadcastDims = broadcastDims;
      ctx.bufferMap.set(op.getResult(0), srcBuf);
      continue;
    }

    const rule = getLoweringRule(op.opName, target, context);
    if (!rule) throw new Error(`No lowering rule defined for op: ${op.opName}`);

    const inputs = new Array(op.numOperands);
    for (let i = 0; i < op.numOperands; i++) inputs[i] = ctx.getOrAllocBuffer(op.getOperand(i));
    const outputs = new Array(op.numResults);
    for (let i = 0; i < op.numResults; i++) outputs[i] = ctx.getOrAllocBuffer(op.getResult(i));

    const stmt = rule(ctx, op, inputs, outputs);
    if (stmt) stmts.push(stmt);
  }

  for (const { src, dst } of copyPairs) {
    const { loopVars, loopBinds, indices, extentNodes } = makeLoopNest(ctx, src.shape, src);
    const load = new BufferLoadNode(src, indices);
    const store = new BufferStoreNode(dst, indices, load);
    const block = new BlockNode(ctx.blockName('copy_block'), loopBinds, [{ buffer: src }], [{ buffer: dst }], store);
    stmts.push(wrapInLoops(block, loopVars, src.shape, extentNodes));
  }

  const shapeParams = [];
  const seenSp = new Set();
  for (const sp of ctx.shapeParams.values()) {
    if (seenSp.has(sp.name)) continue;
    seenSp.add(sp.name);
    shapeParams.push(sp);
  }
  for (const [name, node] of ctx.symVars) {
    if (!seenSp.has(node.name)) {
      throw new Error(`Symbolic dimension '${name}' has no input dimension to bind it to at runtime`);
    }
  }
  for (const sp of shapeParams) params.push(sp);

  const primFunc = new PrimFunc(graphFunc.name, params, stmts.length === 1 ? stmts[0] : new SeqNode(stmts), bufferMap, shapeParams, new Map(ctx.shapeParams));
  if (graphFunc._partitionTarget) primFunc._partitionTarget = graphFunc._partitionTarget;
  return primFunc;
}
