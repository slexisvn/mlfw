import { PrimFunc, SeqNode, BufferStoreNode, BufferLoadNode, BlockNode } from '../../ir/tensor/nodes.js';
import { topoSortOpSet } from '../../ir/graph/graph_algorithms.js';
import { registry } from '../../ir/graph/ops.js';

import { LoweringContext, registerLoweringRule, hasLoweringRule, getLoweringRule, lowerConstant, expandConstantStores, isConstantOp, makeLoopNest, wrapInLoops } from './lowering_registry.js';
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
import { FuncAttr } from '../../ir/func_attrs.js';
import type { GraphFunction } from '../../ir/graph/function.js';
import type { Operation } from '../../ir/graph/operation.js';
import type { Value } from '../../ir/graph/value.js';
import type { Block } from '../../ir/graph/block.js';
import type { Buffer } from '../../ir/tensor/buffer.js';
import type { ConstBuffer } from './lowering_registry.js';
import type { TirNode, VariableNode } from '../../ir/tensor/nodes.js';
import type { TensorType } from '../../ir/graph/types.js';
import type { CompilerContext } from '../../pipeline/compiler_context.js';
import type { CompileTarget } from '../../pipeline/pipeline_types.js';

const BROADCAST_VIEW_SAFE_EXTRA = ['compare', 'select', 'clamp', 'convert', 'copy_to_device', 'dot', 'fusion'];

for (const opName of [...elementwiseOpNames(), ...BROADCAST_VIEW_SAFE_EXTRA]) {
  if (registry.has(opName)) registry.registerOpAttr(opName, 'broadcastViewSafe', true);
}

function broadcastViewSafeForUser(value: Value, user: Operation, visited: Set<string> = new Set()): boolean {
  const def = registry.get(user.opName);
  if (!def || !def.getAttr('broadcastViewSafe')) return false;
  if (user.opName !== 'fusion') return true;
  const region = user.regions[0];
  if (!region) return false;
  const block = region.entryBlock as Block;
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

function topologicalOps(graphFunc: GraphFunction): Operation[] {
  return topoSortOpSet([...graphFunc.ops()], 'ignore');
}

export function lowerGraphToPrimFunc(graphFunc: GraphFunction, target: CompileTarget | null = null, context: CompilerContext | null = null): PrimFunc {
  const ctx = new LoweringContext();
  ctx.target = target;
  const params: VariableNode[] = [];
  const bufferMap = new Map<VariableNode, Buffer>();

  for (const arg of graphFunc.args) {
    const v = ctx.allocVar('arg');
    params.push(v);
    bufferMap.set(v, ctx.getOrAllocBuffer(arg));
  }

  const retOp = graphFunc.getReturnOp() as Operation;
  const inputBuffers = new Set<Buffer>();
  for (const arg of graphFunc.args) {
    inputBuffers.add(ctx.getOrAllocBuffer(arg));
  }

  const copyPairs: { src: Buffer; dst: Buffer }[] = [];
  const usedReturnBuffers = new Set<Buffer>();
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

  const returnedValues = new Set<Value>();
  for (let i = 0; i < retOp.numOperands; i++) {
    returnedValues.add(retOp.getOperand(i));
  }

  const stmts: TirNode[] = [];

  for (const op of graphFunc.ops()) {
    if (!isConstantOp(op.opName)) continue;
    const stmt = lowerConstant(ctx, op);
    if (stmt) stmts.push(stmt);
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
      const outShape = (op.getResult(0).type as TensorType).shape;
      const dims = op.getAttr<readonly number[]>('broadcast_dimensions');
      let broadcastDims: readonly number[];
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

    const inputs: Buffer[] = new Array(op.numOperands);
    for (let i = 0; i < op.numOperands; i++) inputs[i] = ctx.getOrAllocBuffer(op.getOperand(i));
    const outputs: Buffer[] = new Array(op.numResults);
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

  const boundBuffers = new Set<Buffer>(bufferMap.values());
  const constBuffers: ConstBuffer[] = [];
  for (const cb of ctx.constBuffers) {
    if (boundBuffers.has(cb.buffer)) {
      stmts.push(expandConstantStores(cb.buffer, cb.data, ctx.blockName('constant_block')));
      continue;
    }
    const v = ctx.allocVar('const');
    params.push(v);
    bufferMap.set(v, cb.buffer);
    boundBuffers.add(cb.buffer);
    constBuffers.push(cb);
  }

  const shapeParams: VariableNode[] = [];
  const seenSp = new Set<string>();
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
  if (constBuffers.length > 0) primFunc.setAttr(FuncAttr.CONST_BUFFERS, constBuffers);
  if (graphFunc._partitionTarget) primFunc.setAttr(FuncAttr.PARTITION_TARGET, graphFunc._partitionTarget);
  return primFunc;
}
