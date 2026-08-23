import { FloatImmNode, IntImmNode, CompareNode, BufferStoreNode, BufferLoadNode, BlockNode, SeqNode, IfThenElseNode, CallExternNode, CastNode, LetStmtNode } from '../../../ir/tensor/nodes.js';

import { getLoweringRule, makeLoopNest, wrapInLoops, computeBroadcastIndices, bufRefs, lowerConstant, isConstantOp, buildReduceGeometry, splitReduceDims, emitReduceMeanDiv } from '../lowering_registry.js';
import { buildElementwiseExpr, elementwiseOpNames } from './elementwise.js';
import { getReduceCombiner } from './reduction.js';
import { buildQuantizeExpr, buildDequantizeExpr } from '../quant_math.js';
import { isBroadcastOp, isReductionOp } from '../../../ir/graph/op_traits.js';
import { walk as irWalk } from '../../../ir/ir_visitor.js';
import type { BufferOwner, LoweringContext } from '../lowering_registry.js';
import type { TirNode, VariableNode } from '../../../ir/tensor/nodes.js';
import type { Operation } from '../../../ir/graph/operation.js';
import type { Buffer } from '../../../ir/tensor/buffer.js';
import type { Value } from '../../../ir/graph/value.js';
import type { Block } from '../../../ir/graph/block.js';
import type { Shape, TensorType } from '../../../ir/graph/types.js';
import type { BufferRegionLike } from '../../../ir/tensor/buffer.js';
import type { IRNode } from '../../../ir/ir_visitor.js';

export type InlineFusionBuilder = (innerOp: Operation, args: readonly TirNode[], dtype: string) => TirNode;
export type FusionLoweringStrategy = 'inline-loop' | 'reduction-nest' | 'per-op';
type CseBinding = { variable: VariableNode; value: TirNode };

const INLINE_FUSION_BUILDERS = new Map<string, InlineFusionBuilder>();

export function registerInlineFusionBuilder(opName: string, builder: InlineFusionBuilder): void {
  INLINE_FUSION_BUILDERS.set(opName, builder);
}

export function canInlineFuse(opName: string): boolean {
  return INLINE_FUSION_BUILDERS.has(opName);
}

export function getInlineFusionBuilder(opName: string): InlineFusionBuilder | undefined {
  return INLINE_FUSION_BUILDERS.get(opName);
}

function initBuiltinFusionBuilders(): void {
  for (const opName of elementwiseOpNames()) {
    INLINE_FUSION_BUILDERS.set(opName, (innerOp, args, dtype) =>
      buildElementwiseExpr(innerOp.opName, args, dtype)
    );
  }

  INLINE_FUSION_BUILDERS.set('compare', (innerOp, args) =>
    new CompareNode(innerOp.getAttr<string>('direction') || 'eq', args[0], args[1])
  );

  INLINE_FUSION_BUILDERS.set('select', (_innerOp, args) =>
    new IfThenElseNode(args[0], args[1], args[2])
  );

  INLINE_FUSION_BUILDERS.set('clamp', (_innerOp, args, dtype) =>
    new CallExternNode('min', [new CallExternNode('max', [args[1], args[0]], dtype), args[2]], dtype)
  );

  INLINE_FUSION_BUILDERS.set('convert', (innerOp, args) =>
    new CastNode(args[0], (innerOp.getOperand(0).type as TensorType).dtype, innerOp.getAttr<string>('target_dtype') || (innerOp.getResult(0).type as TensorType).dtype)
  );

  INLINE_FUSION_BUILDERS.set('broadcast_in_dim', (_innerOp, args) => args[0]);
  INLINE_FUSION_BUILDERS.set('broadcast', (_innerOp, args) => args[0]);

  INLINE_FUSION_BUILDERS.set('iota', () => {
    throw new Error('iota fusion must be handled by the index-aware path in lowerFusion');
  });

  INLINE_FUSION_BUILDERS.set('quantize', (innerOp, args) =>
    buildQuantizeExpr(args[0], {
      scale: innerOp.getAttr<number>('scale') as number,
      zeroPoint: innerOp.getAttr<number>('zero_point') as number,
      targetDtype: innerOp.getAttr<string>('target_dtype') || 'i8',
    }));

  INLINE_FUSION_BUILDERS.set('dequantize', (innerOp, args) =>
    buildDequantizeExpr(args[0], {
      scale: innerOp.getAttr<number>('scale') as number,
      zeroPoint: innerOp.getAttr<number>('zero_point') as number,
      srcDtype: (innerOp.getOperand(0).type as TensorType)?.dtype || 'i8',
      targetDtype: innerOp.getAttr<string>('target_dtype') || 'f32',
    }));
}

const CSE_TRIVIAL = new Set(['BufferLoadNode', 'VariableNode', 'IntImmNode', 'FloatImmNode']);
const INDEX_REPLICATED = new Set(['iota']);
const FUSION_REDUCE_PREFIXES = Object.freeze({ init: 'fi', spatial: 'fa', reduce: 'fr', reduceIter: 'frv' });

function isReplicatedOp(opName: string): boolean {
  return isConstantOp(opName) || INDEX_REPLICATED.has(opName);
}

function collectArgBroadcastDims(entryBlock: Block): Map<number, readonly number[]> {
  const valueDims = new Map<Value, readonly number[]>();
  const record = (val: Value, dims: readonly number[]): void => {
    const shape = tensorShapeOf(val);
    if (!shape || shape.length !== dims.length || valueDims.has(val)) return;
    valueDims.set(val, dims);
  };
  const opsArr = [...entryBlock.ops()];
  for (let k = opsArr.length - 1; k >= 0; k--) {
    const innerOp = opsArr[k];
    if (isBroadcastOp(innerOp.opName)) {
      const dims = innerOp.getAttr<readonly number[]>('broadcast_dimensions');
      if (dims && dims.length > 0) record(innerOp.getOperand(0), dims);
      continue;
    }
    if (innerOp.opName === 'yield' || isConstantOp(innerOp.opName)) continue;
    for (let r = 0; r < innerOp.numResults; r++) {
      const dims = valueDims.get(innerOp.getResult(r));
      if (!dims) continue;
      for (let i = 0; i < innerOp.numOperands; i++) record(innerOp.getOperand(i), dims);
    }
  }
  const argDims = new Map<number, readonly number[]>();
  const blockArgs = entryBlock.arguments;
  for (let i = 0; i < blockArgs.length; i++) {
    const dims = valueDims.get(blockArgs[i]);
    if (dims) argDims.set(i, dims);
  }
  return argDims;
}

function countRegionUses(entryBlock: Block): Map<Value, number> {
  const useCount = new Map<Value, number>();
  for (const innerOp of entryBlock.ops()) {
    for (let i = 0; i < innerOp.numOperands; i++) {
      const val = innerOp.getOperand(i);
      useCount.set(val, (useCount.get(val) || 0) + 1);
    }
  }
  return useCount;
}

function exprDtype(expr: TirNode, val: Value, fallback: string): string {
  if (expr.type === 'CompareNode') return 'i32';
  if (expr.type === 'CastNode') return (expr as unknown as { toDtype: string }).toDtype;
  const t = val.type as TensorType;
  return (t && t.dtype) ? t.dtype : fallback;
}

class RegionExprBuilder {
  private ctx: LoweringContext;
  private useCount: ReadonlyMap<Value, number>;
  private fallbackDtype: string;
  private exprMap: Map<Value, TirNode>;
  private cseVars: Map<Value, VariableNode>;
  private bindings: CseBinding[];

  constructor(ctx: LoweringContext, useCount: ReadonlyMap<Value, number>, fallbackDtype: string) {
    this.ctx = ctx;
    this.useCount = useCount;
    this.fallbackDtype = fallbackDtype;
    this.exprMap = new Map();
    this.cseVars = new Map();
    this.bindings = [];
  }

  bind(value: Value, expr: TirNode): void {
    this.exprMap.set(value, expr);
  }

  bindBlockArgs(entryBlock: Block, inputs: readonly Buffer[], argDims: ReadonlyMap<number, readonly number[]>, shapeRef: Buffer, indices: readonly TirNode[]): void {
    const blockArgs = entryBlock.arguments;
    for (let i = 0; i < blockArgs.length; i++) {
      if (this.exprMap.has(blockArgs[i])) continue;
      const inBuf = inputs[i];
      const explicitDims = argDims.get(i);
      let loadIndices: TirNode[];
      if (explicitDims) {
        loadIndices = new Array(inBuf.shape.length);
        for (let j = 0; j < inBuf.shape.length; j++) {
          loadIndices[j] = inBuf.shape[j] === 1 ? new IntImmNode(0) : indices[explicitDims[j]];
        }
      } else {
        loadIndices = computeBroadcastIndices(inBuf, shapeRef, indices);
      }
      this.exprMap.set(blockArgs[i], new BufferLoadNode(inBuf, loadIndices));
    }
  }

  get(val: Value): TirNode {
    const expr = this.exprMap.get(val);
    if (expr === undefined) {
      throw new Error(`Fusion lowering: unmapped operand from '${val.definingOp ? val.definingOp.opName : 'unknown'}'`);
    }
    if ((this.useCount.get(val) || 0) > 1 && !CSE_TRIVIAL.has(expr.type)) {
      return this.letBind(val, expr, exprDtype(expr, val, this.fallbackDtype));
    }
    return expr;
  }

  letBind(val: Value, expr: TirNode, dtype: string): TirNode {
    if (CSE_TRIVIAL.has(expr.type)) return expr;
    const existing = this.cseVars.get(val);
    if (existing) return existing;
    const v = this.ctx.allocVar('cse', dtype);
    this.cseVars.set(val, v);
    this.bindings.push({ variable: v, value: expr });
    this.exprMap.set(val, v);
    return v;
  }

  evaluate(ops: Iterable<Operation>, indices: readonly TirNode[]): void {
    for (const innerOp of ops) {
      if (isConstantOp(innerOp.opName)) {
        const val = innerOp.getAttr<number>('value');
        this.exprMap.set(innerOp.getResult(0), new FloatImmNode(typeof val === 'number' ? val : 0));
        continue;
      }
      if (innerOp.opName === 'iota') {
        const dim = innerOp.getAttr<number>('iota_dimension') ?? innerOp.getAttr<number>('dimension') ?? 0;
        this.exprMap.set(innerOp.getResult(0), indices[dim]);
        continue;
      }
      const builder = INLINE_FUSION_BUILDERS.get(innerOp.opName);
      if (!builder) throw new Error(`Fusion lowering: unsupported op '${innerOp.opName}' inside fusion body`);
      const args: TirNode[] = new Array(innerOp.numOperands);
      for (let i = 0; i < innerOp.numOperands; i++) args[i] = this.get(innerOp.getOperand(i));
      const innerDtype = (innerOp.getResult(0).type as TensorType).dtype;
      this.exprMap.set(innerOp.getResult(0), builder(innerOp, args, innerDtype));
    }
  }

  wrap(body: TirNode): TirNode {
    let result = body;
    for (let i = this.bindings.length - 1; i >= 0; i--) {
      result = new LetStmtNode(this.bindings[i].variable, this.bindings[i].value, result);
    }
    return result;
  }
}

function loadedBuffers(root: TirNode): BufferRegionLike[] {
  const seen = new Set<Buffer>();
  const refs: BufferRegionLike[] = [];
  irWalk(root as unknown as IRNode, ((node: TirNode) => {
    if (node.type !== 'BufferLoadNode') return;
    const buffer = (node as unknown as { buffer: Buffer }).buffer;
    if (seen.has(buffer)) return;
    seen.add(buffer);
    refs.push({ buffer });
  }) as never);
  return refs;
}

function fusionBuffers(ctx: LoweringContext, op: Operation): { inputs: Buffer[]; outputs: Buffer[] } {
  const inputs: Buffer[] = new Array(op.numOperands);
  for (let i = 0; i < op.numOperands; i++) inputs[i] = ctx.getOrAllocBuffer(op.getOperand(i));
  const outputs: Buffer[] = new Array(op.numResults);
  for (let i = 0; i < op.numResults; i++) outputs[i] = ctx.getOrAllocBuffer(op.getResult(i));
  return { inputs, outputs };
}

function regionBodyOps(entryBlock: Block): { bodyOps: Operation[]; yieldOp: Operation | null } {
  const bodyOps: Operation[] = [];
  let yieldOp: Operation | null = null;
  for (const innerOp of entryBlock.ops()) {
    if (innerOp.opName === 'yield') { yieldOp = innerOp; break; }
    bodyOps.push(innerOp);
  }
  return { bodyOps, yieldOp };
}

function lowerFusion(ctx: LoweringContext, op: Operation): TirNode {
  const { inputs, outputs } = fusionBuffers(ctx, op);
  const outBuf = outputs[0];
  const { loopVars, loopBinds, indices: outIndices, extentNodes } = makeLoopNest(ctx, outBuf.shape, outBuf);

  const entryBlock = op.regions[0].entryBlock as Block;
  const builder = new RegionExprBuilder(ctx, countRegionUses(entryBlock), outBuf.dtype);
  builder.bindBlockArgs(entryBlock, inputs, collectArgBroadcastDims(entryBlock), outBuf, outIndices);

  const { bodyOps, yieldOp } = regionBodyOps(entryBlock);
  builder.evaluate(bodyOps, outIndices);

  const stores: TirNode[] = [];
  if (yieldOp) {
    for (let i = 0; i < yieldOp.numOperands; i++) {
      stores.push(new BufferStoreNode(outputs[i], outIndices, builder.get(yieldOp.getOperand(i))));
    }
  }

  const storeBody = builder.wrap(stores.length === 1 ? stores[0] : new SeqNode(stores));
  const block = new BlockNode(ctx.blockName('fusion_block'), loopBinds, bufRefs(inputs), bufRefs(outputs), storeBody);
  return wrapInLoops(block, loopVars, outBuf.shape, extentNodes);
}

function shapesEqual(a: Shape, b: Shape): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

function tensorShapeOf(val: Value): Shape | null {
  const t = val.type as TensorType;
  return t && t.shape ? t.shape : null;
}

function canLowerAsElementwiseFusion(op: Operation): boolean {
  const region = op.regions[0];
  if (!region) return false;
  for (const innerOp of (region.entryBlock as Block).ops()) {
    if (innerOp.opName === 'yield') continue;
    if (isConstantOp(innerOp.opName)) {
      if (typeof innerOp.getAttr('value') === 'number') continue;
      return false;
    }
    if (!INLINE_FUSION_BUILDERS.has(innerOp.opName)) return false;
  }
  if (op.numResults > 1) {
    const refShape = (op.getResult(0).type as TensorType).shape;
    for (let i = 1; i < op.numResults; i++) {
      if (!shapesEqual((op.getResult(i).type as TensorType).shape, refShape)) return false;
    }
  }
  return true;
}

type ReductionFusionPlan = {
  entryBlock: Block;
  reduceOp: Operation;
  reduceType: string;
  reduceInput: Value;
  reduceResult: Value;
  replicated: Operation[];
  prologue: Operation[];
  epilogue: Operation[];
  fullShape: Shape;
  spatialDims: number[];
  reduceDims: number[];
  anchor: Buffer;
  initExpr: TirNode;
  redOwner: BufferOwner;
  materialize: Map<Value, BufferOwner>;
  epilogueStores: { owner: BufferOwner; value: Value }[];
};

function broadcastDimsOf(op: Operation, srcRank: number, outRank: number): readonly number[] {
  const dims = op.getAttr<readonly number[]>('broadcast_dimensions');
  if (dims && dims.length > 0) return dims;
  const offset = outRank - srcRank;
  return Array.from({ length: srcRank }, (_, i) => i + offset);
}

function dimsEqual(a: readonly number[], b: readonly number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

function reduceDependencyCone(reduceOp: Operation, regionOps: ReadonlySet<Operation>): Set<Operation> {
  const cone = new Set<Operation>();
  const stack: Value[] = [];
  for (let i = 0; i < reduceOp.numOperands; i++) stack.push(reduceOp.getOperand(i));
  while (stack.length > 0) {
    const def = (stack.pop() as Value).definingOp;
    if (!def || def === reduceOp || cone.has(def) || !regionOps.has(def)) continue;
    cone.add(def);
    for (let i = 0; i < def.numOperands; i++) stack.push(def.getOperand(i));
  }
  return cone;
}

function findShapeAnchor(candidates: readonly Buffer[][], shape: Shape): Buffer | null {
  for (const group of candidates) {
    for (const buf of group) {
      if (!buf.broadcastDims && shapesEqual(buf.shape, shape)) return buf;
    }
  }
  return null;
}

function planReductionFusion(op: Operation, inputs: readonly Buffer[], outputs: readonly Buffer[]): ReductionFusionPlan | null {
  const region = op.regions[0];
  if (!region) return null;
  const entryBlock = region.entryBlock as Block;
  const { bodyOps, yieldOp } = regionBodyOps(entryBlock);

  let reduceOp: Operation | null = null;
  for (const innerOp of bodyOps) {
    if (!isReductionOp(innerOp.opName)) continue;
    if (reduceOp || innerOp.opName !== 'reduce' || innerOp.numResults !== 1) return null;
    reduceOp = innerOp;
  }
  if (!reduceOp) return null;

  const reduceType = reduceOp.getAttr<string>('reduce_type') || 'sum';
  if (!getReduceCombiner(reduceType)) return null;

  const reduceInput = reduceOp.getOperand(0);
  const reduceResult = reduceOp.getResult(0);
  const fullShape = tensorShapeOf(reduceInput);
  if (!fullShape) return null;

  const { spatialDims, reduceDims } = splitReduceDims(fullShape.length, reduceOp.getAttr<readonly number[]>('dimensions') || []);
  if (reduceDims.length === 0) return null;
  const spatialShape = spatialDims.map((d) => fullShape[d]);
  const reduceResultShape = tensorShapeOf(reduceResult);
  if (!reduceResultShape || !shapesEqual(reduceResultShape, spatialShape)) return null;

  const regionOps = new Set(bodyOps);
  const cone = reduceDependencyCone(reduceOp, regionOps);
  const replicated: Operation[] = [];
  const prologue: Operation[] = [];
  const epilogue: Operation[] = [];
  for (const innerOp of bodyOps) {
    if (innerOp === reduceOp) continue;
    if (isReplicatedOp(innerOp.opName)) {
      if (isConstantOp(innerOp.opName) && typeof innerOp.getAttr('value') !== 'number') return null;
      replicated.push(innerOp);
      continue;
    }
    if (!INLINE_FUSION_BUILDERS.has(innerOp.opName) || innerOp.numResults !== 1) return null;
    const shape = tensorShapeOf(innerOp.getResult(0));
    if (!shape || !shapesEqual(shape, fullShape)) return null;
    (cone.has(innerOp) ? prologue : epilogue).push(innerOp);
  }

  const yieldSlot = new Map<Value, number>();
  if (yieldOp) {
    for (let i = 0; i < yieldOp.numOperands; i++) {
      const val = yieldOp.getOperand(i);
      if (yieldSlot.has(val)) return null;
      yieldSlot.set(val, i);
    }
  }

  for (const user of reduceResult.getUsers()) {
    if (user === yieldOp) continue;
    if (!isBroadcastOp(user.opName)) return null;
    const outShape = tensorShapeOf(user.getResult(0));
    if (!outShape || !shapesEqual(outShape, fullShape)) return null;
    if (!dimsEqual(broadcastDimsOf(user, spatialShape.length, fullShape.length), spatialDims)) return null;
  }

  const redSlot = yieldSlot.get(reduceResult);
  const redOwner: BufferOwner = redSlot !== undefined ? op.getResult(redSlot) : { type: reduceResult.type };

  const epilogueSet = new Set(epilogue);
  const materialize = new Map<Value, BufferOwner>();
  for (const innerOp of prologue) {
    const val = innerOp.getResult(0);
    const slot = yieldSlot.get(val);
    let needed = slot !== undefined;
    if (!needed) {
      for (const user of val.getUsers()) {
        if (epilogueSet.has(user)) { needed = true; break; }
      }
    }
    if (needed) materialize.set(val, slot !== undefined ? op.getResult(slot) : { type: val.type });
  }

  const epilogueStores: { owner: BufferOwner; value: Value }[] = [];
  for (const [val, slot] of yieldSlot) {
    if (val === reduceResult || materialize.has(val)) continue;
    const shape = tensorShapeOf(val);
    if (!shape || !shapesEqual(shape, fullShape)) return null;
    epilogueStores.push({ owner: op.getResult(slot), value: val });
  }

  const anchor = findShapeAnchor([outputs as Buffer[], inputs as Buffer[]], fullShape);
  if (!anchor) return null;

  const initExpr = scalarInitExpr(entryBlock, inputs, reduceOp.getOperand(1));
  if (!initExpr) return null;

  return {
    entryBlock, reduceOp, reduceType, reduceInput, reduceResult,
    replicated, prologue, epilogue,
    fullShape, spatialDims, reduceDims, anchor, initExpr,
    redOwner, materialize, epilogueStores,
  };
}

function scalarInitExpr(entryBlock: Block, inputs: readonly Buffer[], initVal: Value): TirNode | null {
  const argIdx = entryBlock.arguments.findIndex((arg) => arg === initVal);
  if (argIdx >= 0) {
    const buf = inputs[argIdx];
    if (buf.shape.length === 0) return new BufferLoadNode(buf, []);
    for (const d of buf.shape) if (d !== 1) return null;
    return new BufferLoadNode(buf, buf.shape.map(() => new IntImmNode(0)));
  }
  const def = initVal.definingOp;
  if (def && isConstantOp(def.opName)) {
    const val = def.getAttr<number>('value');
    if (typeof val === 'number') return new FloatImmNode(val);
  }
  return null;
}

function lowerReductionFusion(ctx: LoweringContext, plan: ReductionFusionPlan, inputs: readonly Buffer[], stmts: TirNode[]): void {
  const { entryBlock, reduceType, fullShape, spatialDims, reduceDims, anchor } = plan;
  const redBuf = ctx.getOrAllocBuffer(plan.redOwner);
  const materializeBufs = new Map<Value, Buffer>();
  for (const [val, owner] of plan.materialize) materializeBufs.set(val, ctx.getOrAllocBuffer(owner));

  const argDims = collectArgBroadcastDims(entryBlock);
  const useCount = countRegionUses(entryBlock);
  const combiner = getReduceCombiner(reduceType) as (a: TirNode, b: TirNode, dt: string) => TirNode;
  const geo = buildReduceGeometry(ctx, fullShape, anchor, spatialDims, reduceDims, FUSION_REDUCE_PREFIXES);

  const initBlock = new BlockNode(ctx.blockName('fused_reduce_init'), geo.init.ivs, loadedBuffers(plan.initExpr), [{ buffer: redBuf }],
    new BufferStoreNode(redBuf, geo.init.indices, plan.initExpr));
  stmts.push(spatialDims.length > 0 ? geo.init.wrap(initBlock) : initBlock);

  const accBuilder = new RegionExprBuilder(ctx, useCount, redBuf.dtype);
  accBuilder.bindBlockArgs(entryBlock, inputs, argDims, anchor, geo.fullIdx);
  accBuilder.evaluate(plan.replicated, geo.fullIdx);
  accBuilder.evaluate(plan.prologue, geo.fullIdx);

  const accStores: TirNode[] = [];
  const accWrites = [{ buffer: redBuf }];
  for (const [val, buf] of materializeBufs) {
    const bound = accBuilder.letBind(val, accBuilder.get(val), buf.dtype);
    accStores.push(new BufferStoreNode(buf, geo.fullIdx, bound));
    accWrites.push({ buffer: buf });
  }
  const accLoad = new BufferLoadNode(redBuf, geo.spatial.indices);
  accStores.push(new BufferStoreNode(redBuf, geo.spatial.indices,
    combiner(accLoad, accBuilder.get(plan.reduceInput), redBuf.dtype)));

  const accBody = accBuilder.wrap(accStores.length === 1 ? accStores[0] : new SeqNode(accStores));
  const accBlock = new BlockNode(ctx.blockName('fused_reduce_acc'), geo.accIvs, loadedBuffers(accBody), accWrites, accBody);
  stmts.push(geo.wrapAcc(accBlock));

  if (reduceType === 'mean') {
    stmts.push(emitReduceMeanDiv(ctx, redBuf, fullShape, anchor, spatialDims, reduceDims, 'fm', 'fused_mean_div'));
  }

  if (plan.epilogueStores.length === 0) return;

  const { loopVars, loopBinds, indices: fullIdx, extentNodes } = makeLoopNest(ctx, fullShape, anchor);
  const epiBuilder = new RegionExprBuilder(ctx, useCount, redBuf.dtype);
  epiBuilder.bind(plan.reduceResult, new BufferLoadNode(redBuf, spatialDims.map((d) => fullIdx[d])));
  for (const [val, buf] of materializeBufs) epiBuilder.bind(val, new BufferLoadNode(buf, fullIdx));
  epiBuilder.bindBlockArgs(entryBlock, inputs, argDims, anchor, fullIdx);
  epiBuilder.evaluate(plan.replicated, fullIdx);
  epiBuilder.evaluate(plan.epilogue, fullIdx);

  const epiStores: TirNode[] = [];
  const epiWrites: { buffer: Buffer }[] = [];
  for (const { owner, value } of plan.epilogueStores) {
    const buf = ctx.getOrAllocBuffer(owner);
    epiStores.push(new BufferStoreNode(buf, fullIdx, epiBuilder.get(value)));
    epiWrites.push({ buffer: buf });
  }
  const epiBody = epiBuilder.wrap(epiStores.length === 1 ? epiStores[0] : new SeqNode(epiStores));
  const epiBlock = new BlockNode(ctx.blockName('fused_reduce_epilogue'), loopBinds, loadedBuffers(epiBody), epiWrites, epiBody);
  stmts.push(wrapInLoops(epiBlock, loopVars, fullShape, extentNodes));
}

function lowerFusionAsIndividualOps(ctx: LoweringContext, fusionOp: Operation, stmts: TirNode[]): void {
  const entryBlock = fusionOp.regions[0].entryBlock as Block;
  const valueMap = new Map<Value, BufferOwner>();
  for (let i = 0; i < entryBlock.arguments.length; i++) {
    valueMap.set(entryBlock.arguments[i], fusionOp.getOperand(i));
  }

  const yieldedInner = new Map<Value, Value>();
  for (const op of entryBlock.ops()) {
    if (op.opName === 'yield') {
      for (let i = 0; i < op.numOperands; i++) {
        yieldedInner.set(op.getOperand(i), fusionOp.getResult(i));
      }
      break;
    }
  }

  for (const innerOp of entryBlock.ops()) {
    if (innerOp.opName === 'yield') continue;

    const outerOperands: BufferOwner[] = new Array(innerOp.numOperands);
    for (let i = 0; i < innerOp.numOperands; i++) {
      outerOperands[i] = valueMap.get(innerOp.getOperand(i)) || innerOp.getOperand(i);
    }

    const inputs: Buffer[] = new Array(outerOperands.length);
    for (let i = 0; i < outerOperands.length; i++) inputs[i] = ctx.getOrAllocBuffer(outerOperands[i]);
    const outputs: Buffer[] = new Array(innerOp.numResults);
    for (let i = 0; i < innerOp.numResults; i++) {
      const innerVal = innerOp.getResult(i);
      const outerResult = yieldedInner.get(innerVal);
      if (outerResult) {
        const outBuf = ctx.getOrAllocBuffer(outerResult);
        outputs[i] = outBuf;
        valueMap.set(innerVal, outerResult);
      } else {
        const proxy = { type: innerVal.type };
        outputs[i] = ctx.getOrAllocBuffer(proxy);
        valueMap.set(innerVal, proxy);
      }
    }

    if (isConstantOp(innerOp.opName)) {
      const stmt = lowerConstant(ctx, innerOp);
      if (stmt) stmts.push(stmt);
      continue;
    }

    const rule = getLoweringRule(innerOp.opName);
    if (!rule) {
      throw new Error(`Fusion lowering: no lowering rule for op '${innerOp.opName}' inside fusion body`);
    }
    const stmt = rule(ctx, innerOp, inputs, outputs);
    if (stmt) stmts.push(stmt);
  }
}

export function lowerFusionOp(ctx: LoweringContext, op: Operation, stmts: TirNode[]): FusionLoweringStrategy {
  if (canLowerAsElementwiseFusion(op)) {
    stmts.push(lowerFusion(ctx, op));
    return 'inline-loop';
  }
  const { inputs, outputs } = fusionBuffers(ctx, op);
  const plan = planReductionFusion(op, inputs, outputs);
  if (plan) {
    lowerReductionFusion(ctx, plan, inputs, stmts);
    return 'reduction-nest';
  }
  lowerFusionAsIndividualOps(ctx, op, stmts);
  return 'per-op';
}

export { lowerFusion, canLowerAsElementwiseFusion, lowerFusionAsIndividualOps };

export function register(): void {
  initBuiltinFusionBuilders();
}
