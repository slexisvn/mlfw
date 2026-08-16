import { DYNAMIC } from '../../ir/graph/types.js';
import { SymInt, symVarName } from '../../analysis/sym_int.js';
import { MemoryScope } from '../../ir/tensor/tensor_types.js';
import { Buffer } from '../../ir/tensor/buffer.js';
import { isDtypeInt } from '../../../util/dtype_map.js';
import { ForNode, BlockNode, SeqNode, BufferStoreNode, BufferLoadNode, VariableNode, IntImmNode, FloatImmNode, BlockRealizeNode, IterVarKind, ForKind, MathOpNode, CompareNode, IfThenElseNode, CastNode, mathOp } from '../../ir/tensor/nodes.js';
import { symIntToNode } from '../../ir/tensor/sym_lower.js';

import { isConstantOp } from '../../ir/graph/op_traits.js';
import { registerOpStrategy, getOpStrategy, selectImplementation } from './op_strategy.js';
import type { Dim, IRType, Shape, SymIntValue, TensorType } from '../../ir/graph/types.js';
import type { Value } from '../../ir/graph/value.js';
import type { Operation } from '../../ir/graph/operation.js';
import type { TirNode } from '../../ir/tensor/nodes.js';
import type { BufferRegionLike } from '../../ir/tensor/buffer.js';
import type { OpStrategyTarget } from './op_strategy.js';
import type { CompilerContext } from '../../pipeline/compiler_context.js';
import type { CompileTarget } from '../../pipeline/pipeline_types.js';

export type LoweringRuleFn = (ctx: LoweringContext, op: Operation, inputs: Buffer[], outputs: Buffer[]) => TirNode;
export type BufferOwner = { type: IRType; symbolicShape?: Shape };
export type LayoutIndexMap = Record<string, number>;
export type LoopNest = {
  loopVars: VariableNode[];
  loopBinds: BlockRealizeNode[];
  indices: VariableNode[];
  extentNodes: TirNode[] | null;
};
export type SpatialNest = {
  vars: VariableNode[];
  ivs: BlockRealizeNode[];
  indices: VariableNode[];
  extentNodes: TirNode[];
  wrap(body: TirNode): TirNode;
};
export type DotGeometry = {
  outIdx: VariableNode[];
  lhsIdx: TirNode[];
  rhsIdx: TirNode[];
  allIvs: BlockRealizeNode[];
  wrapAccBody(body: TirNode): TirNode;
};
export type MatmulInitAccOpts = {
  prefix: string;
  initBlockName: string;
  accBlockName: string;
  initVal(): TirNode;
  accLeaf(lhsLoad: TirNode, rhsLoad: TirNode): TirNode;
};
export type MatmulInitAccResult = { geo: DotGeometry; initBody: TirNode; accBody: TirNode };
export type ConvNestOpts = {
  prefix: string;
  blockPrefix: string;
  initVal(): TirNode;
  guardFill(): TirNode;
  leafBuilder(inIdx: TirNode[], kerIdx: TirNode[]): TirNode;
};
export type PointwiseExprBuilder = (op: Operation, loads: BufferLoadNode[], dtype: string) => TirNode;

const GENERIC_PLEVEL = 10;
const TARGET_PLEVEL = 20;

export { isConstantOp };

export { registerOpStrategy, unregisterOpStrategy, getOpStrategy, OpStrategy, OpImplementation, selectImplementation } from './op_strategy.js';

export function hasLoweringRule(opName: string, target?: OpStrategyTarget): boolean {
  if (getOpStrategy(opName, target)) return true;
  return isConstantOp(opName);
}

export function registerLoweringRule(opName: string, ruleFunc: LoweringRuleFn, plevel = GENERIC_PLEVEL): void {
  registerOpStrategy(opName, { name: `${opName}.generic`, compute: ruleFunc, plevel, targetKind: null });
}

export function registerTargetLoweringRule(opName: string, targetKind: string, ruleFunc: LoweringRuleFn, plevel = TARGET_PLEVEL): void {
  registerOpStrategy(opName, { name: `${opName}.${targetKind}`, compute: ruleFunc, plevel, targetKind });
}

export function getLoweringRule(opName: string, target?: OpStrategyTarget, context: CompilerContext | null = null): LoweringRuleFn | undefined {
  if (context) {
    const override = context.getLoweringRule(opName);
    if (override) return override;
  }
  const impl = selectImplementation(opName, target);
  return impl ? impl.compute as LoweringRuleFn : undefined;
}

export class LoweringContext {
  declare target?: CompileTarget | null;
  bufferMap: Map<BufferOwner, Buffer>;
  varCounter: number;
  shapeParams: Map<string, VariableNode>;
  symbolToVar: Map<SymIntValue, VariableNode>;
  symVars: Map<string, VariableNode>;
  private _blockCounter: number;

  constructor() {
    this.bufferMap = new Map();
    this.varCounter = 0;
    this.shapeParams = new Map();
    this.symbolToVar = new Map();
    this.symVars = new Map();
    this._blockCounter = 0;
  }

  blockName(hint: string): string {
    return `${hint}_${this._blockCounter++}`;
  }

  allocVar(nameHint: string, dtype = 'int32'): VariableNode {
    return new VariableNode(`${nameHint}_${this.varCounter++}`, dtype);
  }

  getOrAllocBuffer(value: BufferOwner): Buffer {
    let buf = this.bufferMap.get(value);
    if (buf) return buf;
    const t = value.type as TensorType;
    const shape = t.shape || [];
    const dtype = t.dtype || 'f32';
    const strides = t.layout ? t.layout.computeStrides(shape) : null;
    buf = new Buffer(`buf_${this.varCounter++}`, shape, dtype, MemoryScope.GLOBAL, strides);
    if (value.symbolicShape) buf.symbolicShape = value.symbolicShape;
    this.bufferMap.set(value, buf);
    this.declareBuffer(buf);
    return buf;
  }

  allocFreshBuffer(value: BufferOwner): Buffer {
    const t = value.type as TensorType;
    const shape = t.shape || [];
    const dtype = t.dtype || 'f32';
    const strides = t.layout ? t.layout.computeStrides(shape) : null;
    const buf = new Buffer(`buf_${this.varCounter++}`, shape, dtype, MemoryScope.GLOBAL, strides);
    if (value.symbolicShape) buf.symbolicShape = value.symbolicShape;
    this.declareBuffer(buf);
    return buf;
  }

  declareBuffer(buf: Buffer): void {
    for (let i = 0; i < buf.shape.length; i++) {
      const d = buf.shape[i];
      if (d === DYNAMIC) this.extentNode(DYNAMIC, buf, i);
      else if (d instanceof SymInt) this._registerSymIntDim(buf, i, d as SymIntValue);
    }
  }

  _symVarNode(name: string): VariableNode {
    let v = this.symVars.get(name);
    if (!v) {
      v = new VariableNode(symVarName(name), 'int32');
      this.symVars.set(name, v);
    }
    return v;
  }

  _registerSymIntDim(buf: Buffer, dimIdx: number, sym: SymIntValue): void {
    for (const name of SymInt.freeVars(sym)) this._symVarNode(name);
    if (sym.type === 'var') {
      const key = `${buf.name}:${dimIdx}`;
      if (!this.shapeParams.has(key)) this.shapeParams.set(key, this._symVarNode(sym.name as string));
    }
  }

  symIntToExtentNode(sym: SymIntValue): TirNode {
    return symIntToNode(sym, (name: string) => this._symVarNode(name));
  }

  _shapeParamVar(buf: Buffer, dimIdx: number): VariableNode {
    const key = dimIdx >= 0 ? `${buf.name}:${dimIdx}` : `${buf.name}:dyn`;
    let v = this.shapeParams.get(key);
    if (v) return v;
    const sym = buf.symbolicShape && dimIdx >= 0 && typeof buf.symbolicShape[dimIdx] !== 'number'
      ? buf.symbolicShape[dimIdx] as SymIntValue : null;
    if (sym !== null && this.symbolToVar.has(sym)) {
      v = this.symbolToVar.get(sym) as VariableNode;
    } else {
      v = this.allocVar(`_ds`);
      if (sym !== null) this.symbolToVar.set(sym, v);
    }
    this.shapeParams.set(key, v);
    return v;
  }

  extentNode(dim: Dim, buf: Buffer, dimIdx = -1): TirNode {
    if (dim instanceof SymInt) return this.symIntToExtentNode(dim as SymIntValue);
    if (dim !== DYNAMIC) return new IntImmNode(dim as number);
    return this._shapeParamVar(buf, dimIdx);
  }

  extentNodes(shape: Shape, buf: Buffer): TirNode[] {
    const nodes: TirNode[] = new Array(shape.length);
    for (let i = 0; i < shape.length; i++) {
      nodes[i] = this.extentNode(shape[i], buf, i);
    }
    return nodes;
  }

  allocVarArray(prefix: string, count: number): VariableNode[] {
    const arr: VariableNode[] = new Array(count);
    for (let i = 0; i < count; i++) arr[i] = this.allocVar(`${prefix}${i}`);
    return arr;
  }

  allocBindArray(prefix: string, loopVars: readonly VariableNode[]): BlockRealizeNode[] {
    const ivs: BlockRealizeNode[] = new Array(loopVars.length);
    for (let i = 0; i < loopVars.length; i++) {
      ivs[i] = new BlockRealizeNode(this.allocVar(`${prefix}${i}`), loopVars[i]);
    }
    return ivs;
  }
}

export function markCommReduce(ivs: BlockRealizeNode[]): BlockRealizeNode[] {
  for (const iv of ivs) iv.kind = IterVarKind.COMM_REDUCE;
  return ivs;
}

export function makeLoopNest(ctx: LoweringContext, shape: Shape, buf: Buffer | null): LoopNest {
  const n = shape.length;
  const loopVars = ctx.allocVarArray('i', n);
  const loopBinds = ctx.allocBindArray('v', loopVars);
  const indices: VariableNode[] = new Array(n);
  for (let i = 0; i < n; i++) indices[i] = loopBinds[i].iterVar;
  const extentNodes = buf ? ctx.extentNodes(shape, buf) : null;
  return { loopVars, loopBinds, indices, extentNodes };
}

export function wrapLoopsWithNodes(body: TirNode, loopVars: readonly VariableNode[], extentNodes: readonly TirNode[]): TirNode {
  let result: TirNode = body;
  for (let i = loopVars.length - 1; i >= 0; i--) {
    result = new ForNode(loopVars[i], new IntImmNode(0), extentNodes[i], ForKind.SERIAL, result);
  }
  return result;
}

export function wrapLoops(body: TirNode, loopVars: readonly VariableNode[], extents: Shape): TirNode {
  let result: TirNode = body;
  for (let i = loopVars.length - 1; i >= 0; i--) {
    result = new ForNode(loopVars[i], new IntImmNode(0), new IntImmNode(extents[i] as number), ForKind.SERIAL, result);
  }
  return result;
}

export function wrapInLoops(body: TirNode, loopVars: readonly VariableNode[], shape: Shape, extentNodes: readonly TirNode[] | null): TirNode {
  if (extentNodes) return wrapLoopsWithNodes(body, loopVars, extentNodes);
  return wrapLoops(body, loopVars, shape);
}

export function emitMatmulInitAcc(ctx: LoweringContext, op: Operation, lhs: Buffer, rhs: Buffer, out: Buffer, { prefix, initBlockName, accBlockName, initVal, accLeaf }: MatmulInitAccOpts): MatmulInitAccResult {
  const geo = buildDotGeometry(ctx, op, lhs, rhs);

  const initNest = buildSpatialNest(ctx, prefix, Array.from({ length: out.shape.length }, (_, i) => i), out.shape, out);
  const initStore = new BufferStoreNode(out, initNest.indices, initVal());
  const initBlock = new BlockNode(ctx.blockName(initBlockName), initNest.ivs, [], [{ buffer: out }], initStore);
  const initBody = initNest.wrap(initBlock);

  let lhsLoad: TirNode = new BufferLoadNode(lhs, geo.lhsIdx);
  let rhsLoad: TirNode = new BufferLoadNode(rhs, geo.rhsIdx);
  const promoteTo = (buf: Buffer, explicit: string | null | undefined): string | null => {
    if (explicit !== undefined && explicit !== null) return explicit;
    return buf.dtype === out.dtype ? null : out.dtype;
  };
  const lhsCast = promoteTo(lhs, op.getAttr<string>('lhs_prologue_cast'));
  const rhsCast = promoteTo(rhs, op.getAttr<string>('rhs_prologue_cast'));
  if (lhsCast) lhsLoad = new CastNode(lhsLoad, lhs.dtype, lhsCast);
  if (rhsCast) rhsLoad = new CastNode(rhsLoad, rhs.dtype, rhsCast);
  const product = accLeaf(lhsLoad, rhsLoad);
  const accExpr = new MathOpNode('+', new BufferLoadNode(out, geo.outIdx), product);
  const accStore = new BufferStoreNode(out, geo.outIdx, accExpr);
  const accBlock = new BlockNode(ctx.blockName(accBlockName), geo.allIvs, [{ buffer: lhs }, { buffer: rhs }], [{ buffer: out }], accStore);
  const accBody = geo.wrapAccBody(accBlock);

  return { geo, initBody, accBody };
}

export function buildConvNest(ctx: LoweringContext, op: Operation, inBuf: Buffer, kerBuf: Buffer, outBuf: Buffer, { prefix, blockPrefix, initVal, guardFill, leafBuilder }: ConvNestOpts): TirNode {
  const strides = op.getAttr<number[]>('strides') as number[];
  const padding = op.getAttr<number[][]>('padding') as number[][];
  const dilation = op.getAttr<number[]>('dilation') || strides.map(() => 1);
  const groups = op.getAttr<number>('groups') || 1;
  const iLayout = parseLayout(op.getAttr<string>('input_layout') as string);
  const kLayout = parseLayout(op.getAttr<string>('kernel_layout') as string);
  const spatialDims = strides.length;
  const batch = inBuf.shape[iLayout['N']];
  const outChannels = kerBuf.shape[kLayout['O']] as number;
  const inChannelsPerGroup = kerBuf.shape[kLayout['I']];
  const outShape = outBuf.shape;

  const initNest = buildSpatialNest(ctx, prefix + 'i', Array.from({ length: outShape.length }, (_, i) => i), outShape, outBuf);
  const initStore = new BufferStoreNode(outBuf, initNest.indices, initVal());
  const initBlock = new BlockNode(ctx.blockName(blockPrefix + '_init'), initNest.ivs, [], [{ buffer: outBuf }], initStore);
  const initBody = initNest.wrap(initBlock);

  const nVar = ctx.allocVar(prefix + 'n');
  const ocVar = ctx.allocVar(prefix + 'oc');
  const icVar = ctx.allocVar(prefix + 'ic');
  const spatialOutVars = ctx.allocVarArray(prefix + 'o', spatialDims);
  const spatialKerVars = ctx.allocVarArray(prefix + 'k', spatialDims);
  const allVars = [nVar, ocVar, ...spatialOutVars, icVar, ...spatialKerVars];
  const allBinds = ctx.allocBindArray(prefix + 'v', allVars);

  const bv = allBinds[0].iterVar;
  const ocv = allBinds[1].iterVar;
  const soBinds = allBinds.slice(2, 2 + spatialDims);
  const icv = allBinds[2 + spatialDims].iterVar;
  const skBinds = allBinds.slice(3 + spatialDims);
  markCommReduce([allBinds[2 + spatialDims], ...skBinds]);

  const outIdx: TirNode[] = new Array(outShape.length);
  outIdx[iLayout['N']] = bv;
  outIdx[iLayout['C']] = ocv;
  const spatialLayoutKeys = Object.keys(iLayout).filter(k => k !== 'N' && k !== 'C').sort();
  for (let s = 0; s < spatialDims; s++) {
    outIdx[iLayout[spatialLayoutKeys[s]]] = soBinds[s].iterVar;
  }

  const inIdx: TirNode[] = new Array(inBuf.shape.length);
  inIdx[iLayout['N']] = bv;
  const groupSize = Math.floor(outChannels / groups);
  if (groups > 1) {
    inIdx[iLayout['C']] = new MathOpNode('+', new MathOpNode('*', new MathOpNode('//', ocv, new IntImmNode(groupSize)), new IntImmNode(inChannelsPerGroup as number)), icv);
  } else {
    inIdx[iLayout['C']] = icv;
  }

  const kerIdx: TirNode[] = new Array(kerBuf.shape.length);
  kerIdx[kLayout['O']] = ocv;
  kerIdx[kLayout['I']] = icv;

  let inBoundsExpr: TirNode | null = null;
  for (let s = 0; s < spatialDims; s++) {
    const key = spatialLayoutKeys[s];
    const kKey = key.toUpperCase();
    const inSpatialIdx = mathOp('+',
      mathOp('*', soBinds[s].iterVar, new IntImmNode(strides[s])),
      mathOp('+',
        mathOp('*', skBinds[s].iterVar, new IntImmNode(dilation[s])),
        new IntImmNode(-padding[s][0])
      )
    );
    inIdx[iLayout[key]] = inSpatialIdx;
    kerIdx[kLayout[kKey]] = skBinds[s].iterVar;
    if (padding[s][0] !== 0 || padding[s][1] !== 0) {
      const ge = new CompareNode('ge', inSpatialIdx, new IntImmNode(0));
      const lt = new CompareNode('lt', inSpatialIdx, new IntImmNode(inBuf.shape[iLayout[key]] as number));
      const dimOk = new MathOpNode('*', ge, lt);
      inBoundsExpr = inBoundsExpr ? new MathOpNode('*', inBoundsExpr, dimOk) : dimOk;
    }
  }

  const product = leafBuilder(inIdx, kerIdx);
  const guardedProduct = inBoundsExpr ? new IfThenElseNode(inBoundsExpr, product, guardFill()) : product;
  const loadOut = new BufferLoadNode(outBuf, outIdx);
  const accExpr = new MathOpNode('+', loadOut, guardedProduct);
  const accStore = new BufferStoreNode(outBuf, outIdx, accExpr);
  const accBlock = new BlockNode(ctx.blockName(blockPrefix + '_acc'), allBinds, [{ buffer: inBuf }, { buffer: kerBuf }], [{ buffer: outBuf }], accStore);

  const kerSpatialSizes: Dim[] = new Array(spatialDims);
  for (let s = 0; s < spatialDims; s++) {
    const kKey = spatialLayoutKeys[s].toUpperCase();
    kerSpatialSizes[s] = kerBuf.shape[kLayout[kKey]];
  }

  let accBody: TirNode = accBlock;
  for (let s = spatialDims - 1; s >= 0; s--) {
    const kKey = spatialLayoutKeys[s].toUpperCase();
    accBody = new ForNode(spatialKerVars[s], new IntImmNode(0), ctx.extentNode(kerSpatialSizes[s], kerBuf, kLayout[kKey]), ForKind.SERIAL, accBody);
  }
  accBody = new ForNode(icVar, new IntImmNode(0), ctx.extentNode(inChannelsPerGroup, kerBuf, kLayout['I']), ForKind.SERIAL, accBody);
  for (let s = spatialDims - 1; s >= 0; s--) {
    const dimIdx = iLayout[spatialLayoutKeys[s]];
    accBody = new ForNode(spatialOutVars[s], new IntImmNode(0), ctx.extentNode(outShape[dimIdx], outBuf, dimIdx), ForKind.SERIAL, accBody);
  }
  accBody = new ForNode(ocVar, new IntImmNode(0), ctx.extentNode(outChannels, kerBuf, kLayout['O']), ForKind.SERIAL, accBody);
  accBody = new ForNode(nVar, new IntImmNode(0), ctx.extentNode(batch, inBuf, iLayout['N']), ForKind.SERIAL, accBody);

  return new SeqNode([initBody, accBody]);
}

export function buildSpatialNest(ctx: LoweringContext, prefix: string, dims: readonly number[], shape: Shape, buf: Buffer): SpatialNest {
  const n = dims.length;
  const vars: VariableNode[] = new Array(n);
  const ivs: BlockRealizeNode[] = new Array(n);
  const indices: VariableNode[] = new Array(n);
  const extentNodes: TirNode[] = new Array(n);
  for (let i = 0; i < n; i++) {
    vars[i] = ctx.allocVar(`${prefix}${dims[i]}`);
    ivs[i] = new BlockRealizeNode(ctx.allocVar(`${prefix}v${dims[i]}`), vars[i]);
    indices[i] = ivs[i].iterVar;
    extentNodes[i] = ctx.extentNode(shape[dims[i]], buf, dims[i]);
  }
  return {
    vars, ivs, indices, extentNodes,
    wrap(body: TirNode): TirNode { return wrapLoopsWithNodes(body, vars, extentNodes); }
  };
}

export function computeBroadcastIndices(inBuf: Buffer, outBuf: Buffer, outIndices: readonly TirNode[]): TirNode[] {
  const inRank = inBuf.shape.length;
  if (inBuf.broadcastDims) {
    const dims = inBuf.broadcastDims;
    const indices: TirNode[] = new Array(inRank);
    for (let j = 0; j < inRank; j++) {
      indices[j] = inBuf.shape[j] === 1 ? new IntImmNode(0) : outIndices[dims[j]];
    }
    return indices;
  }
  const outRank = outBuf.shape.length;
  const offset = outRank - inRank;
  const indices: TirNode[] = new Array(inRank);
  for (let j = 0; j < inRank; j++) {
    indices[j] = inBuf.shape[j] === 1 ? new IntImmNode(0) : outIndices[offset + j];
  }
  return indices;
}

export function bufRefs(bufs: readonly Buffer[]): BufferRegionLike[] {
  const refs: BufferRegionLike[] = new Array(bufs.length);
  for (let i = 0; i < bufs.length; i++) refs[i] = { buffer: bufs[i] };
  return refs;
}

export function concatIterVars<T>(...args: readonly (readonly T[])[]): T[] {
  let total = 0;
  for (let i = 0; i < arguments.length; i++) total += arguments[i].length;
  const result: T[] = new Array(total);
  let idx = 0;
  for (let i = 0; i < arguments.length; i++) {
    const arr = arguments[i];
    for (let j = 0; j < arr.length; j++) result[idx++] = arr[j];
  }
  return result;
}

export function extractIterVars(ivs: readonly BlockRealizeNode[]): VariableNode[] {
  const result: VariableNode[] = new Array(ivs.length);
  for (let i = 0; i < ivs.length; i++) result[i] = ivs[i].iterVar;
  return result;
}

export function lowerPointwise(ctx: LoweringContext, op: Operation, inputs: readonly Buffer[], outputs: readonly Buffer[], exprBuilder: PointwiseExprBuilder): TirNode {
  const outBuf = outputs[0];
  const { loopVars, loopBinds, indices, extentNodes } = makeLoopNest(ctx, outBuf.shape, outBuf);
  const loads: BufferLoadNode[] = new Array(inputs.length);
  for (let i = 0; i < inputs.length; i++) {
    const inIndices = computeBroadcastIndices(inputs[i], outBuf, indices);
    loads[i] = new BufferLoadNode(inputs[i], inIndices);
  }
  const expr = exprBuilder(op, loads, outBuf.dtype);
  const store = new BufferStoreNode(outBuf, indices, expr);
  const block = new BlockNode(ctx.blockName(`${op.opName}_block`), loopBinds, bufRefs(inputs), [{ buffer: outBuf }], store);
  return wrapInLoops(block, loopVars, outBuf.shape, extentNodes);
}

export function lowerConstant(ctx: LoweringContext, op: Operation): TirNode {
  const result = op.getResult(0);
  const rtype = result.type as TensorType;
  const val = op.getAttr<number | readonly number[]>('value');
  const fullShape = (rtype && rtype.shape) || [];
  if (typeof val === 'number' && fullShape.length > 0 && !ctx.bufferMap.has(result)) {
    const dtype = (rtype && rtype.dtype) || 'f32';
    const scalarShape = new Array(fullShape.length).fill(1);
    const buf = new Buffer(`buf_${ctx.varCounter++}`, scalarShape, dtype, MemoryScope.GLOBAL);
    buf.broadcastDims = Array.from({ length: fullShape.length }, (_, i) => i);
    ctx.bufferMap.set(result, buf);
    const node = isDtypeInt(dtype) ? new IntImmNode(val as number) : new FloatImmNode(val as number);
    return new BufferStoreNode(buf, scalarShape.map(() => new IntImmNode(0)), node);
  }
  const outBuf = ctx.getOrAllocBuffer(result);
  const isInt = isDtypeInt(outBuf.dtype);
  const imm = (x: number): TirNode => isInt ? new IntImmNode(x) : new FloatImmNode(x);

  const arrVal = val as ArrayLike<number>;
  if (val && typeof val !== 'number' && typeof arrVal.length === 'number') {
    if (outBuf.shape.length === 0) {
      return new BufferStoreNode(outBuf, [], imm(arrVal[0]));
    }
    const strides: number[] = new Array(outBuf.shape.length);
    let acc = 1;
    for (let d = outBuf.shape.length - 1; d >= 0; d--) { strides[d] = acc; acc *= outBuf.shape[d] as number; }
    const stmts: TirNode[] = [];
    for (let i = 0; i < arrVal.length; i++) {
      const idx: TirNode[] = new Array(outBuf.shape.length);
      for (let d = 0; d < outBuf.shape.length; d++) idx[d] = new IntImmNode(Math.floor(i / strides[d]) % (outBuf.shape[d] as number));
      stmts.push(new BufferStoreNode(outBuf, idx, imm(arrVal[i])));
    }
    return new BlockNode(ctx.blockName(`${op.opName}_block`), [], [], [{ buffer: outBuf }], new SeqNode(stmts));
  }

  const valNode = typeof val === 'number' ? imm(val) : imm(0);
  if (outBuf.shape.length === 0) {
    return new BufferStoreNode(outBuf, [], valNode);
  }
  const { loopVars, loopBinds, indices, extentNodes } = makeLoopNest(ctx, outBuf.shape, outBuf);
  const store = new BufferStoreNode(outBuf, indices, valNode);
  const block = new BlockNode(ctx.blockName(`${op.opName}_block`), loopBinds, [], [{ buffer: outBuf }], store);
  return wrapInLoops(block, loopVars, outBuf.shape, extentNodes);
}

export function parseLayout(str: string): LayoutIndexMap {
  const m: LayoutIndexMap = {};
  for (let i = 0; i < str.length; i++) m[str[i]] = i;
  return m;
}

function physicalDotIndices(buf: Buffer, logicalIdx: readonly TirNode[]): TirNode[] {
  if (!buf.broadcastDims) return logicalIdx as TirNode[];
  const dims = buf.broadcastDims;
  const idx: TirNode[] = new Array(buf.shape.length);
  for (let j = 0; j < buf.shape.length; j++) {
    idx[j] = buf.shape[j] === 1 ? new IntImmNode(0) : logicalIdx[dims[j]];
  }
  return idx;
}

export function buildDotGeometry(ctx: LoweringContext, op: Operation, lhs: Buffer, rhs: Buffer): DotGeometry {
  const lhsContracting = op.getAttr<number[]>('lhs_contracting') || [];
  const rhsContracting = op.getAttr<number[]>('rhs_contracting') || [];
  const lhsBatch = op.getAttr<number[]>('lhs_batch') || [];
  const rhsBatch = op.getAttr<number[]>('rhs_batch') || [];
  const lhsShape = (op.getOperand(0).type as TensorType).shape;
  const rhsShape = (op.getOperand(1).type as TensorType).shape;
  const lhsCSet = new Set(lhsContracting);
  const lhsBSet = new Set(lhsBatch);
  const rhsCSet = new Set(rhsContracting);
  const rhsBSet = new Set(rhsBatch);

  const lhsSpatial: number[] = [];
  for (let i = 0; i < lhsShape.length; i++) {
    if (!lhsCSet.has(i) && !lhsBSet.has(i)) lhsSpatial.push(i);
  }
  const rhsSpatial: number[] = [];
  for (let i = 0; i < rhsShape.length; i++) {
    if (!rhsCSet.has(i) && !rhsBSet.has(i)) rhsSpatial.push(i);
  }

  const bVars = ctx.allocVarArray('b', lhsBatch.length);
  const lsVars = ctx.allocVarArray('ls', lhsSpatial.length);
  const rsVars = ctx.allocVarArray('rs', rhsSpatial.length);
  const cVars = ctx.allocVarArray('c', lhsContracting.length);
  const bIvs = ctx.allocBindArray('vb', bVars);
  const lsIvs = ctx.allocBindArray('vls', lsVars);
  const rsIvs = ctx.allocBindArray('vrs', rsVars);
  const cIvs = markCommReduce(ctx.allocBindArray('vc', cVars));

  const outIdx = concatIterVars(
    extractIterVars(bIvs), extractIterVars(lsIvs), extractIterVars(rsIvs)
  );

  const lhsLogical: TirNode[] = new Array(lhsShape.length);
  for (let i = 0; i < lhsBatch.length; i++) lhsLogical[lhsBatch[i]] = bIvs[i].iterVar;
  for (let i = 0; i < lhsSpatial.length; i++) lhsLogical[lhsSpatial[i]] = lsIvs[i].iterVar;
  for (let i = 0; i < lhsContracting.length; i++) lhsLogical[lhsContracting[i]] = cIvs[i].iterVar;

  const rhsLogical: TirNode[] = new Array(rhsShape.length);
  for (let i = 0; i < rhsBatch.length; i++) rhsLogical[rhsBatch[i]] = bIvs[i].iterVar;
  for (let i = 0; i < rhsSpatial.length; i++) rhsLogical[rhsSpatial[i]] = rsIvs[i].iterVar;
  for (let i = 0; i < rhsContracting.length; i++) rhsLogical[rhsContracting[i]] = cIvs[i].iterVar;

  const lhsIdx = physicalDotIndices(lhs, lhsLogical);
  const rhsIdx = physicalDotIndices(rhs, rhsLogical);

  const allIvs = concatIterVars(bIvs, lsIvs, rsIvs, cIvs);

  const loopGroups = [
    { vars: bVars, dims: lhsBatch, shape: lhsShape, buf: lhs },
    { vars: lsVars, dims: lhsSpatial, shape: lhsShape, buf: lhs },
    { vars: rsVars, dims: rhsSpatial, shape: rhsShape, buf: rhs },
    { vars: cVars, dims: lhsContracting, shape: lhsShape, buf: lhs },
  ];

  function wrapAccBody(body: TirNode): TirNode {
    let result: TirNode = body;
    for (let g = loopGroups.length - 1; g >= 0; g--) {
      const { vars, dims, shape, buf } = loopGroups[g];
      for (let i = vars.length - 1; i >= 0; i--) {
        const extent = buf.broadcastDims ? new IntImmNode(shape[dims[i]] as number) : ctx.extentNode(shape[dims[i]], buf, dims[i]);
        result = new ForNode(vars[i], new IntImmNode(0), extent, ForKind.SERIAL, result);
      }
    }
    return result;
  }

  return { outIdx, lhsIdx, rhsIdx, allIvs, wrapAccBody };
}
