import { DYNAMIC } from '../../ir/graph/types.js';
import { SymInt, symVarName } from '../../analysis/sym_int.js';
import { MemoryScope } from '../../ir/tensor/tensor_types.js';
import { Buffer } from '../../ir/tensor/buffer.js';
import { isDtypeInt } from '../../../util/dtype_map.js';
import { ForNode, BlockNode, SeqNode, BufferStoreNode, BufferLoadNode, VariableNode, IntImmNode, FloatImmNode, BlockRealizeNode, ForKind, MathOpNode, CompareNode, IfThenElseNode, mathOp } from '../../ir/tensor/nodes.js';
import { symIntToNode } from '../../ir/tensor/sym_lower.js';

import { isConstantOp } from '../../ir/graph/op_traits.js';
import { registerOpStrategy, getOpStrategy, selectImplementation } from './op_strategy.js';

const GENERIC_PLEVEL = 10;
const TARGET_PLEVEL = 20;

export { isConstantOp };

export { registerOpStrategy, getOpStrategy, OpStrategy, OpImplementation, selectImplementation } from './op_strategy.js';

export function hasLoweringRule(opName, target) {
  if (getOpStrategy(opName, target)) return true;
  return isConstantOp(opName);
}

export function registerLoweringRule(opName, ruleFunc, plevel = GENERIC_PLEVEL) {
  registerOpStrategy(opName, { name: `${opName}.generic`, compute: ruleFunc, plevel, targetKind: null });
}

export function registerTargetLoweringRule(opName, targetKind, ruleFunc, plevel = TARGET_PLEVEL) {
  registerOpStrategy(opName, { name: `${opName}.${targetKind}`, compute: ruleFunc, plevel, targetKind });
}

export function getLoweringRule(opName, target, context = null) {
  if (context) {
    const override = context.getLoweringRule(opName);
    if (override) return override;
  }
  const impl = selectImplementation(opName, target);
  return impl ? impl.compute : undefined;
}

export class LoweringContext {
  constructor() {
    this.bufferMap = new Map();
    this.varCounter = 0;
    this.shapeParams = new Map();
    this.symbolToVar = new Map();
    this.symVars = new Map();
    this._blockCounter = 0;
  }

  blockName(hint) {
    return `${hint}_${this._blockCounter++}`;
  }

  allocVar(nameHint, dtype = 'int32') {
    return new VariableNode(`${nameHint}_${this.varCounter++}`, dtype);
  }

  getOrAllocBuffer(value) {
    let buf = this.bufferMap.get(value);
    if (buf) return buf;
    const t = value.type;
    const shape = t.shape || [];
    const dtype = t.dtype || 'f32';
    const strides = t.layout ? t.layout.computeStrides(shape) : null;
    buf = new Buffer(`buf_${this.varCounter++}`, shape, dtype, MemoryScope.GLOBAL, strides);
    if (value.symbolicShape) buf.symbolicShape = value.symbolicShape;
    this.bufferMap.set(value, buf);
    this._registerDynamicDims(buf);
    return buf;
  }

  allocFreshBuffer(value) {
    const t = value.type;
    const shape = t.shape || [];
    const dtype = t.dtype || 'f32';
    const strides = t.layout ? t.layout.computeStrides(shape) : null;
    const buf = new Buffer(`buf_${this.varCounter++}`, shape, dtype, MemoryScope.GLOBAL, strides);
    if (value.symbolicShape) buf.symbolicShape = value.symbolicShape;
    this._registerDynamicDims(buf);
    return buf;
  }

  _registerDynamicDims(buf) {
    for (let i = 0; i < buf.shape.length; i++) {
      const d = buf.shape[i];
      if (d === DYNAMIC) this.extentNode(DYNAMIC, buf, i);
      else if (d instanceof SymInt) this._registerSymIntDim(buf, i, d);
    }
  }

  _symVarNode(name) {
    let v = this.symVars.get(name);
    if (!v) {
      v = new VariableNode(symVarName(name), 'int32');
      this.symVars.set(name, v);
    }
    return v;
  }

  _registerSymIntDim(buf, dimIdx, sym) {
    for (const name of SymInt.freeVars(sym)) this._symVarNode(name);
    if (sym.type === 'var') {
      const key = `${buf.name}:${dimIdx}`;
      if (!this.shapeParams.has(key)) this.shapeParams.set(key, this._symVarNode(sym.name));
    }
  }

  symIntToExtentNode(sym) {
    return symIntToNode(sym, (name) => this._symVarNode(name));
  }

  _shapeParamVar(buf, dimIdx) {
    const key = dimIdx >= 0 ? `${buf.name}:${dimIdx}` : `${buf.name}:dyn`;
    let v = this.shapeParams.get(key);
    if (v) return v;
    const sym = buf.symbolicShape && dimIdx >= 0 && typeof buf.symbolicShape[dimIdx] !== 'number'
      ? buf.symbolicShape[dimIdx] : null;
    if (sym !== null && this.symbolToVar.has(sym)) {
      v = this.symbolToVar.get(sym);
    } else {
      v = this.allocVar(`_ds`);
      if (sym !== null) this.symbolToVar.set(sym, v);
    }
    this.shapeParams.set(key, v);
    return v;
  }

  extentNode(dim, buf, dimIdx = -1) {
    if (dim instanceof SymInt) return this.symIntToExtentNode(dim);
    if (dim !== DYNAMIC) return new IntImmNode(dim);
    return this._shapeParamVar(buf, dimIdx);
  }

  extentNodes(shape, buf) {
    const nodes = new Array(shape.length);
    for (let i = 0; i < shape.length; i++) {
      nodes[i] = this.extentNode(shape[i], buf, i);
    }
    return nodes;
  }

  allocVarArray(prefix, count) {
    const arr = new Array(count);
    for (let i = 0; i < count; i++) arr[i] = this.allocVar(`${prefix}${i}`);
    return arr;
  }

  allocBindArray(prefix, loopVars) {
    const ivs = new Array(loopVars.length);
    for (let i = 0; i < loopVars.length; i++) {
      ivs[i] = new BlockRealizeNode(this.allocVar(`${prefix}${i}`), loopVars[i]);
    }
    return ivs;
  }
}

export function makeLoopNest(ctx, shape, buf) {
  const n = shape.length;
  const loopVars = ctx.allocVarArray('i', n);
  const loopBinds = ctx.allocBindArray('v', loopVars);
  const indices = new Array(n);
  for (let i = 0; i < n; i++) indices[i] = loopBinds[i].iterVar;
  const extentNodes = buf ? ctx.extentNodes(shape, buf) : null;
  return { loopVars, loopBinds, indices, extentNodes };
}

export function wrapLoopsWithNodes(body, loopVars, extentNodes) {
  let result = body;
  for (let i = loopVars.length - 1; i >= 0; i--) {
    result = new ForNode(loopVars[i], new IntImmNode(0), extentNodes[i], ForKind.SERIAL, result);
  }
  return result;
}

export function wrapLoops(body, loopVars, extents) {
  let result = body;
  for (let i = loopVars.length - 1; i >= 0; i--) {
    result = new ForNode(loopVars[i], new IntImmNode(0), new IntImmNode(extents[i]), ForKind.SERIAL, result);
  }
  return result;
}

export function wrapInLoops(body, loopVars, shape, extentNodes) {
  if (extentNodes) return wrapLoopsWithNodes(body, loopVars, extentNodes);
  return wrapLoops(body, loopVars, shape);
}

export function emitMatmulInitAcc(ctx, op, lhs, rhs, out, { prefix, initBlockName, accBlockName, initVal, accLeaf }) {
  const geo = buildDotGeometry(ctx, op, lhs, rhs);

  const initNest = buildSpatialNest(ctx, prefix, Array.from({ length: out.shape.length }, (_, i) => i), out.shape, out);
  const initStore = new BufferStoreNode(out, initNest.indices, initVal());
  const initBlock = new BlockNode(ctx.blockName(initBlockName), initNest.ivs, [], [{ buffer: out }], initStore);
  const initBody = initNest.wrap(initBlock);

  const product = accLeaf(new BufferLoadNode(lhs, geo.lhsIdx), new BufferLoadNode(rhs, geo.rhsIdx));
  const accExpr = new MathOpNode('+', new BufferLoadNode(out, geo.outIdx), product);
  const accStore = new BufferStoreNode(out, geo.outIdx, accExpr);
  const accBlock = new BlockNode(ctx.blockName(accBlockName), geo.allIvs, [{ buffer: lhs }, { buffer: rhs }], [{ buffer: out }], accStore);
  const accBody = geo.wrapAccBody(accBlock);

  return { geo, initBody, accBody };
}

export function buildConvNest(ctx, op, inBuf, kerBuf, outBuf, { prefix, blockPrefix, initVal, guardFill, leafBuilder }) {
  const strides = op.getAttr('strides');
  const padding = op.getAttr('padding');
  const dilation = op.getAttr('dilation') || strides.map(() => 1);
  const groups = op.getAttr('groups') || 1;
  const iLayout = parseLayout(op.getAttr('input_layout'));
  const kLayout = parseLayout(op.getAttr('kernel_layout'));
  const spatialDims = strides.length;
  const batch = inBuf.shape[iLayout['N']];
  const outChannels = kerBuf.shape[kLayout['O']];
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

  const outIdx = new Array(outShape.length);
  outIdx[iLayout['N']] = bv;
  outIdx[iLayout['C']] = ocv;
  const spatialLayoutKeys = Object.keys(iLayout).filter(k => k !== 'N' && k !== 'C').sort();
  for (let s = 0; s < spatialDims; s++) {
    outIdx[iLayout[spatialLayoutKeys[s]]] = soBinds[s].iterVar;
  }

  const inIdx = new Array(inBuf.shape.length);
  inIdx[iLayout['N']] = bv;
  const groupSize = Math.floor(outChannels / groups);
  if (groups > 1) {
    inIdx[iLayout['C']] = new MathOpNode('+', new MathOpNode('*', new MathOpNode('//', ocv, new IntImmNode(groupSize)), new IntImmNode(inChannelsPerGroup)), icv);
  } else {
    inIdx[iLayout['C']] = icv;
  }

  const kerIdx = new Array(kerBuf.shape.length);
  kerIdx[kLayout['O']] = ocv;
  kerIdx[kLayout['I']] = icv;

  let inBoundsExpr = null;
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
      const lt = new CompareNode('lt', inSpatialIdx, new IntImmNode(inBuf.shape[iLayout[key]]));
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

  const kerSpatialSizes = new Array(spatialDims);
  for (let s = 0; s < spatialDims; s++) {
    const kKey = spatialLayoutKeys[s].toUpperCase();
    kerSpatialSizes[s] = kerBuf.shape[kLayout[kKey]];
  }

  let accBody = accBlock;
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

export function buildSpatialNest(ctx, prefix, dims, shape, buf) {
  const n = dims.length;
  const vars = new Array(n);
  const ivs = new Array(n);
  const indices = new Array(n);
  const extentNodes = new Array(n);
  for (let i = 0; i < n; i++) {
    vars[i] = ctx.allocVar(`${prefix}${dims[i]}`);
    ivs[i] = new BlockRealizeNode(ctx.allocVar(`${prefix}v${dims[i]}`), vars[i]);
    indices[i] = ivs[i].iterVar;
    extentNodes[i] = ctx.extentNode(shape[dims[i]], buf, dims[i]);
  }
  return {
    vars, ivs, indices, extentNodes,
    wrap(body) { return wrapLoopsWithNodes(body, vars, extentNodes); }
  };
}

export function computeBroadcastIndices(inBuf, outBuf, outIndices) {
  const inRank = inBuf.shape.length;
  if (inBuf.broadcastDims) {
    const dims = inBuf.broadcastDims;
    const indices = new Array(inRank);
    for (let j = 0; j < inRank; j++) {
      indices[j] = inBuf.shape[j] === 1 ? new IntImmNode(0) : outIndices[dims[j]];
    }
    return indices;
  }
  const outRank = outBuf.shape.length;
  const offset = outRank - inRank;
  const indices = new Array(inRank);
  for (let j = 0; j < inRank; j++) {
    indices[j] = inBuf.shape[j] === 1 ? new IntImmNode(0) : outIndices[offset + j];
  }
  return indices;
}

export function bufRefs(bufs) {
  const refs = new Array(bufs.length);
  for (let i = 0; i < bufs.length; i++) refs[i] = { buffer: bufs[i] };
  return refs;
}

export function concatIterVars() {
  let total = 0;
  for (let i = 0; i < arguments.length; i++) total += arguments[i].length;
  const result = new Array(total);
  let idx = 0;
  for (let i = 0; i < arguments.length; i++) {
    const arr = arguments[i];
    for (let j = 0; j < arr.length; j++) result[idx++] = arr[j];
  }
  return result;
}

export function extractIterVars(ivs) {
  const result = new Array(ivs.length);
  for (let i = 0; i < ivs.length; i++) result[i] = ivs[i].iterVar;
  return result;
}

export function lowerPointwise(ctx, op, inputs, outputs, exprBuilder) {
  const outBuf = outputs[0];
  const { loopVars, loopBinds, indices, extentNodes } = makeLoopNest(ctx, outBuf.shape, outBuf);
  const loads = new Array(inputs.length);
  for (let i = 0; i < inputs.length; i++) {
    const inIndices = computeBroadcastIndices(inputs[i], outBuf, indices);
    loads[i] = new BufferLoadNode(inputs[i], inIndices);
  }
  const expr = exprBuilder(op, loads, outBuf.dtype);
  const store = new BufferStoreNode(outBuf, indices, expr);
  const block = new BlockNode(ctx.blockName(`${op.opName}_block`), loopBinds, bufRefs(inputs), [{ buffer: outBuf }], store);
  return wrapInLoops(block, loopVars, outBuf.shape, extentNodes);
}

export function lowerConstant(ctx, op) {
  const result = op.getResult(0);
  const val = op.getAttr('value');
  const fullShape = (result.type && result.type.shape) || [];
  if (typeof val === 'number' && fullShape.length > 0 && !ctx.bufferMap.has(result)) {
    const dtype = (result.type && result.type.dtype) || 'f32';
    const scalarShape = new Array(fullShape.length).fill(1);
    const buf = new Buffer(`buf_${ctx.varCounter++}`, scalarShape, dtype, MemoryScope.GLOBAL);
    buf.broadcastDims = Array.from({ length: fullShape.length }, (_, i) => i);
    ctx.bufferMap.set(result, buf);
    const node = isDtypeInt(dtype) ? new IntImmNode(val) : new FloatImmNode(val);
    return new BufferStoreNode(buf, scalarShape.map(() => new IntImmNode(0)), node);
  }
  const outBuf = ctx.getOrAllocBuffer(result);
  const isInt = isDtypeInt(outBuf.dtype);
  const imm = (x) => isInt ? new IntImmNode(x) : new FloatImmNode(x);

  if (val && typeof val !== 'number' && typeof val.length === 'number') {
    if (outBuf.shape.length === 0) {
      return new BufferStoreNode(outBuf, [], imm(val[0]));
    }
    const strides = new Array(outBuf.shape.length);
    let acc = 1;
    for (let d = outBuf.shape.length - 1; d >= 0; d--) { strides[d] = acc; acc *= outBuf.shape[d]; }
    const stmts = [];
    for (let i = 0; i < val.length; i++) {
      const idx = new Array(outBuf.shape.length);
      for (let d = 0; d < outBuf.shape.length; d++) idx[d] = new IntImmNode(Math.floor(i / strides[d]) % outBuf.shape[d]);
      stmts.push(new BufferStoreNode(outBuf, idx, imm(val[i])));
    }
    return new SeqNode(stmts);
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

export function parseLayout(str) {
  const m = {};
  for (let i = 0; i < str.length; i++) m[str[i]] = i;
  return m;
}

function physicalDotIndices(buf, logicalIdx) {
  if (!buf.broadcastDims) return logicalIdx;
  const dims = buf.broadcastDims;
  const idx = new Array(buf.shape.length);
  for (let j = 0; j < buf.shape.length; j++) {
    idx[j] = buf.shape[j] === 1 ? new IntImmNode(0) : logicalIdx[dims[j]];
  }
  return idx;
}

export function buildDotGeometry(ctx, op, lhs, rhs) {
  const lhsContracting = op.getAttr('lhs_contracting') || [];
  const rhsContracting = op.getAttr('rhs_contracting') || [];
  const lhsBatch = op.getAttr('lhs_batch') || [];
  const rhsBatch = op.getAttr('rhs_batch') || [];
  const lhsShape = op.getOperand(0).type.shape;
  const rhsShape = op.getOperand(1).type.shape;
  const lhsCSet = new Set(lhsContracting);
  const lhsBSet = new Set(lhsBatch);
  const rhsCSet = new Set(rhsContracting);
  const rhsBSet = new Set(rhsBatch);

  const lhsSpatial = [];
  for (let i = 0; i < lhsShape.length; i++) {
    if (!lhsCSet.has(i) && !lhsBSet.has(i)) lhsSpatial.push(i);
  }
  const rhsSpatial = [];
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
  const cIvs = ctx.allocBindArray('vc', cVars);

  const outIdx = concatIterVars(
    extractIterVars(bIvs), extractIterVars(lsIvs), extractIterVars(rsIvs)
  );

  const lhsLogical = new Array(lhsShape.length);
  for (let i = 0; i < lhsBatch.length; i++) lhsLogical[lhsBatch[i]] = bIvs[i].iterVar;
  for (let i = 0; i < lhsSpatial.length; i++) lhsLogical[lhsSpatial[i]] = lsIvs[i].iterVar;
  for (let i = 0; i < lhsContracting.length; i++) lhsLogical[lhsContracting[i]] = cIvs[i].iterVar;

  const rhsLogical = new Array(rhsShape.length);
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

  function wrapAccBody(body) {
    let result = body;
    for (let g = loopGroups.length - 1; g >= 0; g--) {
      const { vars, dims, shape, buf } = loopGroups[g];
      for (let i = vars.length - 1; i >= 0; i--) {
        const extent = buf.broadcastDims ? new IntImmNode(shape[dims[i]]) : ctx.extentNode(shape[dims[i]], buf, dims[i]);
        result = new ForNode(vars[i], new IntImmNode(0), extent, ForKind.SERIAL, result);
      }
    }
    return result;
  }

  return { outIdx, lhsIdx, rhsIdx, allIvs, wrapAccBody };
}
