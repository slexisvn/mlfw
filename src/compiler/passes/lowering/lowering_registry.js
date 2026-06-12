import { TensorType, DYNAMIC } from '../../ir/graph/types.js';
import { MemoryScope } from '../../ir/tensor/tensor_types.js';
import { Buffer } from '../../ir/tensor/buffer.js';
import { isDtypeInt } from '../../../backend/dtype_map.js';
import {
  ForNode, BlockNode, SeqNode, BufferStoreNode, BufferLoadNode,
  VariableNode, IntImmNode, FloatImmNode, MathOpNode, BlockRealizeNode, ForKind,
} from '../../ir/tensor/nodes.js';

const loweringRules = new Map();
const CONSTANT_OPS = new Set(['constant', 'scalar_constant']);

export { CONSTANT_OPS };

export function hasLoweringRule(opName) {
  return loweringRules.has(opName) || CONSTANT_OPS.has(opName);
}

export function registerLoweringRule(opName, ruleFunc) {
  loweringRules.set(opName, ruleFunc);
}

export function getLoweringRule(opName) {
  return loweringRules.get(opName);
}

export class LoweringContext {
  constructor() {
    this.bufferMap = new Map();
    this.varCounter = 0;
    this.shapeParams = new Map();
    this.symbolToVar = new Map();
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
      if (buf.shape[i] === DYNAMIC) this.extentNode(DYNAMIC, buf, i);
    }
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
    if (dim !== DYNAMIC) return new IntImmNode(dim);
    return this._shapeParamVar(buf, dimIdx);
  }

  extentNodes(shape, buf) {
    const nodes = new Array(shape.length);
    for (let i = 0; i < shape.length; i++) {
      nodes[i] = shape[i] === DYNAMIC ? this._shapeParamVar(buf, i) : new IntImmNode(shape[i]);
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
  const outBuf = ctx.getOrAllocBuffer(op.getResult(0));
  const val = op.getAttr('value');
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
