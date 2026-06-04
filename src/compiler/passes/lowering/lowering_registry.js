import { TensorType, DYNAMIC } from '../../ir/graph/types.js';
import { MemoryScope } from '../../ir/tensor/tensor_types.js';
import { Buffer } from '../../ir/tensor/buffer.js';
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
    this.bufferMap.set(value, buf);
    return buf;
  }

  extentNode(dim, buf) {
    if (dim !== DYNAMIC) return new IntImmNode(dim);
    const key = `${buf.name}:${buf.shape.indexOf(dim)}`;
    let v = this.shapeParams.get(key);
    if (!v) {
      v = this.allocVar(`_ds`);
      this.shapeParams.set(key, v);
    }
    return v;
  }

  extentNodes(shape, buf) {
    const nodes = new Array(shape.length);
    for (let i = 0; i < shape.length; i++) {
      if (shape[i] === DYNAMIC) {
        const key = `${buf.name}:${i}`;
        let v = this.shapeParams.get(key);
        if (!v) {
          v = this.allocVar(`_ds`);
          this.shapeParams.set(key, v);
        }
        nodes[i] = v;
      } else {
        nodes[i] = new IntImmNode(shape[i]);
      }
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
    extentNodes[i] = buf ? ctx.extentNode(shape[dims[i]], buf) : new IntImmNode(shape[dims[i]]);
  }
  return {
    vars, ivs, indices, extentNodes,
    wrap(body) { return wrapLoopsWithNodes(body, vars, extentNodes); }
  };
}

export function computeBroadcastIndices(inBuf, outBuf, outIndices) {
  const inRank = inBuf.shape.length;
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
  for (let i = 0; i < inputs.length; i++) loads[i] = new BufferLoadNode(inputs[i], indices);
  const expr = exprBuilder(op, loads, outBuf.dtype);
  const store = new BufferStoreNode(outBuf, indices, expr);
  const block = new BlockNode(`${op.opName}_block`, loopBinds, bufRefs(inputs), [{ buffer: outBuf }], store);
  return wrapInLoops(block, loopVars, outBuf.shape, extentNodes);
}

export function lowerConstant(ctx, op) {
  const outBuf = ctx.getOrAllocBuffer(op.getResult(0));
  const val = op.getAttr('value');
  const valNode = typeof val === 'number' ? new FloatImmNode(val) : new FloatImmNode(0);
  if (outBuf.shape.length === 0) {
    return new BufferStoreNode(outBuf, [], valNode);
  }
  const { loopVars, loopBinds, indices } = makeLoopNest(ctx, outBuf.shape);
  const store = new BufferStoreNode(outBuf, indices, valNode);
  const block = new BlockNode(`${op.opName}_block`, loopBinds, [], [{ buffer: outBuf }], store);
  return wrapInLoops(block, loopVars, outBuf.shape);
}

export function parseLayout(str) {
  const m = {};
  for (let i = 0; i < str.length; i++) m[str[i]] = i;
  return m;
}

export function buildDotGeometry(ctx, op, lhs, rhs) {
  const lhsContracting = op.getAttr('lhs_contracting') || [];
  const rhsContracting = op.getAttr('rhs_contracting') || [];
  const lhsBatch = op.getAttr('lhs_batch') || [];
  const rhsBatch = op.getAttr('rhs_batch') || [];
  const lhsCSet = new Set(lhsContracting);
  const lhsBSet = new Set(lhsBatch);
  const rhsCSet = new Set(rhsContracting);
  const rhsBSet = new Set(rhsBatch);

  const lhsSpatial = [];
  for (let i = 0; i < lhs.shape.length; i++) {
    if (!lhsCSet.has(i) && !lhsBSet.has(i)) lhsSpatial.push(i);
  }
  const rhsSpatial = [];
  for (let i = 0; i < rhs.shape.length; i++) {
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

  const lhsIdx = new Array(lhs.shape.length);
  for (let i = 0; i < lhsBatch.length; i++) lhsIdx[lhsBatch[i]] = bIvs[i].iterVar;
  for (let i = 0; i < lhsSpatial.length; i++) lhsIdx[lhsSpatial[i]] = lsIvs[i].iterVar;
  for (let i = 0; i < lhsContracting.length; i++) lhsIdx[lhsContracting[i]] = cIvs[i].iterVar;

  const rhsIdx = new Array(rhs.shape.length);
  for (let i = 0; i < rhsBatch.length; i++) rhsIdx[rhsBatch[i]] = bIvs[i].iterVar;
  for (let i = 0; i < rhsSpatial.length; i++) rhsIdx[rhsSpatial[i]] = rsIvs[i].iterVar;
  for (let i = 0; i < rhsContracting.length; i++) rhsIdx[rhsContracting[i]] = cIvs[i].iterVar;

  const allIvs = concatIterVars(bIvs, lsIvs, rsIvs, cIvs);

  const loopGroups = [
    { vars: bVars, dims: lhsBatch, shape: lhs.shape },
    { vars: lsVars, dims: lhsSpatial, shape: lhs.shape },
    { vars: rsVars, dims: rhsSpatial, shape: rhs.shape },
    { vars: cVars, dims: lhsContracting, shape: lhs.shape },
  ];

  function wrapAccBody(body) {
    let result = body;
    for (let g = loopGroups.length - 1; g >= 0; g--) {
      const { vars, dims, shape } = loopGroups[g];
      for (let i = vars.length - 1; i >= 0; i--) {
        result = new ForNode(vars[i], new IntImmNode(0), new IntImmNode(shape[dims[i]]), ForKind.SERIAL, result);
      }
    }
    return result;
  }

  return { outIdx, lhsIdx, rhsIdx, allIvs, wrapAccBody };
}
