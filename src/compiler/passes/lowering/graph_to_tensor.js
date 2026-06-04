import { TensorType, DYNAMIC } from '../../ir/graph/types.js';
import { registry } from '../../ir/graph/ops.js';
import { MemoryScope } from '../../ir/tensor/tensor_types.js';
import { Buffer } from '../../ir/tensor/buffer.js';
import {
  PrimFunc, ForNode, BlockNode, SeqNode, BufferStoreNode, BufferLoadNode,
  VariableNode, IntImmNode, FloatImmNode, MathOpNode, CompareNode, BlockRealizeNode, ForKind,
  IfThenElseNode, CallExternNode, CastNode, LetStmtNode, WhileNode
} from '../../ir/tensor/nodes.js';

const BINARY_ARITH = new Set(['+', '-', '*', '/']);

const ELEMENTWISE_OPS = {
  'add': '+', 'sub': '-', 'mul': '*', 'div': '/',
  'max': 'max', 'min': 'min', 'exp': 'exp', 'log': 'log',
  'sqrt': 'sqrt', 'rsqrt': 'rsqrt', 'tanh': 'tanh', 'abs': 'abs',
  'ceil': 'ceil', 'floor': 'floor', 'neg': '-',
  'maximum': 'max', 'minimum': 'min',
  'sin': 'sin', 'cos': 'cos', 'round': 'round', 'sign': 'sign',
  'pow': 'pow', 'rem': 'fmod'
};

const REDUCE_COMBINERS = {
  'sum':  (a, b) => new MathOpNode('+', a, b),
  'mean': (a, b) => new MathOpNode('+', a, b),
  'prod': (a, b) => new MathOpNode('*', a, b),
  'max':  (a, b, dt) => new CallExternNode('max', [a, b], dt),
  'min':  (a, b, dt) => new CallExternNode('min', [a, b], dt)
};

const CONSTANT_OPS = new Set(['constant', 'scalar_constant']);

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

function buildElementwiseExpr(opName, loadArgs, dtype) {
  const jsOp = ELEMENTWISE_OPS[opName];
  if (!jsOp) return null;
  if (loadArgs.length === 2 && BINARY_ARITH.has(jsOp)) {
    return new MathOpNode(jsOp, loadArgs[0], loadArgs[1]);
  }
  if (loadArgs.length === 1 && jsOp === '-') {
    return new MathOpNode('-', new FloatImmNode(0), loadArgs[0]);
  }
  return new CallExternNode(jsOp, loadArgs, dtype);
}

function makeLoopNest(ctx, shape, buf) {
  const n = shape.length;
  const loopVars = ctx.allocVarArray('i', n);
  const loopBinds = ctx.allocBindArray('v', loopVars);
  const indices = new Array(n);
  for (let i = 0; i < n; i++) indices[i] = loopBinds[i].iterVar;
  const extentNodes = buf ? ctx.extentNodes(shape, buf) : null;
  return { loopVars, loopBinds, indices, extentNodes };
}

function wrapLoopsWithNodes(body, loopVars, extentNodes) {
  let result = body;
  for (let i = loopVars.length - 1; i >= 0; i--) {
    result = new ForNode(loopVars[i], new IntImmNode(0), extentNodes[i], ForKind.SERIAL, result);
  }
  return result;
}

function wrapLoops(body, loopVars, extents) {
  let result = body;
  for (let i = loopVars.length - 1; i >= 0; i--) {
    result = new ForNode(loopVars[i], new IntImmNode(0), new IntImmNode(extents[i]), ForKind.SERIAL, result);
  }
  return result;
}

function wrapInLoops(body, loopVars, shape, extentNodes) {
  if (extentNodes) return wrapLoopsWithNodes(body, loopVars, extentNodes);
  return wrapLoops(body, loopVars, shape);
}

function buildSpatialNest(ctx, prefix, dims, shape, buf) {
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

function computeBroadcastIndices(inBuf, outBuf, outIndices) {
  const inRank = inBuf.shape.length;
  const outRank = outBuf.shape.length;
  const offset = outRank - inRank;
  const indices = new Array(inRank);
  for (let j = 0; j < inRank; j++) {
    indices[j] = inBuf.shape[j] === 1 ? new IntImmNode(0) : outIndices[offset + j];
  }
  return indices;
}

function bufRefs(bufs) {
  const refs = new Array(bufs.length);
  for (let i = 0; i < bufs.length; i++) refs[i] = { buffer: bufs[i] };
  return refs;
}

function concatIterVars() {
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

function extractIterVars(ivs) {
  const result = new Array(ivs.length);
  for (let i = 0; i < ivs.length; i++) result[i] = ivs[i].iterVar;
  return result;
}

const loweringRules = new Map();

export function hasLoweringRule(opName) {
  return loweringRules.has(opName) || CONSTANT_OPS.has(opName);
}

export function registerLoweringRule(opName, ruleFunc) {
  loweringRules.set(opName, ruleFunc);
}

function lowerPointwise(ctx, op, inputs, outputs, exprBuilder) {
  const outBuf = outputs[0];
  const { loopVars, loopBinds, indices, extentNodes } = makeLoopNest(ctx, outBuf.shape, outBuf);
  const loads = new Array(inputs.length);
  for (let i = 0; i < inputs.length; i++) loads[i] = new BufferLoadNode(inputs[i], indices);
  const expr = exprBuilder(op, loads, outBuf.dtype);
  const store = new BufferStoreNode(outBuf, indices, expr);
  const block = new BlockNode(`${op.opName}_block`, loopBinds, bufRefs(inputs), [{ buffer: outBuf }], store);
  return wrapInLoops(block, loopVars, outBuf.shape, extentNodes);
}

for (const opName of Object.keys(ELEMENTWISE_OPS)) {
  registerLoweringRule(opName, (ctx, op, inputs, outputs) =>
    lowerPointwise(ctx, op, inputs, outputs, (o, loads, dtype) => buildElementwiseExpr(o.opName, loads, dtype))
  );
}

registerLoweringRule('compare', (ctx, op, inputs, outputs) =>
  lowerPointwise(ctx, op, inputs, outputs, (o, loads) =>
    new CompareNode(o.getAttr('direction') || 'eq', loads[0], loads[1]))
);

registerLoweringRule('select', (ctx, op, inputs, outputs) =>
  lowerPointwise(ctx, op, inputs, outputs, (o, loads) =>
    new IfThenElseNode(loads[0], loads[1], loads[2]))
);

registerLoweringRule('clamp', (ctx, op, inputs, outputs) =>
  lowerPointwise(ctx, op, inputs, outputs, (o, loads, dtype) =>
    new CallExternNode('min', [new CallExternNode('max', [loads[1], loads[0]], dtype), loads[2]], dtype))
);

registerLoweringRule('convert', (ctx, op, inputs, outputs) =>
  lowerPointwise(ctx, op, inputs, outputs, (o, loads) =>
    new CastNode(loads[0], inputs[0].dtype, outputs[0].dtype))
);

registerLoweringRule('broadcast_in_dim', lowerBroadcast);
registerLoweringRule('broadcast', lowerBroadcast);

function lowerBroadcast(ctx, op, inputs, outputs) {
  const inBuf = inputs[0];
  const outBuf = outputs[0];
  const broadcastDims = op.getAttr('broadcast_dimensions') || [];
  const { loopVars, loopBinds, indices: outIndices } = makeLoopNest(ctx, outBuf.shape);
  const inIndices = new Array(inBuf.shape.length);
  for (let i = 0; i < inBuf.shape.length; i++) {
    const mapped = broadcastDims.length > 0 ? broadcastDims[i] : i + (outBuf.shape.length - inBuf.shape.length);
    inIndices[i] = inBuf.shape[i] === 1 ? new IntImmNode(0) : outIndices[mapped];
  }
  const store = new BufferStoreNode(outBuf, outIndices, new BufferLoadNode(inBuf, inIndices));
  const block = new BlockNode(`broadcast_block`, loopBinds, [{ buffer: inBuf }], [{ buffer: outBuf }], store);
  return wrapInLoops(block, loopVars, outBuf.shape);
}

registerLoweringRule('transpose', function lowerTranspose(ctx, op, inputs, outputs) {
  const perm = op.getAttr('permutation');
  const inBuf = inputs[0];
  const outBuf = outputs[0];
  const { loopVars, loopBinds, indices: outIndices } = makeLoopNest(ctx, outBuf.shape);
  const inIndices = new Array(inBuf.shape.length);
  for (let i = 0; i < perm.length; i++) inIndices[perm[i]] = outIndices[i];
  const store = new BufferStoreNode(outBuf, outIndices, new BufferLoadNode(inBuf, inIndices));
  const block = new BlockNode(`transpose_block`, loopBinds, [{ buffer: inBuf }], [{ buffer: outBuf }], store);
  return wrapInLoops(block, loopVars, outBuf.shape);
});

registerLoweringRule('reshape', function lowerReshape(ctx, op, inputs, outputs) {
  const inBuf = inputs[0];
  const outBuf = outputs[0];
  const { loopVars, loopBinds, indices: outIndices } = makeLoopNest(ctx, outBuf.shape);
  let flatIndex = outIndices[outBuf.shape.length - 1];
  let stride = 1;
  for (let i = outBuf.shape.length - 2; i >= 0; i--) {
    stride *= outBuf.shape[i + 1];
    flatIndex = new MathOpNode('+', flatIndex, new MathOpNode('*', outIndices[i], new IntImmNode(stride)));
  }
  const inIndices = new Array(inBuf.shape.length);
  let cur = flatIndex;
  for (let i = inBuf.shape.length - 1; i >= 0; i--) {
    if (i === 0) { inIndices[i] = cur; }
    else {
      inIndices[i] = new MathOpNode('%', cur, new IntImmNode(inBuf.shape[i]));
      cur = new MathOpNode('//', cur, new IntImmNode(inBuf.shape[i]));
    }
  }
  const store = new BufferStoreNode(outBuf, outIndices, new BufferLoadNode(inBuf, inIndices));
  const block = new BlockNode(`reshape_block`, loopBinds, [{ buffer: inBuf }], [{ buffer: outBuf }], store);
  return wrapInLoops(block, loopVars, outBuf.shape);
});

registerLoweringRule('slice', function lowerSlice(ctx, op, inputs, outputs) {
  const inBuf = inputs[0];
  const outBuf = outputs[0];
  const starts = op.getAttr('starts');
  const strides = op.getAttr('strides') || starts.map(() => 1);
  const { loopVars, loopBinds, indices: outIndices } = makeLoopNest(ctx, outBuf.shape);
  const inIndices = new Array(inBuf.shape.length);
  for (let i = 0; i < inBuf.shape.length; i++) {
    const base = new IntImmNode(starts[i]);
    if (strides[i] === 1) {
      inIndices[i] = new MathOpNode('+', base, outIndices[i]);
    } else {
      inIndices[i] = new MathOpNode('+', base, new MathOpNode('*', outIndices[i], new IntImmNode(strides[i])));
    }
  }
  const store = new BufferStoreNode(outBuf, outIndices, new BufferLoadNode(inBuf, inIndices));
  const block = new BlockNode(`slice_block`, loopBinds, [{ buffer: inBuf }], [{ buffer: outBuf }], store);
  return wrapInLoops(block, loopVars, outBuf.shape);
});

registerLoweringRule('pad', function lowerPad(ctx, op, inputs, outputs) {
  const inBuf = inputs[0];
  const padVal = inputs[1];
  const outBuf = outputs[0];
  const low = op.getAttr('low');
  const high = op.getAttr('high');
  const interior = op.getAttr('interior') || low.map(() => 0);
  const { loopVars, loopBinds, indices: outIndices } = makeLoopNest(ctx, outBuf.shape);
  const inIndices = new Array(inBuf.shape.length);
  let inBoundsExpr = new IntImmNode(1);
  for (let i = 0; i < inBuf.shape.length; i++) {
    const shifted = new MathOpNode('+', outIndices[i], new IntImmNode(-low[i]));
    if (interior[i] > 0) {
      const step = interior[i] + 1;
      const modCheck = new MathOpNode('%', shifted, new IntImmNode(step));
      const modIsZero = new CompareNode('eq', modCheck, new IntImmNode(0));
      inBoundsExpr = new MathOpNode('*', inBoundsExpr, modIsZero);
      inIndices[i] = new MathOpNode('//', shifted, new IntImmNode(step));
    } else {
      inIndices[i] = shifted;
    }
    const geZero = new CompareNode('ge', inIndices[i], new IntImmNode(0));
    const ltSize = new CompareNode('lt', inIndices[i], new IntImmNode(inBuf.shape[i]));
    inBoundsExpr = new MathOpNode('*', inBoundsExpr, new MathOpNode('*', geZero, ltSize));
  }
  const loadIn = new BufferLoadNode(inBuf, inIndices);
  const loadPad = new BufferLoadNode(padVal, []);
  const expr = new IfThenElseNode(inBoundsExpr, loadIn, loadPad);
  const store = new BufferStoreNode(outBuf, outIndices, expr);
  const block = new BlockNode(`pad_block`, loopBinds, [{ buffer: inBuf }, { buffer: padVal }], [{ buffer: outBuf }], store);
  return wrapInLoops(block, loopVars, outBuf.shape);
});

registerLoweringRule('concat', function lowerConcat(ctx, op, inputs, outputs) {
  const outBuf = outputs[0];
  const dim = op.getAttr('dimension');
  const stmts = [];
  let offset = 0;
  for (let k = 0; k < inputs.length; k++) {
    const inBuf = inputs[k];
    const { loopVars, loopBinds, indices: inIndices } = makeLoopNest(ctx, inBuf.shape);
    const outIndices = new Array(inBuf.shape.length);
    for (let d = 0; d < inBuf.shape.length; d++) {
      outIndices[d] = d === dim && offset > 0
        ? new MathOpNode('+', inIndices[d], new IntImmNode(offset))
        : inIndices[d];
    }
    const store = new BufferStoreNode(outBuf, outIndices, new BufferLoadNode(inBuf, inIndices));
    const block = new BlockNode(`concat_${k}`, loopBinds, [{ buffer: inBuf }], [{ buffer: outBuf }], store);
    stmts.push(wrapInLoops(block, loopVars, inBuf.shape));
    offset += inBuf.shape[dim];
  }
  return stmts.length === 1 ? stmts[0] : new SeqNode(stmts);
});

registerLoweringRule('iota', function lowerIota(ctx, op, inputs, outputs) {
  const outBuf = outputs[0];
  const iotaDim = op.getAttr('iota_dimension');
  const { loopVars, loopBinds, indices } = makeLoopNest(ctx, outBuf.shape);
  const val = new CastNode(indices[iotaDim], 'index', outBuf.dtype);
  const store = new BufferStoreNode(outBuf, indices, val);
  const block = new BlockNode(`iota_block`, loopBinds, [], [{ buffer: outBuf }], store);
  return wrapInLoops(block, loopVars, outBuf.shape);
});

function parseLayout(str) {
  const m = {};
  for (let i = 0; i < str.length; i++) m[str[i]] = i;
  return m;
}

registerLoweringRule('conv', function lowerConv(ctx, op, inputs, outputs) {
  const inBuf = inputs[0];
  const kerBuf = inputs[1];
  const outBuf = outputs[0];
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

  const initNest = buildSpatialNest(ctx, 'ci', Array.from({ length: outShape.length }, (_, i) => i), outShape);
  const initStore = new BufferStoreNode(outBuf, initNest.indices, new FloatImmNode(0));
  const initBlock = new BlockNode(`conv_init`, initNest.ivs, [], [{ buffer: outBuf }], initStore);
  const initBody = initNest.wrap(initBlock);

  const nVar = ctx.allocVar('cn');
  const ocVar = ctx.allocVar('coc');
  const icVar = ctx.allocVar('cic');
  const spatialOutVars = ctx.allocVarArray('co', spatialDims);
  const spatialKerVars = ctx.allocVarArray('ck', spatialDims);

  const allVars = [nVar, ocVar, ...spatialOutVars, icVar, ...spatialKerVars];
  const allBinds = ctx.allocBindArray('cv', allVars);

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
    const kKey = key.toLowerCase() === 'h' ? 'H' : 'W';
    const inSpatialIdx = new MathOpNode('+',
      new MathOpNode('*', soBinds[s].iterVar, new IntImmNode(strides[s])),
      new MathOpNode('+',
        new MathOpNode('*', skBinds[s].iterVar, new IntImmNode(dilation[s])),
        new IntImmNode(-padding[s][0])
      )
    );
    inIdx[iLayout[key]] = inSpatialIdx;
    kerIdx[kLayout[kKey]] = skBinds[s].iterVar;
    const ge = new CompareNode('ge', inSpatialIdx, new IntImmNode(0));
    const lt = new CompareNode('lt', inSpatialIdx, new IntImmNode(inBuf.shape[iLayout[key]]));
    const dimOk = new MathOpNode('*', ge, lt);
    inBoundsExpr = inBoundsExpr ? new MathOpNode('*', inBoundsExpr, dimOk) : dimOk;
  }

  const loadIn = new BufferLoadNode(inBuf, inIdx);
  const loadKer = new BufferLoadNode(kerBuf, kerIdx);
  const loadOut = new BufferLoadNode(outBuf, outIdx);
  const product = new MathOpNode('*', loadIn, loadKer);
  const guardedProduct = inBoundsExpr ? new IfThenElseNode(inBoundsExpr, product, new FloatImmNode(0)) : product;
  const accExpr = new MathOpNode('+', loadOut, guardedProduct);
  const accStore = new BufferStoreNode(outBuf, outIdx, accExpr);
  const accBlock = new BlockNode(`conv_acc`, allBinds, [{ buffer: inBuf }, { buffer: kerBuf }], [{ buffer: outBuf }], accStore);

  const kerSpatialSizes = new Array(spatialDims);
  for (let s = 0; s < spatialDims; s++) {
    const kKey = spatialLayoutKeys[s].toLowerCase() === 'h' ? 'H' : 'W';
    kerSpatialSizes[s] = kerBuf.shape[kLayout[kKey]];
  }

  let accBody = accBlock;
  for (let s = spatialDims - 1; s >= 0; s--) {
    accBody = new ForNode(spatialKerVars[s], new IntImmNode(0), new IntImmNode(kerSpatialSizes[s]), ForKind.SERIAL, accBody);
  }
  accBody = new ForNode(icVar, new IntImmNode(0), new IntImmNode(inChannelsPerGroup), ForKind.SERIAL, accBody);
  for (let s = spatialDims - 1; s >= 0; s--) {
    accBody = new ForNode(spatialOutVars[s], new IntImmNode(0), new IntImmNode(outShape[iLayout[spatialLayoutKeys[s]]]), ForKind.SERIAL, accBody);
  }
  accBody = new ForNode(ocVar, new IntImmNode(0), new IntImmNode(outChannels), ForKind.SERIAL, accBody);
  accBody = new ForNode(nVar, new IntImmNode(0), new IntImmNode(batch), ForKind.SERIAL, accBody);

  return new SeqNode([initBody, accBody]);
});

registerLoweringRule('gather', function lowerGather(ctx, op, inputs, outputs) {
  const operandBuf = inputs[0];
  const indicesBuf = inputs[1];
  const outBuf = outputs[0];
  const offsetDims = new Set(op.getAttr('offset_dims'));
  const collapsedDims = new Set(op.getAttr('collapsed_slice_dims'));
  const startIndexMap = op.getAttr('start_index_map');
  const sliceSizes = op.getAttr('slice_sizes');
  const indexVectorDim = op.getAttr('index_vector_dim');
  const { loopVars, loopBinds, indices: outIndices } = makeLoopNest(ctx, outBuf.shape);

  const batchIndices = [];
  const offsetIndices = [];
  for (let i = 0; i < outBuf.shape.length; i++) {
    if (offsetDims.has(i)) offsetIndices.push(outIndices[i]);
    else batchIndices.push(outIndices[i]);
  }

  const operandIndices = new Array(operandBuf.shape.length);
  let offsetIdx = 0;
  for (let i = 0; i < operandBuf.shape.length; i++) {
    if (collapsedDims.has(i)) {
      operandIndices[i] = new IntImmNode(0);
    } else {
      operandIndices[i] = offsetIndices[offsetIdx++];
    }
  }

  for (let k = 0; k < startIndexMap.length; k++) {
    const idxLookup = new Array(indicesBuf.shape.length);
    let batchIdx = 0;
    for (let d = 0; d < indicesBuf.shape.length; d++) {
      if (d === indexVectorDim) idxLookup[d] = new IntImmNode(k);
      else idxLookup[d] = batchIndices[batchIdx++];
    }
    const startVal = new BufferLoadNode(indicesBuf, idxLookup);
    const targetDim = startIndexMap[k];
    operandIndices[targetDim] = new MathOpNode('+', operandIndices[targetDim], startVal);
  }

  const store = new BufferStoreNode(outBuf, outIndices, new BufferLoadNode(operandBuf, operandIndices));
  const block = new BlockNode(`gather_block`, loopBinds, [{ buffer: operandBuf }, { buffer: indicesBuf }], [{ buffer: outBuf }], store);
  return wrapInLoops(block, loopVars, outBuf.shape);
});

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
    const rule = loweringRules.get(innerOp.opName);
    if (!rule) throw new Error(`No lowering rule for op '${innerOp.opName}' inside region`);
    const stmt = rule(ctx, innerOp, inputs, outputs);
    if (stmt) stmts.push(stmt);
  }
  return { stmts, yieldBuffers: [] };
}

registerLoweringRule('if', function lowerIf(ctx, op, inputs, outputs) {
  const predBuf = inputs[0];
  const predLoad = new BufferLoadNode(predBuf, []);
  const thenRegion = op.regions[0];
  const elseRegion = op.regions[1];
  const thenArgBufs = [];
  const elseArgBufs = [];
  const thenResult = lowerRegionBody(ctx, thenRegion, thenArgBufs);
  const thenBody = thenResult.stmts.length === 1 ? thenResult.stmts[0] : new SeqNode(thenResult.stmts);
  let elseBody = null;
  if (elseRegion && elseRegion.entryBlock) {
    const elseResult = lowerRegionBody(ctx, elseRegion, elseArgBufs);
    if (elseResult.stmts.length > 0) {
      elseBody = elseResult.stmts.length === 1 ? elseResult.stmts[0] : new SeqNode(elseResult.stmts);
    }
  }
  const copyStmts = [];
  for (let i = 0; i < thenResult.yieldBuffers.length && i < outputs.length; i++) {
    const src = thenResult.yieldBuffers[i];
    if (src !== outputs[i]) {
      ctx.bufferMap.set(op.getResult(i), src);
    }
  }
  const ifNode = new IfThenElseNode(predLoad, thenBody, elseBody);
  if (copyStmts.length > 0) return new SeqNode([ifNode, ...copyStmts]);
  return ifNode;
});

registerLoweringRule('while', function lowerWhile(ctx, op, inputs, outputs) {
  const loopBufs = new Array(inputs.length);
  for (let i = 0; i < inputs.length; i++) loopBufs[i] = inputs[i];
  const condRegion = op.regions[0];
  const bodyRegion = op.regions[1];
  const condVar = ctx.allocVar('_wcond', 'bool');
  const condResult = lowerRegionBody(ctx, condRegion, loopBufs);
  const condBody = condResult.stmts.length === 1 ? condResult.stmts[0] : new SeqNode(condResult.stmts);
  if (condResult.yieldBuffers.length > 0) {
    const condBuf = condResult.yieldBuffers[0];
    const condStore = new BufferStoreNode({ name: condVar.name, shape: [], strides: [], dtype: 'bool' }, [], new BufferLoadNode(condBuf, []));
  }
  const bodyResult = lowerRegionBody(ctx, bodyRegion, loopBufs);
  const loopBody = bodyResult.stmts.length === 1 ? bodyResult.stmts[0] : new SeqNode(bodyResult.stmts);
  for (let i = 0; i < outputs.length && i < bodyResult.yieldBuffers.length; i++) {
    if (bodyResult.yieldBuffers[i] !== outputs[i]) {
      ctx.bufferMap.set(op.getResult(i), bodyResult.yieldBuffers[i]);
    }
  }
  return new WhileNode(condVar, condBody, loopBody);
});

registerLoweringRule('reduce', function lowerReduce(ctx, op, inputs, outputs) {
  const inBuf = inputs[0];
  const initBuf = inputs[1];
  const outBuf = outputs[0];
  const dims = op.getAttr('dimensions') || [];
  const rType = op.getAttr('reduce_type') || 'sum';
  const dimSet = new Set(dims);
  const spatialDims = [];
  const reduceDims = [];
  for (let i = 0; i < inBuf.shape.length; i++) {
    (dimSet.has(i) ? reduceDims : spatialDims).push(i);
  }

  const initNest = buildSpatialNest(ctx, 'si', spatialDims, inBuf.shape);
  const initStore = new BufferStoreNode(outBuf, initNest.indices, new BufferLoadNode(initBuf, []));
  const initBlock = new BlockNode(`reduce_init`, initNest.ivs, [{ buffer: initBuf }], [{ buffer: outBuf }], initStore);
  const initBody = spatialDims.length > 0 ? initNest.wrap(initBlock) : initBlock;

  const accNest = buildSpatialNest(ctx, 'sa', spatialDims, inBuf.shape);
  const rVars = ctx.allocVarArray('r', reduceDims.length);
  const rIvs = ctx.allocBindArray('rv', rVars);
  const inIndices = new Array(inBuf.shape.length);
  for (let i = 0; i < spatialDims.length; i++) inIndices[spatialDims[i]] = accNest.ivs[i].iterVar;
  for (let i = 0; i < reduceDims.length; i++) inIndices[reduceDims[i]] = rIvs[i].iterVar;
  const loadA = new BufferLoadNode(outBuf, accNest.indices);
  const loadB = new BufferLoadNode(inBuf, inIndices);
  const combiner = REDUCE_COMBINERS[rType] || REDUCE_COMBINERS['sum'];
  const store = new BufferStoreNode(outBuf, accNest.indices, combiner(loadA, loadB, outBuf.dtype));
  const rExtents = new Array(reduceDims.length);
  for (let i = 0; i < reduceDims.length; i++) rExtents[i] = inBuf.shape[reduceDims[i]];
  const accBlock = new BlockNode(`reduce_acc`, concatIterVars(accNest.ivs, rIvs), [{ buffer: inBuf }], [{ buffer: outBuf }], store);
  let accBody = wrapLoops(accBlock, rVars, rExtents);
  accBody = accNest.wrap(accBody);

  const parts = [initBody, accBody];

  if (rType === 'mean') {
    let reduceSize = 1;
    for (let i = 0; i < reduceDims.length; i++) reduceSize *= inBuf.shape[reduceDims[i]];
    const meanNest = buildSpatialNest(ctx, 'sm', spatialDims, inBuf.shape);
    const divExpr = new MathOpNode('*', new BufferLoadNode(outBuf, meanNest.indices), new FloatImmNode(1.0 / reduceSize));
    const meanStore = new BufferStoreNode(outBuf, meanNest.indices, divExpr);
    const meanBlock = new BlockNode(`mean_div`, meanNest.ivs, [{ buffer: outBuf }], [{ buffer: outBuf }], meanStore);
    parts.push(spatialDims.length > 0 ? meanNest.wrap(meanBlock) : meanBlock);
  }

  return new SeqNode(parts);
});

function buildDotGeometry(ctx, op, lhs, rhs) {
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

registerLoweringRule('dot', function lowerDot(ctx, op, inputs, outputs) {
  const lhs = inputs[0];
  const rhs = inputs[1];
  const out = outputs[0];
  const geo = buildDotGeometry(ctx, op, lhs, rhs);

  const initNest = buildSpatialNest(ctx, 'di', Array.from({ length: out.shape.length }, (_, i) => i), out.shape);
  const initStore = new BufferStoreNode(out, initNest.indices, new FloatImmNode(0));
  const initBlock = new BlockNode(`matmul_init`, initNest.ivs, [], [{ buffer: out }], initStore);
  const initBody = initNest.wrap(initBlock);

  const accExpr = new MathOpNode('+', new BufferLoadNode(out, geo.outIdx), new MathOpNode('*', new BufferLoadNode(lhs, geo.lhsIdx), new BufferLoadNode(rhs, geo.rhsIdx)));
  const accStore = new BufferStoreNode(out, geo.outIdx, accExpr);
  const accBlock = new BlockNode(`matmul`, geo.allIvs, [{ buffer: lhs }, { buffer: rhs }], [{ buffer: out }], accStore);
  const accBody = geo.wrapAccBody(accBlock);

  return new SeqNode([initBody, accBody]);
});

const EPILOGUE_TAG_LOWERERS = new Map();

EPILOGUE_TAG_LOWERERS.set('bias', (expr, extraInputs, idx, out, epiIdx) => {
  const buf = extraInputs[idx.v++];
  return new MathOpNode('+', expr, new BufferLoadNode(buf, computeBroadcastIndices(buf, out, epiIdx)));
});

EPILOGUE_TAG_LOWERERS.set('residual_add', (expr, extraInputs, idx, out, epiIdx) => {
  const buf = extraInputs[idx.v++];
  return new MathOpNode('+', expr, new BufferLoadNode(buf, computeBroadcastIndices(buf, out, epiIdx)));
});

EPILOGUE_TAG_LOWERERS.set('scale', (expr, extraInputs, idx, out, epiIdx) => {
  const buf = extraInputs[idx.v++];
  return new MathOpNode('*', expr, new BufferLoadNode(buf, computeBroadcastIndices(buf, out, epiIdx)));
});

EPILOGUE_TAG_LOWERERS.set('relu', (expr, extraInputs, idx, out) => {
  return new CallExternNode('max', [expr, new FloatImmNode(0)], out.dtype);
});

EPILOGUE_TAG_LOWERERS.set('clamp', (expr, extraInputs, idx, out, epiIdx) => {
  const lo = extraInputs[idx.v++];
  const hi = extraInputs[idx.v++];
  return new CallExternNode('min', [
    new CallExternNode('max', [expr, new BufferLoadNode(lo, computeBroadcastIndices(lo, out, epiIdx))], out.dtype),
    new BufferLoadNode(hi, computeBroadcastIndices(hi, out, epiIdx))
  ], out.dtype);
});

EPILOGUE_TAG_LOWERERS.set('activation', (expr) => expr);

registerLoweringRule('fused_dot_epilogue', function lowerFusedDotEpilogue(ctx, op, inputs, outputs) {
  const numDotOperands = op.getAttr('num_dot_operands') || 2;
  const lhs = inputs[0];
  const rhs = inputs[1];
  const extraInputs = inputs.slice(numDotOperands);
  const out = outputs[0];
  const epilogueTags = op.getAttr('epilogue_tags') || [];
  const geo = buildDotGeometry(ctx, op, lhs, rhs);

  const initNest = buildSpatialNest(ctx, 'ei', Array.from({ length: out.shape.length }, (_, i) => i), out.shape);
  const initStore = new BufferStoreNode(out, initNest.indices, new FloatImmNode(0));
  const initBlock = new BlockNode(`matmul_init`, initNest.ivs, [], [{ buffer: out }], initStore);
  const initBody = initNest.wrap(initBlock);

  const accExpr = new MathOpNode('+', new BufferLoadNode(out, geo.outIdx), new MathOpNode('*', new BufferLoadNode(lhs, geo.lhsIdx), new BufferLoadNode(rhs, geo.rhsIdx)));
  const accStore = new BufferStoreNode(out, geo.outIdx, accExpr);
  const accBlock = new BlockNode(`matmul_acc`, geo.allIvs, [{ buffer: lhs }, { buffer: rhs }], [{ buffer: out }], accStore);
  const accBody = geo.wrapAccBody(accBlock);

  if (epilogueTags.length === 0) {
    return new SeqNode([initBody, accBody]);
  }

  const epiNest = buildSpatialNest(ctx, 'ep', Array.from({ length: out.shape.length }, (_, i) => i), out.shape);
  const epiIdx = epiNest.indices;

  let expr = new BufferLoadNode(out, epiIdx);
  const idx = { v: 0 };
  for (const tag of epilogueTags) {
    const lowerer = EPILOGUE_TAG_LOWERERS.get(tag);
    if (lowerer) expr = lowerer(expr, extraInputs, idx, out, epiIdx);
  }

  const epiReads = bufRefs([out, ...extraInputs]);
  const epiStore = new BufferStoreNode(out, epiIdx, expr);
  const epiBlock = new BlockNode(`epilogue`, epiNest.ivs, epiReads, [{ buffer: out }], epiStore);
  const epiBody = epiNest.wrap(epiBlock);

  return new SeqNode([initBody, accBody, epiBody]);
});

const INLINE_FUSION_BUILDERS = new Map();

function registerInlineFusionBuilder(opName, builder) {
  INLINE_FUSION_BUILDERS.set(opName, builder);
}

for (const opName of Object.keys(ELEMENTWISE_OPS)) {
  registerInlineFusionBuilder(opName, (innerOp, args, dtype) =>
    buildElementwiseExpr(innerOp.opName, args, dtype)
  );
}

registerInlineFusionBuilder('compare', (innerOp, args) =>
  new CompareNode(innerOp.getAttr('direction') || 'eq', args[0], args[1])
);

registerInlineFusionBuilder('select', (_innerOp, args) =>
  new IfThenElseNode(args[0], args[1], args[2])
);

registerInlineFusionBuilder('clamp', (_innerOp, args, dtype) =>
  new CallExternNode('min', [new CallExternNode('max', [args[1], args[0]], dtype), args[2]], dtype)
);

registerInlineFusionBuilder('convert', (innerOp, args) =>
  new CastNode(args[0], innerOp.getOperand(0).type.dtype, innerOp.getAttr('target_dtype') || innerOp.getResult(0).type.dtype)
);

registerInlineFusionBuilder('broadcast_in_dim', (_innerOp, args) => args[0]);
registerInlineFusionBuilder('broadcast', (_innerOp, args) => args[0]);

export function canInlineFuse(opName) {
  return INLINE_FUSION_BUILDERS.has(opName);
}

const CSE_TRIVIAL = new Set(['BufferLoadNode', 'VariableNode', 'IntImmNode', 'FloatImmNode']);

function lowerFusion(ctx, op) {
  const numInputs = op.numOperands;
  const numOutputs = op.numResults;
  const inputs = new Array(numInputs);
  for (let i = 0; i < numInputs; i++) inputs[i] = ctx.getOrAllocBuffer(op.getOperand(i));
  const outputs = new Array(numOutputs);
  for (let i = 0; i < numOutputs; i++) outputs[i] = ctx.getOrAllocBuffer(op.getResult(i));

  const outBuf = outputs[0];
  const { loopVars, loopBinds, indices: outIndices } = makeLoopNest(ctx, outBuf.shape);
  const exprMap = new Map();

  const entryBlock = op.regions[0].entryBlock;
  for (let i = 0; i < entryBlock.arguments.length; i++) {
    exprMap.set(entryBlock.arguments[i], new BufferLoadNode(inputs[i], computeBroadcastIndices(inputs[i], outBuf, outIndices)));
  }

  const useCount = new Map();
  for (const innerOp of entryBlock.ops()) {
    for (let i = 0; i < innerOp.numOperands; i++) {
      const val = innerOp.getOperand(i);
      useCount.set(val, (useCount.get(val) || 0) + 1);
    }
  }

  const cseVars = new Map();
  let cseCounter = 0;
  const cseStmts = [];

  function getExpr(val) {
    const expr = exprMap.get(val);
    if (expr === undefined) {
      throw new Error(`Fusion lowering: unmapped operand from '${val.definingOp ? val.definingOp.opName : 'unknown'}'`);
    }
    if ((useCount.get(val) || 0) > 1 && !CSE_TRIVIAL.has(expr.type)) {
      if (!cseVars.has(val)) {
        const v = ctx.allocVar(`cse${cseCounter++}`, outBuf.dtype);
        cseVars.set(val, v);
        cseStmts.push({ variable: v, value: expr });
        exprMap.set(val, v);
      }
      return cseVars.get(val);
    }
    return expr;
  }

  const stores = [];
  for (const innerOp of entryBlock.ops()) {
    if (innerOp.opName === 'yield') {
      for (let i = 0; i < innerOp.numOperands; i++) {
        stores.push(new BufferStoreNode(outputs[i], outIndices, getExpr(innerOp.getOperand(i))));
      }
      break;
    }

    if (CONSTANT_OPS.has(innerOp.opName)) {
      const val = innerOp.getAttr('value');
      exprMap.set(innerOp.getResult(0), new FloatImmNode(typeof val === 'number' ? val : 0));
      continue;
    }

    const builder = INLINE_FUSION_BUILDERS.get(innerOp.opName);
    if (!builder) {
      throw new Error(`Fusion lowering: unsupported op '${innerOp.opName}' inside fusion body`);
    }

    const args = new Array(innerOp.numOperands);
    for (let i = 0; i < innerOp.numOperands; i++) args[i] = getExpr(innerOp.getOperand(i));
    exprMap.set(innerOp.getResult(0), builder(innerOp, args, outBuf.dtype));
  }

  let storeBody = stores.length === 1 ? stores[0] : new SeqNode(stores);
  for (let i = cseStmts.length - 1; i >= 0; i--) {
    storeBody = new LetStmtNode(cseStmts[i].variable, cseStmts[i].value, storeBody);
  }

  const block = new BlockNode(`fusion_block`, loopBinds, bufRefs(inputs), bufRefs(outputs), storeBody);
  return wrapInLoops(block, loopVars, outBuf.shape);
}

function lowerConstant(ctx, op) {
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

function canLowerAsElementwiseFusion(op) {
  const region = op.regions[0];
  if (!region) return false;
  for (const innerOp of region.entryBlock.ops()) {
    if (innerOp.opName === 'yield') continue;
    if (!INLINE_FUSION_BUILDERS.has(innerOp.opName)) return false;
  }
  return true;
}

function lowerFusionAsIndividualOps(ctx, fusionOp, stmts) {
  const entryBlock = fusionOp.regions[0].entryBlock;
  const valueMap = new Map();
  for (let i = 0; i < entryBlock.arguments.length; i++) {
    valueMap.set(entryBlock.arguments[i], fusionOp.getOperand(i));
  }

  for (const innerOp of entryBlock.ops()) {
    if (innerOp.opName === 'yield') {
      for (let i = 0; i < innerOp.numOperands; i++) {
        const outerVal = valueMap.get(innerOp.getOperand(i));
        if (outerVal) {
          const srcBuf = ctx.getOrAllocBuffer(outerVal);
          const dstBuf = ctx.getOrAllocBuffer(fusionOp.getResult(i));
          if (srcBuf !== dstBuf) ctx.bufferMap.set(fusionOp.getResult(i), srcBuf);
        }
      }
      continue;
    }

    const outerOperands = new Array(innerOp.numOperands);
    for (let i = 0; i < innerOp.numOperands; i++) {
      outerOperands[i] = valueMap.get(innerOp.getOperand(i)) || innerOp.getOperand(i);
    }

    const inputs = new Array(outerOperands.length);
    for (let i = 0; i < outerOperands.length; i++) inputs[i] = ctx.getOrAllocBuffer(outerOperands[i]);
    const outputs = new Array(innerOp.numResults);
    for (let i = 0; i < innerOp.numResults; i++) {
      const proxy = { type: innerOp.getResult(i).type };
      outputs[i] = ctx.getOrAllocBuffer(proxy);
      valueMap.set(innerOp.getResult(i), proxy);
    }

    if (CONSTANT_OPS.has(innerOp.opName)) {
      stmts.push(lowerConstant(ctx, innerOp));
      continue;
    }

    const rule = loweringRules.get(innerOp.opName);
    if (!rule) {
      throw new Error(`Fusion lowering: no lowering rule for op '${innerOp.opName}' inside fusion body`);
    }
    const stmt = rule(ctx, innerOp, inputs, outputs);
    if (stmt) stmts.push(stmt);
  }
}

export function lowerGraphToPrimFunc(graphFunc) {
  const ctx = new LoweringContext();
  const params = [];
  const bufferMap = new Map();

  for (const arg of graphFunc.args) {
    const v = ctx.allocVar(`arg`);
    params.push(v);
    bufferMap.set(v, ctx.getOrAllocBuffer(arg));
  }

  const retOp = graphFunc.getReturnOp();
  for (let i = 0; i < retOp.numOperands; i++) {
    const v = ctx.allocVar(`ret`);
    params.push(v);
    bufferMap.set(v, ctx.getOrAllocBuffer(retOp.getOperand(i)));
  }

  const stmts = [];

  for (const op of graphFunc.ops()) {
    if (CONSTANT_OPS.has(op.opName)) stmts.push(lowerConstant(ctx, op));
  }

  for (const op of graphFunc.ops()) {
    if (op.opName === 'return' || op.opName === 'yield') continue;
    if (CONSTANT_OPS.has(op.opName)) continue;

    if (op.opName === 'fusion') {
      if (canLowerAsElementwiseFusion(op)) {
        stmts.push(lowerFusion(ctx, op));
      } else {
        lowerFusionAsIndividualOps(ctx, op, stmts);
      }
      continue;
    }

    const rule = loweringRules.get(op.opName);
    if (!rule) throw new Error(`No lowering rule defined for op: ${op.opName}`);

    const inputs = new Array(op.numOperands);
    for (let i = 0; i < op.numOperands; i++) inputs[i] = ctx.getOrAllocBuffer(op.getOperand(i));
    const outputs = new Array(op.numResults);
    for (let i = 0; i < op.numResults; i++) outputs[i] = ctx.getOrAllocBuffer(op.getResult(i));

    const stmt = rule(ctx, op, inputs, outputs);
    if (stmt) stmts.push(stmt);
  }

  const shapeParams = [...ctx.shapeParams.values()];
  for (const sp of shapeParams) params.push(sp);

  return new PrimFunc(graphFunc.name, params, stmts.length === 1 ? stmts[0] : new SeqNode(stmts), bufferMap, shapeParams);
}
