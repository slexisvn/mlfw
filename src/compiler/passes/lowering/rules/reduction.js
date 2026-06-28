import { MathOpNode, FloatImmNode, IntImmNode, BufferStoreNode, BufferLoadNode, BlockNode, SeqNode, CallExternNode, IfThenElseNode, CompareNode, IterVarKind } from '../../../ir/tensor/nodes.js';
import { Buffer } from '../../../ir/tensor/buffer.js';
import { DYNAMIC } from '../../../ir/graph/types.js';
import { registerLoweringRule, buildSpatialNest, wrapLoopsWithNodes, concatIterVars } from '../lowering_registry.js';
import { isDtypeInt } from '../../../../backend/dtype_map.js';

const INT_DTYPE_MIN = { i8: -128, i16: -32768, i32: -2147483648, i64: -2147483648, ui8: 0, ui16: 0, ui32: 0, bool: 0 };
const INT_DTYPE_MAX = { i8: 127, i16: 32767, i32: 2147483647, i64: 2147483647, ui8: 255, ui16: 65535, ui32: 4294967295, bool: 1 };

function argReduceSentinel(dtype, isGt) {
  if (isDtypeInt(dtype)) {
    const v = isGt ? (INT_DTYPE_MIN[dtype] ?? -2147483648) : (INT_DTYPE_MAX[dtype] ?? 2147483647);
    return new IntImmNode(v);
  }
  return new FloatImmNode(isGt ? -Infinity : Infinity);
}

const REDUCE_COMBINERS = {
  'sum':  (a, b) => new MathOpNode('+', a, b),
  'mean': (a, b) => new MathOpNode('+', a, b),
  'prod': (a, b) => new MathOpNode('*', a, b),
  'max':  (a, b, dt) => new CallExternNode('max', [a, b], dt),
  'min':  (a, b, dt) => new CallExternNode('min', [a, b], dt),
  'and':  (a, b) => new MathOpNode('*', a, b),
  'or':   (a, b, dt) => new CallExternNode('max', [a, b], dt)
};

export function register() {
  registerLoweringRule('reduce', (ctx, op, inputs, outputs) => {
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

    const initNest = buildSpatialNest(ctx, 'si', spatialDims, inBuf.shape, inBuf);
    const initStore = new BufferStoreNode(outBuf, initNest.indices, new BufferLoadNode(initBuf, []));
    const initBlock = new BlockNode(ctx.blockName('reduce_init'), initNest.ivs, [{ buffer: initBuf }], [{ buffer: outBuf }], initStore);
    const initBody = spatialDims.length > 0 ? initNest.wrap(initBlock) : initBlock;

    const accNest = buildSpatialNest(ctx, 'sa', spatialDims, inBuf.shape, inBuf);
    const rVars = ctx.allocVarArray('r', reduceDims.length);
    const rIvs = ctx.allocBindArray('rv', rVars);
    for (const iv of rIvs) iv.kind = IterVarKind.COMM_REDUCE;
    const inIndices = new Array(inBuf.shape.length);
    for (let i = 0; i < spatialDims.length; i++) inIndices[spatialDims[i]] = accNest.ivs[i].iterVar;
    for (let i = 0; i < reduceDims.length; i++) inIndices[reduceDims[i]] = rIvs[i].iterVar;
    const loadA = new BufferLoadNode(outBuf, accNest.indices);
    const loadB = new BufferLoadNode(inBuf, inIndices);
    const combiner = REDUCE_COMBINERS[rType];
    if (!combiner) throw new Error(`reduction lowering: unsupported reduce_type '${rType}'`);
    const store = new BufferStoreNode(outBuf, accNest.indices, combiner(loadA, loadB, outBuf.dtype));
    const rExtentNodes = new Array(reduceDims.length);
    for (let i = 0; i < reduceDims.length; i++) rExtentNodes[i] = ctx.extentNode(inBuf.shape[reduceDims[i]], inBuf, reduceDims[i]);
    const accBlock = new BlockNode(ctx.blockName('reduce_acc'), concatIterVars(accNest.ivs, rIvs), [{ buffer: inBuf }], [{ buffer: outBuf }], store);
    let accBody = wrapLoopsWithNodes(accBlock, rVars, rExtentNodes);
    accBody = accNest.wrap(accBody);

    const parts = [initBody, accBody];

    if (rType === 'mean') {
      let staticSize = 1;
      const dynExtents = [];
      for (let i = 0; i < reduceDims.length; i++) {
        const d = inBuf.shape[reduceDims[i]];
        if (d === DYNAMIC) dynExtents.push(ctx.extentNode(DYNAMIC, inBuf, reduceDims[i]));
        else staticSize *= d;
      }
      const meanNest = buildSpatialNest(ctx, 'sm', spatialDims, inBuf.shape, inBuf);
      const meanLoad = new BufferLoadNode(outBuf, meanNest.indices);
      let divExpr;
      if (dynExtents.length === 0) {
        divExpr = new MathOpNode('*', meanLoad, new FloatImmNode(1.0 / staticSize));
      } else {
        let divisor = new IntImmNode(staticSize);
        for (const e of dynExtents) divisor = new MathOpNode('*', divisor, e);
        divExpr = new MathOpNode('/', meanLoad, divisor);
      }
      const meanStore = new BufferStoreNode(outBuf, meanNest.indices, divExpr);
      const meanBlock = new BlockNode(ctx.blockName('mean_div'), meanNest.ivs, [{ buffer: outBuf }], [{ buffer: outBuf }], meanStore);
      parts.push(spatialDims.length > 0 ? meanNest.wrap(meanBlock) : meanBlock);
    }

    return new SeqNode(parts);
  });

  function registerArgReduce(opName, compareFn) {
    registerLoweringRule(opName, (ctx, op, inputs, outputs) => {
      const inBuf = inputs[0];
      const outBuf = outputs[0];
      const axis = op.getAttr('axis');
      const keepDims = op.getAttr('keep_dims') || false;
      const dimSet = new Set([axis]);
      const spatialDims = [];
      const reduceDim = axis;
      for (let i = 0; i < inBuf.shape.length; i++) {
        if (!dimSet.has(i)) spatialDims.push(i);
      }

      const outIndicesFor = (nest) => {
        if (!keepDims) return nest.indices;
        const idx = new Array(inBuf.shape.length);
        for (let i = 0; i < spatialDims.length; i++) idx[spatialDims[i]] = nest.indices[i];
        idx[reduceDim] = new IntImmNode(0);
        return idx;
      };

      const bestValBuf = new Buffer('_argval_' + ctx.varCounter, spatialDims.map(d => inBuf.shape[d]), inBuf.dtype, 'global');
      ctx.varCounter++;

      const initNest = buildSpatialNest(ctx, 'ai', spatialDims, inBuf.shape, inBuf);
      const initValStore = new BufferStoreNode(bestValBuf, initNest.indices, argReduceSentinel(inBuf.dtype, compareFn === 'gt'));
      const initIdxStore = new BufferStoreNode(outBuf, outIndicesFor(initNest), new IntImmNode(0));
      const initBlock = new BlockNode(ctx.blockName('arg_init'), initNest.ivs, [], [{ buffer: bestValBuf }, { buffer: outBuf }], new SeqNode([initValStore, initIdxStore]));
      const initBody = spatialDims.length > 0 ? initNest.wrap(initBlock) : initBlock;

      const accNest = buildSpatialNest(ctx, 'as', spatialDims, inBuf.shape, inBuf);
      const rVar = ctx.allocVar('ar');
      const rBind = ctx.allocBindArray('arv', [rVar]);
      const inIndices = new Array(inBuf.shape.length);
      for (let i = 0; i < spatialDims.length; i++) inIndices[spatialDims[i]] = accNest.ivs[i].iterVar;
      inIndices[reduceDim] = rBind[0].iterVar;

      const loadVal = new BufferLoadNode(inBuf, inIndices);
      const loadBest = new BufferLoadNode(bestValBuf, accNest.indices);
      const isBetter = new CompareNode(compareFn, loadVal, loadBest);
      const newBest = new IfThenElseNode(isBetter, loadVal, loadBest);
      const outAccIndices = outIndicesFor(accNest);
      const loadIdx = new BufferLoadNode(outBuf, outAccIndices);
      const newIdx = new IfThenElseNode(isBetter, rBind[0].iterVar, loadIdx);
      const storeIdx = new BufferStoreNode(outBuf, outAccIndices, newIdx);
      const storeVal = new BufferStoreNode(bestValBuf, accNest.indices, newBest);
      const accBlock = new BlockNode(ctx.blockName('arg_acc'), concatIterVars(accNest.ivs, rBind),
        [{ buffer: inBuf }, { buffer: bestValBuf }], [{ buffer: bestValBuf }, { buffer: outBuf }],
        new SeqNode([storeIdx, storeVal]));
      const rExtent = ctx.extentNode(inBuf.shape[reduceDim], inBuf, reduceDim);
      let accBody = wrapLoopsWithNodes(accBlock, [rVar], [rExtent]);
      accBody = accNest.wrap(accBody);

      return new SeqNode([initBody, accBody]);
    });
  }

  registerArgReduce('argmax', 'gt');
  registerArgReduce('argmin', 'lt');
}
