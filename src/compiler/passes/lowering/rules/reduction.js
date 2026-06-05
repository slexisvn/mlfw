import {
  MathOpNode, FloatImmNode, IntImmNode, BufferStoreNode, BufferLoadNode,
  BlockNode, SeqNode, CallExternNode, ForNode, ForKind, IfThenElseNode, CompareNode
} from '../../../ir/tensor/nodes.js';
import { Buffer } from '../../../ir/tensor/buffer.js';
import {
  registerLoweringRule, buildSpatialNest, wrapLoopsWithNodes,
  concatIterVars, bufRefs
} from '../lowering_registry.js';

const REDUCE_COMBINERS = {
  'sum':  (a, b) => new MathOpNode('+', a, b),
  'mean': (a, b) => new MathOpNode('+', a, b),
  'prod': (a, b) => new MathOpNode('*', a, b),
  'max':  (a, b, dt) => new CallExternNode('max', [a, b], dt),
  'min':  (a, b, dt) => new CallExternNode('min', [a, b], dt)
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
    const initBlock = new BlockNode('reduce_init', initNest.ivs, [{ buffer: initBuf }], [{ buffer: outBuf }], initStore);
    const initBody = spatialDims.length > 0 ? initNest.wrap(initBlock) : initBlock;

    const accNest = buildSpatialNest(ctx, 'sa', spatialDims, inBuf.shape, inBuf);
    const rVars = ctx.allocVarArray('r', reduceDims.length);
    const rIvs = ctx.allocBindArray('rv', rVars);
    const inIndices = new Array(inBuf.shape.length);
    for (let i = 0; i < spatialDims.length; i++) inIndices[spatialDims[i]] = accNest.ivs[i].iterVar;
    for (let i = 0; i < reduceDims.length; i++) inIndices[reduceDims[i]] = rIvs[i].iterVar;
    const loadA = new BufferLoadNode(outBuf, accNest.indices);
    const loadB = new BufferLoadNode(inBuf, inIndices);
    const combiner = REDUCE_COMBINERS[rType] || REDUCE_COMBINERS['sum'];
    const store = new BufferStoreNode(outBuf, accNest.indices, combiner(loadA, loadB, outBuf.dtype));
    const rExtentNodes = new Array(reduceDims.length);
    for (let i = 0; i < reduceDims.length; i++) rExtentNodes[i] = ctx.extentNode(inBuf.shape[reduceDims[i]], inBuf, reduceDims[i]);
    const accBlock = new BlockNode('reduce_acc', concatIterVars(accNest.ivs, rIvs), [{ buffer: inBuf }], [{ buffer: outBuf }], store);
    let accBody = wrapLoopsWithNodes(accBlock, rVars, rExtentNodes);
    accBody = accNest.wrap(accBody);

    const parts = [initBody, accBody];

    if (rType === 'mean') {
      let reduceSize = 1;
      for (let i = 0; i < reduceDims.length; i++) reduceSize *= inBuf.shape[reduceDims[i]];
      const meanNest = buildSpatialNest(ctx, 'sm', spatialDims, inBuf.shape, inBuf);
      const divExpr = new MathOpNode('*', new BufferLoadNode(outBuf, meanNest.indices), new FloatImmNode(1.0 / reduceSize));
      const meanStore = new BufferStoreNode(outBuf, meanNest.indices, divExpr);
      const meanBlock = new BlockNode('mean_div', meanNest.ivs, [{ buffer: outBuf }], [{ buffer: outBuf }], meanStore);
      parts.push(spatialDims.length > 0 ? meanNest.wrap(meanBlock) : meanBlock);
    }

    return new SeqNode(parts);
  });

  function registerArgReduce(opName, compareFn) {
    registerLoweringRule(opName, (ctx, op, inputs, outputs) => {
      const inBuf = inputs[0];
      const outBuf = outputs[0];
      const axis = op.getAttr('axis');
      const dimSet = new Set([axis]);
      const spatialDims = [];
      const reduceDim = axis;
      for (let i = 0; i < inBuf.shape.length; i++) {
        if (!dimSet.has(i)) spatialDims.push(i);
      }

      const bestValBuf = new Buffer('_argval_' + ctx.varCounter, spatialDims.map(d => inBuf.shape[d]), inBuf.dtype, 'global');
      ctx.varCounter++;

      const initNest = buildSpatialNest(ctx, 'ai', spatialDims, inBuf.shape, inBuf);
      const initValStore = new BufferStoreNode(bestValBuf, initNest.indices, new FloatImmNode(compareFn === 'gt' ? -Infinity : Infinity));
      const initIdxStore = new BufferStoreNode(outBuf, initNest.indices, new IntImmNode(0));
      const initBlock = new BlockNode('arg_init', initNest.ivs, [], [{ buffer: bestValBuf }, { buffer: outBuf }], new SeqNode([initValStore, initIdxStore]));
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
      const loadIdx = new BufferLoadNode(outBuf, accNest.indices);
      const newIdx = new IfThenElseNode(isBetter, rBind[0].iterVar, loadIdx);
      const storeIdx = new BufferStoreNode(outBuf, accNest.indices, newIdx);
      const storeVal = new BufferStoreNode(bestValBuf, accNest.indices, newBest);
      const accBlock = new BlockNode('arg_acc', concatIterVars(accNest.ivs, rBind),
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
