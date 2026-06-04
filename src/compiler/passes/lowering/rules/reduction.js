import {
  MathOpNode, FloatImmNode, BufferStoreNode, BufferLoadNode,
  BlockNode, SeqNode, CallExternNode
} from '../../../ir/tensor/nodes.js';
import {
  registerLoweringRule, buildSpatialNest, wrapLoops,
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

    const initNest = buildSpatialNest(ctx, 'si', spatialDims, inBuf.shape);
    const initStore = new BufferStoreNode(outBuf, initNest.indices, new BufferLoadNode(initBuf, []));
    const initBlock = new BlockNode('reduce_init', initNest.ivs, [{ buffer: initBuf }], [{ buffer: outBuf }], initStore);
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
    const accBlock = new BlockNode('reduce_acc', concatIterVars(accNest.ivs, rIvs), [{ buffer: inBuf }], [{ buffer: outBuf }], store);
    let accBody = wrapLoops(accBlock, rVars, rExtents);
    accBody = accNest.wrap(accBody);

    const parts = [initBody, accBody];

    if (rType === 'mean') {
      let reduceSize = 1;
      for (let i = 0; i < reduceDims.length; i++) reduceSize *= inBuf.shape[reduceDims[i]];
      const meanNest = buildSpatialNest(ctx, 'sm', spatialDims, inBuf.shape);
      const divExpr = new MathOpNode('*', new BufferLoadNode(outBuf, meanNest.indices), new FloatImmNode(1.0 / reduceSize));
      const meanStore = new BufferStoreNode(outBuf, meanNest.indices, divExpr);
      const meanBlock = new BlockNode('mean_div', meanNest.ivs, [{ buffer: outBuf }], [{ buffer: outBuf }], meanStore);
      parts.push(spatialDims.length > 0 ? meanNest.wrap(meanBlock) : meanBlock);
    }

    return new SeqNode(parts);
  });
}
