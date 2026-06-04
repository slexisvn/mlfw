import {
  IntImmNode, FloatImmNode, MathOpNode, CompareNode,
  ForNode, ForKind, BufferStoreNode, BufferLoadNode,
  BlockNode, SeqNode, IfThenElseNode, CallExternNode
} from '../../../ir/tensor/nodes.js';
import {
  registerLoweringRule, buildSpatialNest, buildDotGeometry,
  parseLayout, bufRefs, computeBroadcastIndices, makeLoopNest, wrapInLoops
} from '../lowering_registry.js';

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

export function register() {
  registerLoweringRule('dot', (ctx, op, inputs, outputs) => {
    const lhs = inputs[0];
    const rhs = inputs[1];
    const out = outputs[0];
    const geo = buildDotGeometry(ctx, op, lhs, rhs);

    const initNest = buildSpatialNest(ctx, 'di', Array.from({ length: out.shape.length }, (_, i) => i), out.shape);
    const initStore = new BufferStoreNode(out, initNest.indices, new FloatImmNode(0));
    const initBlock = new BlockNode('matmul_init', initNest.ivs, [], [{ buffer: out }], initStore);
    const initBody = initNest.wrap(initBlock);

    const accExpr = new MathOpNode('+', new BufferLoadNode(out, geo.outIdx), new MathOpNode('*', new BufferLoadNode(lhs, geo.lhsIdx), new BufferLoadNode(rhs, geo.rhsIdx)));
    const accStore = new BufferStoreNode(out, geo.outIdx, accExpr);
    const accBlock = new BlockNode('matmul', geo.allIvs, [{ buffer: lhs }, { buffer: rhs }], [{ buffer: out }], accStore);
    const accBody = geo.wrapAccBody(accBlock);

    return new SeqNode([initBody, accBody]);
  });

  registerLoweringRule('conv', (ctx, op, inputs, outputs) => {
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
    const initBlock = new BlockNode('conv_init', initNest.ivs, [], [{ buffer: outBuf }], initStore);
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
    const accBlock = new BlockNode('conv_acc', allBinds, [{ buffer: inBuf }, { buffer: kerBuf }], [{ buffer: outBuf }], accStore);

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

  registerLoweringRule('gather', (ctx, op, inputs, outputs) => {
    const operandBuf = inputs[0];
    const indicesBuf = inputs[1];
    const outBuf = outputs[0];
    const offsetDims = new Set(op.getAttr('offset_dims'));
    const collapsedDims = new Set(op.getAttr('collapsed_slice_dims'));
    const startIndexMap = op.getAttr('start_index_map');
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
    const block = new BlockNode('gather_block', loopBinds, [{ buffer: operandBuf }, { buffer: indicesBuf }], [{ buffer: outBuf }], store);
    return wrapInLoops(block, loopVars, outBuf.shape);
  });

  registerLoweringRule('fused_dot_epilogue', (ctx, op, inputs, outputs) => {
    const numDotOperands = op.getAttr('num_dot_operands') || 2;
    const lhs = inputs[0];
    const rhs = inputs[1];
    const extraInputs = inputs.slice(numDotOperands);
    const out = outputs[0];
    const epilogueTags = op.getAttr('epilogue_tags') || [];
    const geo = buildDotGeometry(ctx, op, lhs, rhs);

    const initNest = buildSpatialNest(ctx, 'ei', Array.from({ length: out.shape.length }, (_, i) => i), out.shape);
    const initStore = new BufferStoreNode(out, initNest.indices, new FloatImmNode(0));
    const initBlock = new BlockNode('matmul_init', initNest.ivs, [], [{ buffer: out }], initStore);
    const initBody = initNest.wrap(initBlock);

    const accExpr = new MathOpNode('+', new BufferLoadNode(out, geo.outIdx), new MathOpNode('*', new BufferLoadNode(lhs, geo.lhsIdx), new BufferLoadNode(rhs, geo.rhsIdx)));
    const accStore = new BufferStoreNode(out, geo.outIdx, accExpr);
    const accBlock = new BlockNode('matmul_acc', geo.allIvs, [{ buffer: lhs }, { buffer: rhs }], [{ buffer: out }], accStore);
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
    const epiBlock = new BlockNode('epilogue', epiNest.ivs, epiReads, [{ buffer: out }], epiStore);
    const epiBody = epiNest.wrap(epiBlock);

    return new SeqNode([initBody, accBody, epiBody]);
  });
}
