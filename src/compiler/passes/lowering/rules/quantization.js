import {
  IntImmNode, FloatImmNode, MathOpNode, CompareNode,
  ForNode, ForKind, BufferStoreNode, BufferLoadNode,
  BlockNode, SeqNode, IfThenElseNode, CastNode, CallExternNode
} from '../../../ir/tensor/nodes.js';
import {
  registerLoweringRule, lowerPointwise, buildSpatialNest,
  buildDotGeometry, parseLayout
} from '../lowering_registry.js';

export function register() {
  registerLoweringRule('quantize', (ctx, op, inputs, outputs) => {
    const scale = op.getAttr('scale');
    const zeroPoint = op.getAttr('zero_point');
    const tgtDtype = op.getAttr('target_dtype') || 'i8';
    const isUnsigned = tgtDtype === 'ui8';
    const bits = 8;
    const cMin = isUnsigned ? 0 : -(1 << (bits - 1));
    const cMax = isUnsigned ? (1 << bits) - 1 : (1 << (bits - 1)) - 1;

    return lowerPointwise(ctx, op, inputs, outputs, (_op, loads) => {
      const scaled = new MathOpNode('/', loads[0], new FloatImmNode(scale));
      const shifted = new MathOpNode('+', scaled, new FloatImmNode(zeroPoint));
      const rounded = new CallExternNode('round', [shifted], 'f32');
      const clamped = new CallExternNode('min', [
        new CallExternNode('max', [rounded, new FloatImmNode(cMin)], 'f32'),
        new FloatImmNode(cMax)
      ], 'f32');
      return new CastNode(clamped, 'f32', tgtDtype);
    });
  });

  registerLoweringRule('dequantize', (ctx, op, inputs, outputs) => {
    const scale = op.getAttr('scale');
    const zeroPoint = op.getAttr('zero_point');
    const srcDtype = inputs[0].dtype || 'i8';
    const tgtDtype = op.getAttr('target_dtype') || 'f32';

    return lowerPointwise(ctx, op, inputs, outputs, (_op, loads) => {
      const asFloat = new CastNode(loads[0], srcDtype, tgtDtype);
      const shifted = new MathOpNode('-', asFloat, new FloatImmNode(zeroPoint));
      return new MathOpNode('*', shifted, new FloatImmNode(scale));
    });
  });

  registerLoweringRule('quantized_dot', (ctx, op, inputs, outputs) => {
    const lhs = inputs[0];
    const rhs = inputs[1];
    const out = outputs[0];
    const geo = buildDotGeometry(ctx, op, lhs, rhs);

    const initNest = buildSpatialNest(ctx, 'qdi', Array.from({ length: out.shape.length }, (_, i) => i), out.shape, out);
    const initStore = new BufferStoreNode(out, initNest.indices, new IntImmNode(0));
    const initBlock = new BlockNode(ctx.blockName('qmatmul_init'), initNest.ivs, [], [{ buffer: out }], initStore);
    const initBody = initNest.wrap(initBlock);

    const lhsLoad = new CastNode(new BufferLoadNode(lhs, geo.lhsIdx), lhs.dtype, 'i32');
    const rhsLoad = new CastNode(new BufferLoadNode(rhs, geo.rhsIdx), rhs.dtype, 'i32');
    const lhsZP = op.getAttr('lhs_zero_point') || 0;
    const rhsZP = op.getAttr('rhs_zero_point') || 0;
    const lhsSub = lhsZP !== 0 ? new MathOpNode('-', lhsLoad, new IntImmNode(lhsZP)) : lhsLoad;
    const rhsSub = rhsZP !== 0 ? new MathOpNode('-', rhsLoad, new IntImmNode(rhsZP)) : rhsLoad;
    const product = new MathOpNode('*', lhsSub, rhsSub);
    const accExpr = new MathOpNode('+', new BufferLoadNode(out, geo.outIdx), product);
    const accStore = new BufferStoreNode(out, geo.outIdx, accExpr);
    const accBlock = new BlockNode(ctx.blockName('qmatmul_acc'), geo.allIvs, [{ buffer: lhs }, { buffer: rhs }], [{ buffer: out }], accStore);
    const accBody = geo.wrapAccBody(accBlock);

    return new SeqNode([initBody, accBody]);
  });

  registerLoweringRule('quantized_conv', (ctx, op, inputs, outputs) => {
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
    const outChannels = kerBuf.shape[kLayout['O']];
    const inChannelsPerGroup = kerBuf.shape[kLayout['I']];
    const batch = inBuf.shape[iLayout['N']];
    const outShape = outBuf.shape;
    const inputZP = op.getAttr('input_zero_point') || 0;
    const kernelZP = op.getAttr('kernel_zero_point') || 0;

    const initNest = buildSpatialNest(ctx, 'qci', Array.from({ length: outShape.length }, (_, i) => i), outShape, outBuf);
    const initStore = new BufferStoreNode(outBuf, initNest.indices, new IntImmNode(0));
    const initBlock = new BlockNode(ctx.blockName('qconv_init'), initNest.ivs, [], [{ buffer: outBuf }], initStore);
    const initBody = initNest.wrap(initBlock);

    const nVar = ctx.allocVar('qcn');
    const ocVar = ctx.allocVar('qcoc');
    const icVar = ctx.allocVar('qcic');
    const spatialOutVars = ctx.allocVarArray('qco', spatialDims);
    const spatialKerVars = ctx.allocVarArray('qck', spatialDims);
    const allVars = [nVar, ocVar, ...spatialOutVars, icVar, ...spatialKerVars];
    const allBinds = ctx.allocBindArray('qcv', allVars);

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

    const loadIn = new CastNode(new BufferLoadNode(inBuf, inIdx), inBuf.dtype, 'i32');
    const loadKer = new CastNode(new BufferLoadNode(kerBuf, kerIdx), kerBuf.dtype, 'i32');
    const inSub = inputZP !== 0 ? new MathOpNode('-', loadIn, new IntImmNode(inputZP)) : loadIn;
    const kerSub = kernelZP !== 0 ? new MathOpNode('-', loadKer, new IntImmNode(kernelZP)) : loadKer;
    const product = new MathOpNode('*', inSub, kerSub);
    const guardedProduct = inBoundsExpr ? new IfThenElseNode(inBoundsExpr, product, new IntImmNode(0)) : product;
    const loadOut = new BufferLoadNode(outBuf, outIdx);
    const accExpr = new MathOpNode('+', loadOut, guardedProduct);
    const accStore = new BufferStoreNode(outBuf, outIdx, accExpr);
    const accBlock = new BlockNode(ctx.blockName('qconv_acc'), allBinds, [{ buffer: inBuf }, { buffer: kerBuf }], [{ buffer: outBuf }], accStore);

    const kerSpatialSizes = new Array(spatialDims);
    for (let s = 0; s < spatialDims; s++) {
      const kKey = spatialLayoutKeys[s].toUpperCase();
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
}
