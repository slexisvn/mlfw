import {
  IntImmNode, FloatImmNode, MathOpNode, CompareNode,
  ForNode, ForKind, BufferStoreNode, BufferLoadNode,
  BlockNode, SeqNode, IfThenElseNode, CallExternNode, CastNode
} from '../../../ir/tensor/nodes.js';
import { registerLoweringRule, buildSpatialNest, bufRefs, parseLayout } from '../lowering_registry.js';

function spatialCount(ov, stride, kernel, pad, inExtent) {
  const start = new MathOpNode('-', new MathOpNode('*', ov, new IntImmNode(stride)), new IntImmNode(pad));
  const end = new MathOpNode('+', start, new IntImmNode(kernel));
  const lo = new CallExternNode('max', [start, new IntImmNode(0)], 'index');
  const hi = new CallExternNode('min', [end, new IntImmNode(inExtent)], 'index');
  const raw = new MathOpNode('-', hi, lo);
  return new CallExternNode('max', [raw, new IntImmNode(0)], 'index');
}

function avgPoolDivisorExpr(ohv, owv, sH, sW, kH, kW, padH, padW, inH, inW, outDtype) {
  const hCount = spatialCount(ohv, sH, kH, padH, inH);
  const wCount = spatialCount(owv, sW, kW, padW, inW);
  const countI = new MathOpNode('*', hCount, wCount);
  return new CastNode(countI, 'index', outDtype);
}

export function register() {
  registerLoweringRule('pool2d', (ctx, op, inputs, outputs) => {
    const inBuf = inputs[0];
    const outBuf = outputs[0];
    const poolType = op.getAttr('pool_type');
    const kernelSize = op.getAttr('kernel_size');
    const strides = op.getAttr('strides');
    const padding = op.getAttr('padding');
    const countIncludePad = op.getAttr('count_include_pad') || false;
    const layout = parseLayout(op.getAttr('layout') || 'NCHW');
    const nAx = layout['N'];
    const cAx = layout['C'];
    const hAx = layout['H'];
    const wAx = layout['W'];

    const batch = inBuf.shape[nAx];
    const channels = inBuf.shape[cAx];
    const inH = inBuf.shape[hAx];
    const inW = inBuf.shape[wAx];
    const outH = outBuf.shape[hAx];
    const outW = outBuf.shape[wAx];
    const kH = kernelSize[0];
    const kW = kernelSize[1];
    const sH = strides[0];
    const sW = strides[1];
    const padH = padding[0][0];
    const padW = padding[1][0];
    const isMax = poolType === 'max';
    const initVal = isMax ? new FloatImmNode(-Infinity) : new FloatImmNode(0);

    const initNest = buildSpatialNest(ctx, 'pi', [0, 1, 2, 3], outBuf.shape, outBuf);
    const initStore = new BufferStoreNode(outBuf, initNest.indices, initVal);
    const initBlock = new BlockNode(ctx.blockName('pool_init'), initNest.ivs, [], [{ buffer: outBuf }], initStore);
    const initBody = initNest.wrap(initBlock);

    const nVar = ctx.allocVar('pn');
    const cVar = ctx.allocVar('pc');
    const ohVar = ctx.allocVar('poh');
    const owVar = ctx.allocVar('pow');
    const khVar = ctx.allocVar('pkh');
    const kwVar = ctx.allocVar('pkw');
    const allVars = [nVar, cVar, ohVar, owVar, khVar, kwVar];
    const allBinds = ctx.allocBindArray('pv', allVars);

    const nv = allBinds[0].iterVar;
    const cv = allBinds[1].iterVar;
    const ohv = allBinds[2].iterVar;
    const owv = allBinds[3].iterVar;
    const khv = allBinds[4].iterVar;
    const kwv = allBinds[5].iterVar;

    const ihExpr = new MathOpNode('+', new MathOpNode('*', ohv, new IntImmNode(sH)), new MathOpNode('-', khv, new IntImmNode(padH)));
    const iwExpr = new MathOpNode('+', new MathOpNode('*', owv, new IntImmNode(sW)), new MathOpNode('-', kwv, new IntImmNode(padW)));

    const geH = new CompareNode('ge', ihExpr, new IntImmNode(0));
    const ltH = new CompareNode('lt', ihExpr, new IntImmNode(inH));
    const geW = new CompareNode('ge', iwExpr, new IntImmNode(0));
    const ltW = new CompareNode('lt', iwExpr, new IntImmNode(inW));
    const inBounds = new MathOpNode('*', new MathOpNode('*', geH, ltH), new MathOpNode('*', geW, ltW));

    const outIdx = new Array(4);
    outIdx[nAx] = nv; outIdx[cAx] = cv; outIdx[hAx] = ohv; outIdx[wAx] = owv;
    const inIdx = new Array(4);
    inIdx[nAx] = nv; inIdx[cAx] = cv; inIdx[hAx] = ihExpr; inIdx[wAx] = iwExpr;
    const loadIn = new BufferLoadNode(inBuf, inIdx);
    const loadOut = new BufferLoadNode(outBuf, outIdx);

    let accExpr;
    if (isMax) {
      const guarded = new IfThenElseNode(inBounds, loadIn, new FloatImmNode(-Infinity));
      accExpr = new CallExternNode('max', [loadOut, guarded], outBuf.dtype);
    } else {
      const guarded = new IfThenElseNode(inBounds, loadIn, new FloatImmNode(0));
      accExpr = new MathOpNode('+', loadOut, guarded);
    }

    const accStore = new BufferStoreNode(outBuf, outIdx, accExpr);
    const accBlock = new BlockNode(ctx.blockName('pool_acc'), allBinds, [{ buffer: inBuf }], [{ buffer: outBuf }], accStore);

    let accBody = accBlock;
    accBody = new ForNode(kwVar, new IntImmNode(0), new IntImmNode(kW), ForKind.SERIAL, accBody);
    accBody = new ForNode(khVar, new IntImmNode(0), new IntImmNode(kH), ForKind.SERIAL, accBody);
    accBody = new ForNode(owVar, new IntImmNode(0), new IntImmNode(outW), ForKind.SERIAL, accBody);
    accBody = new ForNode(ohVar, new IntImmNode(0), new IntImmNode(outH), ForKind.SERIAL, accBody);
    accBody = new ForNode(cVar, new IntImmNode(0), new IntImmNode(channels), ForKind.SERIAL, accBody);
    accBody = new ForNode(nVar, new IntImmNode(0), new IntImmNode(batch), ForKind.SERIAL, accBody);

    const parts = [initBody, accBody];

    if (!isMax) {
      const divNest = buildSpatialNest(ctx, 'pd', [0, 1, 2, 3], outBuf.shape, outBuf);
      const dOh = divNest.indices[hAx];
      const dOw = divNest.indices[wAx];
      const divisor = countIncludePad
        ? new FloatImmNode(kH * kW)
        : avgPoolDivisorExpr(dOh, dOw, sH, sW, kH, kW, padH, padW, inH, inW, outBuf.dtype);
      const loaded = new BufferLoadNode(outBuf, divNest.indices);
      const divExpr = countIncludePad
        ? new MathOpNode('*', loaded, new FloatImmNode(1.0 / (kH * kW)))
        : new MathOpNode('/', loaded, divisor);
      const divStore = new BufferStoreNode(outBuf, divNest.indices, divExpr);
      const divBlock = new BlockNode(ctx.blockName('pool_div'), divNest.ivs, [{ buffer: outBuf }], [{ buffer: outBuf }], divStore);
      parts.push(divNest.wrap(divBlock));
    }

    return new SeqNode(parts);
  });
}
