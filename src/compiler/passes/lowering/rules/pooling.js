import {
  IntImmNode, FloatImmNode, MathOpNode, CompareNode,
  ForNode, ForKind, BufferStoreNode, BufferLoadNode,
  BlockNode, SeqNode, IfThenElseNode, CallExternNode
} from '../../../ir/tensor/nodes.js';
import { registerLoweringRule, buildSpatialNest, bufRefs } from '../lowering_registry.js';

export function register() {
  registerLoweringRule('pool2d', (ctx, op, inputs, outputs) => {
    const inBuf = inputs[0];
    const outBuf = outputs[0];
    const poolType = op.getAttr('pool_type');
    const kernelSize = op.getAttr('kernel_size');
    const strides = op.getAttr('strides');
    const padding = op.getAttr('padding');
    const countIncludePad = op.getAttr('count_include_pad') || false;

    const batch = inBuf.shape[0];
    const channels = inBuf.shape[1];
    const inH = inBuf.shape[2];
    const inW = inBuf.shape[3];
    const outH = outBuf.shape[2];
    const outW = outBuf.shape[3];
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
    const initBlock = new BlockNode('pool_init', initNest.ivs, [], [{ buffer: outBuf }], initStore);
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

    const outIdx = [nv, cv, ohv, owv];
    const inIdx = [nv, cv, ihExpr, iwExpr];
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
    const accBlock = new BlockNode('pool_acc', allBinds, [{ buffer: inBuf }], [{ buffer: outBuf }], accStore);

    let accBody = accBlock;
    accBody = new ForNode(kwVar, new IntImmNode(0), new IntImmNode(kW), ForKind.SERIAL, accBody);
    accBody = new ForNode(khVar, new IntImmNode(0), new IntImmNode(kH), ForKind.SERIAL, accBody);
    accBody = new ForNode(owVar, new IntImmNode(0), new IntImmNode(outW), ForKind.SERIAL, accBody);
    accBody = new ForNode(ohVar, new IntImmNode(0), new IntImmNode(outH), ForKind.SERIAL, accBody);
    accBody = new ForNode(cVar, new IntImmNode(0), new IntImmNode(channels), ForKind.SERIAL, accBody);
    accBody = new ForNode(nVar, new IntImmNode(0), new IntImmNode(batch), ForKind.SERIAL, accBody);

    const parts = [initBody, accBody];

    if (!isMax) {
      const divVal = countIncludePad ? kH * kW : kH * kW;
      const divNest = buildSpatialNest(ctx, 'pd', [0, 1, 2, 3], outBuf.shape, outBuf);
      const divExpr = new MathOpNode('*', new BufferLoadNode(outBuf, divNest.indices), new FloatImmNode(1.0 / divVal));
      const divStore = new BufferStoreNode(outBuf, divNest.indices, divExpr);
      const divBlock = new BlockNode('pool_div', divNest.ivs, [{ buffer: outBuf }], [{ buffer: outBuf }], divStore);
      parts.push(divNest.wrap(divBlock));
    }

    return new SeqNode(parts);
  });
}
