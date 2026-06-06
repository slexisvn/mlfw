import {
  IntImmNode, FloatImmNode, MathOpNode, CastNode,
  ForNode, ForKind, BufferStoreNode, BufferLoadNode,
  BlockNode, CallExternNode
} from '../../../ir/tensor/nodes.js';
import { registerLoweringRule, buildSpatialNest } from '../lowering_registry.js';

export function register() {
  registerLoweringRule('resize', (ctx, op, inputs, outputs) => {
    const inBuf = inputs[0];
    const outBuf = outputs[0];
    const method = op.getAttr('method');
    const inH = inBuf.shape[2];
    const inW = inBuf.shape[3];
    const outH = outBuf.shape[2];
    const outW = outBuf.shape[3];

    const nest = buildSpatialNest(ctx, 'rz', [0, 1, 2, 3], outBuf.shape, outBuf);
    const [nv, cv, ohv, owv] = nest.indices;

    const scaleH = inH / outH;
    const scaleW = inW / outW;

    let loadExpr;
    if (method === 'nearest') {
      const srcH = new CallExternNode('floor', [new MathOpNode('*', ohv, new FloatImmNode(scaleH))], 'f32');
      const srcW = new CallExternNode('floor', [new MathOpNode('*', owv, new FloatImmNode(scaleW))], 'f32');
      const clampH = new CastNode(new CallExternNode('min', [new CallExternNode('max', [srcH, new FloatImmNode(0)], 'f32'), new FloatImmNode(inH - 1)], 'f32'), 'f32', 'i32');
      const clampW = new CastNode(new CallExternNode('min', [new CallExternNode('max', [srcW, new FloatImmNode(0)], 'f32'), new FloatImmNode(inW - 1)], 'f32'), 'f32', 'i32');
      loadExpr = new BufferLoadNode(inBuf, [nv, cv, clampH, clampW]);
    } else {
      const srcHf = new MathOpNode('*', ohv, new FloatImmNode(scaleH));
      const srcWf = new MathOpNode('*', owv, new FloatImmNode(scaleW));
      const h0 = new CallExternNode('floor', [srcHf], 'f32');
      const w0 = new CallExternNode('floor', [srcWf], 'f32');
      const h1 = new CallExternNode('min', [new MathOpNode('+', h0, new FloatImmNode(1)), new FloatImmNode(inH - 1)], 'f32');
      const w1 = new CallExternNode('min', [new MathOpNode('+', w0, new FloatImmNode(1)), new FloatImmNode(inW - 1)], 'f32');
      const hFrac = new MathOpNode('-', srcHf, h0);
      const wFrac = new MathOpNode('-', srcWf, w0);
      const clH0 = new CallExternNode('max', [h0, new FloatImmNode(0)], 'f32');
      const clW0 = new CallExternNode('max', [w0, new FloatImmNode(0)], 'f32');
      const tl = new BufferLoadNode(inBuf, [nv, cv, clH0, clW0]);
      const tr = new BufferLoadNode(inBuf, [nv, cv, clH0, w1]);
      const bl = new BufferLoadNode(inBuf, [nv, cv, h1, clW0]);
      const br = new BufferLoadNode(inBuf, [nv, cv, h1, w1]);
      const oneMinusW = new MathOpNode('-', new FloatImmNode(1), wFrac);
      const oneMinusH = new MathOpNode('-', new FloatImmNode(1), hFrac);
      const top = new MathOpNode('+', new MathOpNode('*', tl, oneMinusW), new MathOpNode('*', tr, wFrac));
      const bot = new MathOpNode('+', new MathOpNode('*', bl, oneMinusW), new MathOpNode('*', br, wFrac));
      loadExpr = new MathOpNode('+', new MathOpNode('*', top, oneMinusH), new MathOpNode('*', bot, hFrac));
    }

    const store = new BufferStoreNode(outBuf, nest.indices, loadExpr);
    const block = new BlockNode(ctx.blockName('resize_block'), nest.ivs, [{ buffer: inBuf }], [{ buffer: outBuf }], store);
    return nest.wrap(block);
  });
}
