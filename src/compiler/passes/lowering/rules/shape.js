import {
  IntImmNode, MathOpNode, BufferStoreNode, BufferLoadNode,
  BlockNode, SeqNode, CastNode, CompareNode, IfThenElseNode, mathOp
} from '../../../ir/tensor/nodes.js';
import {
  registerLoweringRule, makeLoopNest, wrapInLoops
} from '../lowering_registry.js';

export function register() {
  registerLoweringRule('broadcast_in_dim', lowerBroadcast);
  registerLoweringRule('broadcast', lowerBroadcast);

  registerLoweringRule('transpose', (ctx, op, inputs, outputs) => {
    const perm = op.getAttr('permutation');
    const inBuf = inputs[0];
    const outBuf = outputs[0];
    const { loopVars, loopBinds, indices: outIndices, extentNodes } = makeLoopNest(ctx, outBuf.shape, outBuf);
    const inIndices = new Array(inBuf.shape.length);
    for (let i = 0; i < perm.length; i++) inIndices[perm[i]] = outIndices[i];
    const store = new BufferStoreNode(outBuf, outIndices, new BufferLoadNode(inBuf, inIndices));
    const block = new BlockNode(ctx.blockName('transpose_block'), loopBinds, [{ buffer: inBuf }], [{ buffer: outBuf }], store);
    return wrapInLoops(block, loopVars, outBuf.shape, extentNodes);
  });

  registerLoweringRule('reverse', (ctx, op, inputs, outputs) => {
    const dims = new Set(op.getAttr('dimensions'));
    const inBuf = inputs[0];
    const outBuf = outputs[0];
    const { loopVars, loopBinds, indices: outIndices, extentNodes } = makeLoopNest(ctx, outBuf.shape, outBuf);
    const inIndices = new Array(inBuf.shape.length);
    for (let i = 0; i < inBuf.shape.length; i++) {
      inIndices[i] = dims.has(i)
        ? new MathOpNode('-', new MathOpNode('-', extentNodes[i], new IntImmNode(1)), outIndices[i])
        : outIndices[i];
    }
    const store = new BufferStoreNode(outBuf, outIndices, new BufferLoadNode(inBuf, inIndices));
    const block = new BlockNode(ctx.blockName('reverse_block'), loopBinds, [{ buffer: inBuf }], [{ buffer: outBuf }], store);
    return wrapInLoops(block, loopVars, outBuf.shape, extentNodes);
  });

  registerLoweringRule('reshape', (ctx, op, inputs, outputs) => {
    const inBuf = inputs[0];
    const outBuf = outputs[0];
    const { loopVars, loopBinds, indices: outIndices, extentNodes } = makeLoopNest(ctx, outBuf.shape, outBuf);

    let inIndices;
    const isIdentity = inBuf.shape.length === outBuf.shape.length &&
      inBuf.shape.every((d, i) => d === outBuf.shape[i]);

    if (isIdentity) {
      inIndices = outIndices;
    } else {
      for (let i = 1; i < inBuf.shape.length; i++) if (typeof inBuf.shape[i] !== 'number' || inBuf.shape[i] < 0) throw new Error('reshape lowering requires static non-leading input dims');
      for (let i = 1; i < outBuf.shape.length; i++) if (typeof outBuf.shape[i] !== 'number' || outBuf.shape[i] < 0) throw new Error('reshape lowering requires static non-leading output dims');
      let flatIndex = outIndices[outBuf.shape.length - 1];
      let stride = 1;
      for (let i = outBuf.shape.length - 2; i >= 0; i--) {
        stride *= outBuf.shape[i + 1];
        flatIndex = mathOp('+', flatIndex, mathOp('*', outIndices[i], new IntImmNode(stride)));
      }
      inIndices = new Array(inBuf.shape.length);
      let cur = flatIndex;
      for (let i = inBuf.shape.length - 1; i >= 0; i--) {
        if (i === 0) { inIndices[i] = cur; }
        else {
          inIndices[i] = mathOp('%', cur, new IntImmNode(inBuf.shape[i]));
          cur = mathOp('//', cur, new IntImmNode(inBuf.shape[i]));
        }
      }
    }

    const store = new BufferStoreNode(outBuf, outIndices, new BufferLoadNode(inBuf, inIndices));
    const block = new BlockNode(ctx.blockName('reshape_block'), loopBinds, [{ buffer: inBuf }], [{ buffer: outBuf }], store);
    return wrapInLoops(block, loopVars, outBuf.shape, extentNodes);
  });

  registerLoweringRule('slice', (ctx, op, inputs, outputs) => {
    const inBuf = inputs[0];
    const outBuf = outputs[0];
    const starts = op.getAttr('starts');
    const strides = op.getAttr('strides') || starts.map(() => 1);
    const { loopVars, loopBinds, indices: outIndices, extentNodes } = makeLoopNest(ctx, outBuf.shape, outBuf);
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
    const block = new BlockNode(ctx.blockName('slice_block'), loopBinds, [{ buffer: inBuf }], [{ buffer: outBuf }], store);
    return wrapInLoops(block, loopVars, outBuf.shape, extentNodes);
  });

  registerLoweringRule('pad', (ctx, op, inputs, outputs) => {
    const inBuf = inputs[0];
    const padVal = inputs[1];
    const outBuf = outputs[0];
    const low = op.getAttr('low');
    const interior = op.getAttr('interior') || low.map(() => 0);
    const { loopVars, loopBinds, indices: outIndices, extentNodes } = makeLoopNest(ctx, outBuf.shape, outBuf);
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
    const block = new BlockNode(ctx.blockName('pad_block'), loopBinds, [{ buffer: inBuf }, { buffer: padVal }], [{ buffer: outBuf }], store);
    return wrapInLoops(block, loopVars, outBuf.shape, extentNodes);
  });

  registerLoweringRule('concat', (ctx, op, inputs, outputs) => {
    const outBuf = outputs[0];
    const dim = op.getAttr('dimension');
    const stmts = [];
    let offset = 0;
    for (let k = 0; k < inputs.length; k++) {
      const inBuf = inputs[k];
      const { loopVars, loopBinds, indices: inIndices, extentNodes } = makeLoopNest(ctx, inBuf.shape, inBuf);
      const outIndices = new Array(inBuf.shape.length);
      for (let d = 0; d < inBuf.shape.length; d++) {
        outIndices[d] = d === dim && offset > 0
          ? new MathOpNode('+', inIndices[d], new IntImmNode(offset))
          : inIndices[d];
      }
      const store = new BufferStoreNode(outBuf, outIndices, new BufferLoadNode(inBuf, inIndices));
      const block = new BlockNode(ctx.blockName('concat'), loopBinds, [{ buffer: inBuf }], [{ buffer: outBuf }], store);
      stmts.push(wrapInLoops(block, loopVars, inBuf.shape, extentNodes));
      offset += inBuf.shape[dim];
    }
    return stmts.length === 1 ? stmts[0] : new SeqNode(stmts);
  });

  registerLoweringRule('iota', (ctx, op, inputs, outputs) => {
    const outBuf = outputs[0];
    const iotaDim = op.getAttr('iota_dimension');
    const { loopVars, loopBinds, indices, extentNodes } = makeLoopNest(ctx, outBuf.shape, outBuf);
    const val = new CastNode(indices[iotaDim], 'index', outBuf.dtype);
    const store = new BufferStoreNode(outBuf, indices, val);
    const block = new BlockNode(ctx.blockName('iota_block'), loopBinds, [], [{ buffer: outBuf }], store);
    return wrapInLoops(block, loopVars, outBuf.shape, extentNodes);
  });
}

function lowerBroadcast(ctx, op, inputs, outputs) {
  const inBuf = inputs[0];
  const outBuf = outputs[0];
  const broadcastDims = op.getAttr('broadcast_dimensions') || [];
  const { loopVars, loopBinds, indices: outIndices, extentNodes } = makeLoopNest(ctx, outBuf.shape, outBuf);
  const inIndices = new Array(inBuf.shape.length);
  for (let i = 0; i < inBuf.shape.length; i++) {
    const mapped = broadcastDims.length > 0 ? broadcastDims[i] : i + (outBuf.shape.length - inBuf.shape.length);
    inIndices[i] = inBuf.shape[i] === 1 ? new IntImmNode(0) : outIndices[mapped];
  }
  const store = new BufferStoreNode(outBuf, outIndices, new BufferLoadNode(inBuf, inIndices));
  const block = new BlockNode(ctx.blockName('broadcast_block'), loopBinds, [{ buffer: inBuf }], [{ buffer: outBuf }], store);
  return wrapInLoops(block, loopVars, outBuf.shape, extentNodes);
}
