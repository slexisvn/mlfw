import {
  IntImmNode, FloatImmNode, MathOpNode, CompareNode,
  ForNode, ForKind, BufferStoreNode, BufferLoadNode,
  BlockNode, SeqNode, IfThenElseNode, CallExternNode, CastNode, mathOp
} from '../../../ir/tensor/nodes.js';
import { ScalarType, isFloatType } from '../../../ir/graph/types.js';
import type { ScalarDType } from '../../../ir/graph/types.js';
import {
  registerLoweringRule, getLoweringRule, buildSpatialNest, buildDotGeometry,
  parseLayout, bufRefs, computeBroadcastIndices, makeLoopNest, wrapInLoops, buildConvNest, emitMatmulInitAcc
} from '../lowering_registry.js';
import type { LoweringRuleFn } from '../lowering_registry.js';
import type { TirNode } from '../../../ir/tensor/nodes.js';
import type { Buffer } from '../../../ir/tensor/buffer.js';

type EpilogueCursor = { v: number };
type EpilogueTagLowerer = (expr: TirNode, extraInputs: readonly Buffer[], idx: EpilogueCursor, out: Buffer, epiIdx: readonly TirNode[]) => TirNode;

function asIndexValue(load: TirNode, dtype: string): TirNode {
  return isFloatType(dtype as ScalarDType) ? new CastNode(load, dtype, ScalarType.I32) : load;
}

const EPILOGUE_TAG_LOWERERS = new Map<string, EpilogueTagLowerer>();

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

EPILOGUE_TAG_LOWERERS.set('neg', (expr) => new MathOpNode('-', expr));

EPILOGUE_TAG_LOWERERS.set('exp', (expr, extraInputs, idx, out) => new CallExternNode('exp', [expr], out.dtype));

EPILOGUE_TAG_LOWERERS.set('tanh', (expr, extraInputs, idx, out) => new CallExternNode('tanh', [expr], out.dtype));

EPILOGUE_TAG_LOWERERS.set('sqrt', (expr, extraInputs, idx, out) => new CallExternNode('sqrt', [expr], out.dtype));

EPILOGUE_TAG_LOWERERS.set('abs', (expr, extraInputs, idx, out) => new CallExternNode('abs', [expr], out.dtype));

EPILOGUE_TAG_LOWERERS.set('log', (expr, extraInputs, idx, out) => new CallExternNode('log', [expr], out.dtype));

EPILOGUE_TAG_LOWERERS.set('activation', (expr) => expr);

export function register(): void {
  registerLoweringRule('dot', (ctx, op, inputs, outputs) => {
    const { initBody, accBody } = emitMatmulInitAcc(ctx, op, inputs[0], inputs[1], outputs[0], {
      prefix: 'di',
      initBlockName: 'matmul_init',
      accBlockName: 'matmul',
      initVal: () => new FloatImmNode(0),
      accLeaf: (a, b) => new MathOpNode('*', a, b),
    });
    return new SeqNode([initBody, accBody]);
  });

  registerLoweringRule('conv', (ctx, op, inputs, outputs) => {
    const inBuf = inputs[0];
    const kerBuf = inputs[1];
    return buildConvNest(ctx, op, inBuf, kerBuf, outputs[0], {
      prefix: 'c',
      blockPrefix: 'conv',
      initVal: () => new FloatImmNode(0),
      guardFill: () => new FloatImmNode(0),
      leafBuilder: (inIdx, kerIdx) => new MathOpNode('*', new BufferLoadNode(inBuf, inIdx), new BufferLoadNode(kerBuf, kerIdx)),
    });
  });

  registerLoweringRule('gather', (ctx, op, inputs, outputs) => {
    const operandBuf = inputs[0];
    const indicesBuf = inputs[1];
    const outBuf = outputs[0];
    const offsetDims = new Set(op.getAttr<readonly number[]>('offset_dims'));
    const collapsedDims = new Set(op.getAttr<readonly number[]>('collapsed_slice_dims'));
    const startIndexMap = op.getAttr<readonly number[]>('start_index_map') as readonly number[];
    const indexVectorDim = op.getAttr<number>('index_vector_dim');
    const { loopVars, loopBinds, indices: outIndices, extentNodes } = makeLoopNest(ctx, outBuf.shape, outBuf);

    const batchIndices: TirNode[] = [];
    const offsetIndices: TirNode[] = [];
    for (let i = 0; i < outBuf.shape.length; i++) {
      if (offsetDims.has(i)) offsetIndices.push(outIndices[i]);
      else batchIndices.push(outIndices[i]);
    }

    const operandIndices: TirNode[] = new Array(operandBuf.shape.length);
    let offsetIdx = 0;
    for (let i = 0; i < operandBuf.shape.length; i++) {
      if (collapsedDims.has(i)) {
        operandIndices[i] = new IntImmNode(0);
      } else {
        operandIndices[i] = offsetIndices[offsetIdx++];
      }
    }

    for (let k = 0; k < startIndexMap.length; k++) {
      const idxLookup: TirNode[] = new Array(indicesBuf.shape.length);
      let batchIdx = 0;
      for (let d = 0; d < indicesBuf.shape.length; d++) {
        if (d === indexVectorDim) idxLookup[d] = new IntImmNode(k);
        else idxLookup[d] = batchIndices[batchIdx++];
      }
      const startVal = asIndexValue(new BufferLoadNode(indicesBuf, idxLookup), indicesBuf.dtype);
      const targetDim = startIndexMap[k];
      operandIndices[targetDim] = new MathOpNode('+', operandIndices[targetDim], startVal);
    }

    const store = new BufferStoreNode(outBuf, outIndices, new BufferLoadNode(operandBuf, operandIndices));
    const block = new BlockNode(ctx.blockName('gather_block'), loopBinds, [{ buffer: operandBuf }, { buffer: indicesBuf }], [{ buffer: outBuf }], store);
    return wrapInLoops(block, loopVars, outBuf.shape, extentNodes);
  });

  registerLoweringRule('scatter', (ctx, op, inputs, outputs) => {
    const operandBuf = inputs[0];
    const indicesBuf = inputs[1];
    const updatesBuf = inputs[2];
    const outBuf = outputs[0];
    const insertedWindowDims = new Set(op.getAttr<readonly number[]>('inserted_window_dims'));
    const scatterDimsToOperandDims = op.getAttr<readonly number[]>('scatter_dims_to_operand_dims') as readonly number[];
    const indexVectorDim = op.getAttr<number>('index_vector_dim');
    const updateWindowDims = new Set(op.getAttr<readonly number[]>('update_window_dims'));

    const copyNest = makeLoopNest(ctx, operandBuf.shape, operandBuf);
    const copyStore = new BufferStoreNode(outBuf, copyNest.indices, new BufferLoadNode(operandBuf, copyNest.indices));
    const copyBlock = new BlockNode(ctx.blockName('scatter_copy'), copyNest.loopBinds, [{ buffer: operandBuf }], [{ buffer: outBuf }], copyStore);
    const copyBody = wrapInLoops(copyBlock, copyNest.loopVars, operandBuf.shape, copyNest.extentNodes);

    const { loopVars: uVars, loopBinds: uBinds, indices: uIndices, extentNodes: uExtents } = makeLoopNest(ctx, updatesBuf.shape, updatesBuf);

    const batchIndices: TirNode[] = [];
    const windowIndices: TirNode[] = [];
    for (let i = 0; i < updatesBuf.shape.length; i++) {
      if (updateWindowDims.has(i)) windowIndices.push(uIndices[i]);
      else batchIndices.push(uIndices[i]);
    }

    const operandIndices: TirNode[] = new Array(operandBuf.shape.length);
    let windowIdx = 0;
    for (let i = 0; i < operandBuf.shape.length; i++) {
      if (insertedWindowDims.has(i)) {
        operandIndices[i] = new IntImmNode(0);
      } else {
        operandIndices[i] = windowIndices[windowIdx++];
      }
    }

    for (let k = 0; k < scatterDimsToOperandDims.length; k++) {
      const idxLookup: TirNode[] = new Array(indicesBuf.shape.length);
      let batchIdx = 0;
      for (let d = 0; d < indicesBuf.shape.length; d++) {
        if (d === indexVectorDim) idxLookup[d] = new IntImmNode(k);
        else idxLookup[d] = batchIndices[batchIdx++];
      }
      const startVal = asIndexValue(new BufferLoadNode(indicesBuf, idxLookup), indicesBuf.dtype);
      const targetDim = scatterDimsToOperandDims[k];
      operandIndices[targetDim] = new MathOpNode('+', operandIndices[targetDim], startVal);
    }

    const updateLoad = new BufferLoadNode(updatesBuf, uIndices);
    const existingLoad = new BufferLoadNode(outBuf, operandIndices);
    const combined = new MathOpNode('+', existingLoad, updateLoad);
    const scatterStore = new BufferStoreNode(outBuf, operandIndices, combined);
    const scatterBlock = new BlockNode(ctx.blockName('scatter_update'), uBinds, [{ buffer: updatesBuf }, { buffer: indicesBuf }], [{ buffer: outBuf }], scatterStore);
    const scatterBody = wrapInLoops(scatterBlock, uVars, updatesBuf.shape, uExtents);

    return new SeqNode([copyBody, scatterBody]);
  });

  registerLoweringRule('fused_dot_epilogue', (ctx, op, inputs, outputs) => {
    const numDotOperands = op.getAttr<number>('num_dot_operands') || 2;
    const lhs = inputs[0];
    const rhs = inputs[1];
    const extraInputs = inputs.slice(numDotOperands);
    const out = outputs[0];
    const epilogueTags = op.getAttr<readonly string[]>('epilogue_tags') || [];

    const { initBody, accBody } = emitMatmulInitAcc(ctx, op, lhs, rhs, out, {
      prefix: 'ei',
      initBlockName: 'matmul_init',
      accBlockName: 'matmul_acc',
      initVal: () => new FloatImmNode(0),
      accLeaf: (a, b) => new MathOpNode('*', a, b),
    });

    if (epilogueTags.length === 0) {
      return new SeqNode([initBody, accBody]);
    }

    const epiNest = buildSpatialNest(ctx, 'ep', Array.from({ length: out.shape.length }, (_, i) => i), out.shape, out);
    const epiIdx = epiNest.indices;

    let expr: TirNode = new BufferLoadNode(out, epiIdx);
    const idx = { v: 0 };
    for (const tag of epilogueTags) {
      const lowerer = EPILOGUE_TAG_LOWERERS.get(tag);
      if (lowerer) expr = lowerer(expr, extraInputs, idx, out, epiIdx);
    }

    const epiReads = bufRefs([out, ...extraInputs]);
    const epiStore = new BufferStoreNode(out, epiIdx, expr);
    const epiBlock = new BlockNode(ctx.blockName('epilogue'), epiNest.ivs, epiReads, [{ buffer: out }], epiStore);
    const epiBody = epiNest.wrap(epiBlock);

    return new SeqNode([initBody, accBody, epiBody]);
  });

  registerLoweringRule('cublas_gemm', getLoweringRule('dot') as LoweringRuleFn);
}
