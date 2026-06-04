import {
  FloatImmNode, MathOpNode, CompareNode,
  BufferStoreNode, BufferLoadNode,
  BlockNode, SeqNode, IfThenElseNode, CallExternNode, CastNode, LetStmtNode
} from '../../../ir/tensor/nodes.js';
import { Buffer } from '../../../ir/tensor/buffer.js';
import {
  registerLoweringRule, getLoweringRule, makeLoopNest, wrapInLoops,
  computeBroadcastIndices, bufRefs, lowerConstant, CONSTANT_OPS
} from '../lowering_registry.js';
import { buildElementwiseExpr, ELEMENTWISE_OPS } from './elementwise.js';

const INLINE_FUSION_BUILDERS = new Map();

export function registerInlineFusionBuilder(opName, builder) {
  INLINE_FUSION_BUILDERS.set(opName, builder);
}

export function canInlineFuse(opName) {
  return INLINE_FUSION_BUILDERS.has(opName);
}

export function getInlineFusionBuilder(opName) {
  return INLINE_FUSION_BUILDERS.get(opName);
}

function initBuiltinFusionBuilders() {
  for (const opName of Object.keys(ELEMENTWISE_OPS)) {
    INLINE_FUSION_BUILDERS.set(opName, (innerOp, args, dtype) =>
      buildElementwiseExpr(innerOp.opName, args, dtype)
    );
  }

  INLINE_FUSION_BUILDERS.set('compare', (innerOp, args) =>
    new CompareNode(innerOp.getAttr('direction') || 'eq', args[0], args[1])
  );

  INLINE_FUSION_BUILDERS.set('select', (_innerOp, args) =>
    new IfThenElseNode(args[0], args[1], args[2])
  );

  INLINE_FUSION_BUILDERS.set('clamp', (_innerOp, args, dtype) =>
    new CallExternNode('min', [new CallExternNode('max', [args[1], args[0]], dtype), args[2]], dtype)
  );

  INLINE_FUSION_BUILDERS.set('convert', (innerOp, args) =>
    new CastNode(args[0], innerOp.getOperand(0).type.dtype, innerOp.getAttr('target_dtype') || innerOp.getResult(0).type.dtype)
  );

  INLINE_FUSION_BUILDERS.set('broadcast_in_dim', (_innerOp, args) => args[0]);
  INLINE_FUSION_BUILDERS.set('broadcast', (_innerOp, args) => args[0]);

  INLINE_FUSION_BUILDERS.set('quantize', (innerOp, args) => {
    const scale = innerOp.getAttr('scale');
    const zp = innerOp.getAttr('zero_point');
    const tgtDtype = innerOp.getAttr('target_dtype') || 'i8';
    const isUnsigned = tgtDtype === 'ui8';
    const cMin = isUnsigned ? 0 : -128;
    const cMax = isUnsigned ? 255 : 127;
    const scaled = new MathOpNode('/', args[0], new FloatImmNode(scale));
    const shifted = new MathOpNode('+', scaled, new FloatImmNode(zp));
    const rounded = new CallExternNode('round', [shifted], 'f32');
    const clamped = new CallExternNode('min', [
      new CallExternNode('max', [rounded, new FloatImmNode(cMin)], 'f32'),
      new FloatImmNode(cMax)
    ], 'f32');
    return new CastNode(clamped, 'f32', tgtDtype);
  });

  INLINE_FUSION_BUILDERS.set('dequantize', (innerOp, args) => {
    const scale = innerOp.getAttr('scale');
    const zp = innerOp.getAttr('zero_point');
    const srcDtype = innerOp.getOperand(0).type?.dtype || 'i8';
    const tgtDtype = innerOp.getAttr('target_dtype') || 'f32';
    const asFloat = new CastNode(args[0], srcDtype, tgtDtype);
    const shifted = new MathOpNode('-', asFloat, new FloatImmNode(zp));
    return new MathOpNode('*', shifted, new FloatImmNode(scale));
  });
}

const CSE_TRIVIAL = new Set(['BufferLoadNode', 'VariableNode', 'IntImmNode', 'FloatImmNode']);

function lowerFusion(ctx, op) {
  const numInputs = op.numOperands;
  const numOutputs = op.numResults;
  const inputs = new Array(numInputs);
  for (let i = 0; i < numInputs; i++) inputs[i] = ctx.getOrAllocBuffer(op.getOperand(i));
  const outputs = new Array(numOutputs);
  for (let i = 0; i < numOutputs; i++) outputs[i] = ctx.getOrAllocBuffer(op.getResult(i));

  const outBuf = outputs[0];
  const { loopVars, loopBinds, indices: outIndices } = makeLoopNest(ctx, outBuf.shape);
  const exprMap = new Map();

  const entryBlock = op.regions[0].entryBlock;
  for (let i = 0; i < entryBlock.arguments.length; i++) {
    exprMap.set(entryBlock.arguments[i], new BufferLoadNode(inputs[i], computeBroadcastIndices(inputs[i], outBuf, outIndices)));
  }

  const useCount = new Map();
  for (const innerOp of entryBlock.ops()) {
    for (let i = 0; i < innerOp.numOperands; i++) {
      const val = innerOp.getOperand(i);
      useCount.set(val, (useCount.get(val) || 0) + 1);
    }
  }

  const cseVars = new Map();
  let cseCounter = 0;
  const cseStmts = [];

  function getExpr(val) {
    const expr = exprMap.get(val);
    if (expr === undefined) {
      throw new Error(`Fusion lowering: unmapped operand from '${val.definingOp ? val.definingOp.opName : 'unknown'}'`);
    }
    if ((useCount.get(val) || 0) > 1 && !CSE_TRIVIAL.has(expr.type)) {
      if (!cseVars.has(val)) {
        const v = ctx.allocVar(`cse${cseCounter++}`, outBuf.dtype);
        cseVars.set(val, v);
        cseStmts.push({ variable: v, value: expr });
        exprMap.set(val, v);
      }
      return cseVars.get(val);
    }
    return expr;
  }

  const stores = [];
  for (const innerOp of entryBlock.ops()) {
    if (innerOp.opName === 'yield') {
      for (let i = 0; i < innerOp.numOperands; i++) {
        stores.push(new BufferStoreNode(outputs[i], outIndices, getExpr(innerOp.getOperand(i))));
      }
      break;
    }

    if (CONSTANT_OPS.has(innerOp.opName)) {
      const val = innerOp.getAttr('value');
      exprMap.set(innerOp.getResult(0), new FloatImmNode(typeof val === 'number' ? val : 0));
      continue;
    }

    const builder = INLINE_FUSION_BUILDERS.get(innerOp.opName);
    if (!builder) {
      throw new Error(`Fusion lowering: unsupported op '${innerOp.opName}' inside fusion body`);
    }

    const args = new Array(innerOp.numOperands);
    for (let i = 0; i < innerOp.numOperands; i++) args[i] = getExpr(innerOp.getOperand(i));
    exprMap.set(innerOp.getResult(0), builder(innerOp, args, outBuf.dtype));
  }

  let storeBody = stores.length === 1 ? stores[0] : new SeqNode(stores);
  for (let i = cseStmts.length - 1; i >= 0; i--) {
    storeBody = new LetStmtNode(cseStmts[i].variable, cseStmts[i].value, storeBody);
  }

  const block = new BlockNode('fusion_block', loopBinds, bufRefs(inputs), bufRefs(outputs), storeBody);
  return wrapInLoops(block, loopVars, outBuf.shape);
}

function canLowerAsElementwiseFusion(op) {
  const region = op.regions[0];
  if (!region) return false;
  for (const innerOp of region.entryBlock.ops()) {
    if (innerOp.opName === 'yield') continue;
    if (!INLINE_FUSION_BUILDERS.has(innerOp.opName)) return false;
  }
  return true;
}

function lowerFusionAsIndividualOps(ctx, fusionOp, stmts) {
  const entryBlock = fusionOp.regions[0].entryBlock;
  const valueMap = new Map();
  for (let i = 0; i < entryBlock.arguments.length; i++) {
    valueMap.set(entryBlock.arguments[i], fusionOp.getOperand(i));
  }

  for (const innerOp of entryBlock.ops()) {
    if (innerOp.opName === 'yield') {
      for (let i = 0; i < innerOp.numOperands; i++) {
        const outerVal = valueMap.get(innerOp.getOperand(i));
        if (outerVal) {
          const srcBuf = ctx.getOrAllocBuffer(outerVal);
          const dstBuf = ctx.getOrAllocBuffer(fusionOp.getResult(i));
          if (srcBuf !== dstBuf) ctx.bufferMap.set(fusionOp.getResult(i), srcBuf);
        }
      }
      continue;
    }

    const outerOperands = new Array(innerOp.numOperands);
    for (let i = 0; i < innerOp.numOperands; i++) {
      outerOperands[i] = valueMap.get(innerOp.getOperand(i)) || innerOp.getOperand(i);
    }

    const inputs = new Array(outerOperands.length);
    for (let i = 0; i < outerOperands.length; i++) inputs[i] = ctx.getOrAllocBuffer(outerOperands[i]);
    const outputs = new Array(innerOp.numResults);
    for (let i = 0; i < innerOp.numResults; i++) {
      const proxy = { type: innerOp.getResult(i).type };
      outputs[i] = ctx.getOrAllocBuffer(proxy);
      valueMap.set(innerOp.getResult(i), proxy);
    }

    if (CONSTANT_OPS.has(innerOp.opName)) {
      stmts.push(lowerConstant(ctx, innerOp));
      continue;
    }

    const rule = getLoweringRule(innerOp.opName);
    if (!rule) {
      throw new Error(`Fusion lowering: no lowering rule for op '${innerOp.opName}' inside fusion body`);
    }
    const stmt = rule(ctx, innerOp, inputs, outputs);
    if (stmt) stmts.push(stmt);
  }
}

export { lowerFusion, canLowerAsElementwiseFusion, lowerFusionAsIndividualOps };

export function register() {
  initBuiltinFusionBuilders();
}
