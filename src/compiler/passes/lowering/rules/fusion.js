import { FloatImmNode, IntImmNode, MathOpNode, CompareNode, BufferStoreNode, BufferLoadNode, BlockNode, SeqNode, IfThenElseNode, CallExternNode, CastNode, LetStmtNode } from '../../../ir/tensor/nodes.js';

import { getLoweringRule, makeLoopNest, wrapInLoops, computeBroadcastIndices, bufRefs, lowerConstant, isConstantOp } from '../lowering_registry.js';
import { buildElementwiseExpr, elementwiseOpNames } from './elementwise.js';
import { buildQuantizeExpr, buildDequantizeExpr } from '../quant_math.js';
import { isBroadcastOp } from '../../../ir/graph/op_traits.js';

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
  for (const opName of elementwiseOpNames()) {
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

  INLINE_FUSION_BUILDERS.set('iota', () => {
    throw new Error('iota fusion must be handled by the index-aware path in lowerFusion');
  });

  INLINE_FUSION_BUILDERS.set('quantize', (innerOp, args) =>
    buildQuantizeExpr(args[0], {
      scale: innerOp.getAttr('scale'),
      zeroPoint: innerOp.getAttr('zero_point'),
      targetDtype: innerOp.getAttr('target_dtype') || 'i8',
    }));

  INLINE_FUSION_BUILDERS.set('dequantize', (innerOp, args) =>
    buildDequantizeExpr(args[0], {
      scale: innerOp.getAttr('scale'),
      zeroPoint: innerOp.getAttr('zero_point'),
      srcDtype: innerOp.getOperand(0).type?.dtype || 'i8',
      targetDtype: innerOp.getAttr('target_dtype') || 'f32',
    }));
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
  const { loopVars, loopBinds, indices: outIndices, extentNodes } = makeLoopNest(ctx, outBuf.shape, outBuf);
  const exprMap = new Map();

  const entryBlock = op.regions[0].entryBlock;

  const valueDims = new Map();
  const opsArr = [...entryBlock.ops()];
  for (let k = opsArr.length - 1; k >= 0; k--) {
    const innerOp = opsArr[k];
    if (isBroadcastOp(innerOp.opName)) {
      const dims = innerOp.getAttr('broadcast_dimensions');
      if (dims && dims.length > 0) {
        valueDims.set(innerOp.getOperand(0), dims);
      }
      continue;
    }
    if (innerOp.opName === 'yield' || isConstantOp(innerOp.opName)) continue;
    for (let r = 0; r < innerOp.numResults; r++) {
      const dims = valueDims.get(innerOp.getResult(r));
      if (!dims) continue;
      for (let i = 0; i < innerOp.numOperands; i++) {
        if (!valueDims.has(innerOp.getOperand(i))) {
          valueDims.set(innerOp.getOperand(i), dims);
        }
      }
    }
  }
  const argBroadcastDims = new Map();
  const blockArgs = entryBlock.arguments;
  for (let i = 0; i < blockArgs.length; i++) {
    const dims = valueDims.get(blockArgs[i]);
    if (dims) argBroadcastDims.set(i, dims);
  }

  for (let i = 0; i < entryBlock.arguments.length; i++) {
    const explicitDims = argBroadcastDims.get(i);
    let loadIndices;
    if (explicitDims) {
      const inBuf = inputs[i];
      loadIndices = new Array(inBuf.shape.length);
      for (let j = 0; j < inBuf.shape.length; j++) {
        loadIndices[j] = inBuf.shape[j] === 1 ? new IntImmNode(0) : outIndices[explicitDims[j]];
      }
    } else {
      loadIndices = computeBroadcastIndices(inputs[i], outBuf, outIndices);
    }
    exprMap.set(entryBlock.arguments[i], new BufferLoadNode(inputs[i], loadIndices));
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
        const exprDtype = expr.type === 'CompareNode' ? 'i32'
          : expr.type === 'CastNode' ? expr.toDtype
          : outBuf.dtype;
        const v = ctx.allocVar(`cse${cseCounter++}`, exprDtype);
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

    if (isConstantOp(innerOp.opName)) {
      const val = innerOp.getAttr('value');
      exprMap.set(innerOp.getResult(0), new FloatImmNode(typeof val === 'number' ? val : 0));
      continue;
    }

    if (innerOp.opName === 'iota') {
      const dim = innerOp.getAttr('iota_dimension') ?? innerOp.getAttr('dimension') ?? 0;
      exprMap.set(innerOp.getResult(0), outIndices[dim]);
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

  const block = new BlockNode(ctx.blockName('fusion_block'), loopBinds, bufRefs(inputs), bufRefs(outputs), storeBody);
  return wrapInLoops(block, loopVars, outBuf.shape, extentNodes);
}

function shapesEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

function canLowerAsElementwiseFusion(op) {
  const region = op.regions[0];
  if (!region) return false;
  for (const innerOp of region.entryBlock.ops()) {
    if (innerOp.opName === 'yield') continue;
    if (isConstantOp(innerOp.opName)) {
      if (typeof innerOp.getAttr('value') === 'number') continue;
      return false;
    }
    if (!INLINE_FUSION_BUILDERS.has(innerOp.opName)) return false;
  }
  if (op.numResults > 1) {
    const refShape = op.getResult(0).type.shape;
    for (let i = 1; i < op.numResults; i++) {
      if (!shapesEqual(op.getResult(i).type.shape, refShape)) return false;
    }
  }
  return true;
}

function lowerFusionAsIndividualOps(ctx, fusionOp, stmts) {
  const entryBlock = fusionOp.regions[0].entryBlock;
  const valueMap = new Map();
  for (let i = 0; i < entryBlock.arguments.length; i++) {
    valueMap.set(entryBlock.arguments[i], fusionOp.getOperand(i));
  }

  const yieldedInner = new Map();
  for (const op of entryBlock.ops()) {
    if (op.opName === 'yield') {
      for (let i = 0; i < op.numOperands; i++) {
        yieldedInner.set(op.getOperand(i), fusionOp.getResult(i));
      }
      break;
    }
  }

  for (const innerOp of entryBlock.ops()) {
    if (innerOp.opName === 'yield') continue;

    const outerOperands = new Array(innerOp.numOperands);
    for (let i = 0; i < innerOp.numOperands; i++) {
      outerOperands[i] = valueMap.get(innerOp.getOperand(i)) || innerOp.getOperand(i);
    }

    const inputs = new Array(outerOperands.length);
    for (let i = 0; i < outerOperands.length; i++) inputs[i] = ctx.getOrAllocBuffer(outerOperands[i]);
    const outputs = new Array(innerOp.numResults);
    for (let i = 0; i < innerOp.numResults; i++) {
      const innerVal = innerOp.getResult(i);
      const outerResult = yieldedInner.get(innerVal);
      if (outerResult) {
        const outBuf = ctx.getOrAllocBuffer(outerResult);
        outputs[i] = outBuf;
        valueMap.set(innerVal, outerResult);
      } else {
        const proxy = { type: innerVal.type };
        outputs[i] = ctx.getOrAllocBuffer(proxy);
        valueMap.set(innerVal, proxy);
      }
    }

    if (isConstantOp(innerOp.opName)) {
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
