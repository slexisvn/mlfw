import { PrimFunc, SeqNode } from '../../ir/tensor/nodes.js';
import {
  LoweringContext, registerLoweringRule, hasLoweringRule, getLoweringRule,
  lowerConstant, CONSTANT_OPS
} from './lowering_registry.js';
import { register as registerElementwise } from './rules/elementwise.js';
import { register as registerShape } from './rules/shape.js';
import { register as registerReduction } from './rules/reduction.js';
import { register as registerLinalg } from './rules/linalg.js';
import { register as registerControlFlow } from './rules/control_flow.js';
import { register as registerLayout } from './rules/layout.js';
import { register as registerQuantization } from './rules/quantization.js';
import {
  register as registerFusion,
  canInlineFuse, lowerFusion, canLowerAsElementwiseFusion,
  lowerFusionAsIndividualOps, registerInlineFusionBuilder
} from './rules/fusion.js';

registerElementwise();
registerShape();
registerReduction();
registerLinalg();
registerControlFlow();
registerLayout();
registerQuantization();
registerFusion();

export { LoweringContext, hasLoweringRule, registerLoweringRule, canInlineFuse, registerInlineFusionBuilder };

export function lowerGraphToPrimFunc(graphFunc) {
  const ctx = new LoweringContext();
  const params = [];
  const bufferMap = new Map();

  for (const arg of graphFunc.args) {
    const v = ctx.allocVar('arg');
    params.push(v);
    bufferMap.set(v, ctx.getOrAllocBuffer(arg));
  }

  const retOp = graphFunc.getReturnOp();
  for (let i = 0; i < retOp.numOperands; i++) {
    const v = ctx.allocVar('ret');
    params.push(v);
    bufferMap.set(v, ctx.getOrAllocBuffer(retOp.getOperand(i)));
  }

  const stmts = [];

  for (const op of graphFunc.ops()) {
    if (CONSTANT_OPS.has(op.opName)) stmts.push(lowerConstant(ctx, op));
  }

  for (const op of graphFunc.ops()) {
    if (op.opName === 'return' || op.opName === 'yield') continue;
    if (CONSTANT_OPS.has(op.opName)) continue;

    if (op.opName === 'fusion') {
      if (canLowerAsElementwiseFusion(op)) {
        stmts.push(lowerFusion(ctx, op));
      } else {
        lowerFusionAsIndividualOps(ctx, op, stmts);
      }
      continue;
    }

    const rule = getLoweringRule(op.opName);
    if (!rule) throw new Error(`No lowering rule defined for op: ${op.opName}`);

    const inputs = new Array(op.numOperands);
    for (let i = 0; i < op.numOperands; i++) inputs[i] = ctx.getOrAllocBuffer(op.getOperand(i));
    const outputs = new Array(op.numResults);
    for (let i = 0; i < op.numResults; i++) outputs[i] = ctx.getOrAllocBuffer(op.getResult(i));

    const stmt = rule(ctx, op, inputs, outputs);
    if (stmt) stmts.push(stmt);
  }

  const shapeParams = [...ctx.shapeParams.values()];
  for (const sp of shapeParams) params.push(sp);

  return new PrimFunc(graphFunc.name, params, stmts.length === 1 ? stmts[0] : new SeqNode(stmts), bufferMap, shapeParams);
}
