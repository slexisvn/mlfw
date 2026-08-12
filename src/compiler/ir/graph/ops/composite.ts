import { OpDef, OpTrait } from '../op_registry.js';
import type { OpRegistry } from '../op_registry.js';
import type { IRType } from '../types.js';
import type { Operation } from '../operation.js';
import { TensorType, isFloatType } from '../types.js';

function inferSameAsInput(operandTypes: readonly IRType[]): IRType[] | null {
  if (operandTypes.length < 1) return null;
  const inp = operandTypes[0];
  if (!(inp instanceof TensorType)) return null;
  return [new TensorType(inp.shape, inp.dtype)];
}

function inferSameAsInputFloat(operandTypes: readonly IRType[]): IRType[] | null {
  if (operandTypes.length < 1) return null;
  const inp = operandTypes[0];
  if (!(inp instanceof TensorType) || !isFloatType(inp.dtype)) return null;
  return [new TensorType(inp.shape, inp.dtype)];
}

function verifyUnaryFloat(op: Operation): string[] {
  const errs: string[] = [];
  if (op.numOperands < 1) { errs.push(`${op.opName} expects at least 1 operand`); return errs; }
  const inp = op.getOperand(0).type;
  if (inp instanceof TensorType && !isFloatType(inp.dtype)) {
    errs.push(`${op.opName} requires float input, got ${inp.dtype}`);
  }
  return errs;
}

export function register(registry: OpRegistry) {
  registry.register(new OpDef({
    name: 'all_reduce',
    numOperands: 1,
    numResults: 1,
    traits: [OpTrait.OPAQUE],
    attrs: [
      { name: 'reduce_op', type: 'string', required: false },
      { name: 'mesh_axis', type: 'number', required: false }
    ],
    inferResultTypes: inferSameAsInput
  }));

  registry.register(new OpDef({
    name: 'all_gather',
    numOperands: 1,
    numResults: 1,
    traits: [OpTrait.OPAQUE],
    attrs: [
      { name: 'mesh_axis', type: 'number', required: false },
      { name: 'gather_dim', type: 'number', required: false }
    ],
    inferResultTypes(operandTypes, attrMap) {
      const x = operandTypes[0];
      if (!(x instanceof TensorType)) return null;
      const meshAxis = (attrMap && attrMap.has('mesh_axis') ? attrMap.get('mesh_axis') : 0) as number;
      const gatherDim = (attrMap && attrMap.has('gather_dim') ? attrMap.get('gather_dim') : 1) as number;
      const shape = [...x.shape];
      shape[gatherDim] = (shape[gatherDim] as number) * (shape[meshAxis] as number);
      return [new TensorType(shape, x.dtype)];
    }
  }));

  registry.register(new OpDef({
    name: 'scaled_dot_product_attention',
    numOperands: 3,
    numResults: 1,
    traits: [OpTrait.OPAQUE],
    attrs: [
      { name: 'scale', type: 'number', required: true },
      { name: 'causal', type: 'boolean', required: false }
    ],
    inferResultTypes(operandTypes) {
      const q = operandTypes[0], v = operandTypes[2];
      if (!(q instanceof TensorType) || !(v instanceof TensorType)) return null;
      return [new TensorType([...q.shape.slice(0, q.rank - 1), v.shape[v.rank - 1]], q.dtype)];
    }
  }));

  registry.register(new OpDef({
    name: 'softmax',
    numOperands: 1,
    numResults: 1,
    opAttrs: { gpuCapable: true },
    attrs: [{ name: 'axis', type: 'number', required: true }],
    inferResultTypes: inferSameAsInputFloat,
    verify: verifyUnaryFloat
  }));

  registry.register(new OpDef({
    name: 'log_softmax',
    numOperands: 1,
    numResults: 1,
    attrs: [{ name: 'axis', type: 'number', required: true }],
    inferResultTypes: inferSameAsInputFloat,
    verify: verifyUnaryFloat
  }));

  registry.register(new OpDef({
    name: 'gelu',
    numOperands: 1,
    numResults: 1,
    traits: [OpTrait.ELEMENTWISE, OpTrait.SAME_OPERAND_AND_RESULT_TYPE],
    inferResultTypes: inferSameAsInputFloat,
    verify: verifyUnaryFloat
  }));

  registry.register(new OpDef({
    name: 'sigmoid',
    numOperands: 1,
    numResults: 1,
    traits: [OpTrait.ELEMENTWISE, OpTrait.SAME_OPERAND_AND_RESULT_TYPE],
    inferResultTypes: inferSameAsInputFloat,
    verify: verifyUnaryFloat
  }));

  registry.register(new OpDef({
    name: 'silu',
    numOperands: 1,
    numResults: 1,
    traits: [OpTrait.ELEMENTWISE, OpTrait.SAME_OPERAND_AND_RESULT_TYPE],
    inferResultTypes: inferSameAsInputFloat,
    verify: verifyUnaryFloat
  }));

  registry.register(new OpDef({
    name: 'layer_norm',
    numOperands: 3,
    numResults: 1,
    opAttrs: { gpuCapable: true },
    attrs: [
      { name: 'axis', type: 'number', required: true },
      { name: 'epsilon', type: 'number', required: true }
    ],
    inferResultTypes: inferSameAsInputFloat,
    verify(op) {
      const errs = [];
      if (op.numOperands !== 3) { errs.push('layer_norm expects 3 operands (input, gamma, beta)'); return errs; }
      const inp = op.getOperand(0).type;
      if (inp instanceof TensorType && !isFloatType(inp.dtype)) {
        errs.push(`layer_norm requires float input, got ${inp.dtype}`);
      }
      return errs;
    }
  }));

  registry.register(new OpDef({
    name: 'batch_norm',
    numOperands: 5,
    numResults: 1,
    opAttrs: { gpuCapable: true },
    attrs: [
      { name: 'axis', type: 'number', required: true },
      { name: 'epsilon', type: 'number', required: true }
    ],
    inferResultTypes: inferSameAsInputFloat,
    verify(op) {
      const errs = [];
      if (op.numOperands !== 5) { errs.push('batch_norm expects 5 operands (input, gamma, beta, mean, var)'); return errs; }
      const inp = op.getOperand(0).type;
      if (inp instanceof TensorType && !isFloatType(inp.dtype)) {
        errs.push(`batch_norm requires float input, got ${inp.dtype}`);
      }
      return errs;
    }
  }));

  for (const name of ['selu', 'mish', 'hardswish', 'hardsigmoid']) {
    registry.register(new OpDef({
      name,
      numOperands: 1,
      numResults: 1,
      traits: [OpTrait.ELEMENTWISE, OpTrait.SAME_OPERAND_AND_RESULT_TYPE],
      inferResultTypes: inferSameAsInputFloat,
      verify: verifyUnaryFloat
    }));
  }

  registry.register(new OpDef({
    name: 'elu',
    numOperands: 1,
    numResults: 1,
    attrs: [{ name: 'alpha', type: 'number' }],
    traits: [OpTrait.ELEMENTWISE, OpTrait.SAME_OPERAND_AND_RESULT_TYPE],
    inferResultTypes: inferSameAsInputFloat,
    verify: verifyUnaryFloat
  }));

  registry.register(new OpDef({
    name: 'leaky_relu',
    numOperands: 1,
    numResults: 1,
    attrs: [{ name: 'negative_slope', type: 'number' }],
    traits: [OpTrait.ELEMENTWISE, OpTrait.SAME_OPERAND_AND_RESULT_TYPE],
    inferResultTypes: inferSameAsInputFloat,
    verify: verifyUnaryFloat
  }));

  registry.register(new OpDef({
    name: 'celu',
    numOperands: 1,
    numResults: 1,
    attrs: [{ name: 'alpha', type: 'number' }],
    traits: [OpTrait.ELEMENTWISE, OpTrait.SAME_OPERAND_AND_RESULT_TYPE],
    inferResultTypes: inferSameAsInputFloat,
    verify: verifyUnaryFloat
  }));

  registry.register(new OpDef({
    name: 'embedding',
    numOperands: 2,
    numResults: 1,
    inferResultTypes(operandTypes) {
      if (operandTypes.length < 2) return null;
      const weight = operandTypes[0];
      const indices = operandTypes[1];
      if (!(weight instanceof TensorType) || !(indices instanceof TensorType)) return null;
      const shape = [...indices.shape, weight.shape[weight.rank - 1]];
      return [new TensorType(shape, weight.dtype)];
    }
  }));
}
