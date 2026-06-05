import { OpDef, OpTrait } from '../op_registry.js';
import { TensorType, isFloatType } from '../types.js';

function inferSameAsInput(operandTypes) {
  if (operandTypes.length < 1) return null;
  const inp = operandTypes[0];
  if (!(inp instanceof TensorType)) return null;
  return [new TensorType(inp.shape, inp.dtype)];
}

function inferSameAsInputFloat(operandTypes) {
  if (operandTypes.length < 1) return null;
  const inp = operandTypes[0];
  if (!(inp instanceof TensorType) || !isFloatType(inp.dtype)) return null;
  return [new TensorType(inp.shape, inp.dtype)];
}

function verifyUnaryFloat(op) {
  const errs = [];
  if (op.numOperands < 1) { errs.push(`${op.opName} expects at least 1 operand`); return errs; }
  const inp = op.getOperand(0).type;
  if (inp instanceof TensorType && !isFloatType(inp.dtype)) {
    errs.push(`${op.opName} requires float input, got ${inp.dtype}`);
  }
  return errs;
}

export function register(registry) {
  registry.register(new OpDef({
    name: 'softmax',
    numOperands: 1,
    numResults: 1,
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
