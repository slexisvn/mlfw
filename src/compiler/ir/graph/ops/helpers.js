import { TensorType, ScalarType, DYNAMIC, isFloatType } from '../types.js';
import { OpTrait } from '../op_registry.js';

export function inferBinaryElementwise(operandTypes) {
  if (operandTypes.length !== 2) return null;
  const lhs = operandTypes[0], rhs = operandTypes[1];
  if (!(lhs instanceof TensorType) || !(rhs instanceof TensorType)) return null;
  if (lhs.dtype !== rhs.dtype) return null;
  const shape = TensorType.broadcastShape(lhs.shape, rhs.shape);
  if (!shape) return null;
  return [new TensorType(shape, lhs.dtype)];
}

export function inferUnaryElementwise(operandTypes) {
  if (operandTypes.length !== 1) return null;
  const inp = operandTypes[0];
  if (!(inp instanceof TensorType)) return null;
  return [new TensorType(inp.shape, inp.dtype)];
}

export function inferCompare(operandTypes) {
  if (operandTypes.length !== 2) return null;
  const lhs = operandTypes[0], rhs = operandTypes[1];
  if (!(lhs instanceof TensorType) || !(rhs instanceof TensorType)) return null;
  const shape = TensorType.broadcastShape(lhs.shape, rhs.shape);
  if (!shape) return null;
  return [new TensorType(shape, ScalarType.BOOL)];
}

export function inferUnaryFloat(operandTypes) {
  if (operandTypes.length !== 1) return null;
  const t = operandTypes[0];
  if (!(t instanceof TensorType) || !isFloatType(t.dtype)) return null;
  return [new TensorType(t.shape, t.dtype)];
}

export function verifyBinaryElementwise(op) {
  const errs = [];
  if (op.numOperands !== 2) { errs.push(`${op.opName} expects 2 operands, got ${op.numOperands}`); return errs; }
  const lt = op.getOperand(0).type, rt = op.getOperand(1).type;
  if (!(lt instanceof TensorType)) errs.push(`${op.opName} operand 0 is not tensor`);
  if (!(rt instanceof TensorType)) errs.push(`${op.opName} operand 1 is not tensor`);
  if (lt instanceof TensorType && rt instanceof TensorType && lt.dtype !== rt.dtype) {
    errs.push(`${op.opName} dtype mismatch: ${lt.dtype} vs ${rt.dtype}`);
  }
  return errs;
}

export function verifyUnaryElementwise(op) {
  const errs = [];
  if (op.numOperands !== 1) { errs.push(`${op.opName} expects 1 operand, got ${op.numOperands}`); return errs; }
  if (!(op.getOperand(0).type instanceof TensorType)) errs.push(`${op.opName} operand is not tensor`);
  return errs;
}

export function verifyUnaryFloat(op) {
  const errs = verifyUnaryElementwise(op);
  if (errs.length === 0) {
    const dt = op.getOperand(0).type.dtype;
    if (!isFloatType(dt)) errs.push(`${op.opName} requires float type, got ${dt}`);
  }
  return errs;
}

export const binaryArithTraits = [OpTrait.ELEMENTWISE, OpTrait.SAME_OPERAND_AND_RESULT_TYPE];
export const commBinaryArithTraits = [...binaryArithTraits, OpTrait.COMMUTATIVE, OpTrait.ASSOCIATIVE];
