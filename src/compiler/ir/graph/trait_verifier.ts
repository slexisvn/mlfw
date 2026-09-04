import { registry } from './ops.js';
import { OpTrait } from './op_registry.js';
import { TensorType, dimEquals, typeToString } from './types.js';
import { sizesOperandSpan } from './mlir_format.js';
import type { Dim, Shape } from './types.js';
import type { Operation } from './operation.js';
import type { OpTraitValue } from './op_registry.js';

export type TraitVerifyFn = (op: Operation) => string[];

const _traitVerifiers = new Map<OpTraitValue, TraitVerifyFn>();

export function registerTraitVerifier(trait: OpTraitValue, verify: TraitVerifyFn): TraitVerifyFn {
  _traitVerifiers.set(trait, verify);
  return verify;
}

export function unregisterTraitVerifier(trait: OpTraitValue): boolean {
  return _traitVerifiers.delete(trait);
}

export function getTraitVerifier(trait: OpTraitValue): TraitVerifyFn | null {
  return _traitVerifiers.get(trait) || null;
}

export function verifiedTraits(): OpTraitValue[] {
  return [..._traitVerifiers.keys()];
}

export function verifyTraits(op: Operation): string[] {
  const def = registry.get(op.opName);
  if (def === null) return [];
  const errors: string[] = [];
  for (const trait of def.traits) {
    const verify = _traitVerifiers.get(trait);
    if (!verify) continue;
    for (const message of verify(op)) errors.push(`trait '${trait}': ${message}`);
  }
  return errors;
}

function tensorOperands(op: Operation): { index: number; type: TensorType }[] {
  const out: { index: number; type: TensorType }[] = [];
  for (let i = 0; i < op.numOperands; i++) {
    const t = op.getOperand(i).type;
    if (t instanceof TensorType) out.push({ index: i, type: t });
  }
  return out;
}

function tensorResults(op: Operation): { index: number; type: TensorType }[] {
  const out: { index: number; type: TensorType }[] = [];
  for (let i = 0; i < op.numResults; i++) {
    const t = op.getResult(i).type;
    if (t instanceof TensorType) out.push({ index: i, type: t });
  }
  return out;
}

function shapeEquals(a: Shape, b: Shape): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (!dimEquals(a[i], b[i])) return false;
  }
  return true;
}

function shapeToString(shape: Shape): string {
  return `[${shape.map((d: Dim) => String(d)).join(', ')}]`;
}

registerTraitVerifier(OpTrait.SAME_OPERAND_AND_RESULT_TYPE, (op) => {
  const operands = tensorOperands(op);
  const results = tensorResults(op);
  if (operands.length === 0) return [];
  const dtype = operands[0].type.dtype;
  const errors: string[] = [];
  for (const { index, type } of operands) {
    if (type.dtype !== dtype) errors.push(`operand ${index} dtype '${type.dtype}' != operand 0 dtype '${dtype}'`);
  }
  for (const { index, type } of results) {
    if (type.dtype !== dtype) errors.push(`result ${index} dtype '${type.dtype}' != operand 0 dtype '${dtype}'`);
  }
  return errors;
});

registerTraitVerifier(OpTrait.SAME_OPERAND_AND_RESULT_SHAPE, (op) => {
  const operands = tensorOperands(op);
  const results = tensorResults(op);
  if (operands.length === 0) return [];
  const shape = operands[0].type.shape;
  const errors: string[] = [];
  for (const { index, type } of operands) {
    if (!shapeEquals(type.shape, shape)) {
      errors.push(`operand ${index} shape ${shapeToString(type.shape)} != operand 0 shape ${shapeToString(shape)}`);
    }
  }
  for (const { index, type } of results) {
    if (!shapeEquals(type.shape, shape)) {
      errors.push(`result ${index} shape ${shapeToString(type.shape)} != operand 0 shape ${shapeToString(shape)}`);
    }
  }
  return errors;
});

registerTraitVerifier(OpTrait.ELEMENTWISE, (op) => {
  const operands = tensorOperands(op);
  const results = tensorResults(op);
  if (operands.length === 0 || results.length === 0) return [];
  const broadcast = TensorType.broadcastShape(...operands.map((o) => o.type.shape));
  if (broadcast === null) {
    return [`operand shapes are not broadcast-compatible: ${operands.map((o) => shapeToString(o.type.shape)).join(' vs ')}`];
  }
  const errors: string[] = [];
  for (const { index, type } of results) {
    if (!shapeEquals(type.shape, broadcast)) {
      errors.push(`result ${index} shape ${shapeToString(type.shape)} != broadcast of operand shapes ${shapeToString(broadcast)}`);
    }
  }
  return errors;
});

registerTraitVerifier(OpTrait.COMMUTATIVE, (op) => {
  if (op.numOperands !== 2) return [`expects 2 operands to be interchangeable, got ${op.numOperands}`];
  const a = op.getOperand(0).type;
  const b = op.getOperand(1).type;
  if (a instanceof TensorType && b instanceof TensorType && a.dtype !== b.dtype) {
    return [`operands are not interchangeable: dtype '${a.dtype}' vs '${b.dtype}'`];
  }
  return [];
});

registerTraitVerifier(OpTrait.IDEMPOTENT, (op) => {
  if (op.numOperands !== 2) return [`expects 2 operands, got ${op.numOperands}`];
  if (op.numResults !== 1) return [`expects 1 result, got ${op.numResults}`];
  const errors: string[] = [];
  const result = op.getResult(0).type;
  for (let i = 0; i < 2; i++) {
    const operand = op.getOperand(i).type;
    if (operand instanceof TensorType && result instanceof TensorType && operand.dtype !== result.dtype) {
      errors.push(`operand ${i} dtype '${operand.dtype}' != result dtype '${result.dtype}', so folding f(x, x) -> x would not preserve types`);
    }
  }
  return errors;
});

registerTraitVerifier(OpTrait.CONSTANT, (op) => {
  if (op.numOperands !== 0) return [`a constant op must have no operands, got ${op.numOperands}`];
  return [];
});

registerTraitVerifier(OpTrait.TERMINATOR, (op) => {
  const block = op.parentBlock;
  if (block === null) return [];
  if (block.lastOp !== op) return ['a terminator must be the last operation in its block'];
  return [];
});

registerTraitVerifier(OpTrait.VIEW, (op) => {
  const errors: string[] = [];
  const dataOperands = sizesOperandSpan(op)?.start ?? op.numOperands;
  if (dataOperands !== 1) errors.push(`a view op reads exactly 1 operand, got ${dataOperands}`);
  if (op.numResults !== 1) errors.push(`a view op produces exactly 1 result, got ${op.numResults}`);
  if (errors.length > 0) return errors;
  const operand = op.getOperand(0).type;
  const result = op.getResult(0).type;
  if (operand instanceof TensorType && result instanceof TensorType && operand.dtype !== result.dtype) {
    errors.push(`a view op cannot change dtype: ${typeToString(operand)} -> ${typeToString(result)}`);
  }
  return errors;
});
