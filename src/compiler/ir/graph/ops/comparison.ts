import { COMPARE_DIRECTIONS } from '../../../../util/dtype_map.js';
import { OpAttrKey, OpDef, OpTrait } from '../op_registry.js';
import type { OpRegistry } from '../op_registry.js';
import { TensorType, ScalarType } from '../types.js';
import type { IRType } from '../types.js';
import { inferCompare } from './helpers.js';

function inferUnaryBool(operandTypes: readonly IRType[]): IRType[] | null {
  if (operandTypes.length !== 1) return null;
  const inp = operandTypes[0];
  if (!(inp instanceof TensorType) || inp.dtype !== ScalarType.BOOL) return null;
  return [new TensorType(inp.shape, ScalarType.BOOL)];
}

function inferBinaryBool(operandTypes: readonly IRType[]): IRType[] | null {
  if (operandTypes.length !== 2) return null;
  const [lhs, rhs] = operandTypes;
  if (!(lhs instanceof TensorType) || lhs.dtype !== ScalarType.BOOL) return null;
  if (!(rhs instanceof TensorType) || rhs.dtype !== ScalarType.BOOL) return null;
  const shape = TensorType.broadcastShape(lhs.shape, rhs.shape);
  if (!shape) return null;
  return [new TensorType(shape, ScalarType.BOOL)];
}

export function register(registry: OpRegistry) {
  registry.register(new OpDef({
    name: 'compare',
    numOperands: 2,
    numResults: 1,
    attrs: [{ name: 'direction', type: 'string', required: true }],
    traits: [OpTrait.ELEMENTWISE],
    opAttrs: { [OpAttrKey.UNIFIED_OPERANDS]: [0, 1] },
    inferResultTypes: inferCompare,
    verify(op) {
      const errs = [];
      if (op.numOperands !== 2) { errs.push('compare expects 2 operands'); return errs; }
      if (!op.hasAttr('direction')) errs.push('compare missing direction attr');
      else {
        const d = op.getAttr<string>('direction')!;
        if (!COMPARE_DIRECTIONS.has(d)) errs.push(`compare invalid direction: ${d}`);
      }
      return errs;
    }
  }));

  registry.register(new OpDef({
    name: 'select',
    numOperands: 3,
    numResults: 1,
    traits: [OpTrait.ELEMENTWISE],
    opAttrs: { [OpAttrKey.UNIFIED_OPERANDS]: [1, 2] },
    inferResultTypes(operandTypes) {
      if (operandTypes.length !== 3) return null;
      const [pred, onTrue, onFalse] = operandTypes;
      if (!(pred instanceof TensorType) || pred.dtype !== ScalarType.BOOL) return null;
      if (!(onTrue instanceof TensorType) || !(onFalse instanceof TensorType)) return null;
      if (onTrue.dtype !== onFalse.dtype) return null;
      const shape = TensorType.broadcastShape(pred.shape, onTrue.shape, onFalse.shape);
      if (!shape) return null;
      return [new TensorType(shape, onTrue.dtype)];
    }
  }));

  registry.register(new OpDef({
    name: 'where',
    numOperands: 3,
    numResults: 1,
    traits: [OpTrait.ELEMENTWISE],
    opAttrs: { [OpAttrKey.UNIFIED_OPERANDS]: [1, 2] },
    inferResultTypes(operandTypes) {
      if (operandTypes.length !== 3) return null;
      const [cond, x, y] = operandTypes;
      if (!(cond instanceof TensorType) || !(x instanceof TensorType) || !(y instanceof TensorType)) return null;
      if (x.dtype !== y.dtype) return null;
      const shape = TensorType.broadcastShape(cond.shape, x.shape, y.shape);
      if (!shape) return null;
      return [new TensorType(shape, x.dtype)];
    }
  }));

  registry.register(new OpDef({
    name: 'logical_not',
    numOperands: 1,
    numResults: 1,
    traits: [OpTrait.ELEMENTWISE, OpTrait.SAME_OPERAND_AND_RESULT_TYPE, OpTrait.SAME_OPERAND_AND_RESULT_SHAPE],
    inferResultTypes: inferUnaryBool,
  }));

  for (const name of ['logical_and', 'logical_or']) {
    registry.register(new OpDef({
      name,
      numOperands: 2,
      numResults: 1,
      traits: [OpTrait.ELEMENTWISE, OpTrait.SAME_OPERAND_AND_RESULT_TYPE, OpTrait.COMMUTATIVE, OpTrait.ASSOCIATIVE, OpTrait.IDEMPOTENT],
      inferResultTypes: inferBinaryBool,
    }));
  }

  registry.register(new OpDef({
    name: 'clamp',
    numOperands: 3,
    numResults: 1,
    traits: [OpTrait.ELEMENTWISE],
    inferResultTypes(operandTypes) {
      if (operandTypes.length !== 3) return null;
      const x = operandTypes[1];
      if (!(x instanceof TensorType)) return null;
      return [new TensorType(x.shape, x.dtype)];
    }
  }));
}
