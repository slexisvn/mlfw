import { OpDef, OpTrait } from '../op_registry.js';
import { TensorType, ScalarType } from '../types.js';
import { inferCompare } from './helpers.js';

const VALID_DIRECTIONS = new Set(['eq', 'ne', 'lt', 'le', 'gt', 'ge']);

export function register(registry) {
  registry.register(new OpDef({
    name: 'compare',
    numOperands: 2,
    numResults: 1,
    attrs: [{ name: 'direction', type: 'string', required: true }],
    traits: [OpTrait.ELEMENTWISE],
    inferResultTypes: inferCompare,
    verify(op) {
      const errs = [];
      if (op.numOperands !== 2) { errs.push('compare expects 2 operands'); return errs; }
      if (!op.hasAttr('direction')) errs.push('compare missing direction attr');
      else {
        const d = op.getAttr('direction');
        if (!VALID_DIRECTIONS.has(d)) errs.push(`compare invalid direction: ${d}`);
      }
      return errs;
    }
  }));

  registry.register(new OpDef({
    name: 'select',
    numOperands: 3,
    numResults: 1,
    traits: [OpTrait.ELEMENTWISE],
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
    inferResultTypes(operandTypes) {
      if (operandTypes.length !== 3) return null;
      const x = operandTypes[1];
      if (!(x instanceof TensorType)) return null;
      return [new TensorType(x.shape, x.dtype)];
    }
  }));

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
