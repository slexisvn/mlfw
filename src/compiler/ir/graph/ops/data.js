import { OpDef, SideEffectKind, OpTrait } from '../op_registry.js';
import { TensorType, TupleType } from '../types.js';

export function register(registry) {
  registry.register(new OpDef({
    name: 'constant',
    numOperands: 0,
    numResults: 1,
    attrs: [
      { name: 'value', type: 'any', required: true },
      { name: 'tensor_type', type: 'object', required: true }
    ],
    traits: [OpTrait.CONSTANT],
    inferResultTypes(operandTypes, attrs) {
      const tt = attrs.get ? attrs.get('tensor_type') : attrs.tensor_type;
      if (!tt) return null;
      return [tt];
    }
  }));

  registry.register(new OpDef({
    name: 'iota',
    numOperands: 0,
    numResults: 1,
    attrs: [
      { name: 'iota_dimension', type: 'number', required: true },
      { name: 'tensor_type', type: 'object', required: true }
    ],
    inferResultTypes(operandTypes, attrs) {
      const tt = attrs.get ? attrs.get('tensor_type') : attrs.tensor_type;
      return tt ? [tt] : null;
    }
  }));

  registry.register(new OpDef({
    name: 'tuple',
    numOperands: -1,
    numResults: 1,
    inferResultTypes(operandTypes) {
      return [new TupleType(operandTypes)];
    }
  }));

  registry.register(new OpDef({
    name: 'get_tuple_element',
    numOperands: 1,
    numResults: 1,
    attrs: [{ name: 'index', type: 'number', required: true }],
    inferResultTypes(operandTypes, attrs) {
      if (operandTypes.length !== 1) return null;
      const tup = operandTypes[0];
      if (!tup || !tup.elements) return null;
      const idx = attrs.get ? attrs.get('index') : attrs.index;
      if (idx === undefined || idx < 0 || idx >= tup.elements.length) return null;
      return [tup.elements[idx]];
    }
  }));

  registry.register(new OpDef({
    name: 'convert',
    numOperands: 1,
    numResults: 1,
    attrs: [{ name: 'target_dtype', type: 'string', required: true }],
    traits: [OpTrait.ELEMENTWISE],
    inferResultTypes(operandTypes, attrs) {
      if (operandTypes.length !== 1) return null;
      const inp = operandTypes[0];
      if (!(inp instanceof TensorType)) return null;
      const dt = attrs.get ? attrs.get('target_dtype') : attrs.target_dtype;
      return [new TensorType(inp.shape, dt)];
    }
  }));
}
