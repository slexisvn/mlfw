import { OpDef, OpTrait } from '../op_registry.js';
import type { OpAttrRecord, OpDefConfig, OpRegistry } from '../op_registry.js';
import { TensorType, TupleType, ScalarType } from '../types.js';
import { sizesClauseErrors } from '../mlir_format.js';
import type { IRType } from '../types.js';
import type { ScalarDType } from '../types.js';

export function register(registry: OpRegistry) {
  const constantConfig: Omit<OpDefConfig, 'name'> = {
    numOperands: 0,
    numResults: 1,
    attrs: [
      { name: 'value', type: 'any', required: true },
      { name: 'tensor_type', type: 'object', required: true }
    ],
    traits: [OpTrait.CONSTANT],
    inferResultTypes(operandTypes, attrs) {
      const tt = (attrs.get ? attrs.get('tensor_type') : (attrs as unknown as OpAttrRecord).tensor_type) as IRType | undefined;
      if (!tt) return null;
      return [tt];
    }
  };
  for (const name of ['constant', 'scalar_constant']) {
    registry.register(new OpDef({ name, ...constantConfig }));
  }

  registry.register(new OpDef({
    name: 'iota',
    numOperands: -1,
    numResults: 1,
    attrs: [
      { name: 'iota_dimension', type: 'number', required: true },
      { name: 'tensor_type', type: 'object', required: true }
    ],
    verify(op) {
      const type = op.getAttr('tensor_type') as unknown as IRType | undefined;
      if (!(type instanceof TensorType)) return ['iota missing tensor_type'];
      return sizesClauseErrors(op);
    },
    inferResultTypes(operandTypes, attrs) {
      const tt = (attrs.get ? attrs.get('tensor_type') : (attrs as unknown as OpAttrRecord).tensor_type) as IRType | undefined;
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
      const tupleType = tup as TupleType;
      if (!tup || !tupleType.elements) return null;
      const idx = (attrs.get ? attrs.get('index') : (attrs as unknown as OpAttrRecord).index) as number;
      if (idx === undefined || idx < 0 || idx >= tupleType.elements.length) return null;
      return [tupleType.elements[idx]];
    }
  }));

  registry.register(new OpDef({
    name: 'dim',
    numOperands: 1,
    numResults: 1,
    attrs: [{ name: 'dimension', type: 'number', required: true }],
    verify(op) {
      const type = op.getOperand(0).type as TensorType;
      const axis = op.getAttr<number>('dimension') as number;
      if (!Number.isInteger(axis) || axis < 0 || axis >= type.shape.length) {
        return [`dim axis ${axis} is out of range for rank ${type.shape.length}`];
      }
      return [];
    },
    inferResultTypes() {
      return [new TensorType([], 'i64')];
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
      const dt = (attrs.get ? attrs.get('target_dtype') : (attrs as unknown as OpAttrRecord).target_dtype) as ScalarDType;
      return [new TensorType(inp.shape, dt)];
    }
  }));

  registry.register(new OpDef({
    name: 'one_hot',
    numOperands: 1,
    numResults: 1,
    attrs: [
      { name: 'depth', type: 'number', required: true },
      { name: 'axis', type: 'number', required: false },
      { name: 'on_value', type: 'number', required: false },
      { name: 'off_value', type: 'number', required: false },
      { name: 'dtype', type: 'string', required: false }
    ],
    inferResultTypes(operandTypes, attrs) {
      if (operandTypes.length < 1) return null;
      const indices = operandTypes[0];
      if (!(indices instanceof TensorType)) return null;
      const depth = (attrs.get ? attrs.get('depth') : (attrs as unknown as OpAttrRecord).depth) as number;
      const axis = ((attrs.get ? attrs.get('axis') : (attrs as unknown as OpAttrRecord).axis) as number) ?? -1;
      const dtype = ((attrs.get ? attrs.get('dtype') : (attrs as unknown as OpAttrRecord).dtype) as ScalarDType) || ScalarType.F32;
      const shape = [...indices.shape];
      const insertAt = axis < 0 ? shape.length + 1 + axis : axis;
      shape.splice(insertAt, 0, depth);
      return [new TensorType(shape, dtype)];
    }
  }));
}
