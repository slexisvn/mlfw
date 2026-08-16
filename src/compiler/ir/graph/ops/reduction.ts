import { OpAttrKey, OpDef, OpTrait } from '../op_registry.js';
import type { OpAttrMap, OpAttrRecord, OpRegistry } from '../op_registry.js';
import { TensorType, ScalarType, Layout } from '../types.js';
import { LayoutPreference } from '../layout_pref.js';
import type { Operation } from '../operation.js';
import type { IRType } from '../types.js';

const VALID_REDUCE_TYPES = new Set(['sum', 'max', 'min', 'prod', 'mean', 'and', 'or']);

function reduceLayout(op: Operation): LayoutPreference | null {
  const outType = op.getResult(0).type as TensorType;
  if (!outType) return null;
  return new LayoutPreference([null], [Layout.rowMajor(outType.rank)]);
}

export function register(registry: OpRegistry) {
  registry.register(new OpDef({
    name: 'reduce',
    numOperands: 2,
    numResults: 1,
    opAttrs: { [OpAttrKey.LAUNCH_BOUNDARY]: 'reduce', [OpAttrKey.LAYOUT_SENSITIVITY]: 2, [OpAttrKey.INFER_LAYOUT]: reduceLayout },
    attrs: [
      { name: 'dimensions', type: 'array', required: true },
      { name: 'reduce_type', type: 'string', required: true }
    ],
    traits: [OpTrait.REDUCTION],
    hasRegions: true,
    numRegions: 1,
    inferResultTypes(operandTypes, attrs) {
      if (operandTypes.length < 1) return null;
      const inp = operandTypes[0];
      if (!(inp instanceof TensorType)) return null;
      const dims = (attrs.get ? attrs.get('dimensions') : (attrs as unknown as OpAttrRecord).dimensions) as readonly number[];
      if (!dims) return null;
      const dimSet = new Set(dims);
      const newShape = [];
      for (let i = 0; i < inp.rank; i++) {
        if (!dimSet.has(i)) newShape.push(inp.shape[i]);
      }
      return [new TensorType(newShape, inp.dtype)];
    },
    propagateSymbolicShapes(op, shapes) {
      const inShape = shapes.get(op.getOperand(0));
      if (!inShape) return null;
      const dims = op.getAttr<readonly number[]>('dimensions')!;
      if (!dims) return null;
      const dimSet = new Set(dims);
      const resShape = [];
      for (let i = 0; i < inShape.length; i++) {
        if (!dimSet.has(i)) resShape.push(inShape[i]);
      }
      return [resShape];
    },
    verify(op) {
      const errs = [];
      if (!op.hasAttr('dimensions')) errs.push('reduce missing dimensions');
      if (!op.hasAttr('reduce_type')) errs.push('reduce missing reduce_type');
      else {
        const rt = op.getAttr<string>('reduce_type')!;
        if (!VALID_REDUCE_TYPES.has(rt)) errs.push(`reduce invalid reduce_type: ${rt}`);
      }
      return errs;
    }
  }));

  function inferArgReduceTypes(operandTypes: readonly IRType[], attrs: OpAttrMap): IRType[] | null {
    if (operandTypes.length < 1) return null;
    const inp = operandTypes[0];
    if (!(inp instanceof TensorType)) return null;
    const axis = (attrs.get ? attrs.get('axis') : (attrs as unknown as OpAttrRecord).axis) as number;
    if (axis === undefined) return null;
    const keepDims = ((attrs.get ? attrs.get('keep_dims') : (attrs as unknown as OpAttrRecord).keep_dims) as boolean) || false;
    const newShape = [];
    for (let i = 0; i < inp.rank; i++) {
      if (i === axis) { if (keepDims) newShape.push(1); }
      else newShape.push(inp.shape[i]);
    }
    return [new TensorType(newShape, ScalarType.I32)];
  }

  function propagateArgReduceShapes(op: Operation, shapes: ReadonlyMap<unknown, readonly unknown[]>): (readonly unknown[])[] | null {
    const inShape = shapes.get(op.getOperand(0));
    if (!inShape) return null;
    const axis = op.getAttr<number>('axis') as number;
    if (axis === undefined) return null;
    const keepDims = op.getAttr<boolean>('keep_dims') || false;
    const resShape: unknown[] = [];
    for (let i = 0; i < inShape.length; i++) {
      if (i === axis) { if (keepDims) resShape.push(1); }
      else resShape.push(inShape[i]);
    }
    return [resShape];
  }

  registry.register(new OpDef({
    name: 'argmax',
    numOperands: 1,
    numResults: 1,
    attrs: [
      { name: 'axis', type: 'number', required: true },
      { name: 'keep_dims', type: 'boolean', required: false }
    ],
    traits: [OpTrait.REDUCTION],
    inferResultTypes: inferArgReduceTypes,
    propagateSymbolicShapes: propagateArgReduceShapes as never,
  }));

  registry.register(new OpDef({
    name: 'argmin',
    numOperands: 1,
    numResults: 1,
    attrs: [
      { name: 'axis', type: 'number', required: true },
      { name: 'keep_dims', type: 'boolean', required: false }
    ],
    traits: [OpTrait.REDUCTION],
    inferResultTypes: inferArgReduceTypes,
    propagateSymbolicShapes: propagateArgReduceShapes as never,
  }));
}
