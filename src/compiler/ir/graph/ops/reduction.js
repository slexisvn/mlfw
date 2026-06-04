import { OpDef, OpTrait } from '../op_registry.js';
import { TensorType } from '../types.js';

const VALID_REDUCE_TYPES = new Set(['sum', 'max', 'min', 'prod', 'mean', 'and', 'or']);

export function register(registry) {
  registry.register(new OpDef({
    name: 'reduce',
    numOperands: 2,
    numResults: 1,
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
      const dims = attrs.get ? attrs.get('dimensions') : attrs.dimensions;
      if (!dims) return null;
      const dimSet = new Set(dims);
      const newShape = [];
      for (let i = 0; i < inp.rank; i++) {
        if (!dimSet.has(i)) newShape.push(inp.shape[i]);
      }
      return [new TensorType(newShape, inp.dtype)];
    },
    verify(op) {
      const errs = [];
      if (!op.hasAttr('dimensions')) errs.push('reduce missing dimensions');
      if (!op.hasAttr('reduce_type')) errs.push('reduce missing reduce_type');
      else {
        const rt = op.getAttr('reduce_type');
        if (!VALID_REDUCE_TYPES.has(rt)) errs.push(`reduce invalid reduce_type: ${rt}`);
      }
      return errs;
    }
  }));
}
