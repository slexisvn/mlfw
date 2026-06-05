import { OpDef, OpTrait } from '../op_registry.js';
import { TensorType } from '../types.js';

export function register(registry) {
  registry.register(new OpDef({
    name: 'resize',
    numOperands: 1,
    numResults: 1,
    traits: [OpTrait.OPAQUE],
    attrs: [
      { name: 'output_size', type: 'array', required: true },
      { name: 'method', type: 'string', required: true },
      { name: 'coordinate_mode', type: 'string', required: false },
      { name: 'layout', type: 'string', required: false }
    ],
    inferResultTypes(operandTypes, attrs) {
      if (operandTypes.length < 1) return null;
      const inp = operandTypes[0];
      if (!(inp instanceof TensorType) || inp.rank !== 4) return null;
      const outSize = attrs.get ? attrs.get('output_size') : attrs.output_size;
      return [new TensorType([inp.shape[0], inp.shape[1], outSize[0], outSize[1]], inp.dtype)];
    }
  }));
}
