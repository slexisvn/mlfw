import { OpDef, SideEffectKind, OpTrait } from '../op_registry.js';
import { TensorType } from '../types.js';

export function register(registry) {
  registry.register(new OpDef({
    name: 'copy_to_device',
    numOperands: 1,
    numResults: 1,
    attrs: [
      { name: 'src_device', type: 'string', required: true },
      { name: 'dst_device', type: 'string', required: true },
    ],
    sideEffects: SideEffectKind.READ | SideEffectKind.WRITE,
    traits: [OpTrait.INJECTIVE],
    inferResultTypes(operandTypes) {
      if (operandTypes.length !== 1) return null;
      return [operandTypes[0]];
    },
    getFlops() {
      return 0;
    },
  }));
}
