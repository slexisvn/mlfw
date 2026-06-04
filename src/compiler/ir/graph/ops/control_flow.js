import { OpDef, SideEffectKind, OpTrait } from '../op_registry.js';

export function register(registry) {
  registry.register(new OpDef({
    name: 'return',
    numOperands: -1,
    numResults: 0,
    traits: [OpTrait.TERMINATOR]
  }));

  registry.register(new OpDef({
    name: 'yield',
    numOperands: -1,
    numResults: 0,
    traits: [OpTrait.TERMINATOR]
  }));

  registry.register(new OpDef({
    name: 'if',
    numOperands: 1,
    numResults: -1,
    hasRegions: true,
    numRegions: 2,
    sideEffects: SideEffectKind.CONTROL,
    inferResultTypes(operandTypes, attrs, resultTypes) {
      return resultTypes || null;
    }
  }));

  registry.register(new OpDef({
    name: 'while',
    numOperands: -1,
    numResults: -1,
    hasRegions: true,
    numRegions: 2,
    sideEffects: SideEffectKind.CONTROL,
    inferResultTypes(operandTypes) {
      return [...operandTypes];
    }
  }));

  registry.register(new OpDef({
    name: 'custom_call',
    numOperands: -1,
    numResults: -1,
    attrs: [
      { name: 'call_target_name', type: 'string', required: true },
      { name: 'backend_config', type: 'any', required: false }
    ],
    sideEffects: SideEffectKind.WRITE,
    inferResultTypes(operandTypes, attrs, resultTypes) {
      return resultTypes || null;
    }
  }));

  registry.register(new OpDef({
    name: 'fused_dot_epilogue',
    numOperands: -1,
    numResults: 1,
    traits: [OpTrait.OPAQUE],
    attrs: [
      { name: 'lhs_contracting', type: 'array', required: true },
      { name: 'rhs_contracting', type: 'array', required: true },
      { name: 'lhs_batch', type: 'array', required: false },
      { name: 'rhs_batch', type: 'array', required: false },
      { name: 'epilogue_ops', type: 'array', required: true },
      { name: 'epilogue_tags', type: 'array', required: true },
      { name: 'num_dot_operands', type: 'number', required: true },
      { name: 'num_extra_inputs', type: 'number', required: true }
    ],
    inferResultTypes(operandTypes, attrs, resultTypes) {
      return resultTypes || null;
    }
  }));

  registry.register(new OpDef({
    name: 'fusion',
    numOperands: -1,
    numResults: -1,
    hasRegions: true,
    numRegions: 1,
    attrs: [{ name: 'fusion_kind', type: 'string', required: false }],
    inferResultTypes(operandTypes, attrs, resultTypes) {
      return resultTypes || null;
    }
  }));
}
