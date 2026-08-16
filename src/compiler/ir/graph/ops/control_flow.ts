import { OpAttrKey, OpDef, SideEffectKind, OpTrait } from '../op_registry.js';
import type { OpRegistry } from '../op_registry.js';
import type { IRType } from '../types.js';
import type { Operation } from '../operation.js';
import type { GraphFunction } from '../function.js';

export function owningFunctionOf(op: Operation): GraphFunction | null {
  return op.parentBlock ? op.parentBlock._owningFunction() : null;
}

export function calleeFunction(op: Operation, callee: string): GraphFunction | null {
  const owner = owningFunctionOf(op);
  const module = owner ? owner._module : null;
  return module ? module.getFunction(callee) : null;
}

function typesMatch(a: IRType | null | undefined, b: IRType | null | undefined): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  return typeof a.equals === 'function' ? a.equals(b) : false;
}

export function register(registry: OpRegistry) {
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
    opAttrs: { [OpAttrKey.SEQUENTIAL_REGION]: true },
    hasRegions: true,
    numRegions: 2,
    sideEffects: SideEffectKind.CONTROL,
    inferResultTypes(operandTypes) {
      return [...operandTypes];
    }
  }));

  registry.register(new OpDef({
    name: 'scan',
    numOperands: -1,
    numResults: -1,
    opAttrs: { [OpAttrKey.SEQUENTIAL_REGION]: true },
    hasRegions: true,
    numRegions: 1,
    sideEffects: SideEffectKind.CONTROL,
    attrs: [
      { name: 'num_carry', type: 'number', required: true },
      { name: 'num_xs', type: 'number', required: true }
    ],
    inferResultTypes(operandTypes, attrs, resultTypes) {
      return resultTypes || null;
    }
  }));

  registry.register(new OpDef({
    name: 'call',
    numOperands: -1,
    numResults: -1,
    sideEffects: SideEffectKind.CONTROL,
    attrs: [{ name: 'callee', type: 'string', required: true }],
    verify(op) {
      const errs: string[] = [];
      const callee = op.getAttr<string>('callee')!;
      if (typeof callee !== 'string' || callee.length === 0) {
        errs.push('call requires a non-empty string callee');
        return errs;
      }
      const target = calleeFunction(op, callee);
      if (target === null) return errs;
      if (target.name === owningFunctionOf(op)!.name) {
        errs.push(`call to '${callee}' is directly recursive`);
      }
      if (target.inputTypes.length !== op.numOperands) {
        errs.push(`call to '${callee}' passes ${op.numOperands} operands but it takes ${target.inputTypes.length}`);
      }
      if (target.outputTypes.length !== op.numResults) {
        errs.push(`call to '${callee}' has ${op.numResults} results but it returns ${target.outputTypes.length}`);
      }
      const arity = Math.min(target.inputTypes.length, op.numOperands);
      for (let i = 0; i < arity; i++) {
        const got = op.getOperand(i).type;
        const want = target.inputTypes[i];
        if (!typesMatch(got, want)) {
          errs.push(`call to '${callee}' operand ${i} has type ${got} but the callee declares ${want}`);
        }
      }
      const results = Math.min(target.outputTypes.length, op.numResults);
      for (let i = 0; i < results; i++) {
        const got = op.getResult(i).type;
        const want = target.outputTypes[i];
        if (!typesMatch(got, want)) {
          errs.push(`call to '${callee}' result ${i} has type ${got} but the callee returns ${want}`);
        }
      }
      return errs;
    },
    inferResultTypes(operandTypes, attrs, resultTypes) {
      return resultTypes || null;
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
    opAttrs: { [OpAttrKey.LAUNCH_BOUNDARY]: 'matmul' },
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
