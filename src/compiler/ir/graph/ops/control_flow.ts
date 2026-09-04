import { OpAttrKey, OpDef, SideEffectKind, OpTrait } from '../op_registry.js';
import type { OpRegistry } from '../op_registry.js';
import { TensorType } from '../types.js';
import { sizesClauseErrors } from '../mlir_format.js';
import type { IRType } from '../types.js';
import type { Operation } from '../operation.js';
import type { Value } from '../value.js';
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

export function scanGroups(op: Operation): {
  carries: Value[]; xs: Value[]; consts: Value[]; sizes: Value[];
} {
  const numCarry = op.getAttr<number>('num_carry') as number;
  const numXs = op.getAttr<number>('num_xs') as number;
  const numConsts = op.getAttr<number>('num_consts') as number;
  return {
    carries: op.operands.slice(0, numCarry),
    xs: op.operands.slice(numCarry, numCarry + numXs),
    consts: op.operands.slice(numCarry + numXs, numCarry + numXs + numConsts),
    sizes: op.operands.slice(numCarry + numXs + numConsts),
  };
}

function yieldedTypes(op: Operation, regionIndex: number): readonly IRType[] | null {
  const region = op.regions[regionIndex];
  const block = region ? region.entryBlock : null;
  const terminator = block ? block.lastOp : null;
  if (!terminator || !terminator.isTerminator()) return null;
  return terminator.operands.map((v) => v.type);
}

function verifyRegionArgs(op: Operation, label: string): string[] {
  const want = op.def!.inferRegionArgTypes!(op);
  const errs: string[] = [];
  for (let r = 0; r < op.regions.length; r++) {
    const block = op.regions[r].entryBlock;
    if (!block) continue;
    const expected = want[r];
    if (block.arguments.length !== expected.length) {
      errs.push(`${label} body takes ${block.arguments.length} arguments but the op provides ${expected.length}`);
      continue;
    }
    for (let i = 0; i < expected.length; i++) {
      if (!typesMatch(block.arguments[i].type, expected[i])) {
        errs.push(`${label} body argument ${i} has type ${block.arguments[i].type} but the op provides ${expected[i]}`);
      }
    }
  }
  return errs;
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
    numOperands: -1,
    numResults: -1,
    opAttrs: { [OpAttrKey.ISOLATED_REGIONS]: true },
    hasRegions: true,
    numRegions: 2,
    traits: [OpTrait.RECURSIVE_MEMORY_EFFECTS],
    inferRegionArgTypes(op) {
      const inputTypes = op.operands.slice(1).map(v => v.type);
      return op.regions.map(() => inputTypes);
    },
    inferResultTypesFromRegions(op) {
      return yieldedTypes(op, 0);
    },
    verify(op) {
      if (op.numOperands < 1) return ['if requires a predicate operand'];
      return verifyRegionArgs(op, 'if');
    },
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
    traits: [OpTrait.RECURSIVE_MEMORY_EFFECTS],
    inferRegionArgTypes(op) {
      const carried = op.operands.map(v => v.type);
      return op.regions.map(() => carried);
    },
    inferResultTypes(operandTypes) {
      return [...operandTypes];
    }
  }));

  registry.register(new OpDef({
    name: 'scan',
    numOperands: -1,
    numResults: -1,
    opAttrs: { [OpAttrKey.SEQUENTIAL_REGION]: true, [OpAttrKey.ISOLATED_REGIONS]: true },
    hasRegions: true,
    numRegions: 1,
    traits: [OpTrait.RECURSIVE_MEMORY_EFFECTS],
    attrs: [
      { name: 'num_carry', type: 'number', required: true },
      { name: 'num_xs', type: 'number', required: true },
      { name: 'num_consts', type: 'number', required: true }
    ],
    inferRegionArgTypes(op) {
      const groups = scanGroups(op);
      return [[
        ...groups.carries.map(v => v.type),
        ...groups.xs.map(v => (v.type as TensorType).dropLeadingAxis()),
        ...groups.consts.map(v => v.type)
      ]];
    },
    inferResultTypesFromRegions(op) {
      const yielded = yieldedTypes(op, 0);
      if (!yielded) return null;
      const numCarry = op.getAttr<number>('num_carry') as number;
      const xs = op.getOperand(numCarry);
      if (!xs || !(xs.type instanceof TensorType)) return null;
      const steps = xs.type.shape[0];
      return yielded.map((type, i) => (
        i < numCarry || !(type instanceof TensorType)
          ? type
          : new TensorType([steps, ...type.shape], type.dtype)
      ));
    },
    verify(op) {
      const numCarry = op.getAttr<number>('num_carry') as number;
      const numXs = op.getAttr<number>('num_xs') as number;
      if (numXs < 1) return ['scan requires at least one xs input to take its trip count from'];
      if (numCarry + numXs > op.numOperands) {
        return [`scan declares ${numCarry} carries and ${numXs} inputs but has ${op.numOperands} operands`];
      }
      return [...sizesClauseErrors(op), ...verifyRegionArgs(op, 'scan')];
    },
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
