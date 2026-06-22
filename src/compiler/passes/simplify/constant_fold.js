import { FunctionPass, PassResult } from '../pass.js';
import { IRBuilder } from '../../ir/graph/builder.js';
import { registry } from '../../ir/graph/ops.js';
import { OpTrait } from '../../ir/graph/op_registry.js';
import { TraceLevel } from '../../pipeline/trace.js';
import { isIntType } from '../../ir/graph/types.js';

const CONSTANT_OPS = new Set(['constant', 'scalar_constant']);

function isFoldResultRepresentable(value, dtype) {
  if (!isIntType(dtype)) return true;
  if (typeof value !== 'number') return true;
  return Number.isInteger(value) && Number.isSafeInteger(value);
}

function resolveConstantValue(value, visited, memo) {
  if (memo.has(value)) return memo.get(value);
  const result = computeConstantValue(value, visited, memo);
  memo.set(value, result);
  return result;
}

function computeConstantValue(value, visited, memo) {
  const op = value.definingOp;
  if (!op) return undefined;
  if (CONSTANT_OPS.has(op.opName)) return op.getAttr('value');
  if (visited.has(op)) return undefined;
  visited.add(op);

  const def = registry.get(op.opName);
  if (!def || !def.fold) return undefined;
  if (def.sideEffects || (def.hasTrait && def.hasTrait(OpTrait.SIDE_EFFECT))) return undefined;
  if (op.regions.length > 0) return undefined;

  const childValues = new Array(op.numOperands);
  const childOps = new Array(op.numOperands);
  for (let i = 0; i < op.numOperands; i++) {
    const v = resolveConstantValue(op.getOperand(i), visited, memo);
    if (v === undefined) return undefined;
    childValues[i] = v;
    childOps[i] = op.getOperand(i).definingOp;
  }

  try {
    return def.fold(childValues, op.attributes, childOps);
  } catch (e) {
    return undefined;
  }
}

export class ConstantFoldPass extends FunctionPass {
  constructor() {
    super('constant_fold');
  }

  run(func, analysisManager) {
    let changed = false;
    let foldedCount = 0;
    const builder = new IRBuilder(func);
    const memo = new Map();

    for (const op of [...func.opsRecursive()]) {
      if (!op.parentBlock) continue;
      if (CONSTANT_OPS.has(op.opName)) continue;

      const def = registry.get(op.opName);
      if (!def || op.regions.length > 0) continue;
      if (def.sideEffects || (def.hasTrait && def.hasTrait(OpTrait.SIDE_EFFECT))) continue;
      if (def.getMemoryEffects && def.getMemoryEffects(op).length > 0) continue;
      if (!def.fold) continue;
      if (op.numOperands === 0) continue;

      const constValues = new Array(op.numOperands);
      const constOps = new Array(op.numOperands);
      let allResolved = true;

      for (let i = 0; i < op.numOperands; i++) {
        const v = resolveConstantValue(op.getOperand(i), new Set(), memo);
        if (v === undefined) { allResolved = false; break; }
        constValues[i] = v;
        constOps[i] = op.getOperand(i).definingOp;
      }

      if (!allResolved) continue;

      try {
        const resultVal = def.fold(constValues, op.attributes, constOps);
        if (resultVal === undefined) continue;
        if (!isFoldResultRepresentable(resultVal, op.getResult(0).type.dtype)) continue;

        builder.block = op.parentBlock;
        builder.setInsertionPoint(op);
        const newConst = builder.constant(resultVal, op.getResult(0).type);
        op.replaceAllResultsWith([newConst.getResult(0)]);
        op.erase();
        changed = true;
        foldedCount++;
      } catch (e) {
      }
    }

    if (this.trace && this.trace.level >= TraceLevel.DEBUG && foldedCount > 0) {
      this.trace.emit({
        type: 'pass_detail', passName: this.name,
        foldedCount, level: TraceLevel.DEBUG,
      });
    }

    return changed ? PassResult.CHANGED : PassResult.UNCHANGED;
  }
}
