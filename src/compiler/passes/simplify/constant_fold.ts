import { FunctionPass, PassResult } from '../pass.js';
import { IRBuilder } from '../../ir/graph/builder.js';
import { registry } from '../../ir/graph/ops.js';
import { explainer } from '../explain.js';
import { TraceLevel } from '../../support/trace.js';
import { isIntType, TensorType } from '../../ir/graph/types.js';
import { roundToDtype } from '../../../tensor/utils/half.js';
import type { AttrValue, ScalarDType } from '../../ir/graph/types.js';
import type { GraphFunction } from '../../ir/graph/function.js';
import type { Operation } from '../../ir/graph/operation.js';
import type { Value } from '../../ir/graph/value.js';
import type { AnalysisManager } from '../../analysis/analysis_manager.js';
import type { PassResultValue, PassTarget } from '../pass.js';
import type { Block } from '../../ir/graph/block.js';

type FoldMemo = Map<Value, AttrValue | undefined>;

function isConstantProducer(opName: string): boolean {
  const def = registry.get(opName);
  return !!(def && def.isConstant);
}

function coerceFoldResult(value: AttrValue | undefined, dtype: string): AttrValue | undefined {
  if (value === undefined || typeof value !== 'number') return value;
  if (isIntType(dtype as ScalarDType)) {
    return Number.isInteger(value) && Number.isSafeInteger(value) ? value : undefined;
  }
  return roundToDtype(dtype, value);
}

function foldOperation(op: Operation, values: readonly AttrValue[], ops: readonly (Operation | null)[]): AttrValue | undefined {
  const def = registry.get(op.opName);
  if (!def || !def.fold) return undefined;
  try {
    const raw = def.fold(values as AttrValue[], op.attributes, ops as never);
    return coerceFoldResult(raw, (op.getResult(0).type as TensorType).dtype);
  } catch (e) {
    return undefined;
  }
}

function resolveConstantValue(value: Value, visited: Set<Operation>, memo: FoldMemo): AttrValue | undefined {
  if (memo.has(value)) return memo.get(value);
  const result = computeConstantValue(value, visited, memo);
  memo.set(value, result);
  return result;
}

function computeConstantValue(value: Value, visited: Set<Operation>, memo: FoldMemo): AttrValue | undefined {
  const op = value.definingOp;
  if (!op) return undefined;
  if (isConstantProducer(op.opName)) return op.getAttr('value');
  if (visited.has(op)) return undefined;
  visited.add(op);

  const def = registry.get(op.opName);
  if (!def || !def.fold) return undefined;
  if (def.hasSideEffects) return undefined;
  if (op.regions.length > 0) return undefined;

  const childValues: AttrValue[] = new Array(op.numOperands);
  const childOps: (Operation | null)[] = new Array(op.numOperands);
  for (let i = 0; i < op.numOperands; i++) {
    const v = resolveConstantValue(op.getOperand(i), visited, memo);
    if (v === undefined) return undefined;
    childValues[i] = v;
    childOps[i] = op.getOperand(i).definingOp;
  }

  return foldOperation(op, childValues, childOps);
}

export class ConstantFoldPass extends FunctionPass {
  constructor() {
    super('constant_fold');
  }

  override run(func: PassTarget, analysisManager?: AnalysisManager): PassResultValue {
    const graphFunc = func as GraphFunction;
    let changed = false;
    let foldedCount = 0;
    const builder = new IRBuilder(graphFunc);
    const explain = explainer(this.trace, this.name);
    const memo: FoldMemo = new Map();

    for (const op of [...graphFunc.opsRecursive()]) {
      if (!op.parentBlock) continue;
      if (isConstantProducer(op.opName)) continue;

      const def = registry.get(op.opName);
      if (!def || op.regions.length > 0) continue;
      if (def.hasSideEffects) continue;
      if (def.getMemoryEffects && def.getMemoryEffects(op).length > 0) continue;
      if (!def.fold) continue;
      if (op.numOperands === 0) continue;
      const resultType = op.getResult(0).type;
      if (resultType instanceof TensorType && resultType.hasDynamic) continue;

      const constValues: AttrValue[] = new Array(op.numOperands);
      const constOps: (Operation | null)[] = new Array(op.numOperands);
      let allResolved = true;

      for (let i = 0; i < op.numOperands; i++) {
        const v = resolveConstantValue(op.getOperand(i), new Set(), memo);
        if (v === undefined) { allResolved = false; break; }
        constValues[i] = v;
        constOps[i] = op.getOperand(i).definingOp;
      }

      if (!allResolved) continue;

      try {
        const resultVal = foldOperation(op, constValues, constOps);
        if (resultVal === undefined) continue;

        builder.block = op.parentBlock as Block;
        builder.setInsertionPoint(op);
        const newConst = builder.withLocation(op.loc, () => builder.constant(resultVal, op.getResult(0).type as TensorType));
        if (explain) {
          explain(op.opName, 'folded to a constant',
            'every operand was already known at compile time, so the op can run now instead of at every call');
        }
        op.replaceAllResultsWith([newConst.getResult(0)]);
        op.erase();
        changed = true;
        foldedCount++;
      } catch (e) {
        if (this.trace && this.trace.level >= TraceLevel.DEBUG) {
          this.trace.emit({
            type: 'pass_detail', passName: this.name,
            foldError: op.opName, message: (e as Error).message, level: TraceLevel.DEBUG,
          });
        }
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
