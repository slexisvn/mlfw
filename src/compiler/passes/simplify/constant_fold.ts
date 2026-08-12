import { FunctionPass, PassResult } from '../pass.js';
import { IRBuilder } from '../../ir/graph/builder.js';
import { registry } from '../../ir/graph/ops.js';
import { TraceLevel } from '../../pipeline/trace.js';
import { isIntType } from '../../ir/graph/types.js';
import type { AttrValue, ScalarDType, TensorType } from '../../ir/graph/types.js';
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

function isFoldResultRepresentable(value: AttrValue, dtype: string): boolean {
  if (!isIntType(dtype as ScalarDType)) return true;
  if (typeof value !== 'number') return true;
  return Number.isInteger(value) && Number.isSafeInteger(value);
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

  try {
    return def.fold(childValues, op.attributes, childOps as never);
  } catch (e) {
    return undefined;
  }
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
        const resultVal = def.fold(constValues, op.attributes, constOps as never);
        if (resultVal === undefined) continue;
        if (!isFoldResultRepresentable(resultVal, (op.getResult(0).type as TensorType).dtype)) continue;

        builder.block = op.parentBlock as Block;
        builder.setInsertionPoint(op);
        const newConst = builder.constant(resultVal, op.getResult(0).type as TensorType);
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
