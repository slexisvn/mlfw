import { FunctionPass, PassResult } from '../pass.js';
import { registry } from '../../ir/graph/ops.js';
import { TraceLevel } from '../../pipeline/trace.js';
import type { GraphFunction } from '../../ir/graph/function.js';
import type { Block } from '../../ir/graph/block.js';
import type { Operation } from '../../ir/graph/operation.js';
import type { AnalysisManager } from '../../analysis/analysis_manager.js';
import type { PassResultValue, PassTarget } from '../pass.js';

class ScopedOpTable {
  private _scopes: Map<number, Operation[]>[];

  constructor() {
    this._scopes = [];
  }

  push(): void {
    this._scopes.push(new Map());
  }

  pop(): void {
    this._scopes.pop();
  }

  lookup(op: Operation): Operation | null {
    const hash = op.structuralHash();
    for (let i = this._scopes.length - 1; i >= 0; i--) {
      const bucket = this._scopes[i].get(hash);
      if (!bucket) continue;
      for (const candidate of bucket) {
        if (candidate.parentBlock && candidate.structuralEquals(op)) return candidate;
      }
    }
    return null;
  }

  insert(op: Operation): void {
    const hash = op.structuralHash();
    const top = this._scopes[this._scopes.length - 1];
    const bucket = top.get(hash);
    if (bucket) bucket.push(op);
    else top.set(hash, [op]);
  }
}

function isRedundancyCandidate(op: Operation): boolean {
  if (op.regions.length > 0) return false;
  const def = registry.get(op.opName);
  if (!def) return true;
  if (def.hasSideEffects) return false;
  if (def.getMemoryEffects && def.getMemoryEffects(op).length > 0) return false;
  return true;
}

export class CSEPass extends FunctionPass {
  constructor() {
    super('cse');
    this.preservedAnalyses = new Set();
  }

  override run(target: PassTarget, analysisManager?: AnalysisManager): PassResultValue {
    const func = target as GraphFunction;
    const table = new ScopedOpTable();
    let eliminated = 0;
    for (const block of func.body) {
      eliminated += this._simplifyBlock(block, table);
    }

    if (this.trace && this.trace.level >= TraceLevel.DEBUG && eliminated > 0) {
      this.trace.emit({
        type: 'pass_detail', passName: this.name,
        eliminated, level: TraceLevel.DEBUG,
      });
    }

    return eliminated > 0 ? PassResult.CHANGED : PassResult.UNCHANGED;
  }

  _simplifyBlock(block: Block, table: ScopedOpTable): number {
    table.push();
    let eliminated = 0;

    for (const op of [...block.ops()]) {
      if (!op.parentBlock) continue;

      if (isRedundancyCandidate(op)) {
        const existing = table.lookup(op);
        if (existing) {
          op.replaceAllResultsWith(existing.results);
          op.erase();
          eliminated++;
          continue;
        }
        table.insert(op);
      }

      for (const region of op.regions) {
        for (const inner of region.blocks) {
          eliminated += this._simplifyBlock(inner, table);
        }
      }
    }

    table.pop();
    return eliminated;
  }
}
