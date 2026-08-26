import { PassResult } from '../pass.js';
import { IRBuilder } from '../../ir/graph/builder.js';
import { TraceLevel } from '../../pipeline/trace.js';
import { explainer } from '../explain.js';
import type { GraphFunction } from '../../ir/graph/function.js';
import type { Operation } from '../../ir/graph/operation.js';
import type { PatternSet } from '../../ir/rewrite/pattern.js';
import type { TraceLog } from '../../pipeline/trace.js';
import type { PassResultValue } from '../pass.js';

export type ApplyPatternsOpts = Readonly<{
  maxIterations?: number;
  trace?: TraceLog | null;
  category?: string;
}>;

export class PatternApplicator {
  patternSet: PatternSet;

  constructor(patternSet: PatternSet) {
    this.patternSet = patternSet;
  }

  applyPatterns(func: GraphFunction, opts: ApplyPatternsOpts = {}): PassResultValue {
    const { maxIterations = 10, trace = null, category = 'rewrite' } = opts;
    const builder = new IRBuilder(func);
    const explain = explainer(trace, category);
    let totalRewrites = 0;

    const worklist = [...func.opsRecursive()];
    let head = 0;
    const queued = new Set<Operation>(worklist);
    const enqueue = (op: Operation | null | undefined): void => {
      if (!op || !op.parentBlock || queued.has(op)) return;
      queued.add(op);
      worklist.push(op);
    };

    const safetyBudget = Math.max(maxIterations, 1) * Math.max(worklist.length, 1) * 4 + 1000;
    let steps = 0;
    let capped = false;

    while (head < worklist.length) {
      if (++steps > safetyBudget) { capped = true; break; }
      const op = worklist[head++];
      queued.delete(op);
      if (!op.parentBlock) continue;

      const patterns = this.patternSet.getForOp(op.opName);
      for (const pattern of patterns) {
        if (!pattern.match(op)) continue;

        const block = op.parentBlock;
        const prevOp = op._prev;
        const nextOp = op._next;
        const affected: Operation[] = [];
        for (let r = 0; r < op.numResults; r++) {
          for (const u of op.getResult(r).getUsers()) affected.push(u);
        }
        for (let o = 0; o < op.numOperands; o++) {
          const d = op.getOperand(o).definingOp;
          if (d) affected.push(d);
        }

        builder.block = block;
        builder.setInsertionPoint(op);
        const rewritten = op.opName;
        if (!pattern.rewrite(op, builder)) continue;

        totalRewrites++;
        if (explain) explain(rewritten, `matched ${pattern.name}`, pattern.why);
        for (const a of affected) enqueue(a);
        let cur = prevOp ? prevOp._next : block._head;
        let guard = block._size + 2;
        while (cur && cur !== nextOp && guard-- > 0) { enqueue(cur); cur = cur._next; }
        enqueue(op);
        break;
      }
    }

    if (trace) {
      if (capped && trace.level >= TraceLevel.INFO) {
        trace.emit({
          type: 'pass_detail', passName: 'PatternApplicator',
          message: `pattern rewriting hit safety budget (${safetyBudget}) without converging`,
          totalRewrites, level: TraceLevel.INFO,
        });
      }
      if (trace.level >= TraceLevel.DEBUG && totalRewrites > 0) {
        trace.emit({
          type: 'pass_detail', passName: 'PatternApplicator',
          totalRewrites, patternCount: this.patternSet.patterns.length,
          level: TraceLevel.DEBUG,
        });
      }
    }

    return totalRewrites > 0 ? PassResult.CHANGED : PassResult.UNCHANGED;
  }
}
