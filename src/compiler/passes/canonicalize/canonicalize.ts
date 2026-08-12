import { FunctionPass } from '../pass.js';
import { PatternSet } from '../../ir/rewrite/pattern.js';
import { PatternApplicator } from '../rewrite/pattern.js';
import { registry } from '../../ir/graph/ops.js';
import type { GraphFunction } from '../../ir/graph/function.js';
import type { AnalysisManager } from '../../analysis/analysis_manager.js';
import type { PassResultValue, PassTarget } from '../pass.js';

let _cachedPatterns: PatternSet | null = null;

function getCanonicalizationPatterns(): PatternSet {
  if (_cachedPatterns) return _cachedPatterns;
  _cachedPatterns = new PatternSet();
  for (const def of registry.allOps()) {
    if (def.getCanonicalizationPatterns) {
      const opPatterns = def.getCanonicalizationPatterns();
      if (opPatterns) {
        for (const p of opPatterns) {
          _cachedPatterns.add(p);
        }
      }
    }
  }
  return _cachedPatterns;
}

export class CanonicalizePass extends FunctionPass {
  constructor() {
    super('canonicalize');
  }

  override run(func: PassTarget, analysisManager?: AnalysisManager): PassResultValue {
    const patterns = getCanonicalizationPatterns();
    const applicator = new PatternApplicator(patterns);
    return applicator.applyPatterns(func as GraphFunction, 10, this.trace) as PassResultValue;
  }
}

export function resetCanonicalizationCache(): void {
  _cachedPatterns = null;
}
