import { FunctionPass, PassResult } from '../pass.js';
import { PatternSet, PatternApplicator } from '../rewrite/pattern.js';
import { registry } from '../../ir/graph/ops.js';

let _cachedPatterns = null;

function getCanonicalizationPatterns() {
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

  run(func, analysisManager) {
    const patterns = getCanonicalizationPatterns();
    const applicator = new PatternApplicator(patterns);
    return applicator.applyPatterns(func, 10, this.trace);
  }
}

export function resetCanonicalizationCache() {
  _cachedPatterns = null;
}
