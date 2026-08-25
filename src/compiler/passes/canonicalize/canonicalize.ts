import { FunctionPass } from '../pass.js';
import { PatternSet } from '../../ir/rewrite/pattern.js';
import { PatternApplicator } from '../rewrite/pattern.js';
import { registry } from '../../ir/graph/ops.js';
import { OpTrait } from '../../ir/graph/op_registry.js';
import { CommutativeConstantRight, IdempotentSelf, AssociativeConstantReassoc } from '../../ir/graph/patterns.js';
import type { OpDef } from '../../ir/graph/op_registry.js';
import type { Pattern } from '../../ir/rewrite/pattern.js';
import type { GraphFunction } from '../../ir/graph/function.js';
import type { AnalysisManager } from '../../analysis/analysis_manager.js';
import type { PassResultValue, PassTarget } from '../pass.js';

const _cachedPatterns = new Map<boolean, PatternSet>();
const _cachedNames = new Map<boolean, ReadonlySet<string>>();

function traitPatternsFor(def: OpDef, fastMath: boolean): Pattern[] {
  const patterns: Pattern[] = [];
  if (def.isCommutative) {
    patterns.push(new CommutativeConstantRight(def.name));
    if (def.isAssociative && def.fold) patterns.push(new AssociativeConstantReassoc(def.name, fastMath));
  }
  if (def.hasTrait(OpTrait.IDEMPOTENT)) patterns.push(new IdempotentSelf(def.name));
  return patterns;
}

function getCanonicalizationPatterns(fastMath: boolean): PatternSet {
  const cached = _cachedPatterns.get(fastMath);
  if (cached) return cached;
  const set = new PatternSet();
  for (const def of registry.allOps()) {
    for (const p of traitPatternsFor(def, fastMath)) set.add(p);
    if (def.getCanonicalizationPatterns) {
      const opPatterns = def.getCanonicalizationPatterns(fastMath);
      if (opPatterns) {
        for (const p of opPatterns) {
          set.add(p);
        }
      }
    }
  }
  _cachedPatterns.set(fastMath, set);
  return set;
}

export class CanonicalizePass extends FunctionPass {
  fastMath: boolean;

  constructor(opts: Readonly<{ fastMath?: boolean }> = {}) {
    super('canonicalize');
    this.fastMath = !!opts.fastMath;
  }

  override run(func: PassTarget, analysisManager?: AnalysisManager): PassResultValue {
    const patterns = getCanonicalizationPatterns(this.fastMath);
    const applicator = new PatternApplicator(patterns);
    return applicator.applyPatterns(func as GraphFunction, 10, this.trace) as PassResultValue;
  }
}

export function canonicalizationPatternNames(fastMath: boolean): ReadonlySet<string> {
  const cached = _cachedNames.get(fastMath);
  if (cached) return cached;
  const names = new Set<string>();
  for (const pattern of getCanonicalizationPatterns(fastMath).patterns) names.add(pattern.name);
  _cachedNames.set(fastMath, names);
  return names;
}

export function resetCanonicalizationCache(): void {
  _cachedPatterns.clear();
  _cachedNames.clear();
}
