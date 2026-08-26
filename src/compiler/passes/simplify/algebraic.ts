import { FunctionPass } from '../pass.js';
import { PatternSet } from '../../ir/rewrite/pattern.js';
import { PatternApplicator } from '../rewrite/pattern.js';
import * as pat from '../../ir/graph/patterns.js';
import type { Pattern } from '../../ir/rewrite/pattern.js';
import type { GraphFunction } from '../../ir/graph/function.js';
import type { AnalysisManager } from '../../analysis/analysis_manager.js';
import type { PassResultValue, PassTarget } from '../pass.js';

export type AlgebraicSimplificationOpts = Readonly<{
  fastMath?: boolean;
  ownedElsewhere?: ReadonlySet<string> | null;
}>;

function registryDeclaredPatterns(fastMath: boolean): Pattern[] {
  return [
    new pat.AddZero(fastMath),
    new pat.SubZero(),
    new pat.SubSelf(fastMath),
    new pat.MulOne(),
    new pat.MulZero(fastMath),
    new pat.DivOne(),
    new pat.DoubleNeg(),
    new pat.ReshapeReshape(),
  ];
}

function crossOpPatterns(fastMath: boolean): Pattern[] {
  const patterns: Pattern[] = [
    new pat.TransposeTranspose(),
    new pat.MulNegNeg(),
    new pat.AddNegToSub(),
    new pat.SubNegToAdd(),
    new pat.DoubleConvert(fastMath),
  ];
  if (fastMath) {
    patterns.push(new pat.DivSelf(fastMath));
    patterns.push(new pat.ExpLog(fastMath));
    patterns.push(new pat.LogExp(fastMath));
  }
  return patterns;
}

function buildAlgebraicPatterns(fastMath: boolean, ownedElsewhere: ReadonlySet<string> | null): PatternSet {
  const set = new PatternSet();
  for (const pattern of [...registryDeclaredPatterns(fastMath), ...crossOpPatterns(fastMath)]) {
    if (ownedElsewhere && ownedElsewhere.has(pattern.name)) continue;
    set.add(pattern);
  }
  return set;
}

export class AlgebraicSimplificationPass extends FunctionPass {
  patterns: PatternSet;

  constructor(opts: AlgebraicSimplificationOpts = {}) {
    super('algebraic_simplify');
    this.preservedAnalyses = new Set();
    this.patterns = buildAlgebraicPatterns(!!opts.fastMath, opts.ownedElsewhere ?? null);
  }

  override run(func: PassTarget, analysisManager?: AnalysisManager): PassResultValue {
    const applicator = new PatternApplicator(this.patterns);
    return applicator.applyPatterns(func as GraphFunction, { trace: this.trace, category: this.name }) as PassResultValue;
  }
}
