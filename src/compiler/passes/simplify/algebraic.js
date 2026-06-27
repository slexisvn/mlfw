import { FunctionPass } from '../pass.js';
import { PatternSet, PatternApplicator } from '../rewrite/pattern.js';
import * as pat from '../../ir/graph/patterns.js';

function buildAlgebraicPatterns(fastMath) {
  const set = new PatternSet();
  set.add(new pat.AddZero());
  set.add(new pat.SubZero());
  set.add(new pat.SubSelf(fastMath));
  set.add(new pat.MulOne());
  set.add(new pat.MulZero(fastMath));
  set.add(new pat.DivOne());
  set.add(new pat.DoubleNeg());
  set.add(new pat.TransposeTranspose());
  set.add(new pat.ReshapeReshape());
  set.add(new pat.MulNegNeg());
  set.add(new pat.AddNegToSub());
  set.add(new pat.SubNegToAdd());
  set.add(new pat.DoubleConvert());
  if (fastMath) {
    set.add(new pat.DivSelf(fastMath));
    set.add(new pat.ExpLog(fastMath));
    set.add(new pat.LogExp(fastMath));
  }
  return set;
}

const _soundPatterns = buildAlgebraicPatterns(false);
const _fastMathPatterns = buildAlgebraicPatterns(true);

export class AlgebraicSimplificationPass extends FunctionPass {
  constructor(opts = {}) {
    super('algebraic_simplify');
    this.preservedAnalyses = new Set();
    this.patterns = opts.fastMath ? _fastMathPatterns : _soundPatterns;
  }

  run(func, analysisManager) {
    const applicator = new PatternApplicator(this.patterns);
    return applicator.applyPatterns(func, 10, this.trace);
  }
}
