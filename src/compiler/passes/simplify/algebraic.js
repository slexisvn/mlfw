import { FunctionPass } from '../pass.js';
import { PatternSet, PatternApplicator } from '../rewrite/pattern.js';
import { ShapeAnalysis } from '../../analysis/shape_analysis.js';
import * as pat from '../../ir/graph/patterns.js';

const _algebraicPatterns = new PatternSet();
_algebraicPatterns.add(new pat.AddZero());
_algebraicPatterns.add(new pat.SubZero());
_algebraicPatterns.add(new pat.SubSelf());
_algebraicPatterns.add(new pat.MulOne());
_algebraicPatterns.add(new pat.MulZero());
_algebraicPatterns.add(new pat.DivOne());
_algebraicPatterns.add(new pat.DoubleNeg());
_algebraicPatterns.add(new pat.ExpLog());
_algebraicPatterns.add(new pat.LogExp());
_algebraicPatterns.add(new pat.DivSelf());
_algebraicPatterns.add(new pat.TransposeTranspose());
_algebraicPatterns.add(new pat.ReshapeReshape());
_algebraicPatterns.add(new pat.MulNegNeg());
_algebraicPatterns.add(new pat.AddNegToSub());
_algebraicPatterns.add(new pat.SubNegToAdd());
_algebraicPatterns.add(new pat.DoubleConvert());

export class AlgebraicSimplificationPass extends FunctionPass {
  constructor() {
    super('algebraic_simplify');
    this.preservedAnalyses = new Set([ShapeAnalysis]);
  }

  run(func, analysisManager) {
    const applicator = new PatternApplicator(_algebraicPatterns);
    return applicator.applyPatterns(func, 10, this.trace);
  }
}
