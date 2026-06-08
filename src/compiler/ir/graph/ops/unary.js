import { OpDef, OpTrait } from '../op_registry.js';
import * as pat from '../patterns.js';
import {
  inferUnaryElementwise, verifyUnaryElementwise,
  inferUnaryFloat, verifyUnaryFloat
} from './helpers.js';

export function register(registry) {
  for (const name of ['abs', 'floor', 'ceil', 'round', 'sign', 'square', 'reciprocal']) {
    registry.register(new OpDef({
      name,
      numOperands: 1,
      numResults: 1,
      traits: [OpTrait.ELEMENTWISE, OpTrait.SAME_OPERAND_AND_RESULT_TYPE],
      inferResultTypes: inferUnaryElementwise,
      verify: verifyUnaryElementwise
    }));
  }

  registry.register(new OpDef({
    name: 'exp',
    numOperands: 1,
    numResults: 1,
    traits: [OpTrait.ELEMENTWISE, OpTrait.SAME_OPERAND_AND_RESULT_TYPE],
    inferResultTypes: inferUnaryFloat,
    getCanonicalizationPatterns() { return [new pat.ExpLog()]; },
    fold(constValues) { return Math.exp(constValues[0]); },
    verify: verifyUnaryFloat
  }));

  for (const name of ['log', 'sqrt', 'rsqrt', 'tanh', 'sin', 'cos', 'erf', 'log2', 'log10', 'exp2']) {
    registry.register(new OpDef({
      name,
      numOperands: 1,
      numResults: 1,
      traits: [OpTrait.ELEMENTWISE, OpTrait.SAME_OPERAND_AND_RESULT_TYPE],
      inferResultTypes: inferUnaryFloat,
      verify: verifyUnaryFloat
    }));
  }
}
