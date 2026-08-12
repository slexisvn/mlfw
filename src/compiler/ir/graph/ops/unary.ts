import { OpDef, OpTrait } from '../op_registry.js';
import type { OpRegistry } from '../op_registry.js';
import {
  inferUnaryElementwise, verifyUnaryElementwise,
  inferUnaryFloat, verifyUnaryFloat
} from './helpers.js';

export function register(registry: OpRegistry) {
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
    fold(constValues) { return typeof constValues[0] === 'number' ? Math.exp(constValues[0]) : undefined; },
    verify: verifyUnaryFloat
  }));

  for (const name of ['log', 'sqrt', 'rsqrt', 'tanh', 'sin', 'cos', 'erf', 'erfc', 'lgamma', 'gamma', 'log2', 'log10', 'exp2']) {
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
