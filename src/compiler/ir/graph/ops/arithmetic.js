import { OpDef, OpTrait } from '../op_registry.js';
import * as pat from '../patterns.js';
import {
  inferBinaryElementwise, inferUnaryElementwise,
  verifyBinaryElementwise, verifyUnaryElementwise,
  binaryArithTraits, commBinaryArithTraits
} from './helpers.js';

export function register(registry) {
  registry.register(new OpDef({
    name: 'add',
    numOperands: 2,
    numResults: 1,
    traits: commBinaryArithTraits,
    inferResultTypes: inferBinaryElementwise,
    verify: verifyBinaryElementwise,
    getCanonicalizationPatterns() { return [pat.commutativeConstantRightFor('add'), new pat.AddZero()]; },
    fold(constValues) { return constValues[0] + constValues[1]; }
  }));

  registry.register(new OpDef({
    name: 'mul',
    numOperands: 2,
    numResults: 1,
    traits: commBinaryArithTraits,
    inferResultTypes: inferBinaryElementwise,
    verify: verifyBinaryElementwise,
    getCanonicalizationPatterns() { return [pat.commutativeConstantRightFor('mul'), new pat.MulOne(), new pat.MulZero()]; },
    fold(constValues) { return constValues[0] * constValues[1]; }
  }));

  registry.register(new OpDef({
    name: 'sub',
    numOperands: 2,
    numResults: 1,
    traits: binaryArithTraits,
    inferResultTypes: inferBinaryElementwise,
    verify: verifyBinaryElementwise,
    getCanonicalizationPatterns() { return [new pat.SubZero(), new pat.SubSelf()]; },
    fold(constValues) { return constValues[0] - constValues[1]; }
  }));

  registry.register(new OpDef({
    name: 'div',
    numOperands: 2,
    numResults: 1,
    traits: binaryArithTraits,
    inferResultTypes: inferBinaryElementwise,
    verify: verifyBinaryElementwise,
    getCanonicalizationPatterns() { return [new pat.DivOne()]; },
    fold(constValues) { return constValues[0] / constValues[1]; }
  }));

  for (const name of ['rem', 'pow']) {
    registry.register(new OpDef({
      name,
      numOperands: 2,
      numResults: 1,
      traits: binaryArithTraits,
      inferResultTypes: inferBinaryElementwise,
      verify: verifyBinaryElementwise
    }));
  }

  registry.register(new OpDef({
    name: 'neg',
    numOperands: 1,
    numResults: 1,
    traits: [OpTrait.ELEMENTWISE, OpTrait.SAME_OPERAND_AND_RESULT_TYPE],
    inferResultTypes: inferUnaryElementwise,
    verify: verifyUnaryElementwise,
    getCanonicalizationPatterns() { return [new pat.DoubleNeg()]; },
    fold(constValues) { return -constValues[0]; }
  }));

  for (const name of ['maximum', 'minimum']) {
    registry.register(new OpDef({
      name,
      numOperands: 2,
      numResults: 1,
      traits: [...binaryArithTraits, OpTrait.COMMUTATIVE],
      inferResultTypes: inferBinaryElementwise,
      verify: verifyBinaryElementwise
    }));
  }
}
