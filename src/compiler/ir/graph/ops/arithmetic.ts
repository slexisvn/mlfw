import { OpDef, OpTrait } from '../op_registry.js';
import type { FoldFn, OpRegistry } from '../op_registry.js';
import * as pat from '../patterns.js';
import {
  inferBinaryElementwise, inferUnaryElementwise,
  verifyBinaryElementwise, verifyUnaryElementwise,
  binaryArithTraits, commBinaryArithTraits
} from './helpers.js';

function scalarBinaryFold(fn: (a: number, b: number) => number): FoldFn {
  return (constValues) => {
    if (typeof constValues[0] !== 'number' || typeof constValues[1] !== 'number') return undefined;
    return fn(constValues[0], constValues[1]);
  };
}

function scalarUnaryFold(fn: (a: number) => number): FoldFn {
  return (constValues) => (typeof constValues[0] === 'number' ? fn(constValues[0]) : undefined);
}

export function register(registry: OpRegistry) {
  registry.register(new OpDef({
    name: 'add',
    numOperands: 2,
    numResults: 1,
    traits: commBinaryArithTraits,
    inferResultTypes: inferBinaryElementwise,
    verify: verifyBinaryElementwise,
    getCanonicalizationPatterns() { return [pat.commutativeConstantRightFor('add'), new pat.AddZero()]; },
    fold: scalarBinaryFold((a, b) => a + b)
  }));

  registry.register(new OpDef({
    name: 'mul',
    numOperands: 2,
    numResults: 1,
    traits: commBinaryArithTraits,
    inferResultTypes: inferBinaryElementwise,
    verify: verifyBinaryElementwise,
    getCanonicalizationPatterns() { return [pat.commutativeConstantRightFor('mul'), new pat.MulOne(), new pat.MulZero()]; },
    fold: scalarBinaryFold((a, b) => a * b)
  }));

  registry.register(new OpDef({
    name: 'sub',
    numOperands: 2,
    numResults: 1,
    traits: binaryArithTraits,
    inferResultTypes: inferBinaryElementwise,
    verify: verifyBinaryElementwise,
    getCanonicalizationPatterns() { return [new pat.SubZero(), new pat.SubSelf()]; },
    fold: scalarBinaryFold((a, b) => a - b)
  }));

  registry.register(new OpDef({
    name: 'div',
    numOperands: 2,
    numResults: 1,
    traits: binaryArithTraits,
    inferResultTypes: inferBinaryElementwise,
    verify: verifyBinaryElementwise,
    getCanonicalizationPatterns() { return [new pat.DivOne()]; },
    fold: scalarBinaryFold((a, b) => a / b)
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
    fold: scalarUnaryFold((a) => -a)
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
