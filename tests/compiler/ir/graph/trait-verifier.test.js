import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { verifyFunction } from '../../../../src/compiler/ir/graph/verifier.js';
import { verifyTraits, verifiedTraits } from '../../../../src/compiler/ir/graph/trait_verifier.js';
import { Operation } from '../../../../src/compiler/ir/graph/operation.js';
import { registry } from '../../../../src/compiler/ir/graph/ops.js';
import { OpTrait } from '../../../../src/compiler/ir/graph/op_registry.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';

function f32(shape) {
  return new TensorType(shape, ScalarType.F32);
}

function i32(shape) {
  return new TensorType(shape, ScalarType.I32);
}

function messagesFor(opName, operandTypes, resultTypes, attrs = null) {
  const t = f32([4]);
  let built = null;
  const func = buildFunction('probe', [t], [t], (b, args) => {
    const operands = operandTypes.map((ty) => b.constant(0, ty).getResult(0));
    built = new Operation(opName, operands, resultTypes, attrs);
    b.block.pushOp(built);
    b.returnOp([args[0]]);
  });
  return { messages: verifyTraits(built), func, op: built };
}

describe('trait verifiers are registered for the traits the op registry declares', () => {
  it('every trait declared on a registered op that has a verifier is exercised by the registry', () => {
    const verified = new Set(verifiedTraits());
    expect(verified.size).toBeGreaterThan(0);

    const declared = new Set();
    for (const def of registry.allOps()) {
      for (const trait of def.traits) declared.add(trait);
    }

    for (const trait of [
      OpTrait.COMMUTATIVE, OpTrait.ASSOCIATIVE, OpTrait.IDEMPOTENT,
      OpTrait.SAME_OPERAND_AND_RESULT_TYPE, OpTrait.SAME_OPERAND_AND_RESULT_SHAPE,
    ]) {
      expect(declared.has(trait), `trait '${trait}' is declared on no registered op`).toBe(true);
    }
  });

  it('real IR built through the builder verifies clean under every trait', () => {
    const t = f32([4]);
    const func = buildFunction('clean', [t, t], [t], (b, args) => {
      const m = b.maximum(args[0], args[1]);
      const s = b.sqrt(m.getResult(0));
      b.returnOp([b.add(s.getResult(0), args[0]).getResult(0)]);
    });
    expect(verifyFunction(func).map(e => e.message)).toEqual([]);
  });
});

describe('a trait that does not hold is reported', () => {
  it('SAME_OPERAND_AND_RESULT_TYPE catches a result dtype that drifts from the operands', () => {
    const { messages } = messagesFor('add', [f32([4]), f32([4])], [i32([4])]);
    expect(messages.some(m => /same_type/.test(m) && /result 0 dtype 'i32'/.test(m))).toBe(true);
  });

  it('SAME_OPERAND_AND_RESULT_SHAPE catches a unary op that changes shape', () => {
    const { messages } = messagesFor('sqrt', [f32([4])], [f32([8])]);
    expect(messages.some(m => /same_shape/.test(m) && /result 0 shape/.test(m))).toBe(true);
  });

  it('ELEMENTWISE catches a result shape that is not the broadcast of the operands', () => {
    const { messages } = messagesFor('add', [f32([4, 8]), f32([4, 8])], [f32([4, 9])]);
    expect(messages.some(m => /elementwise/.test(m) && /broadcast of operand shapes/.test(m))).toBe(true);
  });

  it('ELEMENTWISE catches operands that cannot broadcast against each other', () => {
    const { messages } = messagesFor('add', [f32([4]), f32([8])], [f32([8])]);
    expect(messages.some(m => /not broadcast-compatible/.test(m))).toBe(true);
  });

  it('COMMUTATIVE catches operands that are not interchangeable', () => {
    const { messages } = messagesFor('maximum', [f32([4]), i32([4])], [f32([4])]);
    expect(messages.some(m => /commutative/.test(m) && /not interchangeable/.test(m))).toBe(true);
  });

  it('IDEMPOTENT catches a result dtype that would break folding f(x, x) -> x', () => {
    const { messages } = messagesFor('maximum', [f32([4]), f32([4])], [i32([4])]);
    expect(messages.some(m => /idempotent/.test(m) && /would not preserve types/.test(m))).toBe(true);
  });

  it('CONSTANT catches a constant op that reads an operand', () => {
    const { messages } = messagesFor('constant', [f32([4])], [f32([4])]);
    expect(messages.some(m => /a constant op must have no operands, got 1/.test(m))).toBe(true);
  });

  it('VIEW catches a view op that changes dtype', () => {
    const { messages } = messagesFor('reshape', [f32([4])], [i32([4])], { new_shape: [4] });
    expect(messages.some(m => /view/.test(m) && /cannot change dtype/.test(m))).toBe(true);
  });

  it('TERMINATOR catches a terminator that is not last in its block', () => {
    const t = f32([4]);
    const func = buildFunction('trailing', [t], [t], (b, args) => {
      b.returnOp([args[0]]);
    });
    const ret = func.getReturnOp();
    expect(verifyTraits(ret)).toEqual([]);

    func.entryBlock.pushOp(new Operation('constant', [], [t], { value: 0, tensor_type: t }));
    expect(verifyTraits(ret).some(m => /terminator/.test(m) && /must be the last operation/.test(m))).toBe(true);
  });

  it('surfaces through verifyFunction, not only through verifyTraits', () => {
    const { func } = messagesFor('add', [f32([4]), f32([4])], [i32([4])]);
    expect(verifyFunction(func).some(e => /trait 'same_type'/.test(e.message))).toBe(true);
  });
});
