import { describe, it, expect } from 'vitest';
import { registry } from '../../../src/compiler/ir/graph/ops.js';
import { OpDef, SideEffectKind } from '../../../src/compiler/ir/graph/op_registry.js';
import { MemoryEffectAnalysis } from '../../../src/compiler/analysis/memory_effect.js';

function makeValue(id) {
  return { id, _uses: [], uses() { return this._uses; } };
}

function makeOp(opName, operands, results) {
  return {
    opName,
    numOperands: operands.length,
    numResults: results.length,
    getOperand(i) { return operands[i]; },
    getResult(i) { return results[i]; },
  };
}

function makeFunc(ops) {
  return { ops() { return ops; } };
}

function registerOnce(name, sideEffects) {
  if (!registry.has(name)) {
    registry.register(new OpDef({ name, numOperands: 1, numResults: 1, sideEffects }));
  }
}

describe('MemoryEffectAnalysis effectKind bitmask gating', () => {
  it('READ-only op does not tag its results as WRITE', () => {
    registerOnce('__test_read_only__', SideEffectKind.READ);
    const operand = makeValue('in');
    const resultVal = makeValue('out');
    const op = makeOp('__test_read_only__', [operand], [resultVal]);
    const func = makeFunc([op]);

    const result = MemoryEffectAnalysis.compute(func);
    expect(result.getReadersOf(operand)).toContain(op);
    expect(result.getWritersOf(resultVal)).not.toContain(op);
    expect(result.getEffectsOn(resultVal)).toHaveLength(0);
  });

  it('WRITE-only op does not tag its operands as READ', () => {
    registerOnce('__test_write_only__', SideEffectKind.WRITE);
    const operand = makeValue('in2');
    const resultVal = makeValue('out2');
    const op = makeOp('__test_write_only__', [operand], [resultVal]);
    const func = makeFunc([op]);

    const result = MemoryEffectAnalysis.compute(func);
    expect(result.getWritersOf(resultVal)).toContain(op);
    expect(result.getReadersOf(operand)).not.toContain(op);
    expect(result.getEffectsOn(operand)).toHaveLength(0);
  });

  it('READ|WRITE op tags both operands and results', () => {
    registerOnce('__test_read_write__', SideEffectKind.READ | SideEffectKind.WRITE);
    const operand = makeValue('in3');
    const resultVal = makeValue('out3');
    const op = makeOp('__test_read_write__', [operand], [resultVal]);
    const func = makeFunc([op]);

    const result = MemoryEffectAnalysis.compute(func);
    expect(result.getReadersOf(operand)).toContain(op);
    expect(result.getWritersOf(resultVal)).toContain(op);
  });
});
