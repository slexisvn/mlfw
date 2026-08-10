import { describe, it, expect } from 'vitest';
import {
  ForNode, BlockNode, BufferStoreNode, BufferLoadNode,
  VariableNode, IntImmNode, MathOpNode, IfThenElseNode, ForKind, PrimFunc,
} from '../../../src/compiler/ir/tensor/nodes.js';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import { ScheduleValidator } from '../../../src/compiler/schedule/validator.js';

const iv = (n) => new VariableNode(n, 'int32');
const add = (a, b) => new MathOpNode('+', a, b);
const sub = (a, b) => new MathOpNode('-', a, b);

function storeFunc(buf, indices, value, varName, extent) {
  const store = new BufferStoreNode(buf, indices, value);
  const block = new BlockNode('b', [], [], [{ buffer: buf }], store);
  const body = new ForNode(iv(varName), new IntImmNode(0), new IntImmNode(extent), ForKind.SERIAL, block);
  return new PrimFunc('f', [], body);
}

const oobMsg = (e) => /out of bounds/.test(e);

describe('ScheduleValidator Analyzer-backed bounds proving', () => {
  it('flags an access that is provably above the upper bound', () => {
    const A = new Buffer('A', [8], 'float32', 'global');
    const f = storeFunc(A, [add(iv('i'), new IntImmNode(10))], new IntImmNode(0), 'i', 8);
    expect(ScheduleValidator.validate(f).some(oobMsg)).toBe(true);
  });

  it('flags an access that is provably negative', () => {
    const A = new Buffer('A', [8], 'float32', 'global');
    const f = storeFunc(A, [sub(iv('i'), new IntImmNode(10))], new IntImmNode(0), 'i', 8);
    expect(ScheduleValidator.validate(f).some(oobMsg)).toBe(true);
  });

  it('flags a provably-out-of-bounds load nested in the store value', () => {
    const A = new Buffer('A', [8], 'float32', 'global');
    const S = new Buffer('S', [8], 'float32', 'global');
    const badLoad = new BufferLoadNode(S, [add(iv('i'), new IntImmNode(8))]);
    const f = storeFunc(A, [iv('i')], badLoad, 'i', 8);
    expect(ScheduleValidator.validate(f).some(oobMsg)).toBe(true);
  });

  it('does not flag a valid in-bounds access', () => {
    const A = new Buffer('A', [8], 'float32', 'global');
    const f = storeFunc(A, [iv('i')], new IntImmNode(0), 'i', 8);
    expect(ScheduleValidator.validate(f).some(oobMsg)).toBe(false);
  });

  it('is sound: stays silent on a partial overflow it cannot prove is always OOB (i in [0,7] makes i+4 span [4,11], overlapping the valid range)', () => {
    const A = new Buffer('A', [8], 'float32', 'global');
    const f = storeFunc(A, [add(iv('i'), new IntImmNode(4))], new IntImmNode(0), 'i', 8);
    expect(ScheduleValidator.validate(f).some(oobMsg)).toBe(false);
  });

  it('suppresses the bounds check under a conditional guard', () => {
    const A = new Buffer('A', [8], 'float32', 'global');
    const store = new BufferStoreNode(A, [add(iv('i'), new IntImmNode(10))], new IntImmNode(0));
    const guard = new MathOpNode('<', iv('i'), new IntImmNode(2));
    const guarded = new IfThenElseNode(guard, store);
    const body = new ForNode(iv('i'), new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, guarded);
    const f = new PrimFunc('f', [], body);
    expect(ScheduleValidator.validate(f).some(oobMsg)).toBe(false);
  });

  it('stays silent when the extent is dynamic (loop var unbounded)', () => {
    const A = new Buffer('A', [8], 'float32', 'global');
    const store = new BufferStoreNode(A, [add(iv('i'), new IntImmNode(10))], new IntImmNode(0));
    const block = new BlockNode('b', [], [], [{ buffer: A }], store);
    const body = new ForNode(iv('i'), new IntImmNode(0), iv('n'), ForKind.SERIAL, block);
    const f = new PrimFunc('f', [], body);
    expect(ScheduleValidator.validate(f).some(oobMsg)).toBe(false);
  });
});
