import { describe, it, expect } from 'vitest';
import { ShapeEnv } from '../../src/tracing/shape_env.js';
import { DYNAMIC } from '../../src/compiler/ir/graph/types.js';
import { SymInt } from '../../src/compiler/ir/sym_int.js';

describe('ShapeEnv.allocate', () => {
  it('returns incrementing symbol names', () => {
    const env = new ShapeEnv();
    expect(env.allocate(0, 0, 4)).toBe('s0');
    expect(env.allocate(0, 1, 8)).toBe('s1');
    expect(env.allocate(1, 0, 16)).toBe('s2');
  });

  it('stores source mapping in symbols', () => {
    const env = new ShapeEnv();
    env.allocate(2, 3, 64);
    const info = env.symbols.get('s0');
    expect(info).toEqual({ hint: 64, inputIdx: 2, dimIdx: 3 });
  });
});

describe('ShapeEnv.produceShapeSpec', () => {
  it('marks dynamic dims with DYNAMIC in irShape and symbol in symShape', () => {
    const env = new ShapeEnv();
    const { irShape, symShape } = env.produceShapeSpec(0, [4, 8, 16], new Set([0, 2]));
    expect(irShape).toEqual([DYNAMIC, 8, DYNAMIC]);
    expect(symShape[0]).toBe('s0');
    expect(symShape[1]).toBe(8);
    expect(symShape[2]).toBe('s2');
  });

  it('all static dims produce concrete values and equality guards', () => {
    const env = new ShapeEnv();
    const { irShape, symShape } = env.produceShapeSpec(0, [3, 5], null);
    expect(irShape).toEqual([3, 5]);
    expect(symShape).toEqual([3, 5]);
    expect(env.guards.length).toBe(2);
    expect(env.guards[0]).toEqual({ lhs: 's0', op: 'eq', rhs: 3 });
    expect(env.guards[1]).toEqual({ lhs: 's1', op: 'eq', rhs: 5 });
  });
});

describe('ShapeEnv.guardRelation', () => {
  it('accumulates guards in order', () => {
    const env = new ShapeEnv();
    env.guardRelation('s0', 'gt', 0);
    env.guardRelation('s1', 'eq', 32);
    env.guardRelation('s0', 'le', 1024);
    expect(env.guards.length).toBe(3);
    expect(env.guards[0]).toEqual({ lhs: 's0', op: 'gt', rhs: 0 });
    expect(env.guards[1]).toEqual({ lhs: 's1', op: 'eq', rhs: 32 });
    expect(env.guards[2]).toEqual({ lhs: 's0', op: 'le', rhs: 1024 });
  });
});

describe('ShapeEnv.guardDivisible', () => {
  it('stores divisibility guard', () => {
    const env = new ShapeEnv();
    env.guardDivisible('s0', 4);
    expect(env.guards[0]).toEqual({ type: 'divisible', sym: 's0', divisor: 4 });
  });
});

describe('ShapeEnv.bindInputShapes + evaluateGuards', () => {
  it('passes when concrete shapes satisfy all guards', () => {
    const env = new ShapeEnv();
    env.produceShapeSpec(0, [2, 8], new Set([0]));
    env.guardRelation('s0', 'gt', 0);

    env.bindInputShapes([{ shape: [4, 8] }]);
    const { passed } = env.evaluateGuards();
    expect(passed).toBe(true);
  });

  it('specializes a size-1 axis even when it is requested dynamic', () => {
    const env = new ShapeEnv();
    const { irShape, symShape } = env.produceShapeSpec(0, [1, 8], new Set([0, 1]));

    expect(symShape[0], 'a size-1 axis carries broadcast meaning and stays concrete').toBe(1);
    expect(irShape[0]).toBe(1);
    expect(typeof symShape[1], 'a size-8 axis is still symbolic').toBe('string');

    env.bindInputShapes([{ shape: [4, 8] }]);
    expect(env.evaluateGuards().passed, 'a different extent must fail the guard and force a recompile').toBe(false);
  });

  it('fails when static dim changes', () => {
    const env = new ShapeEnv();
    env.produceShapeSpec(0, [1, 8], null);

    env.bindInputShapes([{ shape: [1, 16] }]);
    const { passed, failedGuard } = env.evaluateGuards();
    expect(passed).toBe(false);
    expect(failedGuard.op).toBe('eq');
  });

  it('fails when guard is violated', () => {
    const env = new ShapeEnv();
    env.produceShapeSpec(0, [4, 8], new Set([0]));
    env.guardRelation('s0', 'gt', 0);

    env.bindInputShapes([{ shape: [0, 8] }]);
    const { passed } = env.evaluateGuards();
    expect(passed).toBe(false);
  });

  it('evaluates divisibility guards', () => {
    const env = new ShapeEnv();
    env.produceShapeSpec(0, [16], new Set([0]));
    env.guardDivisible('s0', 4);

    env.bindInputShapes([{ shape: [12] }]);
    expect(env.evaluateGuards().passed).toBe(true);

    env.bindInputShapes([{ shape: [13] }]);
    expect(env.evaluateGuards().passed).toBe(false);
  });
});

describe('ShapeEnv.resolveSymbolicShape', () => {
  it('resolves symbol names to bound values', () => {
    const env = new ShapeEnv();
    env.produceShapeSpec(0, [4, 8], new Set([0]));
    env.bindInputShapes([{ shape: [16, 8] }]);
    expect(env.resolveSymbolicShape(['s0', 8])).toEqual([16, 8]);
  });

  it('passes through concrete values unchanged', () => {
    const env = new ShapeEnv();
    expect(env.resolveSymbolicShape([3, 5, 7])).toEqual([3, 5, 7]);
  });
});

describe('ShapeEnv unbound symbols', () => {
  it('reports an unbound symbol by name instead of comparing undefined', () => {
    const env = new ShapeEnv();
    env.produceShapeSpec(0, [4, 8], new Set([0]));
    expect(() => env.evaluateGuards()).toThrow('Unbound symbolic variable: s1');

    env.bindInputShapes([{ shape: [4, 8] }]);
    expect(env.evaluateGuards().passed).toBe(true);
  });

  it('fails the same way for a string symbol and for the equivalent SymInt', () => {
    const env = new ShapeEnv();
    env.produceShapeSpec(0, [4], new Set([0]));
    expect(() => env.resolveSymbolicShape(['s0'])).toThrow('Unbound symbolic variable: s0');
    expect(() => env.resolveSymbolicShape([SymInt.var('s0')])).toThrow('Unbound symbolic variable: s0');

    env.bindInputShapes([{ shape: [16] }]);
    expect(env.resolveSymbolicShape(['s0'])).toEqual([16]);
    expect(env.resolveSymbolicShape([SymInt.var('s0')])).toEqual([16]);
  });

  it('names the missing input when fewer arguments are given than symbols', () => {
    const env = new ShapeEnv();
    env.produceShapeSpec(1, [4, 8], new Set([0]));
    expect(() => env.bindInputShapes([{ shape: [4, 8] }]))
      .toThrow(/symbol 's0' tracks input 1, but only 1 input\(s\) were given/);
  });

  it('names the missing dimension when an input has too low a rank', () => {
    const env = new ShapeEnv();
    env.produceShapeSpec(0, [4, 8], new Set([0]));
    expect(() => env.bindInputShapes([{ shape: [4] }]))
      .toThrow(/symbol 's1' tracks dimension 1 of input 0, which has rank 1/);
  });
});
