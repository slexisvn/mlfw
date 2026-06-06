import { describe, it, expect } from 'vitest';
import { ShapeEnv } from '../../src/tracing/shape_env.js';
import { DYNAMIC } from '../../src/compiler/ir/graph/types.js';

describe('ShapeEnv.createSymbolicSize', () => {
  it('concrete positive hint returns the hint unchanged', () => {
    const env = new ShapeEnv();
    expect(env.createSymbolicSize(42)).toBe(42);
    expect(env.createSymbolicSize(0)).toBe(0);
  });

  it('undefined hint returns DYNAMIC', () => {
    const env = new ShapeEnv();
    expect(env.createSymbolicSize(undefined)).toBe(DYNAMIC);
  });

  it('null hint returns DYNAMIC', () => {
    const env = new ShapeEnv();
    expect(env.createSymbolicSize(null)).toBe(DYNAMIC);
  });

  it('negative hint returns DYNAMIC', () => {
    const env = new ShapeEnv();
    expect(env.createSymbolicSize(-1)).toBe(DYNAMIC);
    expect(env.createSymbolicSize(-999)).toBe(DYNAMIC);
  });

  it('multiple DYNAMIC calls increment symbol ids', () => {
    const env = new ShapeEnv();
    env.createSymbolicSize(undefined);
    env.createSymbolicSize(null);
    env.createSymbolicSize(-1);
    expect(env.symbols.size).toBe(3);
    expect(env.symbols.has('s0')).toBe(true);
    expect(env.symbols.has('s1')).toBe(true);
    expect(env.symbols.has('s2')).toBe(true);
  });
});

describe('ShapeEnv.bindConcrete + resolveShape', () => {
  it('resolves bound symbols to concrete values', () => {
    const env = new ShapeEnv();
    env.createSymbolicSize(undefined);
    env.bindConcrete('s0', 16);
    expect(env.resolveShape(['s0', 32])).toEqual([16, 32]);
  });

  it('unbound symbols stay as DYNAMIC', () => {
    const env = new ShapeEnv();
    env.createSymbolicSize(undefined);
    expect(env.resolveShape(['s0'])).toEqual([DYNAMIC]);
  });

  it('concrete dims pass through unchanged', () => {
    const env = new ShapeEnv();
    expect(env.resolveShape([1, 3, 224, 224])).toEqual([1, 3, 224, 224]);
  });

  it('rebinding a symbol overwrites the previous value', () => {
    const env = new ShapeEnv();
    env.createSymbolicSize(undefined);
    env.bindConcrete('s0', 8);
    env.bindConcrete('s0', 64);
    expect(env.resolveShape(['s0'])).toEqual([64]);
  });
});

describe('ShapeEnv guards accumulation', () => {
  it('guardEquals pushes eq constraint', () => {
    const env = new ShapeEnv();
    env.guardEquals('s0', 32);
    expect(env.guards).toEqual([{ type: 'eq', sym: 's0', value: 32 }]);
  });

  it('guardGreater pushes gt constraint', () => {
    const env = new ShapeEnv();
    env.guardGreater('s0', 0);
    expect(env.guards).toEqual([{ type: 'gt', sym: 's0', value: 0 }]);
  });

  it('mixed guards accumulate in order', () => {
    const env = new ShapeEnv();
    env.guardEquals('s0', 16);
    env.guardGreater('s1', 0);
    env.guard({ type: 'custom', sym: 's2', value: 'even' });
    expect(env.guards.length).toBe(3);
    expect(env.guards[0].type).toBe('eq');
    expect(env.guards[1].type).toBe('gt');
    expect(env.guards[2].type).toBe('custom');
  });
});
