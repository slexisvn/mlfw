import { describe, it, expect } from 'vitest';
import {
  ScalarType,
  resultDtype,
  canCast,
} from '../../src/tensor/types/dtype.js';

describe('resultDtype', () => {
  it('promotes F32 + F64 to F64', () => {
    expect(resultDtype(ScalarType.F32, ScalarType.F64)).toBe(ScalarType.F64);
  });

  it('promotes I32 + F32 to F32 (float wins over int)', () => {
    expect(resultDtype(ScalarType.I32, ScalarType.F32)).toBe(ScalarType.F32);
  });

  it('returns same type when both equal', () => {
    expect(resultDtype(ScalarType.F32, ScalarType.F32)).toBe(ScalarType.F32);
  });

  it('promotes BOOL + I32 to I32', () => {
    expect(resultDtype(ScalarType.BOOL, ScalarType.I32)).toBe(ScalarType.I32);
  });

  it('promotes I8 + I32 to I32', () => {
    expect(resultDtype(ScalarType.I8, ScalarType.I32)).toBe(ScalarType.I32);
  });
});

describe('canCast', () => {
  it('allows BOOL → F32 (widening)', () => {
    expect(canCast(ScalarType.BOOL, ScalarType.F32)).toBe(true);
  });

  it('disallows F32 → I32 (narrowing)', () => {
    expect(canCast(ScalarType.F32, ScalarType.I32)).toBe(false);
  });

  it('disallows F64 → F32 (narrowing)', () => {
    expect(canCast(ScalarType.F64, ScalarType.F32)).toBe(false);
  });

  it('allows I8 → I32 (widening chain)', () => {
    expect(canCast(ScalarType.I8, ScalarType.I32)).toBe(true);
  });

  it('allows same type cast', () => {
    expect(canCast(ScalarType.F32, ScalarType.F32)).toBe(true);
  });

  it('disallows I32 → I16 (narrowing)', () => {
    expect(canCast(ScalarType.I32, ScalarType.I16)).toBe(false);
  });
});
