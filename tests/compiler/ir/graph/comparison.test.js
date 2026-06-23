import { describe, it, expect } from 'vitest';
import { registry } from '../../../../src/compiler/ir/graph/ops.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';

function bool(shape) { return new TensorType(shape, ScalarType.BOOL); }
function f32(shape) { return new TensorType(shape, ScalarType.F32); }

describe('where shape inference broadcast', () => {
  const def = registry.get('where');

  it('broadcasts scalar condition against tensor operands', () => {
    const out = def.inferResultTypes([bool([]), f32([3, 4]), f32([3, 4])]);
    expect(out).not.toBeNull();
    expect(out[0].shape).toEqual([3, 4]);
    expect(out[0].dtype).toBe(ScalarType.F32);
  });

  it('broadcasts mismatched x/y shapes', () => {
    const out = def.inferResultTypes([bool([3, 1]), f32([1, 4]), f32([3, 4])]);
    expect(out[0].shape).toEqual([3, 4]);
  });

  it('returns null for non-broadcastable shapes', () => {
    const out = def.inferResultTypes([bool([3]), f32([4]), f32([4])]);
    expect(out).toBeNull();
  });

  it('returns null for mismatched branch dtypes (mirrors select)', () => {
    const i32 = (shape) => new TensorType(shape, ScalarType.I32);
    expect(def.inferResultTypes([bool([2]), f32([2]), i32([2])])).toBeNull();
    expect(def.inferResultTypes([bool([2]), f32([2]), f32([2])])).not.toBeNull();
  });
});
