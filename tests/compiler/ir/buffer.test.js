import { describe, it, expect } from 'vitest';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import { dtypeBytes } from '../../../src/backend/dtype_map.js';

describe('Buffer.sizeInBytes', () => {
  const dtypes = ['f16', 'f32', 'f64', 'i8', 'i16', 'i32', 'i64', 'ui8', 'bool', 'index'];

  for (const dt of dtypes) {
    it(`reports correct byte size for dtype ${dt}`, () => {
      const shape = [2, 3, 4];
      const numel = shape.reduce((a, b) => a * b, 1);
      const buf = new Buffer(`b_${dt}`, shape, dt, 'global');
      expect(buf.sizeInBytes()).toBe(numel * dtypeBytes(dt));
    });
  }

  it('returns -1 for buffers with dynamic dims', () => {
    const buf = new Buffer('dyn', [2, 'N', 4], 'f32', 'global');
    expect(buf.sizeInBytes()).toBe(-1);
  });

  it('ui8 buffers are sized at 1 byte per element, not 4', () => {
    const buf = new Buffer('q', [16, 16], 'ui8', 'global');
    expect(buf.sizeInBytes()).toBe(256);
  });
});
