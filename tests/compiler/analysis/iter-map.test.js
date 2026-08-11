import { describe, it, expect } from 'vitest';
import { toLinearForm, exactCoverRange, LinearForm } from '../../../src/compiler/analysis/iter_map.js';
import { VariableNode, IntImmNode, mathOp, BufferLoadNode } from '../../../src/compiler/ir/tensor/nodes.js';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';

const v = (n) => new VariableNode(n, 'int32');
const i32 = (n) => new IntImmNode(n);

describe('linear form decomposition', () => {
  it('decomposes an affine index into offset and per-variable coefficients', () => {
    const form = toLinearForm(mathOp('+', mathOp('*', v('i'), i32(32)), mathOp('+', v('j'), i32(5))));
    expect(form.offset).toBe(5);
    expect([...form.terms]).toEqual([['i', 32], ['j', 1]]);
  });

  it('cancels a variable that nets to zero', () => {
    const form = toLinearForm(mathOp('-', v('i'), v('i')));
    expect(form.isConstant).toBe(true);
    expect(form.offset).toBe(0);
  });

  it('refuses a non-linear product of two variables', () => {
    expect(toLinearForm(mathOp('*', v('i'), v('j')))).toBeNull();
  });

  it('refuses division and modulo, which are not affine here', () => {
    expect(toLinearForm(mathOp('//', v('i'), i32(2)))).toBeNull();
    expect(toLinearForm(mathOp('%', v('i'), i32(2)))).toBeNull();
  });

  it('refuses an opaque leaf such as a buffer load', () => {
    const IDX = new Buffer('IDX', [8], 'i32', 'global');
    expect(toLinearForm(new BufferLoadNode(IDX, [v('i')]))).toBeNull();
  });
});

describe('exact cover proves a tiled index spans a contiguous range', () => {
  const ranges = (entries) => new Map(entries);

  it('accepts a bare loop variable', () => {
    expect(exactCoverRange(v('i'), ranges([['i', [0, 8]]]))).toEqual([0, 8]);
  });

  it('accepts a constant as a single-element range', () => {
    expect(exactCoverRange(i32(3), ranges([]))).toEqual([3, 1]);
  });

  it('accepts a split loop nest that tiles the dimension exactly', () => {
    const idx = mathOp('+', mathOp('*', v('io'), i32(32)), v('ii'));
    expect(exactCoverRange(idx, ranges([['io', [0, 4]], ['ii', [0, 32]]]))).toEqual([0, 128]);
  });

  it('accepts a three-level tiling', () => {
    const idx = mathOp('+', mathOp('*', v('a'), i32(16)), mathOp('+', mathOp('*', v('b'), i32(4)), v('c')));
    expect(exactCoverRange(idx, ranges([['a', [0, 2]], ['b', [0, 4]], ['c', [0, 4]]]))).toEqual([0, 32]);
  });

  it('carries a constant offset through', () => {
    const idx = mathOp('+', mathOp('*', v('io'), i32(32)), mathOp('+', v('ii'), i32(7)));
    expect(exactCoverRange(idx, ranges([['io', [0, 4]], ['ii', [0, 32]]]))).toEqual([7, 128]);
  });

  it('rejects a strided index that skips elements', () => {
    expect(exactCoverRange(mathOp('*', v('i'), i32(2)), ranges([['i', [0, 64]]]))).toBeNull();
  });

  it('rejects a tiling whose strides leave a gap', () => {
    const idx = mathOp('+', mathOp('*', v('io'), i32(64)), v('ii'));
    expect(exactCoverRange(idx, ranges([['io', [0, 4]], ['ii', [0, 32]]]))).toBeNull();
  });

  it('rejects a tiling whose strides overlap', () => {
    const idx = mathOp('+', mathOp('*', v('io'), i32(16)), v('ii'));
    expect(exactCoverRange(idx, ranges([['io', [0, 4]], ['ii', [0, 32]]]))).toBeNull();
  });

  it('rejects an unbound variable', () => {
    expect(exactCoverRange(v('k'), ranges([['i', [0, 8]]]))).toBeNull();
  });

  it('rejects a negative coefficient (reversed traversal is not proven here)', () => {
    expect(exactCoverRange(mathOp('-', i32(7), v('i')), ranges([['i', [0, 8]]]))).toBeNull();
  });

  it('shifts a loop whose range does not start at zero into the covered offset', () => {
    expect(exactCoverRange(v('i'), ranges([['i', [2, 8]]]))).toEqual([2, 8]);
    expect(exactCoverRange(mathOp('+', mathOp('*', v('io'), i32(32)), v('ii')),
      ranges([['io', [0, 4]], ['ii', [3, 32]]]))).toEqual([3, 128]);
  });
});

describe('LinearForm algebra', () => {
  it('scales offset and every coefficient', () => {
    const scaled = LinearForm.variable('i').add(LinearForm.constant(2)).scale(3);
    expect(scaled.offset).toBe(6);
    expect(scaled.terms.get('i')).toBe(3);
  });

  it('collapses to the zero constant when scaled by zero', () => {
    const zero = LinearForm.variable('i').scale(0);
    expect(zero.isConstant).toBe(true);
    expect(zero.offset).toBe(0);
  });
});
