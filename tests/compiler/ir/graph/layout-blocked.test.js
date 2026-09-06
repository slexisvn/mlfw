import { describe, it, expect } from 'vitest';
import { Layout, TensorType, ScalarType, typeToString, layoutToString, layoutFromString } from '../../../../src/compiler/ir/graph/types.js';
import { AxeAxis, AxeLayout, iter } from '../../../../src/compiler/ir/layout/axe.js';
import { validateGraphProfile } from '../../../../src/compiler/ir/layout/profiles.js';

const NCHW = [0, 1, 2, 3];
const SHAPE = [2, 8, 3, 5];

function addressOf(layout, shape, coords) {
  return layout.bind(shape).apply(coords, shape)[0].get(AxeAxis.MEM);
}

describe('blocked layouts', () => {
  it('splits the blocked dimension into two iters and leaves the rest alone', () => {
    const layout = Layout.blocked(NCHW, 1, 4);
    expect(layout.axe.shard.length).toBe(5);
    expect(layout.dims).toEqual([0, 1, 1, 2, 3]);
    expect(layout.rank).toBe(4);
    expect(layout.isBlocked()).toBe(true);
  });

  it('addresses NCHW4c the way the channel-blocked formula does', () => {
    const layout = Layout.blocked(NCHW, 1, 4);
    for (const [n, c, h, w] of [[0, 0, 0, 0], [1, 7, 2, 4], [0, 5, 1, 3], [1, 4, 0, 1]]) {
      const expected = n * 120 + Math.floor(c / 4) * 60 + (c % 4) + h * 20 + w * 4;
      expect(addressOf(layout, SHAPE, [n, c, h, w])).toBe(expected);
    }
  });

  it('covers every element exactly once', () => {
    const layout = Layout.blocked(NCHW, 1, 4);
    const seen = new Set();
    for (let n = 0; n < SHAPE[0]; n++) {
      for (let c = 0; c < SHAPE[1]; c++) {
        for (let h = 0; h < SHAPE[2]; h++) {
          for (let w = 0; w < SHAPE[3]; w++) seen.add(addressOf(layout, SHAPE, [n, c, h, w]));
        }
      }
    }
    expect(seen.size).toBe(2 * 8 * 3 * 5);
    expect(Math.max(...seen)).toBe(2 * 8 * 3 * 5 - 1);
  });

  it('reports the same footprint as the dense tensor it stores', () => {
    expect(Layout.blocked(NCHW, 1, 4).bind(SHAPE).footprint()).toBe(2 * 8 * 3 * 5);
  });

  it('stays a valid graph-profile layout', () => {
    expect(validateGraphProfile(Layout.blocked(NCHW, 1, 4).bind(SHAPE), SHAPE)).toEqual([]);
  });

  it('fails validation loudly when the block factor does not divide the dimension', () => {
    const ragged = [2, 6, 3, 5];
    const layout = Layout.blocked(NCHW, 1, 4);
    expect(validateGraphProfile(layout.bind(ragged), ragged))
      .toEqual(['the layout does not group under shape [2, 6, 3, 5]']);
  });

  it('refuses a per-dimension stride vector, because it has none', () => {
    expect(() => Layout.blocked(NCHW, 1, 4).computeStrides(SHAPE))
      .toThrow(/no single stride per dimension/);
  });

  it('refuses a split factor that is not a real split', () => {
    expect(() => Layout.blocked(NCHW, 1, 1)).toThrow(/above 1/);
  });

  it('refuses to block a dimension outside the order', () => {
    expect(() => Layout.blocked([0, 1], 5, 4)).toThrow(/not part of order/);
  });

  it('separates two block factors on the same dimension', () => {
    expect(Layout.blocked(NCHW, 1, 4).equals(Layout.blocked(NCHW, 1, 8))).toBe(false);
    expect(Layout.blocked(NCHW, 1, 4).hash()).not.toBe(Layout.blocked(NCHW, 1, 8).hash());
  });

  it('separates a blocked layout from the plain permutation it is built on', () => {
    expect(Layout.blocked(NCHW, 1, 4).equals(new Layout(NCHW))).toBe(false);
  });

  it('round-trips through the layout text form', () => {
    const layout = Layout.blocked(NCHW, 1, 4);
    expect(layoutToString(layout)).toBe('[0, 1, 2, 3]:1/4');
    expect(layoutFromString(layoutToString(layout)).equals(layout)).toBe(true);
  });

  it('prints inside a tensor type and is never mistaken for the identity', () => {
    const type = new TensorType(SHAPE, ScalarType.F32, Layout.blocked(NCHW, 1, 4));
    expect(type.layout.isIdentity()).toBe(false);
    expect(typeToString(type)).toBe('tensor<2x8x3x5xf32, [0, 1, 2, 3]:1/4>');
  });

  it('sizes a padded blocked tensor by what it spans, not by its element count', () => {
    const padded = [2, 8, 3, 5];
    const dense = new TensorType(padded, ScalarType.F32);
    const blocked = new TensorType(padded, ScalarType.F32, Layout.blocked(NCHW, 1, 4));
    expect(blocked.footprint()).toBe(dense.numel());
    expect(blocked.sizeInBytes()).toBe(dense.sizeInBytes());
  });

  it('refuses a replica set at the graph tier, where nothing consumes one', () => {
    const replicated = new AxeLayout(
      [iter(4, 1, AxeAxis.MEM)],
      [iter(2, 4, AxeAxis.MEM)]
    );
    expect(validateGraphProfile(replicated)).toContain(
      'a graph-level layout may not replicate: nothing below the graph IR consumes a set-valued layout yet');
  });

  it('makes two tensor types of the same shape differ by their layout alone', () => {
    const dense = new TensorType(SHAPE, ScalarType.F32);
    const blocked = new TensorType(SHAPE, ScalarType.F32, Layout.blocked(NCHW, 1, 4));
    expect(dense.equals(blocked)).toBe(false);
    expect(dense.hash()).not.toBe(blocked.hash());
  });
});
