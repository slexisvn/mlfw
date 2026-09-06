import { describe, it, expect } from 'vitest';
import { AxeLayout, AxeAxis, iter, coord } from '../../../../src/compiler/ir/layout/axe.js';
import {
  formatAxeLayout,
  parseAxeLayout,
  permutationOf,
  defaultExprCodec,
  AxeFormatError
} from '../../../../src/compiler/ir/layout/axe_format.js';
import { validateGraphProfile, validateThreadProfile } from '../../../../src/compiler/ir/layout/profiles.js';
import { MemoryScope } from '../../../../src/compiler/ir/tensor/tensor_types.js';
import { SymInt } from '../../../../src/compiler/ir/sym_int.js';

function roundTrip(layout, codec = defaultExprCodec) {
  return parseAxeLayout(formatAxeLayout(layout, codec), codec);
}

describe('the Axe layout text form', () => {
  describe('round-tripping', () => {
    it('carries a plain row-major layout through unchanged', () => {
      const layout = AxeLayout.rowMajor([4, 6]);
      expect(formatAxeLayout(layout)).toBe('(4:6@m, 6:1@m)');
      expect(roundTrip(layout).equals(layout)).toBe(true);
    });

    it('carries replica iters and an offset', () => {
      const layout = new AxeLayout(
        [iter(8, 4, AxeAxis.LANE), iter(2, 1, AxeAxis.WARP)],
        [iter(2, 4, AxeAxis.WARP)],
        coord({ [AxeAxis.WARP]: 5 })
      );
      expect(formatAxeLayout(layout)).toBe('(8:4@lane, 2:1@warp) + [2:4@warp] + {warp:5}');
      expect(roundTrip(layout).equals(layout)).toBe(true);
    });

    it('carries a negative stride', () => {
      const layout = new AxeLayout([iter(4, -1, AxeAxis.MEM)]);
      expect(formatAxeLayout(layout)).toBe('(4:-1@m)');
      expect(roundTrip(layout).equals(layout)).toBe(true);
    });

    it('carries a dotted thread axis', () => {
      const layout = new AxeLayout([iter(32, 1, AxeAxis.THREAD_X)]);
      expect(formatAxeLayout(layout)).toBe('(32:1@thread.x)');
      expect(roundTrip(layout).shard[0].axis).toBe(AxeAxis.THREAD_X);
    });

    it('carries a bare symbolic extent', () => {
      const layout = AxeLayout.rowMajor([SymInt.var('n'), 8]);
      expect(formatAxeLayout(layout)).toBe('(n:8@m, 8:1@m)');
      expect(SymInt.equals(roundTrip(layout).shard[0].extent, SymInt.var('n'))).toBe(true);
    });

    it('carries an empty shard', () => {
      expect(formatAxeLayout(new AxeLayout([]))).toBe('()');
      expect(parseAxeLayout('()').shard.length).toBe(0);
    });
  });

  describe('rejecting malformed text', () => {
    it('names the character it wanted', () => {
      expect(() => parseAxeLayout('(4:6@m')).toThrow(AxeFormatError);
    });

    it('refuses a composite expression the default codec cannot read', () => {
      expect(() => parseAxeLayout('((n * 8):1@m)')).toThrow(/expression codec/);
    });

    it('accepts a composite expression once a codec is injected', () => {
      const codec = {
        format: String,
        parse: text => (text === '(n * 8)' ? SymInt.mul(SymInt.var('n'), 8) : defaultExprCodec.parse(text))
      };
      const parsed = parseAxeLayout('((n * 8):1@m)', codec);
      expect(SymInt.equals(parsed.shard[0].extent, SymInt.mul(SymInt.var('n'), 8))).toBe(true);
    });

    it('refuses two replica sections', () => {
      expect(() => parseAxeLayout('(4:1@m) + [2:1@warp] + [2:1@warp]')).toThrow(/twice/);
    });

    it('refuses trailing text', () => {
      expect(() => parseAxeLayout('(4:1@m) junk')).toThrow(/trailing/);
    });
  });

  describe('recovering the legacy permutation form', () => {
    it('reads a row-major layout back as the identity order', () => {
      expect(permutationOf(AxeLayout.rowMajor([4, 6]), 2)).toEqual([0, 1]);
    });

    it('reads a column-major layout back as the reversed order', () => {
      expect(permutationOf(AxeLayout.fromPermutation([1, 0], [4, 6]), 2)).toEqual([1, 0]);
    });

    it('round-trips every permutation of a rank-3 shape', () => {
      const shape = [2, 3, 5];
      for (const order of [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]) {
        expect(permutationOf(AxeLayout.fromPermutation(order, shape), 3)).toEqual(order);
      }
    });

    it('declines a blocked layout at the rank of the tensor it describes', () => {
      const blocked = AxeLayout.tile(
        AxeLayout.rowMajor([2, 3]),
        [2, 3],
        AxeLayout.rowMajor([4, 4]),
        [4, 4]
      );
      expect(permutationOf(blocked, 2)).toBeNull();
      expect(permutationOf(blocked, 4)).toEqual([0, 2, 1, 3]);
    });

    it('declines a layout that carries replication', () => {
      expect(permutationOf(new AxeLayout([iter(4, 1, AxeAxis.MEM)], [iter(2, 1, AxeAxis.WARP)]), 1)).toBeNull();
    });
  });
});

describe('layout profiles', () => {
  describe('the graph profile', () => {
    it('accepts a row-major layout over a dynamic dimension', () => {
      expect(validateGraphProfile(AxeLayout.rowMajor([SymInt.var('n'), 8]))).toEqual([]);
    });

    it('rejects a hardware axis', () => {
      const layout = new AxeLayout([iter(32, 1, AxeAxis.LANE)]);
      expect(validateGraphProfile(layout)).toEqual([`a graph-level layout may only use the 'm' axis, found 'lane'`]);
    });

    it('rejects a layout that does not group under the tensor shape', () => {
      const errors = validateGraphProfile(AxeLayout.rowMajor([4, 6]), [4, 5]);
      expect(errors).toEqual(['the layout does not group under shape [4, 5]']);
    });
  });

  describe('the thread profile', () => {
    it('accepts a shared tile spread over lanes', () => {
      const layout = new AxeLayout([iter(32, 1, AxeAxis.LANE), iter(4, 32, AxeAxis.MEM)]);
      expect(validateThreadProfile(layout, MemoryScope.SHARED)).toEqual([]);
    });

    it('refuses to put a lane axis on a global buffer', () => {
      const layout = new AxeLayout([iter(32, 1, AxeAxis.LANE)]);
      expect(validateThreadProfile(layout, MemoryScope.GLOBAL))
        .toEqual([`a 'global' buffer may not be laid out over the 'lane' axis`]);
    });

    it('refuses a register axis on a shared buffer', () => {
      const layout = new AxeLayout([iter(2, 1, AxeAxis.REG)]);
      expect(validateThreadProfile(layout, MemoryScope.SHARED))
        .toEqual([`a 'shared' buffer may not be laid out over the 'reg' axis`]);
    });

    it('refuses a symbolic extent', () => {
      const layout = AxeLayout.rowMajor([SymInt.var('n'), 8]);
      expect(validateThreadProfile(layout, MemoryScope.SHARED))
        .toEqual([`a thread-level layout needs constant positive extents, found 'n'`]);
    });

    it('reports an unknown scope once and stops', () => {
      expect(validateThreadProfile(new AxeLayout([iter(4, 1, AxeAxis.MEM)]), 'nowhere'))
        .toEqual([`unknown memory scope 'nowhere'`]);
    });
  });
});
