import { describe, it, expect } from 'vitest';
import { AxeLayout, AxeAxis, iter, coord, coordKey } from '../../../../src/compiler/ir/layout/axe.js';
import { SymInt } from '../../../../src/compiler/ir/sym_int.js';

function lcg(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function pick(rand, items) {
  return items[Math.floor(rand() * items.length) % items.length];
}

function imageOf(layout) {
  const extent = layout.domainExtent;
  const out = [];
  for (let x = 0; x < extent; x++) {
    out.push([...new Set(layout.applyFlat(x).map(coordKey))].sort().join('|'));
  }
  return out;
}

function randomLayout(rand, axes = [AxeAxis.MEM]) {
  const count = 1 + Math.floor(rand() * 3);
  const shard = [];
  let stride = 1;
  for (let i = 0; i < count; i++) {
    const extent = pick(rand, [1, 2, 2, 3, 4]);
    shard.unshift(iter(extent, stride, pick(rand, axes)));
    stride *= extent;
  }
  return new AxeLayout(shard);
}

function withReplica(rand, layout) {
  if (rand() < 0.5) return layout;
  const axis = pick(rand, [...layout.axes()]);
  const offset = coord({ [axis]: Math.floor(rand() * 4) });
  return new AxeLayout(layout.shard, [iter(pick(rand, [2, 3]), pick(rand, [1, 5, 32]), axis)], offset);
}

describe('the Axe layout algebra', () => {
  describe('canonicalization preserves the induced map', () => {
    it('drops unit extents without changing where any index lands', () => {
      const layout = new AxeLayout([
        iter(1, 99, AxeAxis.MEM),
        iter(4, 8, AxeAxis.MEM),
        iter(1, 8, AxeAxis.MEM),
        iter(6, 1, AxeAxis.MEM)
      ]);
      const canon = layout.canonicalize();
      expect(canon.shard).toEqual([iter(4, 8, AxeAxis.MEM), iter(6, 1, AxeAxis.MEM)]);
      expect(imageOf(canon)).toEqual(imageOf(layout));
    });

    it('merges adjacent iters on one axis when the outer stride is the inner span', () => {
      const layout = new AxeLayout([iter(4, 6, AxeAxis.MEM), iter(6, 1, AxeAxis.MEM)]);
      const canon = layout.canonicalize();
      expect(canon.shard).toEqual([iter(24, 1, AxeAxis.MEM)]);
      expect(imageOf(canon)).toEqual(imageOf(layout));
    });

    it('leaves adjacent iters alone when they sit on different axes', () => {
      const layout = new AxeLayout([iter(4, 8, AxeAxis.WARP), iter(8, 1, AxeAxis.LANE)]);
      expect(layout.canonicalize().shard.length).toBe(2);
    });

    it('holds over randomly generated layouts', () => {
      const rand = lcg(20260905);
      for (let trial = 0; trial < 200; trial++) {
        const layout = randomLayout(rand, [AxeAxis.MEM, AxeAxis.LANE]);
        expect(imageOf(layout.canonicalize())).toEqual(imageOf(layout));
      }
    });

    it('is idempotent', () => {
      const rand = lcg(7);
      for (let trial = 0; trial < 200; trial++) {
        const once = randomLayout(rand, [AxeAxis.MEM, AxeAxis.WARP]).canonicalize();
        const twice = once.canonicalize();
        expect(twice.shard).toEqual(once.shard);
        expect(twice.replica).toEqual(once.replica);
      }
    });
  });

  describe('replica normalization', () => {
    it('flips a negative stride and pays for it with an offset', () => {
      const layout = new AxeLayout([iter(4, 1, AxeAxis.MEM)], [iter(3, -2, AxeAxis.WARP)]);
      const canon = layout.canonicalize();
      expect(canon.replica).toEqual([iter(3, 2, AxeAxis.WARP)]);
      expect(canon.offset.get(AxeAxis.WARP)).toBe(-4);
      expect(imageOf(canon)).toEqual(imageOf(layout));
    });

    it('absorbs a replica iter whose stride is the span of another', () => {
      const layout = new AxeLayout([iter(2, 1, AxeAxis.MEM)], [iter(2, 1, AxeAxis.WARP), iter(3, 2, AxeAxis.WARP)]);
      const canon = layout.canonicalize();
      expect(canon.replica).toEqual([iter(6, 1, AxeAxis.WARP)]);
      expect(imageOf(canon)).toEqual(imageOf(layout));
    });

    it('absorbs a replica iter whose stride is any multiple up to the absorbing extent', () => {
      const layout = new AxeLayout([iter(2, 1, AxeAxis.MEM)], [iter(3, 1, AxeAxis.WARP), iter(2, 2, AxeAxis.WARP)]);
      const canon = layout.canonicalize();
      expect(canon.replica).toEqual([iter(5, 1, AxeAxis.WARP)]);
      expect(imageOf(canon)).toEqual(imageOf(layout));
    });

    it('leaves replica iters alone when neither stride is a usable multiple of the other', () => {
      const layout = new AxeLayout([iter(2, 1, AxeAxis.MEM)], [iter(2, 1, AxeAxis.WARP), iter(2, 3, AxeAxis.WARP)]);
      const canon = layout.canonicalize();
      expect(canon.replica).toEqual([iter(2, 1, AxeAxis.WARP), iter(2, 3, AxeAxis.WARP)]);
      expect(canon.satisfiesGapCondition()).toBe(true);
      expect(imageOf(canon)).toEqual(imageOf(layout));
    });

    it('reports a replica set that packs too densely to be canonical', () => {
      const layout = new AxeLayout([iter(2, 1, AxeAxis.MEM)], [iter(3, 2, AxeAxis.WARP), iter(2, 3, AxeAxis.WARP)]);
      const canon = layout.canonicalize();
      expect(canon.replica.length).toBe(2);
      expect(canon.satisfiesGapCondition()).toBe(false);
      expect(imageOf(canon)).toEqual(imageOf(layout));
    });

    it('orders the replica multiset canonically whatever order it was written in', () => {
      const a = new AxeLayout([iter(4, 1, AxeAxis.MEM)], [iter(2, 1, AxeAxis.LANE), iter(2, 1, AxeAxis.WARP)]);
      const b = new AxeLayout([iter(4, 1, AxeAxis.MEM)], [iter(2, 1, AxeAxis.WARP), iter(2, 1, AxeAxis.LANE)]);
      expect(imageOf(a)).toEqual(imageOf(b));
      expect(a.equals(b)).toBe(true);
      expect(a.hash()).toBe(b.hash());
      expect(a.canonicalize().replica).toEqual(b.canonicalize().replica);
    });
  });

  describe('grouping a layout under a shape', () => {
    it('consumes whole iters when the extents already line up', () => {
      const layout = AxeLayout.rowMajor([4, 6]);
      expect(layout.group([4, 6])).toEqual([[iter(4, 6, AxeAxis.MEM)], [iter(6, 1, AxeAxis.MEM)]]);
    });

    it('splits an iter that straddles a dimension boundary, outer part first', () => {
      const layout = new AxeLayout([iter(24, 1, AxeAxis.MEM)]);
      expect(layout.group([4, 6])).toEqual([[iter(4, 6, AxeAxis.MEM)], [iter(6, 1, AxeAxis.MEM)]]);
    });

    it('declines a shape whose product does not match the domain', () => {
      expect(AxeLayout.rowMajor([4, 6]).group([4, 5])).toBeNull();
    });

    it('declines when an extent neither divides nor is divided by what the dimension needs', () => {
      expect(new AxeLayout([iter(6, 1, AxeAxis.MEM)]).group([4, 3])).toBeNull();
    });
  });

  describe('the tile product', () => {
    it('reproduces the block layout worked in the paper', () => {
      const a = new AxeLayout([iter(2, 3, AxeAxis.MEM), iter(3, 1, AxeAxis.MEM)]);
      const b = new AxeLayout([iter(8, 8, AxeAxis.MEM), iter(8, 1, AxeAxis.MEM)]);
      const tiled = AxeLayout.tile(a, [2, 3], b, [8, 8]);
      expect(tiled.shard).toEqual([
        iter(2, 192, AxeAxis.MEM),
        iter(8, 8, AxeAxis.MEM),
        iter(3, 64, AxeAxis.MEM),
        iter(8, 1, AxeAxis.MEM)
      ]);
    });

    it('satisfies f_T(x||y) = f_A(x) * span(f_B) + f_B(y)', () => {
      const a = new AxeLayout([iter(2, 3, AxeAxis.MEM), iter(3, 1, AxeAxis.MEM)]);
      const b = new AxeLayout([iter(4, 4, AxeAxis.MEM), iter(4, 1, AxeAxis.MEM)]);
      const span = b.spanSize(AxeAxis.MEM);
      const tiled = AxeLayout.tile(a, [2, 3], b, [4, 4]);

      for (let i = 0; i < 2; i++) {
        for (let j = 0; j < 3; j++) {
          for (let p = 0; p < 4; p++) {
            for (let q = 0; q < 4; q++) {
              const expected = a.apply([i, j], [2, 3])[0].get(AxeAxis.MEM) * span
                + b.apply([p, q], [4, 4])[0].get(AxeAxis.MEM);
              const actual = tiled.apply([i * 4 + p, j * 4 + q], [8, 12])[0].get(AxeAxis.MEM);
              expect(actual).toBe(expected);
            }
          }
        }
      }
    });

    it('scales the replica and the offset of A by the span of B', () => {
      const a = new AxeLayout([iter(2, 1, AxeAxis.MEM)], [iter(2, 2, AxeAxis.MEM)], coord({ [AxeAxis.MEM]: 1 }));
      const b = new AxeLayout([iter(4, 1, AxeAxis.MEM)]);
      const tiled = AxeLayout.tile(a, [2], b, [4]);
      expect(tiled.replica).toEqual([iter(2, 8, AxeAxis.MEM)]);
      expect(tiled.offset.get(AxeAxis.MEM)).toBe(4);
    });

    it('satisfies the tile identity for the whole set-valued map', () => {
      const rand = lcg(4242);
      for (let trial = 0; trial < 120; trial++) {
        const a = withReplica(rand, randomLayout(rand, [AxeAxis.MEM, AxeAxis.WARP]));
        const b = withReplica(rand, randomLayout(rand, [AxeAxis.MEM, AxeAxis.WARP]));
        const shapeA = [a.domainExtent];
        const shapeB = [b.domainExtent];
        const tiled = AxeLayout.tile(a, shapeA, b, shapeB);
        if (tiled === null) continue;

        for (let x = 0; x < shapeA[0]; x++) {
          for (let y = 0; y < shapeB[0]; y++) {
            const expected = new Set();
            for (const ca of a.applyFlat(x)) {
              for (const cb of b.applyFlat(y)) {
                const merged = new Map();
                for (const axis of new Set([...ca.keys(), ...cb.keys()])) {
                  merged.set(axis, (ca.get(axis) ?? 0) * b.spanSize(axis) + (cb.get(axis) ?? 0));
                }
                expected.add(coordKey(merged));
              }
            }
            const actual = new Set(tiled.applyFlat(x * shapeB[0] + y).map(coordKey));
            expect([...actual].sort()).toEqual([...expected].sort());
          }
        }
      }
    });

    it('declines operands of different rank', () => {
      const a = AxeLayout.rowMajor([2, 3]);
      const b = AxeLayout.rowMajor([4]);
      expect(AxeLayout.tile(a, [2, 3], b, [4])).toBeNull();
    });
  });

  describe('slicing', () => {
    it('recovers the inner tile of a tiled layout', () => {
      const a = new AxeLayout([iter(2, 3, AxeAxis.MEM), iter(3, 1, AxeAxis.MEM)]);
      const b = new AxeLayout([iter(4, 4, AxeAxis.MEM), iter(4, 1, AxeAxis.MEM)]);
      const tiled = AxeLayout.tile(a, [2, 3], b, [4, 4]);
      const sliced = tiled.slice([{ start: 4, extent: 4 }, { start: 8, extent: 4 }], [8, 12]);

      for (let p = 0; p < 4; p++) {
        for (let q = 0; q < 4; q++) {
          expect(sliced.apply([p, q], [4, 4])[0].get(AxeAxis.MEM))
            .toBe(tiled.apply([4 + p, 8 + q], [8, 12])[0].get(AxeAxis.MEM));
        }
      }
    });

    it('takes a partial range of the pivot digit', () => {
      const layout = AxeLayout.rowMajor([8, 8]);
      const sliced = layout.slice([{ start: 3, extent: 4 }, { start: 0, extent: 8 }], [8, 8]);
      expect(sliced.shard).toEqual([iter(4, 8, AxeAxis.MEM), iter(8, 1, AxeAxis.MEM)]);
      expect(sliced.offset.get(AxeAxis.MEM)).toBe(24);
    });

    it('declines a region that wraps past the pivot digit', () => {
      const layout = new AxeLayout([iter(3, 64, AxeAxis.MEM), iter(8, 1, AxeAxis.MEM)]);
      expect(layout.slice([{ start: 12, extent: 12 }], [24])).toBeNull();
    });

    it('agrees with the original map over the whole sliced region', () => {
      const rand = lcg(31337);
      for (let trial = 0; trial < 300; trial++) {
        const layout = randomLayout(rand);
        const shape = [layout.domainExtent];
        const extent = pick(rand, [1, 2, 3, 4, 6, 8, 12]);
        if (extent > shape[0]) continue;
        const start = Math.floor(rand() * (shape[0] - extent + 1));
        const sliced = layout.slice([{ start, extent }], shape);
        if (sliced === null) continue;
        expect(sliced.domainExtent).toBe(extent);
        for (let u = 0; u < extent; u++) {
          expect(sliced.applyFlat(u).map(coordKey)).toEqual(layout.applyFlat(start + u).map(coordKey));
        }
      }
    });
  });

  describe('the worked examples from the paper', () => {
    it('places an 8x16 tensor-core tile on warps 5, 6, 9 and 10', () => {
      const layout = new AxeLayout(
        [iter(8, 4, AxeAxis.LANE), iter(2, 1, AxeAxis.WARP), iter(4, 1, AxeAxis.LANE), iter(2, 1, AxeAxis.REG)],
        [iter(2, 4, AxeAxis.WARP)],
        coord({ [AxeAxis.WARP]: 5 })
      );
      const warps = new Set();
      for (let x = 0; x < layout.domainExtent; x++) {
        for (const c of layout.applyFlat(x)) warps.add(c.get(AxeAxis.WARP));
      }
      expect([...warps].sort((p, q) => p - q)).toEqual([5, 6, 9, 10]);
    });

    it('groups a 64x128 tensor fully sharded over a 2x2 device mesh', () => {
      const layout = new AxeLayout([
        iter(2, 1, 'gpuid'),
        iter(32, 128, AxeAxis.MEM),
        iter(2, 2, 'gpuid'),
        iter(64, 1, AxeAxis.MEM)
      ]);
      expect(layout.domainExtent).toBe(64 * 128);
      expect(layout.group([64, 128])).toEqual([
        [iter(2, 1, 'gpuid'), iter(32, 128, AxeAxis.MEM)],
        [iter(2, 2, 'gpuid'), iter(64, 1, AxeAxis.MEM)]
      ]);
    });

    it('gives every logical element two device coordinates when a shard is replicated', () => {
      const layout = new AxeLayout(
        [iter(2, 1, 'gpuid'), iter(32, 128, AxeAxis.MEM), iter(128, 1, AxeAxis.MEM)],
        [iter(2, 2, 'gpuid')]
      );
      expect(layout.applyFlat(0).length).toBe(2);
      expect(layout.applyFlat(0).map(c => c.get('gpuid')).sort()).toEqual([0, 2]);
    });

    it('slices the block layout down to rows 0:8 and columns 8:24', () => {
      const layout = new AxeLayout([
        iter(2, 192, AxeAxis.MEM),
        iter(8, 8, AxeAxis.MEM),
        iter(3, 64, AxeAxis.MEM),
        iter(8, 1, AxeAxis.MEM)
      ]);
      const sliced = layout.slice([{ start: 0, extent: 8 }, { start: 8, extent: 16 }], [16, 24]);
      expect(sliced.shard).toEqual([
        iter(8, 8, AxeAxis.MEM),
        iter(2, 64, AxeAxis.MEM),
        iter(8, 1, AxeAxis.MEM)
      ]);
      expect(sliced.offset.get(AxeAxis.MEM)).toBe(64);

      for (let i = 0; i < 8; i++) {
        for (let j = 0; j < 16; j++) {
          expect(sliced.apply([i, j], [8, 16])[0].get(AxeAxis.MEM))
            .toBe(layout.apply([i, 8 + j], [16, 24])[0].get(AxeAxis.MEM));
        }
      }
    });
  });

  describe('canonical form uniqueness', () => {
    function enumerateLayouts(domain) {
      const out = [];
      const axes = [AxeAxis.MEM, AxeAxis.LANE];
      const walk = (remaining, prefix) => {
        if (remaining === 1) {
          if (prefix.length > 0) out.push(prefix);
          return;
        }
        for (let extent = 2; extent <= remaining; extent++) {
          if (remaining % extent !== 0) continue;
          for (const axis of axes) {
            for (const stride of [1, 2, 3, 4, 6, 8, 12, -1, -4]) {
              walk(remaining / extent, [...prefix, iter(extent, stride, axis)]);
            }
          }
        }
      };
      walk(domain, []);
      return out;
    }

    it('gives one canonical form to every layout that induces the same map', () => {
      const groups = new Map();
      for (const shard of enumerateLayouts(12)) {
        const layout = new AxeLayout(shard);
        const key = imageOf(layout).join(';');
        const bucket = groups.get(key);
        if (bucket) bucket.push(layout);
        else groups.set(key, [layout]);
      }
      expect(groups.size).toBeGreaterThan(100);

      let checked = 0;
      for (const bucket of groups.values()) {
        if (bucket.length < 2) continue;
        const first = bucket[0].canonicalize();
        for (const other of bucket) {
          const canon = other.canonicalize();
          expect(canon.shard).toEqual(first.shard);
          expect(canon.hash()).toBe(first.hash());
          expect(other.equals(bucket[0])).toBe(true);
          checked++;
        }
      }
      expect(checked).toBeGreaterThan(200);
    });

    it('separates layouts that induce different maps', () => {
      const layouts = enumerateLayouts(8).slice(0, 400).map(shard => new AxeLayout(shard));
      for (let i = 0; i < layouts.length; i += 7) {
        for (let j = 0; j < layouts.length; j += 11) {
          const same = imageOf(layouts[i]).join(';') === imageOf(layouts[j]).join(';');
          expect(layouts[i].equals(layouts[j])).toBe(same);
        }
      }
    });

    it('gives one canonical replica set to every layout with the same fibre, under the gap condition', () => {
      const candidates = [];
      for (const e0 of [2, 3, 4]) {
        for (const s0 of [1, 2, 3, -2]) {
          for (const e1 of [1, 2, 3]) {
            for (const s1 of [1, 2, 3, 4, 6, 8]) {
              candidates.push(new AxeLayout(
                [iter(2, 64, AxeAxis.MEM)],
                [iter(e0, s0, AxeAxis.WARP), iter(e1, s1, AxeAxis.WARP)]
              ));
            }
          }
        }
      }

      const groups = new Map();
      for (const layout of candidates) {
        const key = imageOf(layout).join(';');
        const bucket = groups.get(key);
        if (bucket) bucket.push(layout);
        else groups.set(key, [layout]);
      }

      let checked = 0;
      for (const bucket of groups.values()) {
        const canon = bucket.map(l => l.canonicalize());
        if (!canon.every(c => c.satisfiesGapCondition())) continue;
        for (const c of canon) {
          expect(c.replica).toEqual(canon[0].replica);
          expect(c.offset.get(AxeAxis.WARP) ?? 0).toBe(canon[0].offset.get(AxeAxis.WARP) ?? 0);
          expect(c.hash()).toBe(canon[0].hash());
          checked++;
        }
      }
      expect(checked).toBeGreaterThan(50);
    });
  });

  describe('layout equality', () => {
    it('sees through a merge and a unit extent', () => {
      const a = new AxeLayout([iter(1, 99, AxeAxis.MEM), iter(4, 6, AxeAxis.MEM), iter(6, 1, AxeAxis.MEM)]);
      expect(a.equals(new AxeLayout([iter(24, 1, AxeAxis.MEM)]))).toBe(true);
    });

    it('separates a row-major tensor from the same shape stored column-major', () => {
      expect(AxeLayout.rowMajor([4, 6]).equals(AxeLayout.fromPermutation([1, 0], [4, 6]))).toBe(false);
    });

    it('never reports two layouts equal when the gap condition fails and their images differ', () => {
      const rand = lcg(31337);
      let violations = 0;
      for (let trial = 0; trial < 300; trial++) {
        const base = [iter(pick(rand, [2, 3, 4]), 1, AxeAxis.MEM)];
        const a = new AxeLayout(base, [
          iter(pick(rand, [2, 3]), pick(rand, [1, 2, 3]), AxeAxis.WARP),
          iter(pick(rand, [2, 3]), pick(rand, [1, 2, 3]), AxeAxis.WARP)
        ]);
        const b = new AxeLayout(base, [
          iter(pick(rand, [2, 3]), pick(rand, [1, 2, 3]), AxeAxis.WARP),
          iter(pick(rand, [2, 3]), pick(rand, [1, 2, 3]), AxeAxis.WARP)
        ]);
        if (a.canonicalize().satisfiesGapCondition() && b.canonicalize().satisfiesGapCondition()) continue;
        violations++;
        const same = imageOf(a).join('#') === imageOf(b).join('#');
        expect(a.equals(b)).toBe(same);
      }
      expect(violations).toBeGreaterThan(0);
    });
  });

  describe('symbolic extents', () => {
    it('builds a row-major layout over a dynamic batch dimension', () => {
      const n = SymInt.var('n');
      const layout = AxeLayout.rowMajor([n, 4, 8]);
      expect(layout.shard[0].extent).toBe(n);
      expect(layout.shard[0].stride).toBe(32);
      expect(layout.shard[1].stride).toBe(8);
      expect(layout.isStatic()).toBe(false);
    });

    it('still merges symbolic iters when the stride relation is structural', () => {
      const n = SymInt.var('n');
      const layout = new AxeLayout([iter(n, 8, AxeAxis.MEM), iter(8, 1, AxeAxis.MEM)]);
      expect(layout.canonicalize().shard).toEqual([iter(SymInt.mul(n, 8), 1, AxeAxis.MEM)]);
    });

    it('leaves a merge alone when nothing proves the stride relation', () => {
      const n = SymInt.var('n');
      const m = SymInt.var('m');
      const layout = new AxeLayout([iter(4, m, AxeAxis.MEM), iter(n, 1, AxeAxis.MEM)]);
      expect(layout.canonicalize().shard.length).toBe(2);
    });

    it('takes the merge when a prover can close the gap', () => {
      const n = SymInt.var('n');
      const m = SymInt.var('m');
      const prover = { canProveEqual: (a, b) => SymInt.equals(a, m) && SymInt.equals(b, n) };
      const layout = new AxeLayout([iter(4, m, AxeAxis.MEM), iter(n, 1, AxeAxis.MEM)]);
      expect(layout.canonicalize(prover).shard).toEqual([iter(SymInt.mul(4, n), 1, AxeAxis.MEM)]);
    });

    it('declines to group a layout whose extent is symbolic rather than guessing', () => {
      const n = SymInt.var('n');
      const layout = AxeLayout.rowMajor([n, 8]);
      expect(layout.group([8, 8])).toBeNull();
    });
  });

  describe('memory footprint', () => {
    it('measures a dense tensor as its element count', () => {
      expect(AxeLayout.rowMajor([4, 6]).footprint()).toBe(24);
    });

    it('measures a padded row as the padded span', () => {
      expect(new AxeLayout([iter(4, 8, AxeAxis.MEM), iter(6, 1, AxeAxis.MEM)]).footprint()).toBe(30);
    });

    it('reports an unknown footprint for a symbolic layout', () => {
      expect(AxeLayout.rowMajor([SymInt.var('n'), 8]).footprint()).toBe(-1);
    });
  });
});
