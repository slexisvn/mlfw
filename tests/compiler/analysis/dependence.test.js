import { describe, it, expect } from 'vitest';
import {
  ForNode, BufferStoreNode, BufferLoadNode, VariableNode, IntImmNode, MathOpNode, ForKind, PrimFunc,
} from '../../../src/compiler/ir/tensor/nodes.js';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import { collectBufferAccesses, AccessKind } from '../../../src/compiler/analysis/buffer_access.js';
import {
  accessDependence, dependences, carriesDependence, permutationPreservesDependences,
  Direction, ANY_DIRECTION,
} from '../../../src/compiler/analysis/dependence.js';

const v = (name) => new VariableNode(name, 'int32');
const i32 = (n) => new IntImmNode(n);
const add = (a, b) => new MathOpNode('+', a, b);
const mul = (a, b) => new MathOpNode('*', a, b);
const shift = (name, k) => (k === 0 ? v(name) : add(v(name), i32(k)));
const gcdOf = (a, b) => (b === 0 ? a : gcdOf(b, a % b));

function loopNest(names, extents, body) {
  let node = body;
  for (let d = names.length - 1; d >= 0; d--) {
    node = new ForNode(v(names[d]), i32(0), i32(extents[d]), ForKind.SERIAL, node);
  }
  return node;
}

function selfUpdateFunc(names, extents, shape, writeOffsets, readOffsets) {
  const A = new Buffer('A', shape, 'float32', 'global');
  const writeIdx = writeOffsets.map(([n, k]) => shift(names[n], k));
  const readIdx = readOffsets.map(([n, k]) => shift(names[n], k));
  const store = new BufferStoreNode(A, writeIdx, add(new BufferLoadNode(A, readIdx), i32(1)));
  return new PrimFunc('f', [], loopNest(names, extents, store), new Map([['A', A]]));
}

function bruteForceMasks(extents, shape, writeOffsets, readOffsets) {
  const rank = writeOffsets.length;
  const n = extents.length;
  const masks = new Array(n).fill(0);
  const inRange = (idx) => idx.every((x, d) => x >= 0 && x < shape[d]);
  const points = [];
  const walkPoints = (prefix) => {
    if (prefix.length === n) { points.push([...prefix]); return; }
    for (let x = 0; x < extents[prefix.length]; x++) walkPoints([...prefix, x]);
  };
  walkPoints([]);

  const lexCompare = (a, b) => {
    for (let d = 0; d < a.length; d++) {
      if (a[d] !== b[d]) return a[d] < b[d] ? -1 : 1;
    }
    return 0;
  };

  for (const I of points) {
    const w = writeOffsets.map(([lv, k]) => I[lv] + k);
    if (!inRange(w)) continue;
    for (const J of points) {
      const r = readOffsets.map(([lv, k]) => J[lv] + k);
      if (!inRange(r)) continue;
      let same = true;
      for (let d = 0; d < rank; d++) if (w[d] !== r[d]) { same = false; break; }
      if (!same) continue;
      const writeFirst = lexCompare(I, J) <= 0;
      const earlier = writeFirst ? I : J;
      const later = writeFirst ? J : I;
      for (let d = 0; d < n; d++) {
        const delta = later[d] - earlier[d];
        masks[d] |= delta > 0 ? Direction.LT : (delta === 0 ? Direction.EQ : Direction.GT);
      }
    }
  }
  return masks;
}

function writeReadPair(func) {
  const { byBuffer } = collectBufferAccesses(func.body);
  const accesses = [...byBuffer.values()][0];
  const write = accesses.find((a) => a.kind === AccessKind.WRITE);
  const read = accesses.find((a) => a.kind === AccessKind.READ);
  return { write, read };
}

describe('affine dependence testing against brute-force enumeration of the iteration space', () => {
  const cases = [
    { name: 'A[i] = A[i-1] (true dependence carried by i)', names: ['i'], extents: [8], shape: [8], w: [[0, 0]], r: [[0, -1]] },
    { name: 'A[i] = A[i+1] (anti dependence carried by i)', names: ['i'], extents: [8], shape: [8], w: [[0, 0]], r: [[0, 1]] },
    { name: 'A[i] = A[i] (loop-independent only)', names: ['i'], extents: [8], shape: [8], w: [[0, 0]], r: [[0, 0]] },
    { name: 'A[i][j] = A[i-1][j+1] (skewed stencil)', names: ['i', 'j'], extents: [6, 6], shape: [6, 6], w: [[0, 0], [1, 0]], r: [[0, -1], [1, 1]] },
    { name: 'A[i][j] = A[i][j-1] (carried by the inner loop only)', names: ['i', 'j'], extents: [5, 5], shape: [5, 5], w: [[0, 0], [1, 0]], r: [[0, 0], [1, -1]] },
    { name: 'A[i][j] = A[i-2][j] (distance 2 in i)', names: ['i', 'j'], extents: [5, 4], shape: [5, 4], w: [[0, 0], [1, 0]], r: [[0, -2], [1, 0]] },
  ];

  for (const c of cases) {
    it(`${c.name}: analysis masks match the enumerated direction set`, () => {
      const func = selfUpdateFunc(c.names, c.extents, c.shape, c.w, c.r);
      const { write, read } = writeReadPair(func);
      const dep = accessDependence(write, read);
      const expected = bruteForceMasks(c.extents, c.shape, c.w, c.r);
      expect(dep).not.toBeNull();
      expect(dep.masks).toEqual(expected);
    });
  }

  it('proves independence for a write/read pair whose subscripts can never collide', () => {
    const A = new Buffer('A', [16], 'float32', 'global');
    const i = v('i');
    const store = new BufferStoreNode(A, [mul(i, i32(2))], add(new BufferLoadNode(A, [add(mul(i, i32(2)), i32(1))]), i32(1)));
    const func = new PrimFunc('f', [], loopNest(['i'], [8], store), new Map([['A', A]]));
    const { write, read } = writeReadPair(func);

    expect(bruteForceMasks([8], [16], [[0, 0]], [[0, 0]]).every((m) => m === 0)).toBe(false);
    expect(accessDependence(write, read)).toBeNull();
  });

  it('leaves a reduction axis unconstrained so it reads as loop-carried', () => {
    const C = new Buffer('C', [4, 4], 'float32', 'global');
    const A = new Buffer('A', [4, 4], 'float32', 'global');
    const store = new BufferStoreNode(C, [v('m'), v('n')],
      new MathOpNode('+', new BufferLoadNode(C, [v('m'), v('n')]), new BufferLoadNode(A, [v('m'), v('k')])));
    const body = loopNest(['m', 'n', 'k'], [4, 4, 4], store);
    const func = new PrimFunc('f', [], body, new Map([['C', C], ['A', A]]));
    const deps = dependences(collectBufferAccesses(func.body).byBuffer);

    const loops = [];
    for (let node = func.body; node && node.type === 'ForNode'; node = node.body) loops.push(node);
    expect(carriesDependence(deps, loops[2])).not.toBeNull();
    expect(carriesDependence(deps, loops[0])).toBeNull();
    expect(carriesDependence(deps, loops[1])).toBeNull();
  });
});

describe('a loop extent that is not an integer literal', () => {
  function selfUpdateFuncWithSymbolicExtent(names, extents, shape, writeOffsets, readOffsets, symbolicLevel) {
    const A = new Buffer('A', shape, 'float32', 'global');
    const writeIdx = writeOffsets.map(([n, k]) => shift(names[n], k));
    const readIdx = readOffsets.map(([n, k]) => shift(names[n], k));
    let node = new BufferStoreNode(A, writeIdx, add(new BufferLoadNode(A, readIdx), i32(1)));
    for (let d = names.length - 1; d >= 0; d--) {
      const extent = d === symbolicLevel ? v(`${names[d]}_extent`) : i32(extents[d]);
      node = new ForNode(v(names[d]), i32(0), extent, ForKind.SERIAL, node);
    }
    return new PrimFunc('f', [], node, new Map([['A', A]]));
  }

  function nestLoops(func) {
    const loops = [];
    for (let node = func.body; node && node.type === 'ForNode'; node = node.body) loops.push(node);
    return loops;
  }

  it('keeps the symbolic level in the direction vector instead of truncating the nest there', () => {
    const func = selfUpdateFuncWithSymbolicExtent(['i', 'j'], [6, 6], [6, 6], [[0, 0], [1, 0]], [[0, -1], [1, 0]], 0);
    const { write, read } = writeReadPair(func);
    const dep = accessDependence(write, read);

    expect(dep.loops.map((l) => l.name)).toEqual(['i', 'j']);
    expect(dep.masks).toEqual([Direction.LT, Direction.EQ]);
  });

  it('reports a symbolic reduction level as loop-carried rather than as no dependence at all', () => {
    const C = new Buffer('C', [4, 4], 'float32', 'global');
    const A = new Buffer('A', [4, 4], 'float32', 'global');
    const store = new BufferStoreNode(C, [v('m'), v('n')],
      new MathOpNode('+', new BufferLoadNode(C, [v('m'), v('n')]), new BufferLoadNode(A, [v('m'), v('k')])));
    let body = new ForNode(v('k'), i32(0), v('k_extent'), ForKind.SERIAL, store);
    body = new ForNode(v('n'), i32(0), i32(4), ForKind.SERIAL, body);
    body = new ForNode(v('m'), i32(0), i32(4), ForKind.SERIAL, body);
    const func = new PrimFunc('f', [], body, new Map([['C', C], ['A', A]]));
    const deps = dependences(collectBufferAccesses(func.body).byBuffer);
    const loops = nestLoops(func);

    expect(carriesDependence(deps, loops[2])).not.toBeNull();
    expect(carriesDependence(deps, loops[0])).toBeNull();
    expect(carriesDependence(deps, loops[1])).toBeNull();
  });

  it('drops the distance-against-extent refinement, keeping a dependence a literal extent rules out', () => {
    const literal = selfUpdateFuncWithSymbolicExtent(['i'], [8], [16], [[0, 0]], [[0, 8]], -1);
    const symbolic = selfUpdateFuncWithSymbolicExtent(['i'], [8], [16], [[0, 0]], [[0, 8]], 0);
    const pair = writeReadPair(literal);
    const symbolicPair = writeReadPair(symbolic);

    expect(accessDependence(pair.write, pair.read)).toBeNull();
    expect(accessDependence(symbolicPair.write, symbolicPair.read).masks).toEqual([Direction.LT]);
  });

  it('refuses an interchange whose legality rests on the unknown extent', () => {
    const func = selfUpdateFuncWithSymbolicExtent(['i', 'j'], [6, 6], [6, 6], [[0, 0], [1, 0]], [[0, -1], [1, 1]], 0);
    const deps = dependences(collectBufferAccesses(func.body).byBuffer);
    const loops = nestLoops(func);

    expect(permutationPreservesDependences(deps, loops, [loops[1], loops[0]])).not.toBeNull();
  });
});

describe('Allen-Kennedy permutation legality from direction vectors', () => {
  function nestLoops(func) {
    const loops = [];
    for (let node = func.body; node && node.type === 'ForNode'; node = node.body) loops.push(node);
    return loops;
  }

  it('refuses interchange that would make a (<, >) direction vector lexicographically negative', () => {
    const func = selfUpdateFunc(['i', 'j'], [6, 6], [6, 6], [[0, 0], [1, 0]], [[0, -1], [1, 1]]);
    const deps = dependences(collectBufferAccesses(func.body).byBuffer);
    const loops = nestLoops(func);
    expect(permutationPreservesDependences(deps, loops, [loops[1], loops[0]])).not.toBeNull();
  });

  it('allows interchange when every direction vector stays lexicographically positive', () => {
    const func = selfUpdateFunc(['i', 'j'], [6, 6], [6, 6], [[0, 0], [1, 0]], [[0, -1], [1, -1]]);
    const deps = dependences(collectBufferAccesses(func.body).byBuffer);
    const loops = nestLoops(func);
    expect(permutationPreservesDependences(deps, loops, [loops[1], loops[0]])).toBeNull();
  });

  it('allows any permutation when the only dependence is loop-independent', () => {
    const func = selfUpdateFunc(['i', 'j'], [6, 6], [6, 6], [[0, 0], [1, 0]], [[0, 0], [1, 0]]);
    const deps = dependences(collectBufferAccesses(func.body).byBuffer);
    const loops = nestLoops(func);
    expect(permutationPreservesDependences(deps, loops, [loops[1], loops[0]])).toBeNull();
  });

  it('treats an unanalyzable subscript as every direction and refuses the interchange', () => {
    const A = new Buffer('A', [8, 8], 'float32', 'global');
    const idx = new Buffer('idx', [8], 'int32', 'global');
    const store = new BufferStoreNode(A, [v('i'), v('j')],
      new BufferLoadNode(A, [new BufferLoadNode(idx, [v('i')]), new BufferLoadNode(idx, [v('j')])]));
    const func = new PrimFunc('f', [], loopNest(['i', 'j'], [8, 8], store), new Map([['A', A]]));
    const deps = dependences(collectBufferAccesses(func.body).byBuffer);
    const loops = nestLoops(func);
    const dep = deps.find((d) => d.buffer === A && d.masks.some((m) => m === ANY_DIRECTION));
    expect(dep).toBeDefined();
    expect(permutationPreservesDependences(deps, loops, [loops[1], loops[0]])).not.toBeNull();
  });
});

function linearIndex(names, terms, konst) {
  let expr = i32(konst);
  for (const [level, coeff] of terms) {
    expr = add(expr, coeff === 1 ? v(names[level]) : mul(v(names[level]), i32(coeff)));
  }
  return expr;
}

function flatSelfUpdateFunc(names, extents, size, write, read) {
  const A = new Buffer('A', [size], 'float32', 'global');
  const store = new BufferStoreNode(A, [linearIndex(names, write.terms, write.konst)],
    add(new BufferLoadNode(A, [linearIndex(names, read.terms, read.konst)]), i32(1)));
  return new PrimFunc('f', [], loopNest(names, extents, store), new Map([['A', A]]));
}

function flatBruteForceMasks(extents, size, write, read) {
  const n = extents.length;
  const masks = new Array(n).fill(0);
  const points = [];
  const walkPoints = (prefix) => {
    if (prefix.length === n) { points.push([...prefix]); return; }
    for (let x = 0; x < extents[prefix.length]; x++) walkPoints([...prefix, x]);
  };
  walkPoints([]);
  const evaluate = (spec, point) => spec.terms.reduce((acc, [level, coeff]) => acc + coeff * point[level], spec.konst);
  const lexCompare = (a, b) => {
    for (let d = 0; d < a.length; d++) {
      if (a[d] !== b[d]) return a[d] < b[d] ? -1 : 1;
    }
    return 0;
  };

  for (const I of points) {
    const w = evaluate(write, I);
    if (w < 0 || w >= size) continue;
    for (const J of points) {
      const r = evaluate(read, J);
      if (r < 0 || r >= size) continue;
      if (w !== r) continue;
      const writeFirst = lexCompare(I, J) <= 0;
      const earlier = writeFirst ? I : J;
      const later = writeFirst ? J : I;
      for (let d = 0; d < n; d++) {
        const delta = later[d] - earlier[d];
        masks[d] |= delta > 0 ? Direction.LT : (delta === 0 ? Direction.EQ : Direction.GT);
      }
    }
  }
  return masks;
}

describe('Banerjee bounds prune multi-index subscripts the GCD test cannot', () => {
  const cases = [
    {
      name: 'A[2i+4j] vs A[2i+4j+100] on a 128-wide buffer',
      names: ['i', 'j'], extents: [4, 4], size: 128,
      write: { terms: [[0, 2], [1, 4]], konst: 0 },
      read: { terms: [[0, 2], [1, 4]], konst: 100 },
    },
    {
      name: 'A[8i+j] vs A[8i+j-40] on a 64-wide buffer',
      names: ['i', 'j'], extents: [4, 4], size: 64,
      write: { terms: [[0, 8], [1, 1]], konst: 0 },
      read: { terms: [[0, 8], [1, 1]], konst: -40 },
    },
  ];

  for (const c of cases) {
    it(`${c.name}: GCD admits it, Banerjee proves independence`, () => {
      const expected = flatBruteForceMasks(c.extents, c.size, c.write, c.read);
      expect(expected.every((m) => m === 0)).toBe(true);

      const g = c.write.terms.reduce((acc, [, coeff]) => (acc === 0 ? Math.abs(coeff) : gcdOf(acc, Math.abs(coeff))), 0);
      expect(Math.abs((c.read.konst - c.write.konst) % g)).toBe(0);

      const func = flatSelfUpdateFunc(c.names, c.extents, c.size, c.write, c.read);
      const { write, read } = writeReadPair(func);
      expect(accessDependence(write, read)).toBeNull();
    });
  }

  const refined = [
    {
      name: 'A[4i+j] vs A[4i+j-1]',
      names: ['i', 'j'], extents: [4, 4], size: 16,
      write: { terms: [[0, 4], [1, 1]], konst: 0 },
      read: { terms: [[0, 4], [1, 1]], konst: -1 },
    },
    {
      name: 'A[i+2j] vs A[i+2j+3]',
      names: ['i', 'j'], extents: [4, 4], size: 16,
      write: { terms: [[0, 1], [1, 2]], konst: 0 },
      read: { terms: [[0, 1], [1, 2]], konst: 3 },
    },
  ];

  for (const c of refined) {
    it(`${c.name}: reported directions stay a sound superset of the enumerated ones`, () => {
      const expected = flatBruteForceMasks(c.extents, c.size, c.write, c.read);
      expect(expected.some((m) => m !== 0)).toBe(true);

      const func = flatSelfUpdateFunc(c.names, c.extents, c.size, c.write, c.read);
      const { write, read } = writeReadPair(func);
      const dep = accessDependence(write, read);
      expect(dep).not.toBeNull();
      for (let d = 0; d < expected.length; d++) {
        expect(dep.masks[d] & expected[d]).toBe(expected[d]);
      }
    });
  }

  it('narrows a direction the GCD test would have left wide open', () => {
    const names = ['i', 'j'];
    const extents = [4, 4];
    const write = { terms: [[0, 8], [1, 1]], konst: 0 };
    const read = { terms: [[0, 8], [1, 1]], konst: -8 };
    const func = flatSelfUpdateFunc(names, extents, 64, write, read);
    const { write: w, read: r } = writeReadPair(func);
    const dep = accessDependence(w, r);
    expect(dep).not.toBeNull();
    expect(dep.masks.some((m) => m !== ANY_DIRECTION)).toBe(true);

    const expected = flatBruteForceMasks(extents, 64, write, read);
    for (let d = 0; d < expected.length; d++) {
      expect(dep.masks[d] & expected[d]).toBe(expected[d]);
    }
  });
});
