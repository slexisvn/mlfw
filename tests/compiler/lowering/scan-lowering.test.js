import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { lowerGraphToPrimFunc } from '../../../src/compiler/passes/lowering/graph_to_tensor.js';
import { ForNode, ForKind, BufferStoreNode } from '../../../src/compiler/ir/tensor/nodes.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { CPUTarget, WasmTarget } from '../../../src/backend/target.js';

const SKIP_KEYS = new Set(['_parent', '_parentKey', '_parentIdx']);
function collectNodes(node, predicate) {
  const result = [];
  const visited = new Set();
  const stack = [node];
  while (stack.length > 0) {
    const n = stack.pop();
    if (!n || typeof n !== 'object' || visited.has(n)) continue;
    visited.add(n);
    if (predicate(n)) result.push(n);
    for (const key of Object.keys(n)) {
      if (SKIP_KEYS.has(key)) continue;
      const val = n[key];
      if (Array.isArray(val)) for (const item of val) stack.push(item);
      else if (val && typeof val === 'object') stack.push(val);
    }
  }
  return result;
}

const F32 = ScalarType.F32;

function buildCumsum(T, D) {
  const xsT = new TensorType([T, D], F32);
  const carryT = new TensorType([D], F32);
  const ysT = new TensorType([T, D], F32);
  return buildFunction('sc', [xsT, carryT], [carryT, ysT], (b, args) => {
    const sc = b.scanOp([args[0]], [args[1]], (bb, xt, carry) => {
      const next = bb.add(carry[0], xt[0]).getResult(0);
      return [[next], [next]];
    });
    b.returnOp([sc.getResult(0), sc.getResult(1)]);
  });
}

describe('scan lowering', () => {
  it('lowers to a single serial ForNode with counter-indexed output store', () => {
    const pf = lowerGraphToPrimFunc(buildCumsum(3, 2));
    const serialFors = collectNodes(pf.body, n => n instanceof ForNode && n.kind === ForKind.SERIAL);
    expect(serialFors.length).toBeGreaterThan(0);
    const stores = collectNodes(pf.body, n => n instanceof BufferStoreNode && n.indices.length >= 2);
    expect(stores.length).toBeGreaterThan(0);
  });
});

describe('scan end-to-end (cpu + wasm)', () => {
  for (const [tname, makeTarget] of [['cpu', CPUTarget], ['wasm', WasmTarget]]) {
    for (const T of [1, 3, 5]) {
      it(`cumulative sum, T=${T} on ${tname}`, () => {
        const D = 2;
        const res = compileGraph(buildCumsum(T, D), makeTarget());
        const xs = new Float32Array(T * D);
        for (let i = 0; i < xs.length; i++) xs[i] = i + 1;
        const finalCarry = new Float32Array(D);
        const ys = new Float32Array(T * D);
        res.run('sc', xs, new Float32Array(D), finalCarry, ys);

        const expectedYs = new Float32Array(T * D);
        const acc = new Float32Array(D);
        for (let t = 0; t < T; t++) {
          for (let d = 0; d < D; d++) {
            acc[d] += xs[t * D + d];
            expectedYs[t * D + d] = acc[d];
          }
        }
        for (let i = 0; i < ys.length; i++) expect(ys[i]).toBeCloseTo(expectedYs[i], 5);
        for (let d = 0; d < D; d++) expect(finalCarry[d]).toBeCloseTo(acc[d], 5);
      });
    }
  }
});

describe('scan builder validation', () => {
  it('rejects xs inputs with mismatched leading lengths', () => {
    const ty = (s) => new TensorType(s, ScalarType.F32);
    expect(() => buildFunction('s', [ty([3, 2]), ty([4, 2]), ty([2])], [ty([2])], (b, [xs1, xs2, init]) => {
      const s = b.scanOp([xs1, xs2], [init], (bb, xt, cy) => {
        const nc = bb.add(cy[0], xt[0]).getResult(0);
        return [[nc], [nc]];
      });
      b.returnOp([s.getResult(0)]);
    })).toThrow(/leading length/);
  });
});
