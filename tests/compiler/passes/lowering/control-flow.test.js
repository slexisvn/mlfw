import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { lowerGraphToPrimFunc } from '../../../../src/compiler/passes/lowering/graph_to_tensor.js';
import {
  IfThenElseNode, WhileNode, BufferStoreNode, BufferLoadNode
} from '../../../../src/compiler/ir/tensor/nodes.js';
import { compileGraph } from '../../../../src/compiler/pipeline/compiler.js';
import { CPUTarget, WasmTarget } from '../../../../src/compiler/support/target.js';
import { F32 } from '../../../_utils/ir_fixture.js';

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

const BOOL = ScalarType.BOOL;

describe('if lowering result unification', () => {
  it('both then and else branches write into the same result buffer', () => {
    const t = new TensorType([4], F32);
    const pred = new TensorType([], BOOL);
    const func = buildFunction('f', [pred, t], [t], (b, args) => {
      const ifop = b.ifOp(args[0], [t],
        (tb) => { tb.yieldOp([tb.neg(args[1]).getResult(0)]); },
        (eb) => { eb.yieldOp([eb.exp(args[1]).getResult(0)]); });
      b.returnOp([ifop.getResult(0)]);
    });
    const pf = lowerGraphToPrimFunc(func);

    const ifNodes = collectNodes(pf.body, n => n instanceof IfThenElseNode);
    expect(ifNodes.length).toBe(1);
    const ifn = ifNodes[0];

    expect(ifn.elseBody).not.toBeNull();

    const thenStores = collectNodes(ifn.thenBody, n => n instanceof BufferStoreNode);
    const elseStores = collectNodes(ifn.elseBody, n => n instanceof BufferStoreNode);
    const thenTargets = new Set(thenStores.map(s => s.buffer.name));
    const elseTargets = new Set(elseStores.map(s => s.buffer.name));

    const shared = [...thenTargets].filter(n => elseTargets.has(n));
    expect(shared.length).toBeGreaterThan(0);
  });
});

describe('while lowering loop carry and condition', () => {
  it('stores predicate into condVar and copies body yields back into loop buffers', () => {
    const t = new TensorType([4], F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const wop = b.whileOp([args[0]],
        (cb, cargs) => {
          const anyOp = cb.scalarConstant(1, BOOL);
          cb.yieldOp([anyOp.getResult(0)]);
        },
        (bb, bargs) => {
          bb.yieldOp([bb.neg(bargs[0]).getResult(0)]);
        });
      b.returnOp([wop.getResult(0)]);
    });
    const pf = lowerGraphToPrimFunc(func);

    const whileNodes = collectNodes(pf.body, n => n instanceof WhileNode);
    expect(whileNodes.length).toBe(1);
    const wn = whileNodes[0];

    expect(wn.condVar).toBeDefined();
    expect(wn.condVar.shape).toEqual([]);

    const condStores = collectNodes(wn.condBody, n => n instanceof BufferStoreNode);
    const storesIntoCond = condStores.filter(s => s.buffer === wn.condVar);
    expect(storesIntoCond.length).toBe(1);
    expect(storesIntoCond[0].value).toBeInstanceOf(BufferLoadNode);

    const bodyStores = collectNodes(wn.loopBody, n => n instanceof BufferStoreNode);
    expect(bodyStores.length).toBeGreaterThan(0);
  });
});

describe('control flow end-to-end execution vs reference (cpu+wasm)', () => {
  const F = ScalarType.F32, I32 = ScalarType.I32;

  function buildWhile(K) {
    const xT = new TensorType([4], F);
    return buildFunction('wh', [xT], [xT], (b, a) => {
      const acc0 = b.broadcast(b.scalarConstant(0, F).getResult(0), [4], []).getResult(0);
      const i0 = b.scalarConstant(0, I32).getResult(0);
      const w = b.whileOp([acc0, i0],
        (cb, args) => { cb.yieldOp([cb.compare(args[1], b.scalarConstant(K, I32).getResult(0), 'lt').getResult(0)]); },
        (bb, args) => { bb.yieldOp([bb.add(args[0], a[0]).getResult(0), bb.add(args[1], b.scalarConstant(1, I32).getResult(0)).getResult(0)]); });
      b.returnOp([w.getResult(0)]);
    });
  }

  for (const [tname, makeTarget] of [['cpu', CPUTarget], ['wasm', WasmTarget]]) {
    for (const K of [0, 1, 3, 5]) {
      it(`while accumulates x ${K} times (acc=K*x) on ${tname}`, () => {
        const res = compileGraph(buildWhile(K), makeTarget());
        const out = new Float32Array(4);
        res.run('wh', new Float32Array([1, 2, 3, 4]), out);
        expect(Array.from(out)).toEqual([1 * K, 2 * K, 3 * K, 4 * K]);
      });
    }

    function buildIf(predTrue) {
      const xT = new TensorType([4], F);
      return buildFunction('iff', [xT], [xT], (b, a) => {
        const thr = b.scalarConstant(predTrue ? -100 : 100, F).getResult(0);
        const s = b.reduce(a[0], b.scalarConstant(0, F).getResult(0), [0], 'sum').getResult(0);
        const pred = b.compare(s, thr, 'gt').getResult(0);
        const two = b.broadcast(b.scalarConstant(2, F).getResult(0), [4], []).getResult(0);
        const one = b.broadcast(b.scalarConstant(1, F).getResult(0), [4], []).getResult(0);
        const ifo = b.ifOp(pred, [xT],
          (tb) => { tb.yieldOp([tb.mul(a[0], two).getResult(0)]); },
          (eb) => { eb.yieldOp([eb.add(a[0], one).getResult(0)]); });
        b.returnOp([ifo.getResult(0)]);
      });
    }

    it(`if selects then-branch (x*2) when predicate true on ${tname}`, () => {
      const res = compileGraph(buildIf(true), makeTarget());
      const out = new Float32Array(4);
      res.run('iff', new Float32Array([1, 2, 3, 4]), out);
      expect(Array.from(out)).toEqual([2, 4, 6, 8]);
    });

    it(`if selects else-branch (x+1) when predicate false on ${tname}`, () => {
      const res = compileGraph(buildIf(false), makeTarget());
      const out = new Float32Array(4);
      res.run('iff', new Float32Array([1, 2, 3, 4]), out);
      expect(Array.from(out)).toEqual([2, 3, 4, 5]);
    });
  }
});

describe('decomposition recurses into control-flow regions', () => {
  const I32 = ScalarType.I32;
  const sigmoid = (v) => 1 / (1 + Math.exp(-v));
  const refs = { sigmoid, silu: (v) => v * sigmoid(v), gelu: (v) => v * sigmoid(1.702 * v) };

  for (const op of ['sigmoid', 'gelu', 'silu']) {
    for (const [tname, makeTarget] of [['cpu', CPUTarget], ['wasm', WasmTarget]]) {
      it(`${op} inside an if-branch compiles + runs on ${tname}`, () => {
        const t = new TensorType([4], F32);
        const pred = new TensorType([], BOOL);
        const func = buildFunction('iff', [pred, t], [t], (b, a) => {
          const io = b.ifOp(a[0], [t],
            (tb) => tb.yieldOp([tb[op](a[1]).getResult(0)]),
            (eb) => eb.yieldOp([eb.neg(a[1]).getResult(0)]));
          b.returnOp([io.getResult(0)]);
        });
        const res = compileGraph(func, makeTarget());
        const x = new Float32Array([0.5, -0.5, 1.0, -1.0]);
        const out = new Float32Array(4);
        res.run('iff', new Uint8Array([1]), x, out);
        for (let i = 0; i < 4; i++) expect(out[i]).toBeCloseTo(refs[op](x[i]), 5);
      });
    }
  }

  for (const [tname, makeTarget] of [['cpu', CPUTarget], ['wasm', WasmTarget]]) {
    it(`sigmoid inside a while body compiles + runs on ${tname}`, () => {
      const t = new TensorType([4], F32);
      const func = buildFunction('w', [t], [t], (b, a) => {
        const cnt = b.scalarConstant(2, I32).getResult(0);
        const one = b.scalarConstant(1, I32).getResult(0);
        const zero = b.scalarConstant(0, I32).getResult(0);
        const w = b.whileOp([a[0], cnt],
          (cb, c) => cb.yieldOp([cb.compare(c[1], zero, 'gt').getResult(0)]),
          (bb, c) => bb.yieldOp([bb.sigmoid(c[0]).getResult(0), bb.sub(c[1], one).getResult(0)]));
        b.returnOp([w.getResult(0)]);
      });
      const res = compileGraph(func, makeTarget());
      const x = new Float32Array([0.5, -0.5, 1.0, -1.0]);
      const out = new Float32Array(4);
      res.run('w', x, out);
      for (let i = 0; i < 4; i++) expect(out[i]).toBeCloseTo(sigmoid(sigmoid(x[i])), 5);
    });
  }
});
