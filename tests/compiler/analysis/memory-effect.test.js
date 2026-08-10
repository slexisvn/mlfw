import { describe, it, expect, afterEach } from 'vitest';
import { registry } from '../../../src/compiler/ir/graph/ops.js';
import { OpDef, SideEffectKind } from '../../../src/compiler/ir/graph/op_registry.js';
import { MemoryEffectAnalysis } from '../../../src/compiler/analysis/memory_effect.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { UseDefAnalysis } from '../../../src/compiler/analysis/use_def.js';
import { PostDominanceAnalysis } from '../../../src/compiler/analysis/dominance.js';
import { LivenessAnalysis } from '../../../src/compiler/analysis/liveness.js';
import { scalingRatio, QUADRATIC_RATIO } from '../../_utils/scaling.js';

function makeValue(id) {
  return { id, _uses: [], uses() { return this._uses; } };
}

function makeOp(opName, operands, results) {
  return {
    opName,
    numOperands: operands.length,
    numResults: results.length,
    getOperand(i) { return operands[i]; },
    getResult(i) { return results[i]; },
  };
}

function makeFunc(ops) {
  return { ops() { return ops; } };
}

const registeredTestOps = new Set();

function registerTestOp(name, sideEffects) {
  registry.register(new OpDef({ name, numOperands: 1, numResults: 1, sideEffects }));
  registeredTestOps.add(name);
}

afterEach(() => {
  for (const name of registeredTestOps) registry.unregister(name);
  registeredTestOps.clear();
});

describe('MemoryEffectAnalysis effectKind bitmask gating', () => {
  it('READ-only op does not tag its results as WRITE', () => {
    registerTestOp('__test_read_only__', SideEffectKind.READ);
    const operand = makeValue('in');
    const resultVal = makeValue('out');
    const op = makeOp('__test_read_only__', [operand], [resultVal]);
    const func = makeFunc([op]);

    const result = MemoryEffectAnalysis.compute(func);
    expect(result.getReadersOf(operand)).toContain(op);
    expect(result.getWritersOf(resultVal)).not.toContain(op);
    expect(result.getEffectsOn(resultVal)).toHaveLength(0);
  });

  it('WRITE-only op does not tag its operands as READ', () => {
    registerTestOp('__test_write_only__', SideEffectKind.WRITE);
    const operand = makeValue('in2');
    const resultVal = makeValue('out2');
    const op = makeOp('__test_write_only__', [operand], [resultVal]);
    const func = makeFunc([op]);

    const result = MemoryEffectAnalysis.compute(func);
    expect(result.getWritersOf(resultVal)).toContain(op);
    expect(result.getReadersOf(operand)).not.toContain(op);
    expect(result.getEffectsOn(operand)).toHaveLength(0);
  });

  it('READ|WRITE op tags both operands and results', () => {
    registerTestOp('__test_read_write__', SideEffectKind.READ | SideEffectKind.WRITE);
    const operand = makeValue('in3');
    const resultVal = makeValue('out3');
    const op = makeOp('__test_read_write__', [operand], [resultVal]);
    const func = makeFunc([op]);

    const result = MemoryEffectAnalysis.compute(func);
    expect(result.getReadersOf(operand)).toContain(op);
    expect(result.getWritersOf(resultVal)).toContain(op);
  });
});

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const _ri = (r, lo, hi) => lo + Math.floor(r() * (hi - lo + 1));
const _pick = (r, arr) => arr[Math.floor(r() * arr.length)];
const _UNARY = ['neg', 'relu', 'exp', 'tanh', 'sigmoid', 'abs', 'sqrt'];
const _BINARY = ['add', 'sub', 'mul', 'maximum', 'minimum'];

function genAnalysisGraph(r) {
  const F = ScalarType.F32;
  const t = new TensorType([4, 4], F);
  const nArgs = _ri(r, 2, 3);
  const nOps = _ri(r, 4, 9);
  return buildFunction('g', Array.from({ length: nArgs }, () => t), [t], (b, args) => {
    const e = args.map((a) => ({ v: a, sh: '4,4' }));
    const same = (k) => e.filter((x) => x.sh === k);
    for (let i = 0; i < nOps; i++) {
      const roll = r();
      if (roll < 0.18) { const s = _pick(r, same('4,4')); const red = b.reduce(s.v, b.scalarConstant(0, F).getResult(0), [1], 'sum'); const v4 = red.getResult(0); e.push({ v: v4, sh: '4' }); e.push({ v: b.broadcast(v4, [4, 4], [0]).getResult(0), sh: '4,4' }); continue; }
      if (roll < 0.3) { const s = _pick(r, same('4,4')); const rr = b.reshape(s.v, [16]); e.push({ v: b.reshape(rr.getResult(0), [4, 4]).getResult(0), sh: '4,4' }); continue; }
      if (roll < 0.4) { const s = _pick(r, same('4,4')); e.push({ v: b.transpose(s.v, [1, 0]).getResult(0), sh: '4,4' }); continue; }
      if (roll < 0.7) { const s = _pick(r, e); e.push({ v: b[_pick(r, _UNARY)](s.v).getResult(0), sh: s.sh }); continue; }
      const k = _pick(r, ['4,4', '4']); const c = same(k); if (!c.length) continue;
      e.push({ v: b[_pick(r, _BINARY)](_pick(r, c).v, _pick(r, c).v).getResult(0), sh: k });
    }
    const last = [...e].reverse().find((x) => x.sh === '4,4') || e[0];
    b.returnOp([last.v]);
  });
}

const ANALYSIS_GRAPHS = Array.from({ length: 60 }, (_, s) => genAnalysisGraph(mulberry32(0x5151 + s * 2654435761)));

function analysisCase(func) {
  const allOps = [...func.ops()];
  const retOp = func.getReturnOp();
  const ud = UseDefAnalysis.compute(func);
  const tIdx = new Map();
  ud.topologicalOrder.forEach((op, i) => tIdx.set(op, i));
  return { allOps, retOp, ud, tIdx };
}

describe('graph analyses vs independent brute-force on 60 seeded random graphs', () => {
  it('use_def orders every definition before its users and reports exactly the brute-force user set', () => {
    for (const [s, func] of ANALYSIS_GRAPHS.entries()) {
      const { allOps, retOp, ud, tIdx } = analysisCase(func);
      for (const op of ud.topologicalOrder) {
        for (let i = 0; i < op.numOperands; i++) {
          const d = op.getOperand(i).definingOp;
          if (d && tIdx.has(d)) expect(tIdx.get(d), `seed ${s}: ${d.opName} must precede ${op.opName}`).toBeLessThan(tIdx.get(op));
        }
      }
      const brute = new Map(); for (const op of allOps) brute.set(op, new Set());
      for (const op of allOps) for (let i = 0; i < op.numOperands; i++) { const d = op.getOperand(i).definingOp; if (d && brute.has(d)) brute.get(d).add(op); }
      for (const op of allOps) {
        if (op === retOp) continue;
        const got = [...(ud.opUsers.get(op) || new Set())];
        expect(got.length, `seed ${s}: users of ${op.opName}`).toBe(brute.get(op).size);
        for (const u of got) expect(brute.get(op).has(u), `seed ${s}: ${u.opName} is a spurious user of ${op.opName}`).toBe(true);
      }
    }
  });

  it('postDominates(a, b) holds exactly when a lies on every path from b to the return', () => {
    for (const [s, func] of ANALYSIS_GRAPHS.entries()) {
      const { allOps, retOp, ud } = analysisCase(func);
      const pd = PostDominanceAnalysis.compute(func, { useDef: ud });
      const succ = new Map(); for (const op of allOps) succ.set(op, []);
      for (const op of allOps) for (let i = 0; i < op.numResults; i++) for (const use of op.getResult(i).uses()) if (succ.has(use.user)) succ.get(op).push(use.user);
      succ.set(retOp, []);
      const onAll = (start) => {
        const paths = [];
        const dfs = (n, path, seen) => { if (n === retOp) { paths.push([...path, n]); return; } for (const nx of (succ.get(n) || [])) { if (seen.has(nx)) continue; seen.add(nx); dfs(nx, [...path, n], seen); seen.delete(nx); } };
        dfs(start, [], new Set([start]));
        if (!paths.length) return null;
        let inter = new Set(paths[0]); for (const p of paths.slice(1)) { const ps = new Set(p); inter = new Set([...inter].filter((x) => ps.has(x))); }
        return inter;
      };
      for (const b of allOps) {
        if (b === retOp) continue;
        const inter = onAll(b); if (!inter) continue;
        for (const a of allOps) { if (a === b || a === retOp) continue; expect(pd.postDominates(a, b), `seed ${s}: pd(${a.opName},${b.opName})`).toBe(inter.has(a)); }
      }
    }
  });

  it('every live interval spans exactly [defining index, last use index]', () => {
    for (const [s, func] of ANALYSIS_GRAPHS.entries()) {
      const { ud, tIdx } = analysisCase(func);
      const lv = LivenessAnalysis.compute(func, { useDef: ud });
      for (const op of ud.topologicalOrder) for (let i = 0; i < op.numResults; i++) {
        const v = op.getResult(i); let start = tIdx.get(op), end = start;
        for (const use of v.uses()) { const ui = tIdx.get(use.user); if (ui !== undefined && ui > end) end = ui; }
        const got = lv.intervalOf(v);
        if (got) {
          expect(got.start, `seed ${s}: live start of ${op.opName}`).toBe(start);
          expect(got.end, `seed ${s}: live end of ${op.opName}`).toBe(end);
        }
      }
    }
  });

  it('memory_effect reports every generated op as pure, since the generator emits no side-effecting ops', () => {
    for (const [s, func] of ANALYSIS_GRAPHS.entries()) {
      const { allOps, retOp } = analysisCase(func);
      const me = MemoryEffectAnalysis.compute(func);
      for (const op of allOps) if (op !== retOp) expect(me.hasSideEffect(op), `seed ${s}: ${op.opName} must be pure`).toBe(false);
    }
  });
});

describe('post-dominance on residual/skip connections (LCA over a DAG)', () => {
  it('a skip edge means the bypassed op does NOT post-dominate its source', () => {
    const t = new TensorType([4], ScalarType.F32);
    let a, mid, merge, out;
    const func = buildFunction('f', [t], [t], (b, args) => {
      a = b.relu(args[0]);
      mid = b.relu(a.getResult(0));
      merge = b.add(a.getResult(0), mid.getResult(0));
      out = b.relu(merge.getResult(0));
      b.returnOp([out.getResult(0)]);
    });

    const pd = PostDominanceAnalysis.compute(func);
    expect(pd.postDominates(merge, a)).toBe(true);
    expect(pd.postDominates(out, a)).toBe(true);
    expect(pd.postDominates(out, merge)).toBe(true);
    expect(pd.postDominates(mid, a)).toBe(false);
    expect(pd.immediatePDom(a)).toBe(merge);
  });
});

describe('post-dominance scales sub-quadratically on deep residual chains', () => {
  function residualChain(n) {
    const t = new TensorType([4], ScalarType.F32);
    return buildFunction('f', [t], [t], (b, args) => {
      let x = args[0];
      const tips = [];
      for (let i = 0; i < n; i++) {
        x = b.relu(x).getResult(0);
        if (i % 2 === 0) tips.push(x);
      }
      let s = x;
      for (const tp of tips) s = b.add(s, tp).getResult(0);
      b.returnOp([s]);
    });
  }

  it('doubling the op count does not more than double the time (rules out O(n^2))', () => {
    const { ratio, tSmall, tLarge } = scalingRatio({
      build: residualChain,
      work: (func) => PostDominanceAnalysis.compute(func),
      n: 9000,
    });
    expect(ratio, `n=9000 took ${tSmall.toFixed(1)}ms, n=18000 took ${tLarge.toFixed(1)}ms (ratio ${ratio.toFixed(2)})`)
      .toBeLessThan(QUADRATIC_RATIO);
  });
});
