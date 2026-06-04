import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { TensorType, ScalarType, DYNAMIC } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction, buildModule } from '../../../src/compiler/ir/graph/builder.js';
import {
  AnalysisManager, UseDefAnalysis, UseDefResult,
  ShapeAnalysis, ShapeResult,
  DtypeAnalysis, DtypeResult,
  LivenessAnalysis, LivenessResult,
  AliasAnalysis, AliasAnalysisResult,
  MemoryEffectAnalysis, MemoryEffectResult
} from '../../../src/compiler/analysis/index.js';
import { SymInt } from '../../../src/compiler/analysis/sym_int.js';
import { SideEffectKind } from '../../../src/compiler/ir/graph/op_registry.js';

const f32 = ScalarType.F32;
const i32 = ScalarType.I32;
const f32_2x3 = new TensorType([2, 3], f32);
const f32_2x4 = new TensorType([2, 4], f32);
const f32_4x4 = new TensorType([4, 4], f32);

function makeTestFunc() {
  return buildFunction('test', [f32_4x4, f32_4x4], [f32_4x4], (b, [x, y]) => {
    const a = b.add(x, y);
    const m = b.mul(a.getResult(0), x);
    b.returnOp([m.getResult(0)]);
  });
}

describe('AnalysisManager', () => {
  it('caches and invalidates', () => {
    const func = buildFunction('test_func', [f32_2x3], [f32_2x3], (b, [x]) => {
      const a = b.add(x, x);
      b.returnOp([a.getResult(0)]);
    });
    const manager = new AnalysisManager();
    const useDef1 = manager.getAnalysis(UseDefAnalysis, func);
    const useDef2 = manager.getAnalysis(UseDefAnalysis, func);
    assert.equal(useDef1, useDef2);
    manager.invalidate(func, new Set(['shape']));
    const useDef3 = manager.getAnalysis(UseDefAnalysis, func);
    assert.notEqual(useDef1, useDef3);
    func.bumpVersion();
    const useDef4 = manager.getAnalysis(UseDefAnalysis, func);
    assert.notEqual(useDef3, useDef4);
  });

  it('maintains independent caches for different functions', () => {
    const func1 = buildFunction('f1', [], [], b => b.returnOp([]));
    const func2 = buildFunction('f2', [], [], b => b.returnOp([]));
    const manager = new AnalysisManager();
    assert.notEqual(manager.getAnalysis(UseDefAnalysis, func1), manager.getAnalysis(UseDefAnalysis, func2));
  });

  it('preserves specific analyses during invalidation', () => {
    const func = makeTestFunc();
    const manager = new AnalysisManager();
    const useDef1 = manager.getAnalysis(UseDefAnalysis, func);
    manager.getAnalysis(ShapeAnalysis, func);
    manager.invalidate(func, new Set([UseDefAnalysis]));
    assert.equal(useDef1, manager.getAnalysis(UseDefAnalysis, func));
  });

  it('invalidateAll clears everything', () => {
    const func = makeTestFunc();
    const manager = new AnalysisManager();
    const ud1 = manager.getAnalysis(UseDefAnalysis, func);
    manager.invalidateAll();
    assert.notEqual(ud1, manager.getAnalysis(UseDefAnalysis, func));
  });

  it('resolves dependencies automatically', () => {
    const func = makeTestFunc();
    const manager = new AnalysisManager();
    assert.ok(manager.getAnalysis(LivenessAnalysis, func) instanceof LivenessResult);
    assert.ok(manager.getAnalysis(UseDefAnalysis, func) instanceof UseDefResult);
  });

  it('caches resolved dependencies', () => {
    const func = makeTestFunc();
    const manager = new AnalysisManager();
    const l1 = manager.getAnalysis(LivenessAnalysis, func);
    assert.equal(l1, manager.getAnalysis(LivenessAnalysis, func));
  });

  it('cascade invalidates dependents', () => {
    const func = makeTestFunc();
    const manager = new AnalysisManager();
    const liveness1 = manager.getAnalysis(LivenessAnalysis, func);
    manager.getAnalysis(UseDefAnalysis, func);
    manager.invalidate(func, new Set([ShapeAnalysis]));
    assert.notEqual(liveness1, manager.getAnalysis(LivenessAnalysis, func));
  });

  it('cascade invalidates alias when useDef removed', () => {
    const func = makeTestFunc();
    const manager = new AnalysisManager();
    const alias1 = manager.getAnalysis(AliasAnalysis, func);
    manager.invalidate(func, new Set([ShapeAnalysis]));
    assert.notEqual(alias1, manager.getAnalysis(AliasAnalysis, func));
  });

  it('preserving all deps keeps dependents cached', () => {
    const func = makeTestFunc();
    const manager = new AnalysisManager();
    const l1 = manager.getAnalysis(LivenessAnalysis, func);
    const a1 = manager.getAnalysis(AliasAnalysis, func);
    manager.invalidate(func, new Set([UseDefAnalysis, LivenessAnalysis, AliasAnalysis]));
    assert.equal(l1, manager.getAnalysis(LivenessAnalysis, func));
    assert.equal(a1, manager.getAnalysis(AliasAnalysis, func));
  });
});

describe('UseDefAnalysis', () => {
  it('computes topological order and users', () => {
    const func = buildFunction('test', [f32_2x3, f32_2x3], [f32_2x3], (b, [x, y]) => {
      const a = b.add(x, y);
      const m = b.mul(a.getResult(0), x);
      b.returnOp([m.getResult(0)]);
    });
    const useDef = UseDefAnalysis.compute(func);
    assert.ok(useDef instanceof UseDefResult);
    assert.equal(useDef.topologicalOrder.length, 3);
    assert.equal(useDef.topologicalOrder[0].opName, 'add');
    assert.ok(useDef.opUsers.get(useDef.topologicalOrder[0]).has(useDef.topologicalOrder[1]));
  });

  it('handles isolated ops with zero users', () => {
    const func = buildFunction('test', [f32_2x3], [f32_2x3], (b, [x]) => {
      b.add(x, x);
      b.returnOp([x]);
    });
    const useDef = UseDefAnalysis.compute(func);
    assert.equal(useDef.opUsers.get(func.findOp(op => op.opName === 'add')).size, 0);
  });

  it('computes depth and height', () => {
    const func = buildFunction('test', [f32_2x3], [f32_2x3], (b, [x]) => {
      const a = b.exp(x);
      const m = b.mul(a.getResult(0), x);
      b.returnOp([m.getResult(0)]);
    });
    const useDef = UseDefAnalysis.compute(func);
    assert.equal(useDef.depth.get(func.findOp(o => o.opName === 'exp')), 0);
    assert.equal(useDef.depth.get(func.findOp(o => o.opName === 'mul')), 1);
    assert.equal(useDef.height.get(func.findOp(o => o.opName === 'exp')), 2);
  });

  it('long chain depth/height', () => {
    const func = buildFunction('test', [f32_4x4], [f32_4x4], (b, [x]) => {
      let v = x;
      for (let i = 0; i < 5; i++) v = b.exp(v).getResult(0);
      b.returnOp([v]);
    });
    const useDef = UseDefAnalysis.compute(func);
    assert.equal(useDef.depth.get(useDef.topologicalOrder[0]), 0);
    assert.equal(useDef.depth.get(useDef.topologicalOrder[4]), 4);
    assert.equal(useDef.height.get(useDef.topologicalOrder[0]), 5);
  });

  it('topological order respects all dependencies', () => {
    const func = buildFunction('test', [f32_4x4, f32_4x4], [f32_4x4], (b, [x, y]) => {
      const a = b.add(x, y);
      const m = b.mul(x, a.getResult(0));
      const s = b.sub(a.getResult(0), m.getResult(0));
      b.returnOp([s.getResult(0)]);
    });
    const useDef = UseDefAnalysis.compute(func);
    const idx = (op) => useDef.topologicalOrder.indexOf(op);
    assert.ok(idx(func.findOp(o => o.opName === 'add')) < idx(func.findOp(o => o.opName === 'mul')));
    assert.ok(idx(func.findOp(o => o.opName === 'mul')) < idx(func.findOp(o => o.opName === 'sub')));
  });
});

describe('LivenessAnalysis', () => {
  it('computes intervals', () => {
    const func = buildFunction('test', [f32_2x3], [f32_2x3], (b, [x]) => {
      const a = b.add(x, x);
      const m = b.mul(a.getResult(0), x);
      b.returnOp([m.getResult(0)]);
    });
    const liveness = LivenessAnalysis.compute(func);
    assert.ok(liveness instanceof LivenessResult);
    const intv = liveness.intervals.get(func.findOp(o => o.opName === 'add').getResult(0));
    assert.equal(intv.start, 0);
    assert.equal(intv.end, 1);
  });

  it('extends intervals for multiple users', () => {
    const func = buildFunction('test', [f32_2x3], [f32_2x3], (b, [x]) => {
      const a = b.add(x, x);
      b.mul(a.getResult(0), x);
      b.mul(a.getResult(0), x);
      const ret = b.add(b.mul(a.getResult(0), x).getResult(0), x);
      b.returnOp([ret.getResult(0)]);
    });
    const liveness = LivenessAnalysis.compute(func);
    const aRes = func.findOp(o => o.opName === 'add' && o.getOperand(0) === func.args[0]).getResult(0);
    assert.ok(liveness.intervals.get(aRes).end >= 2);
  });

  it('computes interferences', () => {
    const func = buildFunction('test', [f32_2x3], [f32_2x3], (b, [x]) => {
      const a = b.add(x, x);
      const m = b.mul(x, x);
      const c = b.add(a.getResult(0), m.getResult(0));
      b.returnOp([c.getResult(0)]);
    });
    const liveness = LivenessAnalysis.compute(func);
    const aRes = func.findOp(o => o.opName === 'add' && o.getOperand(0) === func.args[0]).getResult(0);
    const mRes = func.findOp(o => o.opName === 'mul').getResult(0);
    assert.ok(liveness.interfere(aRes, mRes));
  });

  it('peak memory pressure', () => {
    const func = makeTestFunc();
    const liveness = LivenessAnalysis.compute(func);
    assert.ok(liveness.peakPressure > 0);
    assert.ok(liveness.peakOp !== null);
  });

  it('liveAtOp and intervalOf', () => {
    const func = makeTestFunc();
    const liveness = LivenessAnalysis.compute(func);
    const mulOp = func.findOp(o => o.opName === 'mul');
    assert.ok(liveness.liveAtOp(mulOp).has(func.args[0]));
    assert.equal(liveness.intervalOf({ type: f32_4x4 }), null);
  });

  it('diamond dependency', () => {
    const func = buildFunction('test', [f32_4x4], [f32_4x4], (b, [x]) => {
      const a = b.add(x, x);
      const b1 = b.exp(a.getResult(0));
      const b2 = b.log(a.getResult(0));
      const c = b.add(b1.getResult(0), b2.getResult(0));
      b.returnOp([c.getResult(0)]);
    });
    const liveness = LivenessAnalysis.compute(func);
    const addRes = func.findOp(o => o.opName === 'add' && o.getOperand(0) === func.args[0]).getResult(0);
    assert.ok(liveness.intervalOf(addRes).end >= liveness.intervalOf(addRes).start + 1);
  });

  it('accepts deps parameter', () => {
    const func = makeTestFunc();
    const useDef = UseDefAnalysis.compute(func);
    const liveness = LivenessAnalysis.compute(func, { useDef });
    assert.ok(liveness instanceof LivenessResult);
  });
});

describe('AliasAnalysis', () => {
  it('tracks reshape aliases', () => {
    const func = buildFunction('test', [f32_2x3], [new TensorType([6], f32)], (b, [x]) => {
      const r = b.reshape(x, [6]);
      b.returnOp([r.getResult(0)]);
    });
    const alias = AliasAnalysis.compute(func);
    assert.ok(alias instanceof AliasAnalysisResult);
    assert.ok(alias.mayAlias(func.args[0], func.findOp(o => o.opName === 'reshape').getResult(0)));
  });

  it('returns NoAlias for independent tensors', () => {
    const func = buildFunction('test', [f32_2x3, f32_2x3], [f32_2x3], (b, [x, y]) => {
      const add = b.add(x, y);
      b.returnOp([add.getResult(0)]);
    });
    const alias = AliasAnalysis.compute(func);
    assert.equal(alias.mayAlias(func.args[0], func.args[1]), false);
    assert.equal(alias.mayAlias(func.args[0], func.findOp(o => o.opName === 'add').getResult(0)), false);
  });

  it('deep propagation through multiple views', () => {
    const func = buildFunction('test', [f32_2x3], [f32_2x3], (b, [x]) => {
      const r1 = b.reshape(x, [6]);
      const r2 = b.reshape(r1.getResult(0), [3, 2]);
      const t = b._buildOp('transpose', [r2.getResult(0)], [f32_2x3], { permutation: [1, 0] });
      b.returnOp([t.getResult(0)]);
    });
    assert.ok(AliasAnalysis.compute(func).mayAlias(func.args[0], func.findOp(o => o.opName === 'transpose').getResult(0)));
  });

  it('transpose as view alias', () => {
    const func = buildFunction('test', [f32_4x4], [f32_4x4], (b, [x]) => {
      const t = b.transpose(x, [1, 0]);
      b.returnOp([t.getResult(0)]);
    });
    assert.ok(AliasAnalysis.compute(func).mayAlias(func.args[0], func.findOp(o => o.opName === 'transpose').getResult(0)));
  });

  it('slice aliases input', () => {
    const func = buildFunction('test', [new TensorType([8], f32)], [new TensorType([4], f32)], (b, [x]) => {
      const s = b.slice(x, [0], [4]);
      b.returnOp([s.getResult(0)]);
    });
    assert.ok(AliasAnalysis.compute(func).mayAlias(func.args[0], func.findOp(o => o.opName === 'slice').getResult(0)));
  });

  it('arithmetic does not alias', () => {
    const func = makeTestFunc();
    const alias = AliasAnalysis.compute(func);
    assert.ok(!alias.mayAlias(func.args[0], func.findOp(o => o.opName === 'add').getResult(0)));
    assert.ok(!alias.mayAlias(func.findOp(o => o.opName === 'add').getResult(0), func.findOp(o => o.opName === 'mul').getResult(0)));
  });

  it('accepts deps parameter', () => {
    const func = makeTestFunc();
    const useDef = UseDefAnalysis.compute(func);
    assert.ok(AliasAnalysis.compute(func, { useDef }) instanceof AliasAnalysisResult);
  });
});

describe('ShapeAnalysis', () => {
  it('infers broadcast shapes', () => {
    const func = buildFunction('test', [f32_2x3, new TensorType([3], f32)], [f32_2x3], (b, [x, y]) => {
      const bc = b.broadcast(y, [2, 3], [1]);
      const add = b.add(x, bc.getResult(0));
      b.returnOp([add.getResult(0)]);
    });
    const info = ShapeAnalysis.compute(func);
    assert.ok(info instanceof ShapeResult);
    assert.deepEqual(info.shapes.get(func.findOp(o => o.opName === 'broadcast_in_dim').getResult(0)), [2, 3]);
  });

  it('scalar tensors', () => {
    const scalarType = new TensorType([], f32);
    const func = buildFunction('test', [scalarType], [scalarType], (b, [x]) => {
      const exp = b.exp(x);
      b.returnOp([exp.getResult(0)]);
    });
    assert.deepEqual(ShapeAnalysis.compute(func).shapeOf(func.findOp(o => o.opName === 'exp').getResult(0)), []);
  });

  it('infers reduce output shape', () => {
    const func = buildFunction('test', [new TensorType([4, 8], f32)], [new TensorType([4], f32)], (b, [x]) => {
      const red = b.reduce(x, b.scalarConstant(0, f32).getResult(0), [1], 'sum');
      b.returnOp([red.getResult(0)]);
    });
    assert.deepEqual(ShapeAnalysis.compute(func).shapes.get(func.findOp(o => o.opName === 'reduce').getResult(0)), [4]);
  });

  it('infers dot output shape', () => {
    const func = buildFunction('test', [new TensorType([4, 8], f32), new TensorType([8, 3], f32)], [new TensorType([4, 3], f32)], (b, [x, y]) => {
      const d = b.matmul(x, y);
      b.returnOp([d.getResult(0)]);
    });
    assert.deepEqual(ShapeAnalysis.compute(func).shapes.get(func.findOp(o => o.opName === 'dot').getResult(0)), [4, 3]);
  });

  it('dynamic shapes with symbolic dims', () => {
    const dynType = new TensorType([DYNAMIC, 64], f32);
    const func = buildFunction('test', [dynType], [dynType], (b, [x]) => {
      const exp = b.exp(x);
      b.returnOp([exp.getResult(0)]);
    });
    const shape = ShapeAnalysis.compute(func).shapes.get(func.findOp(o => o.opName === 'exp').getResult(0));
    assert.equal(shape[1], 64);
    assert.ok(shape[0] instanceof SymInt);
  });
});

describe('DtypeAnalysis', () => {
  it('detects dtype inconsistencies', () => {
    const func = buildFunction('test', [f32_2x3, new TensorType([2, 3], i32)], [f32_2x3], (b, [x, y]) => {
      b._buildOp('add', [x, y], [f32_2x3]);
      b.returnOp([x]);
    });
    const info = DtypeAnalysis.compute(func);
    assert.ok(info instanceof DtypeResult);
    assert.equal(info.isValid, false);
    assert.ok(info.inconsistencies[0].includes('dtype mismatch'));
  });

  it('validates same dtype', () => {
    const func = buildFunction('test', [f32_2x3, f32_2x3], [f32_2x3], (b, [x, y]) => {
      const op = b.add(x, y);
      b.returnOp([op.getResult(0)]);
    });
    assert.equal(DtypeAnalysis.compute(func).isValid, true);
  });

  it('tracks dtypes through unary ops', () => {
    const func = buildFunction('test', [f32_4x4], [f32_4x4], (b, [x]) => {
      const e = b.exp(x);
      const l = b.log(e.getResult(0));
      b.returnOp([l.getResult(0)]);
    });
    assert.equal(DtypeAnalysis.compute(func).dtypeOf(func.findOp(o => o.opName === 'log').getResult(0)), f32);
  });

  it('validates float-only ops with int input', () => {
    const func = buildFunction('test', [new TensorType([4, 4], i32)], [new TensorType([4, 4], i32)], (b, [x]) => {
      b._buildOp('exp', [x], [new TensorType([4, 4], i32)]);
      b.returnOp([x]);
    });
    const info = DtypeAnalysis.compute(func);
    assert.equal(info.isValid, false);
    assert.ok(info.inconsistencies.some(e => e.includes('float')));
  });
});

describe('MemoryEffectAnalysis', () => {
  it('detects read and write side effects', () => {
    const func = buildFunction('test', [f32_2x3], [f32_2x3], (b, [x]) => {
      b.customCall('my_print', [x], [f32_2x3]);
      b.returnOp([x]);
    });
    const info = MemoryEffectAnalysis.compute(func);
    assert.ok(info instanceof MemoryEffectResult);
    const callOp = func.findOp(o => o.opName === 'custom_call');
    assert.ok(info.hasSideEffect(callOp));
    const effects = info.opEffects.get(callOp);
    assert.ok(effects.some(e => e.kind === SideEffectKind.READ));
    assert.ok(effects.some(e => e.kind === SideEffectKind.WRITE));
  });

  it('fallback memory effects for scatter', () => {
    const func = buildFunction('test', [f32_2x3, f32_2x3], [f32_2x3], (b, [base, updates]) => {
      const idx = b.scalarConstant(0, i32);
      b._buildOp('scatter', [base, idx.getResult(0), updates], [f32_2x3]);
      b.returnOp([base]);
    });
    const info = MemoryEffectAnalysis.compute(func);
    assert.ok(info.hasSideEffect(func.findOp(o => o.opName === 'scatter')));
  });

  it('getEffectsOn indexed O(1)', () => {
    const func = buildFunction('test', [f32_4x4], [f32_4x4], (b, [x]) => {
      b.customCall('f', [x], [f32_4x4]);
      b.returnOp([x]);
    });
    const info = MemoryEffectAnalysis.compute(func);
    assert.ok(info.getEffectsOn(func.args[0]).length > 0);
  });

  it('getReadersOf and getWritersOf', () => {
    const func = buildFunction('test', [f32_4x4], [f32_4x4], (b, [x]) => {
      const call = b.customCall('f', [x], [f32_4x4]);
      b.returnOp([call.getResult(0)]);
    });
    const info = MemoryEffectAnalysis.compute(func);
    const callOp = func.findOp(o => o.opName === 'custom_call');
    assert.ok(info.getReadersOf(func.args[0]).includes(callOp));
    assert.ok(info.getWritersOf(callOp.getResult(0)).includes(callOp));
  });

  it('pure values have no effects', () => {
    const func = makeTestFunc();
    const info = MemoryEffectAnalysis.compute(func);
    assert.equal(info.getEffectsOn(func.findOp(o => o.opName === 'add').getResult(0)).length, 0);
  });
});

describe('SymInt', () => {
  it('symbolic dimension propagation', () => {
    const dynType = new TensorType([DYNAMIC, 64], f32);
    const func = buildFunction('test', [dynType], [new TensorType([64, DYNAMIC], f32)], (b, [x]) => {
      const t = b._buildOp('transpose', [x], [new TensorType([64, DYNAMIC], f32)], { permutation: [1, 0] });
      const r = b.reshape(t.getResult(0), [64, DYNAMIC]);
      b.returnOp([r.getResult(0)]);
    });
    const info = ShapeAnalysis.compute(func);
    const tShape = info.shapes.get(func.findOp(o => o.opName === 'transpose').getResult(0));
    assert.equal(tShape[0], 64);
    assert.ok(tShape[1].name.startsWith('d'));
  });

  it('sub simplifications', () => {
    const d = SymInt.var('d');
    assert.equal(SymInt.sub(5, 3), 2);
    assert.equal(SymInt.sub(d, 0), d);
    assert.equal(SymInt.sub(d, d), 0);
  });

  it('neg simplifications', () => {
    const d = SymInt.var('d');
    assert.equal(SymInt.neg(5), -5);
    assert.equal(SymInt.neg(SymInt.neg(d)), d);
  });

  it('mod/ceilDiv/div self', () => {
    const d = SymInt.var('d');
    assert.equal(SymInt.mod(10, 3), 1);
    assert.equal(SymInt.mod(d, d), 0);
    assert.equal(SymInt.ceilDiv(10, 3), 4);
    assert.equal(SymInt.div(d, d), 1);
    assert.equal(SymInt.add(d, d).type, 'mul');
  });

  it('substitute and evaluate', () => {
    const d = SymInt.var('d');
    assert.equal(SymInt.substitute(SymInt.add(d, 3), 'd', 10), 13);
    assert.equal(SymInt.evaluate(SymInt.add(SymInt.mul(d, 2), SymInt.var('e')), new Map([['d', 3], ['e', 7]])), 13);
    assert.throws(() => SymInt.evaluate(d, new Map()), /Unbound/);
  });

  it('freeVars/isConst/toConst/toString', () => {
    const d = SymInt.var('d');
    assert.equal(SymInt.freeVars(SymInt.add(d, SymInt.var('e'))).size, 2);
    assert.ok(SymInt.isConst(42));
    assert.ok(!SymInt.isConst(d));
    assert.equal(SymInt.toConst(42), 42);
    assert.equal(d.toString(), 'd');
    assert.equal(SymInt.mod(d, 3).toString(), '(d % 3)');
  });

  it('all analyses declare dependencies', () => {
    for (const A of [UseDefAnalysis, LivenessAnalysis, AliasAnalysis, ShapeAnalysis, DtypeAnalysis, MemoryEffectAnalysis]) {
      assert.ok(Array.isArray(A.dependencies));
    }
    assert.ok(LivenessAnalysis.dependencies.includes(UseDefAnalysis));
    assert.ok(AliasAnalysis.dependencies.includes(UseDefAnalysis));
  });
});
