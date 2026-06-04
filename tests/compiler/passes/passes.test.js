import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction, buildModule } from '../../../src/compiler/ir/graph/builder.js';
import {
  PassManager, CanonicalizePass, AlgebraicSimplificationPass,
  ConstantFoldPass, CSEPass, DCEPass
} from '../../../src/compiler/passes/index.js';
import { PatternSet, PatternApplicator, Pattern } from '../../../src/compiler/passes/rewrite/pattern.js';
import { ShapeAnalysis } from '../../../src/compiler/analysis/shape_analysis.js';
import { registry } from '../../../src/compiler/ir/graph/ops.js';

const f32 = ScalarType.F32;
const f32_4x4 = new TensorType([4, 4], f32);
const scalar = new TensorType([], f32);

describe('CanonicalizePass', () => {
  it('folds trivial reshape', () => {
    const func = buildFunction('test', [f32_4x4], [f32_4x4], (b, [x]) => { b.returnOp([b.reshape(x, [4, 4]).getResult(0)]); });
    const mod = buildModule('m', []); mod.addFunction(func);
    new PassManager().addPass(new CanonicalizePass()) || new PassManager();
    const pm = new PassManager(); pm.addPass(new CanonicalizePass()); pm.run(mod);
    assert.equal(func.findOps(o => o.opName === 'reshape').length, 0);
  });

  it('moves constants right for commutative ops', () => {
    const func = buildFunction('test', [scalar], [scalar], (b, [x]) => {
      const c = b.scalarConstant(1, f32);
      b.returnOp([b.add(c.getResult(0), x).getResult(0)]);
    });
    const mod = buildModule('m', []); mod.addFunction(func);
    const pm = new PassManager(); pm.addPass(new CanonicalizePass()); pm.run(mod);
    assert.equal(func.findOp(o => o.opName === 'add').getOperand(1).definingOp.opName, 'constant');
  });

  it('folds trivial transpose', () => {
    const func = buildFunction('test', [f32_4x4], [f32_4x4], (b, [x]) => {
      b.returnOp([b._buildOp('transpose', [x], [f32_4x4], { permutation: [0, 1] }).getResult(0)]);
    });
    const mod = buildModule('m', []); mod.addFunction(func);
    const pm = new PassManager(); pm.addPass(new CanonicalizePass()); pm.run(mod);
    assert.equal(func.findOps(o => o.opName === 'transpose').length, 0);
  });

  it('folds consecutive reshapes', () => {
    const func = buildFunction('test', [f32_4x4], [new TensorType([2, 8], f32)], (b, [x]) => {
      const r1 = b.reshape(x, [16]);
      b.returnOp([b.reshape(r1.getResult(0), [2, 8]).getResult(0)]);
    });
    const mod = buildModule('m', []); mod.addFunction(func);
    const pm = new PassManager(); pm.addPass(new CanonicalizePass()); pm.addPass(new DCEPass()); pm.run(mod);
    const reshapes = func.findOps(o => o.opName === 'reshape');
    assert.equal(reshapes.length, 1);
    assert.equal(reshapes[0].getOperand(0), func.args[0]);
  });

  it('does NOT simplify x/x', () => {
    const func = buildFunction('test', [f32_4x4], [f32_4x4], (b, [x]) => { b.returnOp([b.div(x, x).getResult(0)]); });
    const mod = buildModule('m', []); mod.addFunction(func);
    const pm = new PassManager(); pm.addPass(new CanonicalizePass()); pm.run(mod);
    assert.equal(func.findOps(o => o.opName === 'div').length, 1);
  });

  it('does not swap for non-commutative sub', () => {
    const func = buildFunction('test', [scalar], [scalar], (b, [x]) => {
      b.returnOp([b.sub(b.scalarConstant(1, f32).getResult(0), x).getResult(0)]);
    });
    const mod = buildModule('m', []); mod.addFunction(func);
    const pm = new PassManager(); pm.addPass(new CanonicalizePass()); pm.run(mod);
    assert.equal(func.findOp(o => o.opName === 'sub').getOperand(0).definingOp.opName, 'constant');
  });
});

describe('AlgebraicSimplificationPass', () => {
  for (const [name, build, check] of [
    ['double neg', (b, x) => b.neg(b.neg(x).getResult(0)), f => f.findOps(o => o.opName === 'neg').length === 0],
    ['x + 0', (b, x) => b.add(x, b.scalarConstant(0, f32).getResult(0)), f => f.findOps(o => o.opName === 'add').length === 0],
    ['x * 1', (b, x) => b.mul(x, b.scalarConstant(1, f32).getResult(0)), f => f.findOps(o => o.opName === 'mul').length === 0],
    ['x * 0', (b, x) => b.mul(x, b.scalarConstant(0, f32).getResult(0)), f => f.findOps(o => o.opName === 'mul').length === 0],
    ['x - x', (b, x) => b.sub(x, x), f => f.findOps(o => o.opName === 'sub').length === 0],
    ['exp(log(x))', (b, x) => b.exp(b.log(x).getResult(0)), f => f.findOps(o => o.opName === 'exp').length === 0],
    ['x / x', (b, x) => b.div(x, x), f => f.findOps(o => o.opName === 'div').length === 0],
    ['div(x,1)', (b, x) => b.div(x, b.scalarConstant(1, f32).getResult(0)), f => f.findOps(o => o.opName === 'div').length === 0],
    ['sub(x,0)', (b, x) => b.sub(x, b.scalarConstant(0, f32).getResult(0)), f => f.findOps(o => o.opName === 'sub').length === 0],
  ]) {
    it(`simplifies ${name}`, () => {
      const func = buildFunction('test', [scalar], [scalar], (b, [x]) => {
        b.returnOp([build(b, x).getResult(0)]);
      });
      const mod = buildModule('m', []); mod.addFunction(func);
      const pm = new PassManager(); pm.addPass(new AlgebraicSimplificationPass()); pm.addPass(new DCEPass()); pm.run(mod);
      assert.ok(check(func));
    });
  }

  it('simplifies log(exp(x))', () => {
    const func = buildFunction('test', [f32_4x4], [f32_4x4], (b, [x]) => {
      b.returnOp([b.log(b.exp(x).getResult(0)).getResult(0)]);
    });
    const mod = buildModule('m', []); mod.addFunction(func);
    const pm = new PassManager(); pm.addPass(new AlgebraicSimplificationPass()); pm.addPass(new DCEPass()); pm.run(mod);
    assert.equal(func.findOps(o => o.opName === 'log').length, 0);
  });

  it('neg(a)*neg(b) -> a*b', () => {
    const func = buildFunction('test', [scalar, scalar], [scalar], (b, [x, y]) => {
      b.returnOp([b.mul(b.neg(x).getResult(0), b.neg(y).getResult(0)).getResult(0)]);
    });
    const mod = buildModule('m', []); mod.addFunction(func);
    const pm = new PassManager(); pm.addPass(new AlgebraicSimplificationPass()); pm.addPass(new DCEPass()); pm.run(mod);
    assert.equal(func.findOps(o => o.opName === 'neg').length, 0);
    assert.equal(func.findOps(o => o.opName === 'mul').length, 1);
  });

  it('a+neg(b) -> a-b', () => {
    const func = buildFunction('test', [scalar, scalar], [scalar], (b, [x, y]) => {
      b.returnOp([b.add(x, b.neg(y).getResult(0)).getResult(0)]);
    });
    const mod = buildModule('m', []); mod.addFunction(func);
    const pm = new PassManager(); pm.addPass(new AlgebraicSimplificationPass()); pm.addPass(new DCEPass()); pm.run(mod);
    assert.equal(func.findOps(o => o.opName === 'add').length, 0);
    assert.equal(func.findOps(o => o.opName === 'sub').length, 1);
  });

  it('a-neg(b) -> a+b', () => {
    const func = buildFunction('test', [scalar, scalar], [scalar], (b, [x, y]) => {
      b.returnOp([b.sub(x, b.neg(y).getResult(0)).getResult(0)]);
    });
    const mod = buildModule('m', []); mod.addFunction(func);
    const pm = new PassManager(); pm.addPass(new AlgebraicSimplificationPass()); pm.addPass(new DCEPass()); pm.run(mod);
    assert.equal(func.findOps(o => o.opName === 'sub').length, 0);
    assert.equal(func.findOps(o => o.opName === 'add').length, 1);
  });

  it('transpose(transpose(x)) folds', () => {
    const func = buildFunction('test', [f32_4x4], [f32_4x4], (b, [x]) => {
      b.returnOp([b.transpose(b.transpose(x, [1, 0]).getResult(0), [1, 0]).getResult(0)]);
    });
    const mod = buildModule('m', []); mod.addFunction(func);
    const pm = new PassManager(); pm.addPass(new AlgebraicSimplificationPass()); pm.addPass(new CanonicalizePass()); pm.addPass(new DCEPass()); pm.run(mod);
    assert.equal(func.findOps(o => o.opName === 'transpose').length, 0);
  });

  it('preserves ShapeAnalysis', () => {
    assert.ok(new AlgebraicSimplificationPass().preservedAnalyses.has(ShapeAnalysis));
  });
});

describe('ConstantFoldPass', () => {
  it('folds scalar add', () => {
    const func = buildFunction('test', [], [scalar], (b) => {
      b.returnOp([b.add(b.scalarConstant(2, f32).getResult(0), b.scalarConstant(3, f32).getResult(0)).getResult(0)]);
    });
    const mod = buildModule('m', []); mod.addFunction(func);
    const pm = new PassManager(); pm.addPass(new ConstantFoldPass()); pm.run(mod);
    assert.equal(func.findOps(o => o.opName === 'add').length, 0);
  });

  it('folds reshape of constant', () => {
    const func = buildFunction('test', [], [new TensorType([2, 2], f32)], (b) => {
      const c = b._buildOp('constant', [], [new TensorType([4], f32)], { value: [1, 2, 3, 4], tensor_type: new TensorType([4], f32) });
      b.returnOp([b.reshape(c.getResult(0), [2, 2]).getResult(0)]);
    });
    const mod = buildModule('m', []); mod.addFunction(func);
    const pm = new PassManager(); pm.addPass(new ConstantFoldPass()); pm.run(mod);
    assert.equal(func.findOps(o => o.opName === 'reshape').length, 0);
  });

  it('folds chain of constant arithmetic', () => {
    const func = buildFunction('test', [], [scalar], (b) => {
      const a = b.add(b.scalarConstant(2, f32).getResult(0), b.scalarConstant(3, f32).getResult(0));
      b.returnOp([b.mul(a.getResult(0), b.scalarConstant(4, f32).getResult(0)).getResult(0)]);
    });
    const mod = buildModule('m', []); mod.addFunction(func);
    const pm = new PassManager(); pm.addPass(new ConstantFoldPass()); pm.addPass(new ConstantFoldPass()); pm.addPass(new DCEPass()); pm.run(mod);
    assert.equal(func.findOps(o => o.opName === 'add').length, 0);
    assert.equal(func.findOps(o => o.opName === 'mul').length, 0);
  });

  it('does not fold non-constant inputs', () => {
    const func = buildFunction('test', [scalar], [scalar], (b, [x]) => {
      b.returnOp([b.add(x, b.scalarConstant(2, f32).getResult(0)).getResult(0)]);
    });
    const mod = buildModule('m', []); mod.addFunction(func);
    const pm = new PassManager(); pm.addPass(new ConstantFoldPass()); pm.run(mod);
    assert.equal(func.findOps(o => o.opName === 'add').length, 1);
  });

  it('passes constOps to fold', () => {
    let received = null;
    const origFold = registry.get('add').fold;
    registry.get('add').fold = (v, a, ops) => { received = ops; return origFold(v, a); };
    try {
      const func = buildFunction('test', [], [scalar], (b) => {
        b.returnOp([b.add(b.scalarConstant(2, f32).getResult(0), b.scalarConstant(3, f32).getResult(0)).getResult(0)]);
      });
      const mod = buildModule('m', []); mod.addFunction(func);
      const pm = new PassManager(); pm.addPass(new ConstantFoldPass()); pm.run(mod);
      assert.equal(received.length, 2);
      assert.equal(received[0].opName, 'constant');
    } finally { registry.get('add').fold = origFold; }
  });
});

describe('CSEPass', () => {
  it('eliminates common subexpressions', () => {
    const func = buildFunction('test', [f32_4x4], [f32_4x4], (b, [x]) => {
      b.returnOp([b.add(b.mul(x, x).getResult(0), b.mul(x, x).getResult(0)).getResult(0)]);
    });
    const mod = buildModule('m', []); mod.addFunction(func);
    const pm = new PassManager(); pm.addPass(new CSEPass()); pm.run(mod);
    assert.equal(func.findOps(o => o.opName === 'mul').length, 1);
  });

  it('does not eliminate side-effecting ops', () => {
    const func = buildFunction('test', [f32_4x4], [f32_4x4], (b, [x]) => {
      const c1 = b.customCall('p', [x], [f32_4x4]);
      const c2 = b.customCall('p', [x], [f32_4x4]);
      b.returnOp([b.add(c1.getResult(0), c2.getResult(0)).getResult(0)]);
    });
    const mod = buildModule('m', []); mod.addFunction(func);
    const pm = new PassManager(); pm.addPass(new CSEPass()); pm.run(mod);
    assert.equal(func.findOps(o => o.opName === 'custom_call').length, 2);
  });

  it('eliminates redundant constants', () => {
    const func = buildFunction('test', [scalar], [scalar], (b, [x]) => {
      b.returnOp([b.add(b.scalarConstant(42, f32).getResult(0), b.scalarConstant(42, f32).getResult(0)).getResult(0)]);
    });
    const mod = buildModule('m', []); mod.addFunction(func);
    const pm = new PassManager(); pm.addPass(new CSEPass()); pm.run(mod);
    assert.equal(func.findOps(o => o.opName === 'constant').length, 1);
  });

  it('preserves ShapeAnalysis', () => {
    assert.ok(new CSEPass().preservedAnalyses.has(ShapeAnalysis));
  });
});

describe('DCEPass', () => {
  it('removes dead code', () => {
    const func = buildFunction('test', [f32_4x4], [f32_4x4], (b, [x]) => { b.mul(x, x); b.returnOp([x]); });
    const mod = buildModule('m', []); mod.addFunction(func);
    const pm = new PassManager(); pm.addPass(new DCEPass()); pm.run(mod);
    assert.equal(func.findOps(o => o.opName === 'mul').length, 0);
  });

  it('removes long dead chain', () => {
    const func = buildFunction('test', [f32_4x4], [f32_4x4], (b, [x]) => {
      let v = x; for (let i = 0; i < 10; i++) v = b.exp(v).getResult(0);
      b.returnOp([x]);
    });
    const mod = buildModule('m', []); mod.addFunction(func);
    const pm = new PassManager(); pm.addPass(new DCEPass()); pm.run(mod);
    assert.equal(func.findOps(o => o.opName === 'exp').length, 0);
  });

  it('preserves side-effecting ops', () => {
    const func = buildFunction('test', [f32_4x4], [f32_4x4], (b, [x]) => { b.customCall('p', [x], [f32_4x4]); b.returnOp([x]); });
    const dce = new DCEPass(); dce.run(func, null);
    assert.equal(func.findOps(o => o.opName === 'custom_call').length, 1);
  });

  it('works without analysisManager', () => {
    const func = buildFunction('test', [f32_4x4], [f32_4x4], (b, [x]) => { b.mul(x, x); b.returnOp([x]); });
    new DCEPass().run(func, null);
    assert.equal(func.findOps(o => o.opName === 'mul').length, 0);
  });

  it('preserves ShapeAnalysis', () => {
    assert.ok(new DCEPass().preservedAnalyses.has(ShapeAnalysis));
  });
});

describe('PatternSet', () => {
  it('getForOp dispatches by rootOpName', () => {
    const ps = new PatternSet();
    class AP extends Pattern { constructor() { super('a', 5); this.rootOpName = 'add'; } }
    class MP extends Pattern { constructor() { super('m', 3); this.rootOpName = 'mul'; } }
    class GP extends Pattern { constructor() { super('g', 1); } }
    ps.add(new AP()); ps.add(new MP()); ps.add(new GP());
    const forAdd = ps.getForOp('add');
    assert.ok(forAdd.some(p => p.name === 'a'));
    assert.ok(forAdd.some(p => p.name === 'g'));
    assert.ok(!forAdd.some(p => p.name === 'm'));
    assert.equal(ps.getForOp('sub').length, 1);
  });

  it('sorted by benefit', () => {
    const ps = new PatternSet();
    class P1 extends Pattern { constructor() { super('p1', 1); this.rootOpName = 'add'; } }
    class P2 extends Pattern { constructor() { super('p2', 10); this.rootOpName = 'add'; } }
    ps.add(new P1()); ps.add(new P2());
    assert.equal(ps.getForOp('add')[0].name, 'p2');
  });

  it('rootOpName patterns only called for matching ops', () => {
    let addCalls = 0, mulCalls = 0;
    class CA extends Pattern { constructor() { super('ca', 1); this.rootOpName = 'add'; } match() { addCalls++; return false; } }
    class CM extends Pattern { constructor() { super('cm', 1); this.rootOpName = 'mul'; } match() { mulCalls++; return false; } }
    const ps = new PatternSet(); ps.add(new CA()); ps.add(new CM());
    const func = buildFunction('test', [scalar, scalar], [scalar], (b, [x, y]) => { b.returnOp([b.add(x, y).getResult(0)]); });
    new PatternApplicator(ps).applyPatterns(func, 1);
    assert.ok(addCalls > 0);
    assert.equal(mulCalls, 0);
  });
});

describe('Combined pipeline', () => {
  it('cleans up complex graph to identity', () => {
    const func = buildFunction('test', [scalar], [scalar], (b, [x]) => {
      const z = b.scalarConstant(0, f32); const one = b.scalarConstant(1, f32);
      const a = b.add(x, z.getResult(0)); const m = b.mul(a.getResult(0), one.getResult(0));
      b.returnOp([b.neg(b.neg(m.getResult(0)).getResult(0)).getResult(0)]);
    });
    const mod = buildModule('m', []); mod.addFunction(func);
    const pm = new PassManager();
    pm.addPass(new CanonicalizePass()); pm.addPass(new AlgebraicSimplificationPass());
    pm.addPass(new ConstantFoldPass()); pm.addPass(new CSEPass()); pm.addPass(new DCEPass());
    pm.run(mod);
    assert.equal(func.getReturnOp().getOperand(0), func.args[0]);
  });

  it('exp(log(x)) + 0 * 1 = x', () => {
    const func = buildFunction('test', [scalar], [scalar], (b, [x]) => {
      const e = b.exp(b.log(x).getResult(0));
      const a = b.add(e.getResult(0), b.scalarConstant(0, f32).getResult(0));
      b.returnOp([b.mul(a.getResult(0), b.scalarConstant(1, f32).getResult(0)).getResult(0)]);
    });
    const mod = buildModule('m', []); mod.addFunction(func);
    const pm = new PassManager();
    pm.addPass(new AlgebraicSimplificationPass()); pm.addPass(new CanonicalizePass()); pm.addPass(new DCEPass());
    pm.run(mod);
    assert.equal(func.getReturnOp().getOperand(0), func.args[0]);
  });
});
