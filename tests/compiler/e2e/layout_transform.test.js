import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';

import { TensorType, ScalarType, Layout } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { BlockedLayout } from '../../../src/compiler/ir/graph/blocked_layout.js';
import { LayoutTransformPass } from '../../../src/compiler/passes/layout/layout_transform.js';
import { LayoutPolicy } from '../../../src/compiler/passes/layout/layout_policy.js';
import { PassManager } from '../../../src/compiler/passes/pass_manager.js';
import { GraphModule } from '../../../src/compiler/ir/graph/module.js';
import { CPUTarget, GPUTarget } from '../../../src/compiler/backend/target.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';

const f32 = ScalarType.F32;

function hasOpNamed(func, name) {
  for (const op of func.ops()) if (op.opName === name) return true;
  return false;
}

function runLayoutPass(func, target) {
  const mod = new GraphModule('test');
  mod.addFunction(func);
  const pm = new PassManager();
  pm.addPass(new LayoutTransformPass({ target }));
  return pm.run(mod);
}

describe('BlockedLayout', () => {
  it('computes physical shape for NCHWc', () => {
    const bl = new BlockedLayout([0, 1, 2, 3], new Map([[1, 8]]));
    const phys = bl.getPhysicalShape([2, 64, 28, 28]);
    assert.deepStrictEqual(phys, [2, 8, 28, 28, 8]);
  });

  it('reports correct physical rank', () => {
    const bl = new BlockedLayout([0, 1, 2, 3], new Map([[1, 8]]));
    assert.equal(bl.physicalRank, 5);
    assert.equal(bl.logicalRank, 4);
  });

  it('isIdentity for trivial layout', () => {
    const bl = new BlockedLayout([0, 1, 2]);
    assert.ok(bl.isIdentity());
    assert.ok(!bl.isBlocked());
  });

  it('isBlocked for blocked layout', () => {
    const bl = new BlockedLayout([0, 1], new Map([[0, 4]]));
    assert.ok(bl.isBlocked());
    assert.ok(!bl.isIdentity());
  });

  it('equals works correctly', () => {
    const a = new BlockedLayout([0, 1, 2], new Map([[1, 8]]));
    const b = new BlockedLayout([0, 1, 2], new Map([[1, 8]]));
    const c = new BlockedLayout([0, 1, 2], new Map([[1, 4]]));
    assert.ok(a.equals(b));
    assert.ok(!a.equals(c));
  });

  it('fromLayout wraps plain Layout', () => {
    const layout = Layout.rowMajor(3);
    const bl = BlockedLayout.fromLayout(layout);
    assert.ok(bl.isIdentity());
    assert.ok(!bl.isBlocked());
  });
});

describe('LayoutPolicy', () => {
  it('returns NHWC preference for conv on CPU', () => {
    const policy = new LayoutPolicy(CPUTarget());
    const func = buildFunction('conv_pref',
      [new TensorType([1, 3, 28, 28], f32), new TensorType([16, 3, 3, 3], f32)],
      [new TensorType([1, 16, 26, 26], f32)],
      (b, [input, kernel]) => {
        const conv = b.conv(input, kernel, [1, 1], [[0, 0], [0, 0]]);
        b.returnOp([conv.getResult(0)]);
      }
    );
    const convOp = [...func.ops()].find(op => op.opName === 'conv');
    const pref = policy.getPreference(convOp);
    assert.ok(pref);
    assert.ok(pref.outputs.length > 0);
  });

  it('returns col-major RHS preference for dot on CPU', () => {
    const policy = new LayoutPolicy(CPUTarget());
    const func = buildFunction('dot_pref',
      [new TensorType([4, 8], f32), new TensorType([8, 4], f32)],
      [new TensorType([4, 4], f32)],
      (b, [a, w]) => {
        const mm = b.matmul(a, w);
        b.returnOp([mm.getResult(0)]);
      }
    );
    const dotOp = [...func.ops()].find(op => op.opName === 'dot');
    const pref = policy.getPreference(dotOp);
    assert.ok(pref);
    assert.ok(pref.inputs[1]);
  });

  it('returns null for elementwise ops', () => {
    const policy = new LayoutPolicy(CPUTarget());
    const func = buildFunction('ew_pref', [new TensorType([4, 4], f32)], [new TensorType([4, 4], f32)],
      (b, [x]) => { const e = b.exp(x); b.returnOp([e.getResult(0)]); }
    );
    const expOp = [...func.ops()].find(op => op.opName === 'exp');
    assert.equal(policy.getPreference(expOp), null);
  });
});

describe('LayoutTransformPass', () => {
  it('inserts layout_transform for conv on CPU', () => {
    const func = buildFunction('conv_layout',
      [new TensorType([1, 3, 28, 28], f32), new TensorType([16, 3, 3, 3], f32)],
      [new TensorType([1, 16, 26, 26], f32)],
      (b, [input, kernel]) => {
        const conv = b.conv(input, kernel, [1, 1], [[0, 0], [0, 0]]);
        b.returnOp([conv.getResult(0)]);
      }
    );
    const result = runLayoutPass(func, CPUTarget());
    assert.equal(result.changed, true);
    assert.ok(hasOpNamed(func, 'layout_transform'));
  });

  it('inserts layout_transform for dot RHS on CPU', () => {
    const func = buildFunction('dot_layout',
      [new TensorType([4, 8], f32), new TensorType([8, 4], f32)],
      [new TensorType([4, 4], f32)],
      (b, [a, w]) => {
        const mm = b.matmul(a, w);
        b.returnOp([mm.getResult(0)]);
      }
    );
    const result = runLayoutPass(func, CPUTarget());
    assert.equal(result.changed, true);
    assert.ok(hasOpNamed(func, 'layout_transform'));
  });

  it('compiles end-to-end with layout optimization', () => {
    const func = buildFunction('e2e_layout',
      [new TensorType([4, 8], f32), new TensorType([8, 4], f32)],
      [new TensorType([4, 4], f32)],
      (b, [a, w]) => {
        const mm = b.matmul(a, w);
        b.returnOp([mm.getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget(), {
      enableLayoutOptimization: true,
      enableFusion: false,
    });
    assert.ok(compiled.listKernels().includes('e2e_layout'));
  });
});
