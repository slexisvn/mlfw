import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';

import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { DecompositionPass, hasDecomposition } from '../../../src/compiler/passes/decompose/decomposition_pass.js';
import { PassResult } from '../../../src/compiler/passes/pass.js';
import { PassManager } from '../../../src/compiler/passes/pass_manager.js';
import { GraphModule } from '../../../src/compiler/ir/graph/module.js';
import { CPUTarget } from '../../../src/backend/target.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';

const f32 = ScalarType.F32;

function collectOpNames(func) {
  const names = [];
  for (const op of func.ops()) names.push(op.opName);
  return names;
}

function runDecompose(func) {
  const mod = new GraphModule('test');
  mod.addFunction(func);
  const pm = new PassManager();
  pm.addPass(new DecompositionPass());
  return pm.run(mod);
}

describe('DecompositionPass', () => {
  it('decomposes softmax into primitives', () => {
    const func = buildFunction('sm', [new TensorType([2, 10], f32)], [new TensorType([2, 10], f32)],
      (b, [x]) => {
        const s = b.softmax(x, -1);
        b.returnOp([s.getResult(0)]);
      }
    );
    const ops = collectOpNames(func);
    assert.ok(ops.includes('softmax'));

    const result = runDecompose(func);
    assert.equal(result.changed, true);

    const after = collectOpNames(func);
    assert.ok(!after.includes('softmax'));
    assert.ok(after.includes('reduce'));
    assert.ok(after.includes('exp'));
    assert.ok(after.includes('div'));
    assert.ok(after.includes('sub'));
  });

  it('decomposes log_softmax into primitives', () => {
    const func = buildFunction('lsm', [new TensorType([2, 10], f32)], [new TensorType([2, 10], f32)],
      (b, [x]) => {
        const s = b.logSoftmax(x, -1);
        b.returnOp([s.getResult(0)]);
      }
    );
    runDecompose(func);
    const after = collectOpNames(func);
    assert.ok(!after.includes('log_softmax'));
    assert.ok(after.includes('log'));
    assert.ok(after.includes('exp'));
    assert.ok(after.includes('sub'));
  });

  it('decomposes gelu into primitives', () => {
    const func = buildFunction('ge', [new TensorType([4, 4], f32)], [new TensorType([4, 4], f32)],
      (b, [x]) => {
        const g = b.gelu(x);
        b.returnOp([g.getResult(0)]);
      }
    );
    runDecompose(func);
    const after = collectOpNames(func);
    assert.ok(!after.includes('gelu'));
    assert.ok(after.includes('exp'));
    assert.ok(after.includes('mul'));
    assert.ok(after.includes('div'));
  });

  it('decomposes sigmoid into primitives', () => {
    const func = buildFunction('sig', [new TensorType([4, 4], f32)], [new TensorType([4, 4], f32)],
      (b, [x]) => {
        const s = b.sigmoid(x);
        b.returnOp([s.getResult(0)]);
      }
    );
    runDecompose(func);
    const after = collectOpNames(func);
    assert.ok(!after.includes('sigmoid'));
    assert.ok(after.includes('exp'));
    assert.ok(after.includes('neg'));
  });

  it('decomposes silu into primitives', () => {
    const func = buildFunction('sil', [new TensorType([4, 4], f32)], [new TensorType([4, 4], f32)],
      (b, [x]) => {
        const s = b.silu(x);
        b.returnOp([s.getResult(0)]);
      }
    );
    runDecompose(func);
    const after = collectOpNames(func);
    assert.ok(!after.includes('silu'));
    assert.ok(after.includes('mul'));
    assert.ok(after.includes('exp'));
  });

  it('decomposes layer_norm into primitives', () => {
    const func = buildFunction('ln',
      [new TensorType([2, 10], f32), new TensorType([10], f32), new TensorType([10], f32)],
      [new TensorType([2, 10], f32)],
      (b, [x, g, beta]) => {
        const ln = b.layernorm(x, g, beta, -1, 1e-5);
        b.returnOp([ln.getResult(0)]);
      }
    );
    runDecompose(func);
    const after = collectOpNames(func);
    assert.ok(!after.includes('layer_norm'));
    assert.ok(after.includes('reduce'));
    assert.ok(after.includes('rsqrt'));
    assert.ok(after.includes('mul'));
    assert.ok(after.includes('add'));
  });

  it('decomposes batch_norm into primitives', () => {
    const func = buildFunction('bn',
      [new TensorType([2, 8, 4, 4], f32), new TensorType([8], f32), new TensorType([8], f32),
       new TensorType([8], f32), new TensorType([8], f32)],
      [new TensorType([2, 8, 4, 4], f32)],
      (b, [x, gamma, beta, mean, variance]) => {
        const bn = b.batchnorm(x, gamma, beta, mean, variance, 1, 1e-5);
        b.returnOp([bn.getResult(0)]);
      }
    );
    runDecompose(func);
    const after = collectOpNames(func);
    assert.ok(!after.includes('batch_norm'));
    assert.ok(after.includes('rsqrt'));
    assert.ok(after.includes('sub'));
    assert.ok(after.includes('mul'));
    assert.ok(after.includes('add'));
  });

  it('returns UNCHANGED when no composite ops present', () => {
    const func = buildFunction('no_comp', [new TensorType([4, 4], f32)], [new TensorType([4, 4], f32)],
      (b, [x]) => {
        const e = b.exp(x);
        b.returnOp([e.getResult(0)]);
      }
    );
    const result = runDecompose(func);
    assert.equal(result.changed, false);
  });

  it('decomposed graph compiles end-to-end', () => {
    const func = buildFunction('e2e_sm',
      [new TensorType([2, 10], f32)],
      [new TensorType([2, 10], f32)],
      (b, [x]) => {
        const s = b.softmax(x, -1);
        b.returnOp([s.getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget(), { enableFusion: false });
    assert.ok(compiled.listKernels().includes('e2e_sm'));
  });

  it('has decomposition rules for all composite ops', () => {
    for (const name of ['softmax', 'log_softmax', 'gelu', 'sigmoid', 'silu', 'layer_norm', 'batch_norm']) {
      assert.ok(hasDecomposition(name), `missing decomposition for ${name}`);
    }
  });
});
