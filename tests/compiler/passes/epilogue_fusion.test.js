import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { CPUTarget } from '../../../src/compiler/backend/target.js';
import { RuntimeTensor } from '../../../src/compiler/runtime/runtime.js';
import { compileGraph, Compiler } from '../../../src/compiler/pipeline/compiler.js';
import { EpilogueFusionPass } from '../../../src/compiler/passes/fusion/epilogue_fusion.js';
import { PassResult } from '../../../src/compiler/passes/pass.js';
import { printFunction } from '../../../src/compiler/ir/printer/printer.js';

const f32 = ScalarType.F32;

describe('EpilogueFusionPass pattern detection', () => {
  it('detects dot + bias pattern', () => {
    const func = buildFunction('dot_bias',
      [new TensorType([2, 3], f32), new TensorType([3, 4], f32), new TensorType([4], f32)],
      [new TensorType([2, 4], f32)],
      (b, [x, w, bias]) => {
        const dot = b.matmul(x, w);
        const bcast = b.broadcast(bias, [2, 4], [1]);
        const result = b.add(dot.getResult(0), bcast.getResult(0));
        b.returnOp([result.getResult(0)]);
      }
    );

    const pass = new EpilogueFusionPass();
    const result = pass.run(func);
    assert.equal(result, PassResult.CHANGED);

    const fusedOp = func.findOp(op => op.opName === 'fused_dot_epilogue');
    assert.ok(fusedOp, 'should create fused_dot_epilogue op');
    const tags = fusedOp.getAttr('epilogue_tags');
    assert.ok(tags.includes('bias'), `tags should include bias, got: ${tags}`);
  });

  it('detects dot + bias + relu pattern', () => {
    const func = buildFunction('dot_bias_relu',
      [new TensorType([2, 3], f32), new TensorType([3, 4], f32), new TensorType([4], f32)],
      [new TensorType([2, 4], f32)],
      (b, [x, w, bias]) => {
        const dot = b.matmul(x, w);
        const bcast = b.broadcast(bias, [2, 4], [1]);
        const biased = b.add(dot.getResult(0), bcast.getResult(0));
        const result = b.relu(biased.getResult(0));
        b.returnOp([result.getResult(0)]);
      }
    );

    const pass = new EpilogueFusionPass();
    const result = pass.run(func);
    assert.equal(result, PassResult.CHANGED);

    const fusedOp = func.findOp(op => op.opName === 'fused_dot_epilogue');
    assert.ok(fusedOp);
  });

  it('does not fuse when no epilogue ops follow dot', () => {
    const func = buildFunction('dot_only',
      [new TensorType([2, 3], f32), new TensorType([3, 2], f32)],
      [new TensorType([2, 2], f32)],
      (b, [x, w]) => {
        b.returnOp([b.matmul(x, w).getResult(0)]);
      }
    );

    const pass = new EpilogueFusionPass();
    const result = pass.run(func);
    assert.equal(result, PassResult.UNCHANGED);
  });
});

describe('E2E: fused_dot_epilogue lowering', () => {
  it('dot + bias produces correct results', () => {
    const func = buildFunction('gemm_bias',
      [new TensorType([2, 3], f32), new TensorType([3, 4], f32), new TensorType([4], f32)],
      [new TensorType([2, 4], f32)],
      (b, [x, w, bias]) => {
        const dot = b.matmul(x, w);
        const bcast = b.broadcast(bias, [2, 4], [1]);
        const result = b.add(dot.getResult(0), bcast.getResult(0));
        b.returnOp([result.getResult(0)]);
      }
    );

    new EpilogueFusionPass().run(func);
    const compiled = compileGraph(func, CPUTarget(), { enableFusion: false });

    const X = RuntimeTensor.fromArray([1,0,0, 0,1,0], [2, 3]);
    const W = RuntimeTensor.fromArray([1,2,3,4, 5,6,7,8, 9,10,11,12], [3, 4]);
    const bias = RuntimeTensor.fromArray([100, 200, 300, 400], [4]);
    const out = RuntimeTensor.zeros([2, 4]);
    compiled.run('gemm_bias', X, W, bias, out);
    assert.equal(out.data[0], 1 + 100);
    assert.equal(out.data[1], 2 + 200);
    assert.equal(out.data[2], 3 + 300);
    assert.equal(out.data[3], 4 + 400);
    assert.equal(out.data[4], 5 + 100);
    assert.equal(out.data[5], 6 + 200);
    assert.equal(out.data[6], 7 + 300);
    assert.equal(out.data[7], 8 + 400);
  });

  it('dot + bias + relu produces correct results', () => {
    const func = buildFunction('gemm_bias_relu',
      [new TensorType([2, 3], f32), new TensorType([3, 2], f32), new TensorType([2], f32)],
      [new TensorType([2, 2], f32)],
      (b, [x, w, bias]) => {
        const dot = b.matmul(x, w);
        const bcast = b.broadcast(bias, [2, 2], [1]);
        const biased = b.add(dot.getResult(0), bcast.getResult(0));
        const result = b.relu(biased.getResult(0));
        b.returnOp([result.getResult(0)]);
      }
    );

    new EpilogueFusionPass().run(func);
    const compiled = compileGraph(func, CPUTarget(), { enableFusion: false });

    const X = RuntimeTensor.fromArray([1,0,0, 0,1,0], [2, 3]);
    const W = RuntimeTensor.fromArray([-10, 5, 3, -2, 0, 0], [3, 2]);
    const bias = RuntimeTensor.fromArray([1, 1], [2]);
    const out = RuntimeTensor.zeros([2, 2]);
    compiled.run('gemm_bias_relu', X, W, bias, out);
    assert.equal(out.data[0], Math.max(-10 + 1, 0));
    assert.equal(out.data[1], Math.max(5 + 1, 0));
    assert.equal(out.data[2], Math.max(3 + 1, 0));
    assert.equal(out.data[3], Math.max(-2 + 1, 0));
  });

  it('generates fused_dot_epilogue in source', () => {
    const func = buildFunction('check_src',
      [new TensorType([2, 3], f32), new TensorType([3, 2], f32), new TensorType([2], f32)],
      [new TensorType([2, 2], f32)],
      (b, [x, w, bias]) => {
        const dot = b.matmul(x, w);
        const bcast = b.broadcast(bias, [2, 2], [1]);
        b.returnOp([b.add(dot.getResult(0), bcast.getResult(0)).getResult(0)]);
      }
    );

    new EpilogueFusionPass().run(func);
    const compiled = compileGraph(func, CPUTarget(), { enableFusion: false });
    const source = compiled.getSource('check_src');
    assert.ok(source.includes('vls') && source.includes('epv'),
      `source should contain matmul (vls) and epilogue (epv) variables`);
  });
});
