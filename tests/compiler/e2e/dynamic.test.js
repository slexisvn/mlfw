import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { TensorType, ScalarType, DYNAMIC } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { CPUTarget } from '../../../src/backend/target.js';
import { RuntimeTensor } from '../../../src/compiler/runtime/runtime.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { lowerGraphToPrimFunc } from '../../../src/compiler/passes/lowering/graph_to_tensor.js';

const f32 = ScalarType.F32;

function close(a, b, eps = 1e-3) { return Math.abs(a - b) < eps; }

describe('Dynamic shape: lowering produces shape params', () => {
  it('DYNAMIC dim creates shape param in PrimFunc', () => {
    const func = buildFunction('dyn_add',
      [new TensorType([DYNAMIC], f32), new TensorType([DYNAMIC], f32)],
      [new TensorType([DYNAMIC], f32)],
      (b, [x, y]) => {
        b.returnOp([b.add(x, y).getResult(0)]);
      }
    );
    const pf = lowerGraphToPrimFunc(func);
    assert.ok(pf.shapeParams.length > 0, 'should have shape params for DYNAMIC dims');
    assert.ok(pf.params.length > pf.bufferMap.size, 'total params should exceed buffer params');
  });

  it('static dims produce no shape params', () => {
    const func = buildFunction('static_add',
      [new TensorType([4], f32), new TensorType([4], f32)],
      [new TensorType([4], f32)],
      (b, [x, y]) => {
        b.returnOp([b.add(x, y).getResult(0)]);
      }
    );
    const pf = lowerGraphToPrimFunc(func);
    assert.equal(pf.shapeParams.length, 0);
  });
});

describe('Dynamic shape: CPU codegen includes shape params', () => {
  it('function signature contains shape param names', () => {
    const func = buildFunction('dyn_sig',
      [new TensorType([DYNAMIC, 4], f32)],
      [new TensorType([DYNAMIC, 4], f32)],
      (b, [x]) => {
        b.returnOp([b.exp(x).getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget({ enableEpilogueFusion: false }), {
      enableFusion: false, enableEpilogueFusion: false
    });
    const source = compiled.getSource('dyn_sig');
    assert.ok(source.includes('_ds'), `source should contain shape param _ds, got:\n${source.substring(0, 300)}`);
  });
});

describe('Dynamic shape: GPU codegen includes shape params', () => {
  it('CUDA kernel signature has int shape param', async () => {
    const { GPUTarget } = await import('../../../src/backend/target.js');
    const func = buildFunction('gpu_dyn',
      [new TensorType([DYNAMIC, 4], f32)],
      [new TensorType([DYNAMIC, 4], f32)],
      (b, [x]) => {
        b.returnOp([b.exp(x).getResult(0)]);
      }
    );
    const compiled = compileGraph(func, GPUTarget({ enableEpilogueFusion: false }), {
      enableFusion: false, enableEpilogueFusion: false
    });
    const source = compiled.getSource('gpu_dyn');
    assert.ok(source.includes('int _ds'), `CUDA should contain int shape param, got:\n${source.substring(0, 300)}`);
  });
});
