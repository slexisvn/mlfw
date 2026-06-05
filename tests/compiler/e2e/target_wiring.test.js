import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { CPUTarget, GPUTarget } from '../../../src/backend/target.js';
import { RuntimeTensor } from '../../../src/compiler/runtime/runtime.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { FusionPass } from '../../../src/compiler/passes/fusion/fusion_pass.js';

const f32 = ScalarType.F32;

describe('Target wiring: FusionPass receives target capabilities', () => {
  it('FusionPass constructed with target propagates libraryOps to cost model', () => {
    const target = GPUTarget();
    const pass = new FusionPass({ target });
    assert.ok(pass.costModel.libraryOps.has('dot'), 'cost model should have dot as library op from GPU target');
    assert.ok(pass.costModel.libraryOps.has('conv'), 'cost model should have conv as library op from GPU target');
  });

  it('FusionPass constructed with target propagates libraryOps to legality', () => {
    const target = GPUTarget();
    const pass = new FusionPass({ target });
    assert.ok(pass.legality.libraryOps.has('dot'));
  });

  it('FusionPass constructed with target propagates sharedMemory', () => {
    const target = GPUTarget({ sharedMemoryBytes: 12345 });
    const pass = new FusionPass({ target });
    assert.equal(pass.legality.maxSharedMemory, 12345);
    assert.equal(pass.costModel.maxSharedMemory, 12345);
  });

  it('FusionPass constructed with target propagates registersPerThread', () => {
    const target = GPUTarget({ registersPerThread: 128 });
    const pass = new FusionPass({ target });
    assert.equal(pass.costModel.maxRegistersPerThread, 128);
  });

  it('FusionPass constructed with target propagates bandwidth/compute', () => {
    const target = GPUTarget({ memoryBandwidthGBs: 1500, computeTFLOPs: 40 });
    const pass = new FusionPass({ target });
    assert.equal(pass.costModel.memoryBandwidthGBs, 1500);
    assert.equal(pass.costModel.computeTFLOPs, 40);
  });

  it('cost override in fusionConfig takes precedence over target', () => {
    const target = GPUTarget();
    const pass = new FusionPass({
      target,
      cost: { launchOverheadUs: 99 }
    });
    assert.equal(pass.costModel.launchOverheadUs, 99);
    assert.ok(pass.costModel.libraryOps.has('dot'), 'target libraryOps should still be present');
  });

  it('CPU target has no library ops by default', () => {
    const target = CPUTarget();
    const pass = new FusionPass({ target });
    assert.equal(pass.costModel.libraryOps.size, 0);
  });
});

describe('Target wiring: full pipeline integration', () => {
  it('GPU pipeline: dot is not fused by generic FusionPass (library op)', () => {
    const func = buildFunction('gpu_dot_nofuse',
      [new TensorType([4, 4], f32), new TensorType([4, 4], f32)],
      [new TensorType([4, 4], f32)],
      (b, [x, y]) => {
        const dot = b.dot(x, y, [1], [0]);
        const add = b.add(dot.getResult(0), x);
        b.returnOp([add.getResult(0)]);
      }
    );

    const compiled = compileGraph(func, GPUTarget({ enableEpilogueFusion: false }));
    const source = compiled.getSource('gpu_dot_nofuse');
    assert.ok(source.includes('__global__'));
  });

  it('CPU pipeline: elementwise chain fuses correctly with target wiring', () => {
    const func = buildFunction('cpu_chain',
      [new TensorType([8], f32), new TensorType([8], f32)],
      [new TensorType([8], f32)],
      (b, [x, y]) => {
        const a = b.add(x, y);
        const m = b.mul(a.getResult(0), x);
        b.returnOp([m.getResult(0)]);
      }
    );

    const compiled = compileGraph(func, CPUTarget(), { enableFusion: true });
    const X = RuntimeTensor.fromArray([1,2,3,4,5,6,7,8], [8]);
    const Y = RuntimeTensor.fromArray([1,1,1,1,1,1,1,1], [8]);
    const out = RuntimeTensor.zeros([8]);
    compiled.run('cpu_chain', X, Y, out);
    for (let i = 0; i < 8; i++) {
      const x = i + 1, y = 1;
      assert.equal(out.data[i], (x + y) * x);
    }
  });

  it('allowReductionFusion=false blocks reduction fusion in pipeline', () => {
    const func = buildFunction('no_reduce_fuse',
      [new TensorType([4, 4], f32), new TensorType([4, 4], f32)],
      [new TensorType([4], f32)],
      (b, [x, y]) => {
        const add = b.add(x, y);
        const init = b.scalarConstant(0, f32);
        b.returnOp([b.reduce(add.getResult(0), init.getResult(0), [1], 'sum').getResult(0)]);
      }
    );

    const compiled = compileGraph(func, CPUTarget(), {
      enableFusion: true,
      fusionConfig: { allowReductionFusion: false }
    });
    const source = compiled.getSource('no_reduce_fuse');
    assert.ok(!source.includes('fusion_block'),
      'should not produce fused block when reduction fusion is disabled');
  });
});
