import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { GraphFunction } from '../../../src/compiler/ir/graph/function.js';
import { IRBuilder } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { FusionLegality } from '../../../src/compiler/passes/fusion/fusion_analysis.js';
import { FusionGroupBuilder } from '../../../src/compiler/passes/fusion/fusion_groups.js';

describe('Horizontal Fusion', () => {
  it('fuses independent ops with same shape and shared input', () => {
    const f32 = ScalarType.F32;
    const f32_2x3 = new TensorType([2, 3], f32);
    
    const func = buildFunction('test_horizontal', [f32_2x3], [f32_2x3, f32_2x3], (b, [x]) => {
      const a = b.exp(x);
      const c = b.log(x);
      b.returnOp([a.getResult(0), c.getResult(0)]);
    });

    const builder = new FusionGroupBuilder(new FusionLegality({}));
    const horizontalGroups = builder.buildHorizontalGroups(func);
    
    assert.equal(horizontalGroups.length, 1);
    const group = horizontalGroups[0];
    assert.equal(group.size, 2);
    
    const opNames = group.ops.map(o => o.opName).sort();
    assert.deepEqual(opNames, ['exp', 'log']);
  });
  it('fuses multiple horizontal operations (horizontal)', () => {
    const f32 = ScalarType.F32;
    const f32_4x4 = new TensorType([4, 4], f32);
    const func = buildFunction('test_multi_horizontal', [f32_4x4], [f32_4x4, f32_4x4, f32_4x4], (b, [x]) => {
      const y1 = b.add(x, x);
      const y2 = b.sub(x, x);
      const y3 = b.mul(x, x);
      b.returnOp([y1.getResult(0), y2.getResult(0), y3.getResult(0)]);
    });

    const builder = new FusionGroupBuilder(new FusionLegality({}));
    const horizontal = builder.buildHorizontalGroups(func);
    assert.equal(horizontal.length, 1);
    assert.equal(horizontal[0].size, 3);
  });

  it('fuses independent elementwise operations with identical output shapes (horizontal)', () => {
    const f32 = ScalarType.F32;
    const f32_4x4 = new TensorType([4, 4], f32);
    const func = buildFunction('test_independent_fusion', [f32_4x4, f32_4x4], [f32_4x4, f32_4x4], (b, [x, y]) => {
      const o1 = b.exp(x);
      const o2 = b.exp(y);
      b.returnOp([o1.getResult(0), o2.getResult(0)]);
    });

    const builder = new FusionGroupBuilder(new FusionLegality({}));
    const horizontal = builder.buildHorizontalGroups(func);
    assert.equal(horizontal.length, 1);
    assert.equal(horizontal[0].size, 2);
  });

  it('does not fuse operations with different output shapes (horizontal)', () => {
    const f32 = ScalarType.F32;
    const f32_4x4 = new TensorType([4, 4], f32);
    const f32_2x2 = new TensorType([2, 2], f32);
    const func = buildFunction('test_diff_shapes', [f32_4x4], [f32_4x4, f32_2x2], (b, [x]) => {
      const o1 = b.exp(x);
      const o2 = b.reshape(x, [2, 2]);
      b.returnOp([o1.getResult(0), o2.getResult(0)]);
    });

    const builder = new FusionGroupBuilder(new FusionLegality({}));
    const horizontal = builder.buildHorizontalGroups(func);
    assert.equal(horizontal.length, 0);
  });

  it('does not fuse opaque operations even if they share inputs (horizontal)', () => {
    const f32 = ScalarType.F32;
    const f32_4x4 = new TensorType([4, 4], f32);
    const func = buildFunction('test_opaque_no_fuse', [f32_4x4, f32_4x4], [f32_4x4, f32_4x4], (b, [x, y]) => {
      const o1 = b.dot(x, y, [1], [0]);
      const o2 = b.dot(x, y, [1], [0]);
      b.returnOp([o1.getResult(0), o2.getResult(0)]);
    });

    const builder = new FusionGroupBuilder(new FusionLegality({}));
    const horizontal = builder.buildHorizontalGroups(func);
    assert.equal(horizontal.length, 0);
  });
});
