import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { lowerGraphToPrimFunc } from '../../../src/compiler/passes/lowering/graph_to_tensor.js';
import { printTensorIR } from '../../../src/compiler/ir/tensor/printer.js';
import { TensorVerifier } from '../../../src/compiler/ir/tensor/verifier.js';
import { FusionPass } from '../../../src/compiler/passes/fusion/fusion_pass.js';

const f32 = ScalarType.F32;
const f32_4x4 = new TensorType([4, 4], f32);

describe('TensorIR Lowering', () => {
  it('lowers elementwise add to nested loops', () => {
    const f32_2x3 = new TensorType([2, 3], f32);
    const func = buildFunction('test_add', [f32_2x3, f32_2x3], [f32_2x3], (b, [x, y]) => {
      b.returnOp([b.add(x, y).getResult(0)]);
    });
    const printed = printTensorIR(lowerGraphToPrimFunc(func));
    assert.ok(printed.includes('for i0_'));
    assert.ok(printed.includes('block add_block'));
    assert.ok(printed.includes('+'));
  });

  it('lowers reduce with Set-based dim lookup', () => {
    const func = buildFunction('test', [new TensorType([4, 8], f32)], [new TensorType([4], f32)], (b, [x]) => {
      b.returnOp([b.reduce(x, b.scalarConstant(0, f32).getResult(0), [1], 'sum').getResult(0)]);
    });
    const printed = printTensorIR(lowerGraphToPrimFunc(func));
    assert.ok(printed.includes('reduce_acc'));
    assert.ok(printed.includes('+'));
  });

  it('lowers max reduction', () => {
    const func = buildFunction('test', [new TensorType([8], f32)], [new TensorType([], f32)], (b, [x]) => {
      b.returnOp([b.reduce(x, b.scalarConstant(-Infinity, f32).getResult(0), [0], 'max').getResult(0)]);
    });
    assert.ok(printTensorIR(lowerGraphToPrimFunc(func)).includes('max('));
  });

  it('lowers scalar constant without loop nest', () => {
    const func = buildFunction('test', [], [new TensorType([], f32)], (b) => {
      b.returnOp([b.scalarConstant(42, f32).getResult(0)]);
    });
    const printed = printTensorIR(lowerGraphToPrimFunc(func));
    assert.ok(!printed.includes('for '));
    assert.ok(printed.includes('42'));
  });

  it('lowers elementwise chain to verified IR', () => {
    const func = buildFunction('test', [f32_4x4, f32_4x4], [f32_4x4], (b, [x, y]) => {
      const a = b.add(x, y);
      const m = b.mul(a.getResult(0), x);
      b.returnOp([b.exp(m.getResult(0)).getResult(0)]);
    });
    assert.equal(new TensorVerifier().verify(lowerGraphToPrimFunc(func)).length, 0);
  });
});

describe('TensorIR Lowering for Dot', () => {
  it('lowers standard 2D matmul', () => {
    const func = buildFunction('test', [new TensorType([2, 3], f32), new TensorType([3, 4], f32)], [new TensorType([2, 4], f32)], (b, [x, y]) => {
      b.returnOp([b.dot(x, y, [1], [0]).getResult(0)]);
    });
    const printed = printTensorIR(lowerGraphToPrimFunc(func));
    assert.ok(printed.includes('block matmul'));
    assert.ok(printed.includes('*'));
    assert.ok(printed.includes('+'));
  });

  it('lowers batch matmul (3D)', () => {
    const func = buildFunction('test', [new TensorType([5, 2, 3], f32), new TensorType([5, 3, 4], f32)], [new TensorType([5, 2, 4], f32)], (b, [x, y]) => {
      b.returnOp([b.dot(x, y, [2], [1], [0], [0]).getResult(0)]);
    });
    const printed = printTensorIR(lowerGraphToPrimFunc(func));
    assert.ok(printed.includes('bind vb0_'));
  });
});

describe('TensorIR Lowering for Fusion', () => {
  it('lowers elementwise fusion to single loop nest', () => {
    const func = buildFunction('test', [f32_4x4, f32_4x4], [f32_4x4], (b, [x, y]) => {
      const add = b.add(x, y);
      b.returnOp([b.mul(add.getResult(0), x).getResult(0)]);
    });
    new FusionPass({ cost: { launchOverheadUs: 100 } }).run(func, null);
    const printed = printTensorIR(lowerGraphToPrimFunc(func));
    assert.ok(printed.includes('block fusion_block'));
    assert.ok(printed.includes('*'));
    assert.ok(printed.includes('+'));
  });

  it('lowers horizontal fusion to multi-store', () => {
    const func = buildFunction('test', [f32_4x4], [f32_4x4, f32_4x4], (b, [x]) => {
      b.returnOp([b.exp(x).getResult(0), b.log(x).getResult(0)]);
    });
    new FusionPass({ cost: { launchOverheadUs: 100 } }).run(func, null);
    const printed = printTensorIR(lowerGraphToPrimFunc(func));
    assert.ok(printed.includes('block fusion_block'));
    assert.ok(printed.includes('exp('));
    assert.ok(printed.includes('log('));
  });
});
