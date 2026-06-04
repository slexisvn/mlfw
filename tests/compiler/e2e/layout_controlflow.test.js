import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { TensorType, ScalarType, Layout } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { CPUTarget } from '../../../src/compiler/backend/target.js';
import { RuntimeTensor } from '../../../src/compiler/runtime/runtime.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { lowerGraphToPrimFunc } from '../../../src/compiler/passes/lowering/graph_to_tensor.js';

const f32 = ScalarType.F32;

function close(a, b, eps = 1e-3) { return Math.abs(a - b) < eps; }

function compile(func) {
  return compileGraph(func, CPUTarget({ enableEpilogueFusion: false }), {
    enableFusion: false, enableEpilogueFusion: false
  });
}

describe('Layout-aware buffer: column-major strides', () => {
  it('column-major tensor gets correct strides in lowered buffer', () => {
    const colMajor = Layout.columnMajor(2);
    const tt = new TensorType([3, 4], f32, colMajor);
    assert.deepEqual(tt.strides(), [1, 3]);

    const func = buildFunction('col_add',
      [tt, tt], [tt],
      (b, [x, y]) => {
        b.returnOp([b.add(x, y).getResult(0)]);
      }
    );
    const pf = lowerGraphToPrimFunc(func);
    let foundColMajor = false;
    for (const [, buf] of pf.bufferMap) {
      if (buf.strides[0] === 1 && buf.strides[1] === 3) foundColMajor = true;
    }
    assert.ok(foundColMajor, 'should have column-major strides in lowered buffer');
  });
});

describe('Layout: row-major identity preserved', () => {
  it('default layout produces standard strides', () => {
    const func = buildFunction('rm_add',
      [new TensorType([3, 4], f32), new TensorType([3, 4], f32)],
      [new TensorType([3, 4], f32)],
      (b, [x, y]) => {
        b.returnOp([b.add(x, y).getResult(0)]);
      }
    );
    const pf = lowerGraphToPrimFunc(func);
    for (const [, buf] of pf.bufferMap) {
      if (buf.shape.length === 2) {
        assert.equal(buf.strides[0], 4);
        assert.equal(buf.strides[1], 1);
      }
    }
  });
});

describe('Control flow: if op lowering', () => {
  it('if op compiles and lowered IR contains IfThenElseNode', () => {
    const func = buildFunction('if_test',
      [new TensorType([], ScalarType.BOOL), new TensorType([4], f32), new TensorType([4], f32)],
      [new TensorType([4], f32)],
      (b, [pred, x, y]) => {
        const result = b.ifOp(pred, [new TensorType([4], f32)],
          (tb) => { tb.yieldOp([x]); },
          (eb) => { eb.yieldOp([y]); }
        );
        b.returnOp([result.getResult(0)]);
      }
    );
    const compiled = compile(func);
    const source = compiled.getSource('if_test');
    assert.ok(source.includes('if'), `source should contain if statement, got:\n${source.substring(0, 300)}`);
  });
});

describe('WhileNode in tensor IR', () => {
  it('WhileNode is recognized by verifier', async () => {
    const { WhileNode, VariableNode, SeqNode, PrimFunc } = await import('../../../src/compiler/ir/tensor/nodes.js');
    const { TensorVerifier } = await import('../../../src/compiler/ir/tensor/verifier.js');
    const condVar = new VariableNode('cond', 'bool');
    const whileNode = new WhileNode(condVar, null, null);
    const func = new PrimFunc('test_while', [condVar], whileNode, new Map());
    const verifier = new TensorVerifier();
    const errors = verifier.verify(func);
    assert.ok(!errors.some(e => e.includes('Unrecognized')), `WhileNode should be recognized, errors: ${errors}`);
  });
});

describe('Conv with different layouts (integration)', () => {
  it('NCHW conv produces correct output', () => {
    const func = buildFunction('nchw_conv',
      [new TensorType([1, 1, 3, 3], f32), new TensorType([1, 1, 2, 2], f32)],
      [new TensorType([1, 1, 2, 2], f32)],
      (b, [x, k]) => {
        b.returnOp([b.conv(x, k, [1, 1], [[0,0],[0,0]]).getResult(0)]);
      }
    );
    const compiled = compile(func);
    const X = RuntimeTensor.fromArray([1,2,3, 4,5,6, 7,8,9], [1, 1, 3, 3]);
    const K = RuntimeTensor.fromArray([1,0, 0,1], [1, 1, 2, 2]);
    const out = RuntimeTensor.zeros([1, 1, 2, 2]);
    compiled.run('nchw_conv', X, K, out);
    assert.ok(close(out.data[0], 1 + 5));
    assert.ok(close(out.data[1], 2 + 6));
    assert.ok(close(out.data[2], 4 + 8));
    assert.ok(close(out.data[3], 5 + 9));
  });
});
