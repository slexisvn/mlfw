import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { TensorType, ScalarType, DYNAMIC } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { CPUTarget, GPUTarget, WasmTarget } from '../../../src/backend/target.js';
import { RuntimeTensor } from '../../../src/compiler/runtime/runtime.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { lowerGraphToPrimFunc } from '../../../src/compiler/passes/lowering/graph_to_tensor.js';

const f32 = ScalarType.F32;

function close(a, b, eps = 1e-3) { return Math.abs(a - b) < eps; }

function buildDynAdd() {
  return buildFunction('dyn_add',
    [new TensorType([DYNAMIC, 4], f32), new TensorType([DYNAMIC, 4], f32)],
    [new TensorType([DYNAMIC, 4], f32)],
    (b, [x, y]) => {
      b.returnOp([b.add(x, y).getResult(0)]);
    }
  );
}

function buildDynMul() {
  return buildFunction('dyn_mul',
    [new TensorType([DYNAMIC], f32), new TensorType([DYNAMIC], f32)],
    [new TensorType([DYNAMIC], f32)],
    (b, [x, y]) => {
      b.returnOp([b.mul(x, y).getResult(0)]);
    }
  );
}

function buildDynExp() {
  return buildFunction('dyn_exp',
    [new TensorType([DYNAMIC, 3], f32)],
    [new TensorType([DYNAMIC, 3], f32)],
    (b, [x]) => {
      b.returnOp([b.exp(x).getResult(0)]);
    }
  );
}

describe('Dynamic shapes: lowering', () => {
  it('multi-dim with one dynamic creates shape params', () => {
    const func = buildDynAdd();
    const pf = lowerGraphToPrimFunc(func);
    assert.ok(pf.shapeParams.length > 0);
    assert.ok(pf.shapeParamMap.size > 0);
  });

  it('shapeParamMap keys reference correct buffer dims', () => {
    const func = buildDynAdd();
    const pf = lowerGraphToPrimFunc(func);
    let hasDim0 = false;
    for (const key of pf.shapeParamMap.keys()) {
      if (key.endsWith(':0')) hasDim0 = true;
    }
    assert.ok(hasDim0, 'should have shape param for dim 0');
  });
});

describe('Dynamic shapes: CPU codegen', () => {
  it('generates valid JS with shape params in signature', () => {
    const func = buildDynAdd();
    const compiled = compileGraph(func, CPUTarget(), {
      enableFusion: false, enableEpilogueFusion: false
    });
    const source = compiled.getSource('dyn_add');
    assert.ok(source.includes('_ds'), 'should contain shape param');
    assert.ok(source.includes('function dyn_add'), 'should have function declaration');
  });

  it('generated code is syntactically valid JS', () => {
    const func = buildDynAdd();
    const compiled = compileGraph(func, CPUTarget(), {
      enableFusion: false, enableEpilogueFusion: false
    });
    const source = compiled.getSource('dyn_add');
    assert.doesNotThrow(() => new Function('return ' + source)());
  });
});

describe('Dynamic shapes: CPU end-to-end execution', () => {
  it('add with batch=2', () => {
    const func = buildDynAdd();
    const compiled = compileGraph(func, CPUTarget(), {
      enableFusion: false, enableEpilogueFusion: false
    });
    const x = RuntimeTensor.fromArray([1, 2, 3, 4, 5, 6, 7, 8], [2, 4], f32);
    const y = RuntimeTensor.fromArray([10, 20, 30, 40, 50, 60, 70, 80], [2, 4], f32);
    const out = RuntimeTensor.zeros([2, 4], f32);
    compiled.run('dyn_add', x, y, out);
    assert.ok(close(out.data[0], 11));
    assert.ok(close(out.data[3], 44));
    assert.ok(close(out.data[7], 88));
  });

  it('add with batch=5', () => {
    const func = buildDynAdd();
    const compiled = compileGraph(func, CPUTarget(), {
      enableFusion: false, enableEpilogueFusion: false
    });
    const n = 5 * 4;
    const xData = new Array(n).fill(0).map((_, i) => i);
    const yData = new Array(n).fill(0).map((_, i) => i * 2);
    const x = RuntimeTensor.fromArray(xData, [5, 4], f32);
    const y = RuntimeTensor.fromArray(yData, [5, 4], f32);
    const out = RuntimeTensor.zeros([5, 4], f32);
    compiled.run('dyn_add', x, y, out);
    for (let i = 0; i < n; i++) {
      assert.ok(close(out.data[i], xData[i] + yData[i]), `mismatch at ${i}`);
    }
  });

  it('same compiled module, different batch sizes', () => {
    const func = buildDynAdd();
    const compiled = compileGraph(func, CPUTarget(), {
      enableFusion: false, enableEpilogueFusion: false
    });

    for (const batch of [1, 3, 7, 16]) {
      const n = batch * 4;
      const xData = new Array(n).fill(1);
      const yData = new Array(n).fill(2);
      const x = RuntimeTensor.fromArray(xData, [batch, 4], f32);
      const y = RuntimeTensor.fromArray(yData, [batch, 4], f32);
      const out = RuntimeTensor.zeros([batch, 4], f32);
      compiled.run('dyn_add', x, y, out);
      for (let i = 0; i < n; i++) {
        assert.ok(close(out.data[i], 3), `batch=${batch} mismatch at ${i}`);
      }
    }
  });

  it('1D dynamic mul', () => {
    const func = buildDynMul();
    const compiled = compileGraph(func, CPUTarget(), {
      enableFusion: false, enableEpilogueFusion: false
    });
    for (const size of [1, 5, 100]) {
      const xData = new Array(size).fill(3);
      const yData = new Array(size).fill(7);
      const x = RuntimeTensor.fromArray(xData, [size], f32);
      const y = RuntimeTensor.fromArray(yData, [size], f32);
      const out = RuntimeTensor.zeros([size], f32);
      compiled.run('dyn_mul', x, y, out);
      for (let i = 0; i < size; i++) {
        assert.ok(close(out.data[i], 21), `size=${size} mismatch at ${i}`);
      }
    }
  });

  it('dynamic exp', () => {
    const func = buildDynExp();
    const compiled = compileGraph(func, CPUTarget(), {
      enableFusion: false, enableEpilogueFusion: false
    });
    for (const batch of [1, 4, 10]) {
      const n = batch * 3;
      const xData = new Array(n).fill(0);
      const x = RuntimeTensor.fromArray(xData, [batch, 3], f32);
      const out = RuntimeTensor.zeros([batch, 3], f32);
      compiled.run('dyn_exp', x, out);
      for (let i = 0; i < n; i++) {
        assert.ok(close(out.data[i], 1.0), `exp(0) should be 1, batch=${batch} idx=${i}`);
      }
    }
  });
});

describe('Dynamic shapes: GPU codegen', () => {
  it('generates CUDA with dynamic shape params', () => {
    const func = buildDynAdd();
    const compiled = compileGraph(func, GPUTarget({ enableEpilogueFusion: false }), {
      enableFusion: false, enableEpilogueFusion: false
    });
    const source = compiled.getSource('dyn_add');
    assert.ok(source.includes('int _ds'), 'should have int shape param');
    assert.ok(source.includes('__global__'), 'should have CUDA kernel');
  });
});

describe('Dynamic shapes: WASM codegen', () => {
  it('generates WAT with shape params in function signature', () => {
    const func = buildDynMul();
    const compiled = compileGraph(func, WasmTarget(), {
      enableFusion: false, enableEpilogueFusion: false
    });
    const source = compiled.getSource('dyn_mul');
    assert.ok(source.includes('(module'), 'should be WASM module');
    assert.ok(source.includes('param i32'), 'should have i32 params');
  });
});

describe('Dynamic shapes: memory planning', () => {
  it('handles dynamic buffer sizes without crashing', () => {
    const func = buildDynAdd();
    assert.doesNotThrow(() => {
      compileGraph(func, CPUTarget(), {
        enableFusion: false, enableEpilogueFusion: false
      });
    });
  });
});

describe('Dynamic shapes: verifier', () => {
  it('passes verification with dynamic dims', () => {
    const func = buildDynAdd();
    assert.doesNotThrow(() => {
      compileGraph(func, CPUTarget(), {
        verify: true, enableFusion: false, enableEpilogueFusion: false
      });
    });
  });

  it('passes verification with 1D dynamic', () => {
    const func = buildDynMul();
    assert.doesNotThrow(() => {
      compileGraph(func, CPUTarget(), {
        verify: true, enableFusion: false, enableEpilogueFusion: false
      });
    });
  });
});
