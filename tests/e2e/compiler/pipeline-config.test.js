import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { Compiler, compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { CPUTarget, GPUTarget } from '../../../src/backend/target.js';
import { GraphModule } from '../../../src/compiler/ir/graph/module.js';
import { TraceLevel } from '../../../src/compiler/pipeline/trace.js';

function compile(func, opts = {}) {
  return compileGraph(func, CPUTarget(), opts);
}

describe('multi-function module compilation', () => {
  it('compiles module with two functions into two kernels', () => {
    const t = new TensorType([4], ScalarType.F32);
    const f1 = buildFunction('add_fn', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const f2 = buildFunction('neg_fn', [t], [t], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });

    const mod = new GraphModule('test');
    mod.addFunction(f1);
    mod.addFunction(f2);

    const compiler = new Compiler({ target: CPUTarget() });
    const result = compiler.compile(mod);
    expect(result.succeeded).toBe(true);

    const kernels = result.listKernels();
    expect(kernels).toContain('add_fn');
    expect(kernels).toContain('neg_fn');

    const a = new Float32Array([1, 2, 3, 4]);
    const b = new Float32Array([10, 20, 30, 40]);
    const out1 = new Float32Array(4);
    result.run('add_fn', a, b, out1);
    expect(Array.from(out1)).toEqual([11, 22, 33, 44]);

    const out2 = new Float32Array(4);
    result.run('neg_fn', a, out2);
    expect(Array.from(out2)).toEqual([-1, -2, -3, -4]);
  });
});

describe('fusion disabled', () => {
  it('without fusion: separate loops and temp buffers; with fusion: single loop', () => {
    const t = new TensorType([4], ScalarType.F32);
    const mkFunc = (name) => buildFunction(name, [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.neg(sum.getResult(0)).getResult(0)]);
    });

    const unfused = compile(mkFunc('unfused'), { fusion: { enabled: false } });
    const fused = compile(mkFunc('fused'));
    const srcU = unfused.getSource('unfused');
    const srcF = fused.getSource('fused');

    const loopsU = (srcU.match(/\bfor\s*\(/g) || []).length;
    const loopsF = (srcF.match(/\bfor\s*\(/g) || []).length;
    expect(loopsU).toBe(2);
    expect(loopsF).toBe(1);

    const tempsU = (srcU.match(/new Float32Array/g) || []).length;
    const tempsF = (srcF.match(/new Float32Array/g) || []).length;
    expect(tempsU).toBeGreaterThanOrEqual(1);
    expect(tempsF).toBe(0);

    const a = new Float32Array([1, 2, 3, 4]);
    const b = new Float32Array([10, 20, 30, 40]);
    const out1 = new Float32Array(4);
    const out2 = new Float32Array(4);
    unfused.run('unfused', a, b, out1);
    fused.run('fused', a, b, out2);
    expect(Array.from(out1)).toEqual([-11, -22, -33, -44]);
    expect(Array.from(out2)).toEqual([-11, -22, -33, -44]);
  });
});

describe('trace collects compilation events', () => {
  it('trace sink receives pass-level events at VERBOSE', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('traced', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.neg(sum.getResult(0)).getResult(0)]);
    });

    const events = [];
    const result = compile(func, {
      trace: {
        level: TraceLevel.VERBOSE,
        sink: (e) => events.push(e),
      }
    });

    expect(result.succeeded).toBe(true);
    const passEvents = events.filter(e => e.type === 'pass');
    expect(passEvents.length).toBeGreaterThan(0);
    expect(passEvents[0]).toHaveProperty('passName');
    expect(passEvents[0]).toHaveProperty('changed');
    expect(passEvents[0]).toHaveProperty('durationMs');
  });
});

describe('constant folding', () => {
  it('compile-time constants folded to literals in generated source', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('cf', [t], [t], (b, args) => {
      const c1 = b.scalarConstant(2, ScalarType.F32);
      const c2 = b.scalarConstant(3, ScalarType.F32);
      const prod = b.mul(c1.getResult(0), c2.getResult(0));
      const bc = b.broadcast(prod.getResult(0), [4], []);
      b.returnOp([b.mul(args[0], bc.getResult(0)).getResult(0)]);
    });

    const r = compile(func);
    const src = r.getSource('cf');
    expect(src).toMatch(/\b6\b/);
    const out = new Float32Array(4);
    r.run('cf', new Float32Array([1, 2, 3, 4]), out);
    expect(Array.from(out)).toEqual([6, 12, 18, 24]);
  });
});

describe('GPU target compilation (codegen only — no execution)', () => {
  it('produces CUDA kernel source for matmul', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 4], ScalarType.F32);
    const out = new TensorType([4, 4], ScalarType.F32);

    const func = buildFunction('gpu_mm', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    const result = compileGraph(func, GPUTarget());
    expect(result.succeeded).toBe(true);
    const source = result.getSource('gpu_mm');
    expect(source.length).toBeGreaterThan(0);
  });
});

describe('layout optimization', () => {
  it('layout-optimized matmul produces same results as default', () => {
    const lhs = new TensorType([2, 3], ScalarType.F32);
    const rhs = new TensorType([3, 2], ScalarType.F32);
    const out = new TensorType([2, 2], ScalarType.F32);

    const func1 = buildFunction('mm_default', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    const func2 = buildFunction('mm_layout', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    const r1 = compile(func1);
    const r2 = compile(func2, { optimization: { layout: true } });
    expect(r1.succeeded).toBe(true);
    expect(r2.succeeded).toBe(true);

    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    const b = new Float32Array([7, 8, 9, 10, 11, 12]);
    const c1 = new Float32Array(4);
    const c2 = new Float32Array(4);
    r1.run('mm_default', a, b, c1);
    r2.run('mm_layout', a, b, c2);
    expect(Array.from(c1)).toEqual([58, 64, 139, 154]);
    expect(Array.from(c2)).toEqual(Array.from(c1));
  });
});
