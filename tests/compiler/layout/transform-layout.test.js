import { describe, it, expect } from 'vitest';
import { buildFunction, IRBuilder } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType, Layout } from '../../../src/compiler/ir/graph/types.js';
import { LayoutTransformPass } from '../../../src/compiler/passes/layout/layout_transform.js';
import { PassResult } from '../../../src/compiler/passes/pass.js';
import { CPUTarget, CUDATarget, WasmTarget } from '../../../src/backend/target.js';
import { GraphFunction } from '../../../src/compiler/ir/graph/function.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';

const LAYOUT_AWARE = ['dot', 'conv', 'matmul', 'reduce'];
const CPUTargetL = (o = {}) => CPUTarget({ layoutAwareOps: LAYOUT_AWARE, ...o });
const CUDATargetL = (o = {}) => CUDATarget({ layoutAwareOps: LAYOUT_AWARE, ...o });
const WasmTargetL = (o = {}) => WasmTarget({ layoutAwareOps: LAYOUT_AWARE, ...o });

function run(func, target) {
  return new LayoutTransformPass({ target }).run(func);
}

function findOps(func, opName) {
  const result = [];
  for (const op of func.ops()) {
    if (op.opName === opName) result.push(op);
  }
  return result;
}

describe('LayoutTransformPass', () => {
  it('returns UNCHANGED when no target is set', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    expect(run(func, null)).toBe(PassResult.UNCHANGED);
  });

  it('returns UNCHANGED when no layout conversions needed', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    expect(run(func, CPUTargetL())).toBe(PassResult.UNCHANGED);
  });

  it('inserts layout_transform for CPU dot RHS (row-major -> column-major)', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const out = new TensorType([4, 6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    expect(run(func, CPUTargetL())).toBe(PassResult.CHANGED);

    const transforms = findOps(func, 'layout_transform');
    expect(transforms.length).toBe(1);

    const srcLayout = transforms[0].getAttr('src_layout');
    const dstLayout = transforms[0].getAttr('dst_layout');
    expect(srcLayout).toEqual([0, 1]);
    expect(dstLayout).toEqual([1, 0]);
  });

  it('dot RHS operand is rewired through the layout_transform', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const out = new TensorType([4, 6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    run(func, CPUTargetL());

    const dotOps = findOps(func, 'dot');
    expect(dotOps.length).toBe(1);
    expect(dotOps[0].getOperand(1).definingOp.opName).toBe('layout_transform');
  });

  it('layout_transform result type has the target layout', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const out = new TensorType([4, 6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    run(func, CPUTargetL());

    const transforms = findOps(func, 'layout_transform');
    const resultType = transforms[0].getResult(0).type;
    expect(resultType.layout.equals(Layout.columnMajor(2))).toBe(true);
    expect(resultType.shape).toEqual([8, 6]);
    expect(resultType.dtype).toBe(ScalarType.F32);
  });

  it('GPU dot does NOT insert layout_transform (both sides row-major)', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const out = new TensorType([4, 6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    expect(run(func, CUDATargetL())).toBe(PassResult.UNCHANGED);
    expect(findOps(func, 'layout_transform').length).toBe(0);
  });

  it('deduplicates identical transforms — same value + same layout pair inserts once', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const out = new TensorType([4, 6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out, out], (b, args) => {
      const d1 = b.matmul(args[0], args[1]);
      const d2 = b.matmul(args[0], args[1]);
      b.returnOp([d1.getResult(0), d2.getResult(0)]);
    });

    run(func, CPUTargetL());

    const transforms = findOps(func, 'layout_transform');
    expect(transforms.length).toBe(1);
  });

  it('conv on GPU inserts NCHW->NHWC transform for input', () => {
    const inp = new TensorType([1, 3, 32, 32], ScalarType.F32);
    const ker = new TensorType([16, 3, 3, 3], ScalarType.F32);
    const out = new TensorType([1, 16, 30, 30], ScalarType.F32);
    const func = buildFunction('f', [inp, ker], [out], (b, args) => {
      b.returnOp([b.conv(args[0], args[1], [1, 1], [0, 0, 0, 0]).getResult(0)]);
    });

    expect(run(func, CUDATargetL())).toBe(PassResult.CHANGED);

    const transforms = findOps(func, 'layout_transform');
    expect(transforms.length).toBeGreaterThanOrEqual(1);

    const convOp = findOps(func, 'conv')[0];
    const inputSource = convOp.getOperand(0).definingOp;
    expect(inputSource.opName).toBe('layout_transform');

    const dstLayout = inputSource.getAttr('dst_layout');
    expect(dstLayout).toEqual([0, 2, 3, 1]);
  });

  it('elementwise after dot does NOT trigger extra transform (inherits layout)', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const out = new TensorType([4, 6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs, out], [out], (b, args) => {
      const d = b.matmul(args[0], args[1]);
      b.returnOp([b.add(d.getResult(0), args[2]).getResult(0)]);
    });

    run(func, CPUTargetL());

    const transforms = findOps(func, 'layout_transform');
    const addOp = findOps(func, 'add')[0];
    for (let i = 0; i < addOp.numOperands; i++) {
      const src = addOp.getOperand(i).definingOp;
      if (src) {
        expect(src.opName).not.toBe('layout_transform');
      }
    }
  });
});

describe('LayoutTransformPass — cost-benefit profitability', () => {
  it('inserts transform for large dot RHS (benefit > cost)', () => {
    const lhs = new TensorType([64, 128], ScalarType.F32);
    const rhs = new TensorType([128, 64], ScalarType.F32);
    const out = new TensorType([64, 64], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    expect(run(func, CPUTargetL())).toBe(PassResult.CHANGED);
    expect(findOps(func, 'layout_transform').length).toBeGreaterThanOrEqual(1);
  });

  it('inserts transform for conv on GPU (compute-intensive)', () => {
    const inp = new TensorType([1, 3, 64, 64], ScalarType.F32);
    const ker = new TensorType([32, 3, 3, 3], ScalarType.F32);
    const out = new TensorType([1, 32, 62, 62], ScalarType.F32);
    const func = buildFunction('f', [inp, ker], [out], (b, args) => {
      b.returnOp([b.conv(args[0], args[1], [1, 1], [0, 0, 0, 0]).getResult(0)]);
    });

    expect(run(func, CUDATargetL())).toBe(PassResult.CHANGED);
    expect(findOps(func, 'layout_transform').length).toBeGreaterThanOrEqual(1);
  });

  it('dot transform count does not exceed needed conversions', () => {
    const lhs = new TensorType([32, 64], ScalarType.F32);
    const rhs = new TensorType([64, 32], ScalarType.F32);
    const out = new TensorType([32, 32], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    run(func, CPUTargetL());
    const transforms = findOps(func, 'layout_transform');
    expect(transforms.length).toBeLessThanOrEqual(2);
  });
});

const F = ScalarType.F32;
function buildAuto(name, inTypes, build) {
  const probe = new GraphFunction(name, inTypes, []);
  const out = build(new IRBuilder(probe), probe.args).getResult(0);
  return { func: buildFunction(name, inTypes, [out.type], (b, a) => { b.returnOp([build(b, a).getResult(0)]); }), outNumel: out.type.shape.reduce((x, y) => x * y, 1) };
}
const numel = (s) => s.reduce((a, b) => a * b, 1);

const LAYOUT_METAMORPHIC = [
  { name: 'matmul', inTypes: [[4, 5], [5, 6]], build: (b, a) => b.matmul(a[0], a[1]) },
  { name: 'double_matmul', inTypes: [[3, 4], [4, 5], [5, 2]], build: (b, a) => b.matmul(b.matmul(a[0], a[1]).getResult(0), a[2]) },
  { name: 'matmul_bias_relu', inTypes: [[4, 5], [5, 6], [6]], build: (b, a) => b.relu(b.add(b.matmul(a[0], a[1]).getResult(0), b.broadcast(a[2], [4, 6], [1]).getResult(0)).getResult(0)) },
  { name: 'conv', inTypes: [[1, 4, 7, 7], [4, 4, 3, 3]], build: (b, a) => b.conv(a[0], a[1], [1, 1], [[0, 0], [0, 0]]) },
  { name: 'conv_groups', inTypes: [[1, 4, 7, 7], [4, 2, 3, 3]], build: (b, a) => b.conv(a[0], a[1], [1, 1], [[1, 1], [1, 1]], { groups: 2 }) },
  { name: 'conv_relu_pool', inTypes: [[1, 3, 8, 8], [4, 3, 3, 3]], build: (b, a) => b.pool2d(b.relu(b.conv(a[0], a[1], [1, 1], [[0, 0], [0, 0]]).getResult(0)).getResult(0), 'max', [2, 2], [2, 2], [[0, 0], [0, 0]]) },
];

describe('layout optimization is semantics-preserving: layout ON == layout OFF (cpu+wasm)', () => {
  let seed = 99;
  const rng = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
  for (const spec of LAYOUT_METAMORPHIC) {
    for (const [tname, makeTarget] of [['cpu', CPUTargetL], ['wasm', WasmTargetL]]) {
      it(`${spec.name} on ${tname}`, () => {
        const inTypes = spec.inTypes.map((sh) => new TensorType(sh, F));
        const built = buildAuto(spec.name, inTypes, spec.build);
        const inputs = inTypes.map((t) => { const a = new Float32Array(numel(t.shape)); for (let i = 0; i < a.length; i++) a[i] = -1 + 2 * rng(); return a; });
        const outs = {};
        for (const lay of [false, true]) {
          const res = compileGraph(built.func, makeTarget(), { optimization: { layout: lay } });
          const out = new Float32Array(built.outNumel);
          res.run(spec.name, ...inputs, out);
          outs[lay] = out;
        }
        for (let i = 0; i < built.outNumel; i++) {
          const relErr = Math.abs(outs[false][i] - outs[true][i]) / (1 + Math.abs(outs[false][i]));
          expect(relErr, `${spec.name}/${tname} idx ${i}: off=${outs[false][i]} on=${outs[true][i]}`).toBeLessThan(1e-5);
        }
      });
    }
  }
});
