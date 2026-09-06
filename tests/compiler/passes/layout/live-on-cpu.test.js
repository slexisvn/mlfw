import { describe, it, expect } from 'vitest';
import { buildFunction, IRBuilder } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType } from '../../../../src/compiler/ir/graph/types.js';
import { LayoutTransformPass } from '../../../../src/compiler/passes/layout/layout_transform.js';
import { PassResult } from '../../../../src/compiler/passes/pass.js';
import { CPUTarget } from '../../../../src/compiler/support/target.js';
import { GraphFunction } from '../../../../src/compiler/ir/graph/function.js';
import { compileGraph } from '../../../../src/compiler/pipeline/compiler.js';
import { lowerGraphToPrimFunc } from '../../../../src/compiler/passes/lowering/graph_to_tensor.js';
import { optimizationCandidates } from '../../../../src/compiler/pipeline/opt_gate.js';
import { collect } from '../../../../src/compiler/ir/ir_visitor.js';
import { F32 as F } from '../../../_utils/ir_fixture.js';

const numel = (shape) => shape.reduce((a, b) => a * b, 1);

function buildAuto(name, inTypes, build) {
  const probe = new GraphFunction(name, inTypes, []);
  const out = build(new IRBuilder(probe), probe.args).getResult(0);
  return {
    func: buildFunction(name, inTypes, [out.type], (b, a) => { b.returnOp([build(b, a).getResult(0)]); }),
    outNumel: numel(out.type.shape)
  };
}

function countOps(func, opName) {
  let n = 0;
  for (const op of func.ops()) if (op.opName === opName) n++;
  return n;
}

function blockedConv(inShape, kernelShape) {
  const input = new TensorType(inShape, F);
  const kernel = new TensorType(kernelShape, F);
  const probe = new GraphFunction('c', [input, kernel], []);
  const outType = new IRBuilder(probe).conv(probe.args[0], probe.args[1], [1, 1], [[0, 0], [0, 0]]).getResult(0).type;
  return buildFunction('c', [input, kernel], [outType], (b, args) => {
    b.returnOp([b.conv(args[0], args[1], [1, 1], [[0, 0], [0, 0]]).getResult(0)]);
  });
}

const CASES = [
  { name: 'matmul', inTypes: [[8, 12], [12, 6]], build: (b, a) => b.matmul(a[0], a[1]) },
  { name: 'conv', inTypes: [[1, 4, 7, 7], [4, 4, 3, 3]], build: (b, a) => b.conv(a[0], a[1], [1, 1], [[0, 0], [0, 0]]) },
  { name: 'blocked_conv', inTypes: [[1, 16, 7, 7], [8, 16, 3, 3]], build: (b, a) => b.conv(a[0], a[1], [1, 1], [[0, 0], [0, 0]]) },
  { name: 'blocked_conv_padded', inTypes: [[2, 8, 6, 6], [8, 8, 3, 3]], build: (b, a) => b.conv(a[0], a[1], [2, 2], [[1, 1], [1, 1]]) },
  {
    name: 'blocked_conv_chain',
    inTypes: [[1, 8, 6, 6], [8, 8, 3, 3], [8, 8, 3, 3]],
    build: (b, a) => b.conv(b.relu(b.conv(a[0], a[1], [1, 1], [[1, 1], [1, 1]]).getResult(0)).getResult(0), a[2], [1, 1], [[1, 1], [1, 1]])
  },
  {
    name: 'blocked_conv_deep_chain',
    inTypes: [[1, 16, 6, 6], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    build: (b, a) => {
      let x = a[0];
      for (let i = 1; i <= 3; i++) {
        const c = b.conv(x, a[i], [1, 1], [[1, 1], [1, 1]]).getResult(0);
        x = i === 3 ? c : b.relu(c).getResult(0);
      }
      return { getResult: () => x };
    },
    tol: 5e-5
  }
];

describe('the shipped CPU target reaches the layout optimization', () => {
  it('declares the ops whose benefit the cost model is allowed to count', () => {
    expect([...CPUTarget().layoutAwareOps].sort()).toEqual(['conv', 'dot']);
  });

  it('offers layout as a measurable candidate to the optimization gate', () => {
    expect(optimizationCandidates(CPUTarget()).map(c => c.name)).toContain('layout');
  });

  it('rewrites a matmul without any test-only target override', () => {
    const lhs = new TensorType([8, 12], F);
    const rhs = new TensorType([12, 6], F);
    const out = new TensorType([8, 6], F);
    const func = buildFunction('m', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    expect(new LayoutTransformPass({ target: CPUTarget() }).run(func)).toBe(PassResult.CHANGED);
    expect(countOps(func, 'layout_transform')).toBeGreaterThan(0);
  });

  it('materialises a blocked conv layout as a higher-rank buffer', () => {
    const func = blockedConv([1, 8, 7, 7], [8, 8, 3, 3]);
    const target = CPUTarget({ preferredConvLayout: { order: [0, 1, 2, 3], block: { dim: 1, factor: 4 } } });

    expect(new LayoutTransformPass({ target }).run(func)).toBe(PassResult.CHANGED);
    const transform = [...func.ops()].find(op => op.opName === 'layout_transform');
    expect(transform.getAttr('dst_block')).toEqual([1, 4]);
    expect(transform.getResult(0).type.layout.isBlocked()).toBe(true);

    expect(transform.getResult(0).type.layout.storage([1, 8, 7, 7])).toEqual({
      shape: [1, 2, 7, 7, 4],
      strides: [392, 196, 28, 4, 1]
    });

    const prim = lowerGraphToPrimFunc(func, target);
    const stores = collect(prim, n => n.type === 'BufferStoreNode' && n.buffer.rank === 5);
    expect(stores.length).toBeGreaterThan(0);
    expect(stores[0].indices.length).toBe(5);
    expect(stores[0].buffer.shape).toEqual([1, 2, 7, 7, 4]);
    expect(stores[0].buffer.strides).toEqual([392, 196, 28, 4, 1]);
  });

  it('splits the channel index in the generated kernel without any target override', () => {
    const on = compileGraph(blockedConv([1, 16, 7, 7], [8, 16, 3, 3]), CPUTarget(), { optimization: { layout: true } });
    const off = compileGraph(blockedConv([1, 16, 7, 7], [8, 16, 3, 3]), CPUTarget(), { optimization: { layout: false } });

    expect(on.getSource('c')).toMatch(/\/ 8\) \| 0\) \* 392\)/);
    expect(on.getSource('c')).toMatch(/% 8\)/);
    expect(off.getSource('c')).not.toMatch(/% 8\)/);
  });

  it('makes the channel block the innermost conv loop with both operands unit stride', () => {
    const src = compileGraph(blockedConv([1, 16, 7, 7], [8, 16, 3, 3]), CPUTarget(), { optimization: { layout: true } }).getSource('c');
    const inner = src.match(/for \(let (\w+) = 0; \1 < 8; \1\+\+\) \{\s*\n(.*_acc.*)\n/);

    expect(inner).not.toBeNull();
    expect(inner[2].match(new RegExp(`\\+ ${inner[1]}\\)\\]`, 'g'))).toHaveLength(2);
  });

  it('blocks the kernel alongside the input and unblocks only at the frontier', () => {
    const func = blockedConv([1, 16, 7, 7], [8, 16, 3, 3]);
    new LayoutTransformPass({ target: CPUTarget() }).run(func);
    const transforms = [...func.ops()].filter(op => op.opName === 'layout_transform');

    expect(transforms.filter(op => op.hasAttr('dst_block')).map(op => op.getAttr('dst_block'))).toEqual([[1, 8], [1, 8]]);
    expect(transforms.filter(op => op.hasAttr('src_block') && !op.hasAttr('dst_block')).length).toBe(1);
    expect([...func.ops()].find(op => op.opName === 'conv').getResult(0).type.layout.isBlocked()).toBe(true);
  });

  it('keeps a conv chain blocked end to end instead of round-tripping between convs', () => {
    const inp = new TensorType([1, 16, 8, 8], F);
    const k = new TensorType([16, 16, 3, 3], F);
    const probe = new GraphFunction('c', [inp, k, k], []);
    const chain = (b, a) => b.conv(b.relu(b.conv(a[0], a[1], [1, 1], [[1, 1], [1, 1]]).getResult(0)).getResult(0), a[2], [1, 1], [[1, 1], [1, 1]]);
    const outType = chain(new IRBuilder(probe), probe.args).getResult(0).type;
    const func = buildFunction('c', [inp, k, k], [outType], (b, a) => { b.returnOp([chain(b, a).getResult(0)]); });

    new LayoutTransformPass({ target: CPUTarget() }).run(func);
    const unblock = [...func.ops()].filter(op => op.opName === 'layout_transform' && !op.hasAttr('dst_block'));

    expect(unblock.length).toBe(1);
    expect([...func.ops()].find(op => op.opName === 'maximum').getResult(0).type.layout.isBlocked()).toBe(true);
  });

  it('unblocks once however deep the chain runs', () => {
    const inp = new TensorType([1, 16, 8, 8], F);
    const k = new TensorType([16, 16, 3, 3], F);
    const args = [inp, k, k, k, k];
    const chain = (b, a) => {
      let x = a[0];
      for (let i = 1; i <= 4; i++) {
        const c = b.conv(x, a[i], [1, 1], [[1, 1], [1, 1]]).getResult(0);
        x = i === 4 ? c : b.relu(c).getResult(0);
      }
      return x;
    };
    const probe = new GraphFunction('c', args, []);
    const outType = chain(new IRBuilder(probe), probe.args).type;
    const func = buildFunction('c', args, [outType], (b, a) => { b.returnOp([chain(b, a)]); });

    new LayoutTransformPass({ target: CPUTarget() }).run(func);
    const unblock = [...func.ops()].filter(op => op.opName === 'layout_transform' && !op.hasAttr('dst_block'));
    const convs = [...func.ops()].filter(op => op.opName === 'conv');

    expect(unblock.length).toBe(1);
    expect(convs.length).toBe(4);
    expect(convs.every(op => op.getResult(0).type.layout.isBlocked())).toBe(true);
  });

  it('declines to block a channel count the factor does not divide', () => {
    const func = blockedConv([1, 12, 7, 7], [8, 12, 3, 3]);
    const target = CPUTarget({ preferredConvLayout: { order: [0, 1, 2, 3], block: { dim: 1, factor: 8 } } });

    new LayoutTransformPass({ target }).run(func);
    for (const op of func.ops()) {
      if (op.opName === 'layout_transform') expect(op.hasAttr('dst_block')).toBe(false);
    }
  });

  it('leaves a graph alone when no op in it is layout aware', () => {
    const t = new TensorType([8, 12], F);
    const func = buildFunction('e', [t], [t], (b, args) => {
      b.returnOp([b.relu(args[0]).getResult(0)]);
    });

    expect(new LayoutTransformPass({ target: CPUTarget() }).run(func)).toBe(PassResult.UNCHANGED);
  });

  for (const spec of CASES) {
    it(`keeps ${spec.name} numerically identical with the optimization on`, () => {
      let seed = 7;
      const rng = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
      const inTypes = spec.inTypes.map(sh => new TensorType(sh, F));
      const built = buildAuto(spec.name, inTypes, spec.build);
      const inputs = inTypes.map(t => {
        const a = new Float32Array(numel(t.shape));
        for (let i = 0; i < a.length; i++) a[i] = -1 + 2 * rng();
        return a;
      });

      const outs = {};
      for (const layout of [false, true]) {
        const res = compileGraph(built.func, CPUTarget(), { optimization: { layout } });
        const out = new Float32Array(built.outNumel);
        res.run(spec.name, ...inputs, out);
        outs[layout] = out;
      }

      for (let i = 0; i < built.outNumel; i++) {
        const relErr = Math.abs(outs[false][i] - outs[true][i]) / (1 + Math.abs(outs[false][i]));
        expect(relErr, `${spec.name} idx ${i}: off=${outs[false][i]} on=${outs[true][i]}`).toBeLessThan(spec.tol || 1e-5);
      }
    });
  }
});
