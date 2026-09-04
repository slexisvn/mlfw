import { describe, it, expect } from 'vitest';
import { buildFunction, IRBuilder } from '../../../../src/compiler/ir/graph/builder.js';
import { GraphFunction } from '../../../../src/compiler/ir/graph/function.js';
import { TensorType, DYNAMIC } from '../../../../src/compiler/ir/graph/types.js';
import { capturedValues } from '../../../../src/compiler/ir/graph/graph_algorithms.js';
import { verifyFunction } from '../../../../src/compiler/ir/graph/verifier.js';
import { PassResult } from '../../../../src/compiler/passes/pass.js';
import { broadcastDimsFor, ExplicitBroadcastPass } from '../../../../src/compiler/passes/normalize/explicit_broadcast.js';
import { IsolateRegionsPass } from '../../../../src/compiler/passes/normalize/isolate_regions.js';
import { MaterializeShapesPass } from '../../../../src/compiler/passes/normalize/materialize_shapes.js';
import { ShapeRefinementPass } from '../../../../src/compiler/passes/normalize/refine_shapes.js';
import { compileGraph } from '../../../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../../../src/compiler/support/target.js';
import { RuntimeTensor } from '../../../../src/runtime/runtime.js';

const f32 = (shape) => new TensorType(shape, 'f32');
const bool = new TensorType([], 'bool');
const errors = (func) => verifyFunction(func).map((e) => e.message);

function runCpu(func, inputs) {
  const outputs = func.outputTypes.map((type) => new Float32Array(type.shape.reduce((n, d) => n * d, 1)));
  const compiled = compileGraph(func, CPUTarget());
  compiled.run(func.name, ...inputs, ...outputs);
  return outputs.map((data) => [...data]);
}

describe('ExplicitBroadcastPass', () => {
  it.each([
    [[], [2, 3], []],
    [[3], [2, 3], [1]],
    [[2, 1], [2, 3], [0, 1]],
    [[2, 3], [2, 3], null],
    [[2, 3], [3], null],
    [[2], [2, 3], null],
    [[DYNAMIC, 1], [DYNAMIC, 4], [0, 1]],
  ])('maps %j onto %j as %j', (from, to, dims) => {
    expect(broadcastDimsFor(f32(from), f32(to))).toEqual(dims);
  });

  it('inserts broadcasts before consumers, preserves symbols and is idempotent', () => {
    const t = f32([DYNAMIC, 3]);
    const func = buildFunction('wide', [t, f32([3])], [t], (b, [x, bias]) => {
      x.symbolicShape = ['batch', 3];
      b.returnOp([b.add(x, bias).getResult(0)]);
    });
    const pass = new ExplicitBroadcastPass();
    expect(pass.run(func)).toBe(PassResult.CHANGED);
    const [wide, add] = func.opsArray();
    expect(wide.opName).toBe('broadcast_in_dim');
    expect(wide.getAttr('broadcast_dimensions')).toEqual([1]);
    expect(wide.getResult(0).symbolicShape).toEqual(['batch', 3]);
    expect(add.operands).toEqual([func.args[0], wide.getResult(0)]);
    expect(func.args[1].getUsers()).toEqual([wide]);
    expect(pass.run(func)).toBe(PassResult.UNCHANGED);
    expect(errors(func)).toEqual([]);
  });

  it('widens operands inside nested branches and retains predicate dtype', () => {
    const t = f32([2, 3]);
    const func = buildFunction('nested', [bool, bool, t, f32([])], [t], (b, [gate, pred, x, scalar]) => {
      const branch = b.ifOp(gate, null,
        (bb) => bb.yieldOp([bb.select(pred, x, scalar).getResult(0)]),
        (bb) => bb.yieldOp([x]));
      b.returnOp(branch.results);
    });
    new ExplicitBroadcastPass().run(func);
    const select = [...func.opsRecursive()].find((op) => op.opName === 'select');
    expect(select.operands.map((v) => v.type.shape)).toEqual([[2, 3], [2, 3], [2, 3]]);
    expect(select.getOperand(0).type.dtype).toBe('bool');
    expect(errors(func)).toEqual([]);
    const x = new Float32Array([1, -2, 3, -4, 5, -6]);
    const compiled = compileGraph(func, CPUTarget());
    const output = new Float32Array(6);
    for (const [gate, pred, expected] of [
      [1, 1, [1, -2, 3, -4, 5, -6]],
      [1, 0, [10, 10, 10, 10, 10, 10]],
      [0, 0, [1, -2, 3, -4, 5, -6]],
    ]) {
      compiled.run('nested', new Uint8Array([gate]), new Uint8Array([pred]), x, new Float32Array([10]), output);
      expect([...output], 'gate=' + gate + ', predicate=' + pred).toEqual(expected);
    }
  });
});

describe('IsolateRegionsPass', () => {
  it('lifts a shared capture once and sinks constants only into branches that use them', () => {
    const t = f32([2]);
    const func = buildFunction('branch', [bool, t], [t], (b, [pred, x]) => {
      const constant = b.constant(new Float32Array([2, 3]), t).getResult(0);
      const branch = b.ifOp(pred, null,
        (bb) => bb.yieldOp([bb.add(bb.add(x, x).getResult(0), constant).getResult(0)]),
        (bb) => bb.yieldOp([bb.neg(x).getResult(0)]));
      b.returnOp(branch.results);
    });
    const branch = func.findOp((op) => op.opName === 'if');
    const pass = new IsolateRegionsPass();
    expect(pass.run(func)).toBe(PassResult.CHANGED);
    expect(branch.operands).toEqual(func.args);
    expect(capturedValues(branch)).toEqual([]);
    const [thenBlock, elseBlock] = branch.regions.map((r) => r.entryBlock);
    expect(thenBlock.arguments).toHaveLength(1);
    expect(elseBlock.arguments).toHaveLength(1);
    expect(thenBlock.firstOp.opName).toBe('constant');
    expect(elseBlock.firstOp.opName).toBe('neg');
    const add = thenBlock.opsArray().find((op) => op.opName === 'add');
    expect(add.operands).toEqual([thenBlock.arguments[0], thenBlock.arguments[0]]);
    expect(pass.run(func)).toBe(PassResult.UNCHANGED);
    expect(errors(func)).toEqual([]);
    const compiled = compileGraph(func, CPUTarget());
    const output = new Float32Array(2);
    compiled.run('branch', new Uint8Array([1]), new Float32Array([4, -5]), output);
    expect([...output]).toEqual([10, -7]);
    compiled.run('branch', new Uint8Array([0]), new Float32Array([4, -5]), output);
    expect([...output]).toEqual([-4, 5]);
  });

  it('lifts captures through nested regions from the inside out', () => {
    const t = f32([2]);
    const func = buildFunction('nested', [bool, bool, t], [t], (b, [outerPred, innerPred, x]) => {
      const outer = b.ifOp(outerPred, null, (bb) => {
        const inner = bb.ifOp(innerPred, null,
          (ib) => ib.yieldOp([ib.neg(x).getResult(0)]),
          (ib) => ib.yieldOp([ib.abs(x).getResult(0)]));
        bb.yieldOp(inner.results);
      }, (bb) => bb.yieldOp([x]));
      b.returnOp(outer.results);
    });
    new IsolateRegionsPass().run(func);
    const branches = [...func.opsRecursive()].filter((op) => op.opName === 'if');
    expect(branches).toHaveLength(2);
    for (const branch of branches) expect(capturedValues(branch)).toEqual([]);
    expect(errors(func)).toEqual([]);
    const compiled = compileGraph(func, CPUTarget());
    const output = new Float32Array(2);
    for (const [outer, inner, expected] of [[1, 1, [-3, 4]], [1, 0, [3, 4]], [0, 1, [3, -4]]]) {
      compiled.run('nested', new Uint8Array([outer]), new Uint8Array([inner]), new Float32Array([3, -4]), output);
      expect([...output], 'outer=' + outer + ', inner=' + inner).toEqual(expected);
    }
  });

  it('appends scan captures after existing consts without changing carry/xs order', () => {
    const t = f32([2]);
    const func = buildFunction('scan', [t, f32([3, 2]), t, t], [t], (b, [carry, xs, weight, bias]) => {
      const scan = b.scanOp([carry], [xs], (bb, c, x, k) => {
        const next = bb.add(bb.add(c[0], x[0]).getResult(0), bb.mul(k[0], bias).getResult(0));
        return [[next.getResult(0)], []];
      }, [weight]);
      b.returnOp(scan.results);
    });
    new IsolateRegionsPass().run(func);
    const scan = func.findOp((op) => op.opName === 'scan');
    expect(scan.operands).toEqual(func.args);
    expect(scan.getAttr('num_consts')).toBe(2);
    expect(scan.regions[0].entryBlock.arguments).toHaveLength(4);
    expect(capturedValues(scan)).toEqual([]);
    expect(errors(func)).toEqual([]);
    expect(runCpu(func, [
      new Float32Array([10, 20]), new Float32Array([1, 2, 3, 4, 5, 6]),
      new Float32Array([2, 3]), new Float32Array([4, 5]),
    ])).toEqual([[43, 77]]);
  });
});

function dynamicReshape() {
  return buildFunction('reshape', [f32([DYNAMIC, 2])], [f32([2, DYNAMIC])], (b, [x]) => {
    x.symbolicShape = ['batch', 2];
    const reshape = b.reshape(x, [2, DYNAMIC]);
    reshape.getResult(0).symbolicShape = [2, 'batch'];
    b.returnOp(reshape.results);
  });
}

describe('MaterializeShapesPass', () => {
  it('uses the source axis of a symbol when reshape moves it to another axis', () => {
    const func = dynamicReshape();
    const pass = new MaterializeShapesPass();
    expect(pass.run(func)).toBe(PassResult.CHANGED);
    const [dim, reshape] = func.opsArray();
    expect(dim.opName).toBe('dim');
    expect(dim.getOperand(0)).toBe(func.args[0]);
    expect(dim.getAttr('dimension')).toBe(0);
    expect(dim.getResult(0).type.equals(new TensorType([], 'i64'))).toBe(true);
    expect(reshape.operands).toEqual([func.args[0], dim.getResult(0)]);
    expect(pass.run(func)).toBe(PassResult.UNCHANGED);
    expect(errors(func)).toEqual([]);
  });

  it('obtains a broadcast size from an earlier operand of its consumer', () => {
    const t = f32([DYNAMIC, 2]);
    const func = buildFunction('broadcast', [t], [t], (b, [x]) => {
      const wide = b.splat(1, t);
      b.returnOp([b.add(x, wide.getResult(0)).getResult(0)]);
    });
    new MaterializeShapesPass().run(func);
    const wide = func.findOp((op) => op.opName === 'broadcast_in_dim');
    const dim = wide.getOperand(1).definingOp;
    expect(dim.getOperand(0)).toBe(func.args[0]);
    expect(dim.getAttr('dimension')).toBe(0);
    expect(func.opsArray().indexOf(dim)).toBeLessThan(func.opsArray().indexOf(wide));
    expect(errors(func)).toEqual([]);
    const compiled = compileGraph(func, CPUTarget());
    for (const rows of [2, 5]) {
      const input = Float32Array.from({ length: rows * 2 }, (_, i) => i - 4);
      const output = new Float32Array(rows * 2).fill(NaN);
      compiled.run('broadcast', new RuntimeTensor(input, [rows, 2], 'f32'), new RuntimeTensor(output, [rows, 2], 'f32'));
      expect([...output]).toEqual([...input].map((x) => x + 1));
    }
  });

  it('reuses a dim value for repeated symbols on different results', () => {
    const step = f32([DYNAMIC]);
    const func = buildFunction('scan_sizes', [step, f32([3, DYNAMIC])], [step, f32([3, DYNAMIC])], (b, [c, xs]) => {
      c.symbolicShape = ['width'];
      xs.symbolicShape = [3, 'width'];
      const scan = b.scanOp([c], [xs], (bb, carry, x) => [[carry[0]], [x[0]]]);
      scan.getResult(0).symbolicShape = ['width'];
      scan.getResult(1).symbolicShape = [3, 'width'];
      b.returnOp(scan.results);
    });
    new MaterializeShapesPass().run(func);
    const scan = func.findOp((op) => op.opName === 'scan');
    expect(func.opsArray().filter((op) => op.opName === 'dim')).toHaveLength(1);
    expect(scan.getOperand(2)).toBe(scan.getOperand(3));
    expect(scan.getAttr('num_consts')).toBe(0);
    expect(errors(func)).toEqual([]);
  });

  it('reports an unresolvable result extent', () => {
    const t = f32([DYNAMIC]);
    const func = buildFunction('unknown', [], [t], (b) => b.returnOp(b.iota(0, t).results));
    expect(() => new MaterializeShapesPass().run(func)).toThrow(/iota.*result 0 axis 0.*nothing in scope carries/);
  });
});

describe('ShapeRefinementPass', () => {
  it('specializes moved dimensions, derived attrs and the function signature', () => {
    const func = dynamicReshape();
    new MaterializeShapesPass().run(func);
    const dim = func.findOp((op) => op.opName === 'dim');
    const pass = new ShapeRefinementPass([[5, 2]]);
    expect(pass.run(func)).toBe(PassResult.CHANGED);
    const reshape = func.findOp((op) => op.opName === 'reshape');
    expect(func.inputTypes[0].shape).toEqual([5, 2]);
    expect(func.args[0].type.shape).toEqual([5, 2]);
    expect(func.outputTypes[0].shape).toEqual([2, 5]);
    expect(reshape.getAttr('new_shape')).toEqual([2, 5]);
    expect(reshape.numOperands).toBe(1);
    expect(dim.parentBlock).toBeNull();
    expect(func.args[0].useCount).toBe(1);
    expect(pass.run(func)).toBe(PassResult.UNCHANGED);
    expect(errors(func)).toEqual([]);
    const input = new Float32Array([1, -2, 3, 4, -5, 6, 7, -8, 9, 10]);
    expect(runCpu(func, [input])).toEqual([[1, -2, 3, 4, -5, 6, 7, -8, 9, 10]]);
  });

  it('refines explicit branch arguments and propagates yielded result types', () => {
    const t = f32([DYNAMIC, 2]);
    const func = buildFunction('branch', [bool, t], [t], (b, [pred, x]) => {
      const branch = b.ifOp(pred, null,
        (bb, [arg]) => bb.yieldOp([bb.neg(arg).getResult(0)]),
        (bb, [arg]) => bb.yieldOp([bb.abs(arg).getResult(0)]), [x]);
      b.returnOp(branch.results);
    });
    new ShapeRefinementPass([[], [7, 2]]).run(func);
    const branch = func.findOp((op) => op.opName === 'if');
    for (const region of branch.regions) expect(region.entryBlock.arguments[0].type.shape).toEqual([7, 2]);
    expect(branch.getResult(0).type.shape).toEqual([7, 2]);
    expect(func.outputTypes[0].shape).toEqual([7, 2]);
    expect(errors(func)).toEqual([]);
    const input = Float32Array.from({ length: 14 }, (_, i) => i - 8);
    expect(runCpu(func, [new Uint8Array([0]), input])).toEqual([[8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5]]);
  });

  it('specializes scan carry, step, const and stacked result types', () => {
    const step = f32([DYNAMIC]);
    const func = buildFunction('scan', [step, f32([3, DYNAMIC]), step], [step, f32([3, DYNAMIC])], (b, [c, xs, k]) => {
      const scan = b.scanOp([c], [xs], (bb, carry, x, constants) => {
        const next = bb.add(carry[0], bb.mul(x[0], constants[0]).getResult(0)).getResult(0);
        return [[next], [next]];
      }, [k]);
      b.returnOp(scan.results);
    });
    new ShapeRefinementPass([[4], [3, 4], [4]]).run(func);
    const scan = func.findOp((op) => op.opName === 'scan');
    expect(scan.regions[0].entryBlock.arguments.map((a) => a.type.shape)).toEqual([[4], [4], [4]]);
    expect(func.outputTypes.map((t) => t.shape)).toEqual([[4], [3, 4]]);
    expect(errors(func)).toEqual([]);
    expect(runCpu(func, [
      new Float32Array([10, 20, 30, 40]),
      new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
      new Float32Array([2, -1, 0, 3]),
    ])).toEqual([[40, 2, 30, 112], [12, 18, 30, 52, 22, 12, 30, 76, 40, 2, 30, 112]]);
  });

  it.each([
    [[], /takes 1 arguments but 0 shapes/],
    [[[3]], /rank 2.*rank 1/],
    [[[3, 9]], /axis 1.*contradicts/],
    [[[DYNAMIC, 2]], /still has a dynamic extent/],
  ])('rejects incompatible shapes %j', (shapes, message) => {
    expect(() => new ShapeRefinementPass(shapes).run(dynamicReshape())).toThrow(message);
  });

  it('keeps a size producer that has another live consumer', () => {
    const func = new GraphFunction('shared_size', [f32([DYNAMIC])], [f32([DYNAMIC]), new TensorType([], 'i64')]);
    const b = new IRBuilder(func);
    const size = b.dim(func.args[0], 0).getResult(0);
    const result = b.broadcast(b.scalarConstant(1, 'f32').getResult(0), [DYNAMIC], [], [size]);
    b.returnOp([result.getResult(0), size]);
    new ShapeRefinementPass([[6]]).run(func);
    expect(result.numOperands).toBe(1);
    expect(result.getAttr('result_shape')).toEqual([6]);
    expect(size.definingOp.parentBlock).toBe(func.entryBlock);
    expect(size.useCount).toBe(1);
    expect(errors(func)).toEqual([]);
  });
});
