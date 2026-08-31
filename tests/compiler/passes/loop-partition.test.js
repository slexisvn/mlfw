import { describe, it, expect } from 'vitest';
import {
  PrimFunc, BlockNode, ForNode, BufferStoreNode, BufferLoadNode,
  IfThenElseNode, MathOpNode, VariableNode, IntImmNode, FloatImmNode, ForKind,
} from '../../../src/compiler/ir/tensor/nodes.js';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import { Schedule } from '../../../src/compiler/schedule/schedule.js';
import { LoopPartitionPass } from '../../../src/compiler/passes/loop_partition/loop_partition.js';
import { BackendPipeline } from '../../../src/backend/pipeline.js';
import { CPUTarget } from '../../../src/compiler/support/target.js';
import { collect } from '../../../src/compiler/ir/ir_visitor.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';

const v = (n) => new VariableNode(n, 'int32');
const c = (n) => new IntImmNode(n);

function scaleBlockFunc(name, E) {
  const In = new Buffer('A', [E], 'f32', 'global');
  const Out = new Buffer('C', [E], 'f32', 'global');
  const x = v('x');
  const store = new BufferStoreNode(Out, [x], new MathOpNode('*', new BufferLoadNode(In, [x]), new FloatImmNode(2)));
  const block = new BlockNode('e', [{ iterVar: x, binding: x }], [{ buffer: In }], [{ buffer: Out }], store);
  const loop = new ForNode(x, c(0), c(E), ForKind.SERIAL, block);
  return new PrimFunc(name, [], loop, new Map([['A', In], ['C', Out]]));
}

function compileRun1(func, inArr, E) {
  const src = new BackendPipeline(CPUTarget()).compile(func).source;
  const fn = new Function('return ' + src)();
  const out = new Float32Array(E);
  fn(inArr, out);
  return out;
}

function manualSplitGuard(E, F) {
  const In = new Buffer('A', [E], 'f32', 'global');
  const Out = new Buffer('C', [E], 'f32', 'global');
  const o = v('o');
  const i = v('i');
  const flat = new MathOpNode('+', new MathOpNode('*', o, c(F)), i);
  const store = new BufferStoreNode(Out, [flat], new MathOpNode('*', new BufferLoadNode(In, [flat]), new FloatImmNode(2)));
  const guard = new MathOpNode('<', flat, c(E));
  const inner = new ForNode(i, c(0), c(F), ForKind.SERIAL, new IfThenElseNode(guard, store));
  const outer = new ForNode(o, c(0), c(Math.ceil(E / F)), ForKind.SERIAL, inner);
  return new PrimFunc('manual', [], outer, new Map([['A', In], ['C', Out]]));
}

describe('LoopPartitionPass', () => {
  it('splits a split-guard loop into a guard-free main loop + epilogue', () => {
    const E = 10, F = 4;
    const fn = manualSplitGuard(E, F);
    new LoopPartitionPass().run(fn, {});

    expect(fn.body.type).toBe('SeqNode');
    expect(fn.body.stmts).toHaveLength(2);

    const [main, epi] = fn.body.stmts;
    expect(main.type).toBe('ForNode');
    expect(main.extent.value).toBe(Math.floor(E / F));      // 2
    expect(main.body.type).toBe('ForNode');
    expect(main.body.extent.value).toBe(F);                 // 4
    expect(epi.type).toBe('ForNode');
    expect(epi.extent.value).toBe(E % F);                   // 2

    expect(collect(fn.body, (n) => n.type === 'IfThenElseNode')).toHaveLength(0);
  });

  it('the partitioned loop computes the same values as the guarded loop', () => {
    const E = 10, F = 4;
    const input = Float32Array.from({ length: E }, (_, k) => Math.sin(k * 1.3));
    const ref = input.map((x) => x * 2);

    const guarded = manualSplitGuard(E, F);
    const outGuarded = compileRun1(guarded, input, E);

    const partitioned = manualSplitGuard(E, F);
    new LoopPartitionPass().run(partitioned, {});
    const outPartitioned = compileRun1(partitioned, input, E);

    for (let k = 0; k < E; k++) {
      expect(Math.abs(outGuarded[k] - ref[k])).toBeLessThan(1e-5);
      expect(Math.abs(outPartitioned[k] - ref[k])).toBeLessThan(1e-5);
    }
  });

  it('matches a real Schedule.split remainder guard and stays numerically equivalent', () => {
    const E = 13, F = 4;
    const input = Float32Array.from({ length: E }, (_, k) => Math.cos(k * 0.9));
    const ref = input.map((x) => x * 2);

    const a = new Schedule(scaleBlockFunc('a', E));
    a.split(a.getLoops('e')[0], F);
    const outGuarded = compileRun1(a.func, input, E);

    const b = new Schedule(scaleBlockFunc('b', E));
    b.split(b.getLoops('e')[0], F);
    new LoopPartitionPass().run(b.func, {});

    expect(b.func.body.type).toBe('SeqNode');
    expect(collect(b.func.body, (n) => n.type === 'IfThenElseNode')).toHaveLength(0);

    const outPartitioned = compileRun1(b.func, input, E);
    for (let k = 0; k < E; k++) {
      expect(Math.abs(outGuarded[k] - ref[k])).toBeLessThan(1e-5);
      expect(Math.abs(outPartitioned[k] - ref[k])).toBeLessThan(1e-5);
    }
  });

  it('is a no-op when there is no bounds guard (divisible split)', () => {
    const E = 12, F = 4;
    const s = new Schedule(scaleBlockFunc('d', E));
    s.split(s.getLoops('e')[0], F);
    const before = s.func.body.type;
    new LoopPartitionPass().run(s.func, {});
    expect(s.func.body.type).toBe(before);
    expect(s.func.body.type).not.toBe('SeqNode');
  });

  it('is opt-in: the flag is off by default and does not change a normal compile', () => {
    const t = new TensorType([10], ScalarType.F32);
    const mk = (name) => buildFunction(name, [t], [t], (b, args) => {
      b.returnOp([b.relu(args[0]).getResult(0)]);
    });
    const input = Float32Array.from({ length: 10 }, (_, k) => k - 5);

    const off = compileGraph(mk('off'), CPUTarget());
    const on = compileGraph(mk('on'), CPUTarget(), { scheduling: { enabled: true }, optimization: { loopPartition: true } });

    const o1 = new Float32Array(10), o2 = new Float32Array(10);
    off.run('off', input, o1);
    on.run('on', input, o2);
    expect(Array.from(o2)).toEqual(Array.from(o1));
  });
});
