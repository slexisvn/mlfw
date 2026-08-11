import { describe, it, expect } from 'vitest';
import {
  PrimFunc, BlockNode, ForNode, BufferStoreNode, BufferLoadNode,
  MathOpNode, VariableNode, IntImmNode, FloatImmNode, ForKind,
} from '../../../../src/compiler/ir/tensor/nodes.js';
import { Buffer } from '../../../../src/compiler/ir/tensor/buffer.js';
import { AccumulatorDetectionPass } from '../../../../src/compiler/passes/lowering/accumulator_pass.js';
import { detectAccumulator } from '../../../../src/compiler/passes/lowering/accumulator.js';
import { lowerToLIR } from '../../../../src/compiler/passes/lowering/tensor_to_lir.js';
import { WasmTarget, CPUTarget } from '../../../../src/backend/target.js';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../../../src/compiler/pipeline/compiler.js';

const v = (n) => new VariableNode(n, 'int32');
const c = (n) => new IntImmNode(n);

function reduceFunc(name) {
  const accBuf = new Buffer('acc', [4], 'f32', 'global');
  const inBuf = new Buffer('inp', [4, 8], 'f32', 'global');
  const load = new BufferLoadNode(accBuf, [v('i')]);
  const inLoad = new BufferLoadNode(inBuf, [v('i'), v('j')]);
  const add = new MathOpNode('+', load, inLoad);
  const store = new BufferStoreNode(accBuf, [v('i')], add);
  const block = new BlockNode('red', [{ iterVar: v('i'), binding: v('outer_i') }],
    [{ buffer: inBuf }], [{ buffer: accBuf }], store);
  const innerLoop = new ForNode(v('j'), c(0), c(8), ForKind.SERIAL, block);
  const outerLoop = new ForNode(v('outer_i'), c(0), c(4), ForKind.SERIAL, innerLoop);
  return new PrimFunc(name, ['p0', 'p1'], outerLoop, new Map([['p0', inBuf], ['p1', accBuf]]));
}

describe('AccumulatorDetectionPass', () => {
  it('annotates the reduction loop and a null for the outer loop', () => {
    const pf = reduceFunc('reduce');
    const outer = pf.body;
    const inner = outer.body;
    new AccumulatorDetectionPass().run(pf, {});

    expect(outer.accumulator).toBeNull();
    expect(inner.accumulator).not.toBeNull();
    expect(inner.accumulator.op).toBe('+');
  });

  it('annotation produces the SAME LIRAccumulatorNode as the inline fallback', () => {
    const annotated = reduceFunc('a');
    new AccumulatorDetectionPass().run(annotated, {});
    const lirA = lowerToLIR(annotated, WasmTarget());

    const plain = reduceFunc('b');
    const lirB = lowerToLIR(plain, WasmTarget());

    const accA = lirA.body.body;
    const accB = lirB.body.body;
    expect(accA.type).toBe('LIRAccumulatorNode');
    expect(accB.type).toBe('LIRAccumulatorNode');
    expect(accA.localName).toBe(accB.localName);
    expect(accA.op).toBe(accB.op);
    expect(accA.dtype).toBe(accB.dtype);
  });

  it('matches the standalone detector decision exactly', () => {
    const pf = reduceFunc('c');
    const inner = pf.body.body;
    const direct = detectAccumulator(inner);
    new AccumulatorDetectionPass().run(pf, {});
    expect(!!pf.body.body.accumulator).toBe(!!direct);
  });

  it('is opt-in: detectAccumulators flag does not change compiled output', () => {
    const inT = new TensorType([4, 8], ScalarType.F32);
    const outT = new TensorType([4], ScalarType.F32);
    const mk = (name) => buildFunction(name, [inT], [outT], (b, a) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(a[0], zero.getResult(0), [1], 'sum').getResult(0)]);
    });
    const input = Float32Array.from({ length: 32 }, (_, k) => Math.sin(k * 0.5));

    const off = compileGraph(mk('off'), CPUTarget());
    const on = compileGraph(mk('on'), CPUTarget(), { optimization: { detectAccumulators: true } });

    const o1 = new Float32Array(4), o2 = new Float32Array(4);
    off.run('off', input, o1);
    on.run('on', input, o2);
    expect(Array.from(o2)).toEqual(Array.from(o1));
  });
});
