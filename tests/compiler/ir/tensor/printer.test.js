import { describe, it, expect } from 'vitest';
import { printTensorIR } from '../../../../src/compiler/ir/tensor/printer.js';
import { Buffer } from '../../../../src/compiler/ir/tensor/buffer.js';
import {
  TensorNode, PrimFunc, ForNode, BlockNode, BlockRealizeNode, SeqNode,
  BufferStoreNode, BufferLoadNode, IfThenElseNode, LetStmtNode, AllocateNode,
  EvaluateNode, WhileNode, SyncThreadsNode, VecCopyNode,
  MathOpNode, CompareNode, CastNode, CallExternNode,
  VariableNode, IntImmNode, FloatImmNode,
  ForKind, IterVarKind,
} from '../../../../src/compiler/ir/tensor/nodes.js';

function buf(name, shape, dtype = 'f32') {
  return new Buffer(name, shape, dtype, 'global');
}

function idx(name) {
  return new VariableNode(name, 'index');
}

function lines(node) {
  return printTensorIR(node).split('\n');
}

describe('printTensorIR renders a prim_func', () => {
  it('prints the signature, the buffer map and the body at increasing indentation', () => {
    const x = buf('x', [4]);
    const y = buf('y', [4]);
    const argX = idx('arg_x');
    const argY = idx('arg_y');
    const loop = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL,
      new BufferStoreNode(y, [idx('i')], new BufferLoadNode(x, [idx('i')])));
    const func = new PrimFunc('copy', [argX, argY], loop, new Map([[argX, x], [argY, y]]));

    expect(lines(func)).toEqual([
      'prim_func copy(arg_x, arg_y) {',
      '  x = buffer_map(arg_x, shape=[4], dtype=f32)',
      '  y = buffer_map(arg_y, shape=[4], dtype=f32)',
      '  for i in 0..4 {',
      '    y[i] = x[i]',
      '  }',
      '}',
    ]);
  });

  it('prints a statement sequence one statement per line', () => {
    const y = buf('y', [2]);
    const seq = new SeqNode([
      new BufferStoreNode(y, [new IntImmNode(0)], new FloatImmNode(1.5)),
      new BufferStoreNode(y, [new IntImmNode(1)], new FloatImmNode(2.5)),
    ]);

    expect(lines(seq)).toEqual(['y[0] = 1.5', 'y[1] = 2.5']);
  });
});

describe('printTensorIR renders loops with their schedule annotation', () => {
  const body = new EvaluateNode(new IntImmNode(0));

  it('leaves a serial loop unannotated', () => {
    const loop = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, body);
    expect(lines(loop)[0]).toBe('for i in 0..8 {');
  });

  it('annotates parallel and vectorized loops', () => {
    const parallel = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(8), ForKind.PARALLEL, body);
    const vectorized = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(8), ForKind.VECTORIZED, body);

    expect(lines(parallel)[0]).toBe('for i in 0..8 @parallel {');
    expect(lines(vectorized)[0]).toBe('for i in 0..8 @vectorized {');
  });

  it('adds the thread tag of a thread-bound loop', () => {
    const loop = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(8), ForKind.THREAD_BINDING, body, 'threadIdx.x');
    expect(lines(loop)[0]).toBe('for i in 0..8 @thread_binding [threadIdx.x] {');
  });

  it('prints a symbolic extent as an expression', () => {
    const extent = new MathOpNode('*', idx('n'), new IntImmNode(4));
    const loop = new ForNode(idx('i'), new IntImmNode(0), extent, ForKind.SERIAL, body);
    expect(lines(loop)[0]).toBe('for i in 0..(n * 4) {');
  });
});

describe('printTensorIR renders a block with its bindings and regions', () => {
  it('prints iter var bindings, reads, writes and the init body', () => {
    const x = buf('x', [4]);
    const y = buf('y', [1]);
    const spatial = new BlockRealizeNode(idx('v'), idx('i'), IterVarKind.DATA_PAR);
    const reduce = new BlockRealizeNode(idx('r'), idx('k'), IterVarKind.COMM_REDUCE);
    const block = new BlockNode(
      'sum_block',
      [spatial, reduce],
      [{ buffer: x }],
      [{ buffer: y }],
      new BufferStoreNode(y, [new IntImmNode(0)], new BufferLoadNode(x, [idx('r')])),
      new BufferStoreNode(y, [new IntImmNode(0)], new FloatImmNode(0)),
    );

    expect(lines(block)).toEqual([
      'block sum_block {',
      '  bind v = i',
      '  bind r:CommReduce = k',
      '  reads([x[...]])',
      '  writes([y[...]])',
      '  init {',
      '    y[0] = 0',
      '  }',
      '  y[0] = x[r]',
      '}',
    ]);
  });

  it('prints a standalone binding without inventing a block around it', () => {
    const binding = new BlockRealizeNode(idx('v'), new MathOpNode('+', idx('i'), new IntImmNode(1)));
    expect(printTensorIR(binding)).toBe('bind v = (i + 1)');
  });
});

describe('printTensorIR renders the remaining statement nodes', () => {
  it('prints both arms of an if/else', () => {
    const y = buf('y', [1]);
    const node = new IfThenElseNode(
      new CompareNode('lt', idx('i'), new IntImmNode(4)),
      new BufferStoreNode(y, [new IntImmNode(0)], new FloatImmNode(1)),
      new BufferStoreNode(y, [new IntImmNode(0)], new FloatImmNode(0)),
    );

    expect(lines(node)).toEqual([
      'if ((i < 4)) {',
      '  y[0] = 1',
      '} else {',
      '  y[0] = 0',
      '}',
    ]);
  });

  it('prints a let binding above the body it scopes', () => {
    const y = buf('y', [1]);
    const t = new VariableNode('t', 'f32');
    const node = new LetStmtNode(t, new FloatImmNode(2), new BufferStoreNode(y, [new IntImmNode(0)], t));

    expect(lines(node)).toEqual(['let t = 2', 'y[0] = t']);
  });

  it('prints an allocation with its shape and scope', () => {
    const tmp = new Buffer('tmp', [4, 8], 'f32', 'shared');
    const node = new AllocateNode(tmp, 'shared', new EvaluateNode(new IntImmNode(0)));

    expect(lines(node)).toEqual(['allocate tmp[4, 8] (shared) {', '  evaluate 0', '}']);
  });

  it('prints a while loop with the statement that recomputes its condition', () => {
    const flag = buf('flag', [1], 'i32');
    const y = buf('y', [1]);
    const node = new WhileNode(
      flag,
      new BufferStoreNode(flag, [new IntImmNode(0)], new CompareNode('lt', idx('i'), new IntImmNode(4))),
      new BufferStoreNode(y, [new IntImmNode(0)], new FloatImmNode(1)),
    );

    expect(lines(node)).toEqual([
      'while flag {',
      '  cond {',
      '    flag[0] = (i < 4)',
      '  }',
      '  y[0] = 1',
      '}',
    ]);
  });

  it('prints a vector copy with its width, and a thread barrier', () => {
    const dst = buf('dst', [16]);
    const src = buf('src', [16]);
    const copy = new VecCopyNode(dst, idx('i'), src, idx('j'), 4);

    expect(printTensorIR(copy)).toBe('dst[i] = vec_copy<4>(src[j])');
    expect(printTensorIR(new SyncThreadsNode())).toBe('sync_threads()');
  });
});

describe('printTensorIR renders expressions', () => {
  it('parenthesises binary math and keeps the operator of unary math', () => {
    const binary = new MathOpNode('+', new MathOpNode('*', idx('i'), new IntImmNode(2)), idx('j'));
    const unary = new MathOpNode('-', idx('i'));

    expect(printTensorIR(binary)).toBe('((i * 2) + j)');
    expect(printTensorIR(unary)).toBe('(-i)');
  });

  it('prints comparisons with their C operator', () => {
    expect(printTensorIR(new CompareNode('ge', idx('i'), new IntImmNode(0)))).toBe('(i >= 0)');
    expect(printTensorIR(new CompareNode('ne', idx('i'), idx('j')))).toBe('(i != j)');
  });

  it('prints casts, extern calls and multi-index buffer loads', () => {
    const x = buf('x', [4, 8]);
    const cast = new CastNode(idx('i'), 'i32', 'f32');
    const call = new CallExternNode('tanh', [new BufferLoadNode(x, [idx('i'), idx('j')])], 'f32');

    expect(printTensorIR(cast)).toBe('cast<f32>(i)');
    expect(printTensorIR(call)).toBe('tanh(x[i, j])');
  });
});

describe('printTensorIR reports a node it cannot render', () => {
  it('names the unhandled node type instead of printing nothing', () => {
    class MysteryNode extends TensorNode {}
    expect(printTensorIR(new MysteryNode())).toBe('[UnknownNode: MysteryNode]');
  });

  it('prints nothing at all for a missing child', () => {
    expect(printTensorIR(null)).toBe('');
  });
});
