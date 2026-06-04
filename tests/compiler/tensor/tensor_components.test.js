import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { Buffer, BufferRegion } from '../../../src/compiler/ir/tensor/buffer.js';
import { PrimFunc, ForNode, BlockNode, BufferStoreNode, BufferLoadNode, MathOpNode, IntImmNode, FloatImmNode, VariableNode, BlockRealizeNode, IfThenElseNode, LetStmtNode, AllocateNode, CallExternNode } from '../../../src/compiler/ir/tensor/nodes.js';
import { printTensorIR } from '../../../src/compiler/ir/tensor/printer.js';
import { TensorVerifier } from '../../../src/compiler/ir/tensor/verifier.js';
import { MemoryScope } from '../../../src/compiler/ir/tensor/tensor_types.js';

describe('TensorIR Components', () => {
  it('builds, prints, and verifies a basic TensorIR AST', () => {
    const bufA = new Buffer('A', [10], 'f32', MemoryScope.GLOBAL);
    const bufB = new Buffer('B', [10], 'f32', MemoryScope.GLOBAL);
    const varI = new VariableNode('i', 'int32');
    const iterI = new VariableNode('vi', 'int32');
    const bindI = new BlockRealizeNode(iterI, varI);
    const loadA = new BufferLoadNode(bufA, [iterI]);
    const math = new MathOpNode('+', loadA, new IntImmNode(1));
    const storeB = new BufferStoreNode(bufB, [iterI], math);
    const block = new BlockNode('block_b', [bindI], [{buffer: bufA}], [{buffer: bufB}], storeB);
    const loop = new ForNode(varI, new IntImmNode(0), new IntImmNode(10), 'serial', block);
    const func = new PrimFunc('test_func', [], loop, new Map());
    assert.equal(new TensorVerifier().verify(func).length, 0);
    const printed = printTensorIR(func);
    assert.ok(printed.includes('prim_func test_func()'));
    assert.ok(printed.includes('for i in 0..10'));
    assert.ok(printed.includes('block block_b'));
    assert.ok(printed.includes('B[vi] = (A[vi] + 1)'));
  });

  it('verifier detects unbound loop variables', () => {
    const varI = new VariableNode('i', 'int32');
    const func = new PrimFunc('bad', [], new BufferLoadNode(new Buffer('A', [1], 'f32', 'global'), [varI]), new Map());
    const errors = new TensorVerifier().verify(func);
    assert.equal(errors.length, 1);
    assert.ok(errors[0].includes('Unbound variable'));
  });
});

describe('TensorVerifier node coverage', () => {
  it('IfThenElseNode', () => {
    const v = new VariableNode('i', 'int32');
    const buf = new Buffer('A', [10], 'f32', 'global');
    const ifN = new IfThenElseNode(new MathOpNode('<', v, new IntImmNode(5)),
      new BufferStoreNode(buf, [v], new FloatImmNode(1)),
      new BufferStoreNode(buf, [v], new FloatImmNode(0)));
    const func = new PrimFunc('t', [], new ForNode(v, new IntImmNode(0), new IntImmNode(10), 'serial', ifN), new Map());
    assert.equal(new TensorVerifier().verify(func).length, 0);
  });

  it('LetStmtNode', () => {
    const v = new VariableNode('i', 'int32');
    const j = new VariableNode('j', 'int32');
    const buf = new Buffer('A', [10], 'f32', 'global');
    const letN = new LetStmtNode(j, new IntImmNode(42), new BufferStoreNode(buf, [j], new FloatImmNode(1)));
    const func = new PrimFunc('t', [], new ForNode(v, new IntImmNode(0), new IntImmNode(10), 'serial', letN), new Map());
    assert.equal(new TensorVerifier().verify(func).length, 0);
  });

  it('AllocateNode', () => {
    const buf = new Buffer('tmp', [8], 'f32', 'local');
    const v = new VariableNode('i', 'int32');
    const alloc = new AllocateNode(buf, 'local', new ForNode(v, new IntImmNode(0), new IntImmNode(8), 'serial', new BufferStoreNode(buf, [v], new FloatImmNode(0))));
    assert.equal(new TensorVerifier().verify(new PrimFunc('t', [], alloc, new Map())).length, 0);
  });

  it('CallExternNode', () => {
    const v = new VariableNode('i', 'int32');
    const buf = new Buffer('A', [10], 'f32', 'global');
    const call = new CallExternNode('exp', [new BufferLoadNode(buf, [v])], 'f32');
    const func = new PrimFunc('t', [], new ForNode(v, new IntImmNode(0), new IntImmNode(10), 'serial', new BufferStoreNode(buf, [v], call)), new Map());
    assert.equal(new TensorVerifier().verify(func).length, 0);
  });
});

describe('TensorIRPrinter node coverage', () => {
  it('IfThenElseNode', () => {
    const p = printTensorIR(new IfThenElseNode(new MathOpNode('<', new VariableNode('i', 'int32'), new IntImmNode(5)), new FloatImmNode(1)));
    assert.ok(p.includes('if'));
  });

  it('LetStmtNode', () => {
    const p = printTensorIR(new LetStmtNode(new VariableNode('x', 'int32'), new IntImmNode(42), new FloatImmNode(0)));
    assert.ok(p.includes('let x = 42'));
  });

  it('AllocateNode', () => {
    const p = printTensorIR(new AllocateNode(new Buffer('tmp', [8, 8], 'f32', 'shared'), 'shared', new FloatImmNode(0)));
    assert.ok(p.includes('allocate tmp'));
  });

  it('thread binding', () => {
    const p = printTensorIR(new ForNode(new VariableNode('i', 'int32'), new IntImmNode(0), new IntImmNode(32), 'thread_binding', new FloatImmNode(0), 'threadIdx.x'));
    assert.ok(p.includes('@thread_binding'));
    assert.ok(p.includes('[threadIdx.x]'));
  });
});

describe('Buffer', () => {
  it('numel and sizeInBytes', () => {
    const buf = new Buffer('A', [4, 8], 'f32', 'global');
    assert.equal(buf.numel(), 32);
    assert.equal(buf.sizeInBytes(), 128);
  });

  it('scalar buffer', () => {
    const buf = new Buffer('s', [], 'f32', 'global');
    assert.equal(buf.isScalar, true);
    assert.equal(buf.numel(), 1);
    assert.equal(buf.sizeInBytes(), 4);
  });

  it('dtype sizes', () => {
    assert.equal(new Buffer('a', [10], 'f16', 'global').sizeInBytes(), 20);
    assert.equal(new Buffer('a', [10], 'f64', 'global').sizeInBytes(), 80);
    assert.equal(new Buffer('a', [10], 'i8', 'global').sizeInBytes(), 10);
  });
});
