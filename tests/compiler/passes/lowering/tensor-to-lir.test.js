import { describe, it, expect } from 'vitest';
import { lowerToLIR } from '../../../../src/compiler/passes/lowering/tensor_to_lir.js';
import { Buffer } from '../../../../src/compiler/ir/tensor/buffer.js';
import {
  PrimFunc, ForNode, BlockNode, SeqNode,
  BufferStoreNode, BufferLoadNode, AllocateNode,
  LetStmtNode, CallExternNode, IfThenElseNode, WhileNode, EvaluateNode,
  MathOpNode, CompareNode, CastNode,
  VariableNode, IntImmNode, FloatImmNode,
  ForKind,
} from '../../../../src/compiler/ir/tensor/nodes.js';
import { WasmTarget } from '../../../../src/compiler/support/target.js';

function buf(name, shape, dtype = 'f32') {
  return new Buffer(name, shape, dtype, 'global');
}

function idx(name) {
  return new VariableNode(name, 'index');
}

function makePrimFunc(name, params, body, bufferMap, shapeParams = []) {
  return new PrimFunc(name, params, body, bufferMap, shapeParams);
}

describe('lowerToLIR — basic structure', () => {
  it('returns LIRFunc with correct name and params', () => {
    const a = buf('a', [4]);
    const b = buf('b', [4]);
    const load = new BufferLoadNode(a, [idx('i')]);
    const store = new BufferStoreNode(b, [idx('i')], load);
    const block = new BlockNode('blk', [], [{ buffer: a }], [{ buffer: b }], store);
    const loop = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    const pf = makePrimFunc('copy', ['p0', 'p1'], loop, new Map([['p0', a], ['p1', b]]));

    const lir = lowerToLIR(pf, WasmTarget());
    expect(lir.type).toBe('LIRFunc');
    expect(lir.name).toBe('copy');
    expect(lir.params).toEqual(['p0', 'p1']);
  });

  it('preserves bufferMap and shapeParams', () => {
    const a = buf('a', [4]);
    const store = new BufferStoreNode(a, [idx('i')], new FloatImmNode(0));
    const block = new BlockNode('blk', [], [], [{ buffer: a }], store);
    const loop = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    const pf = makePrimFunc('test', ['p0'], loop, new Map([['p0', a]]));

    const lir = lowerToLIR(pf, WasmTarget());
    expect(lir.bufferMap.get('p0').name).toBe('a');
    expect(lir.shapeParams).toEqual([]);
  });

  it('includes metadata with locals, memoryLayout, etc.', () => {
    const a = buf('a', [4]);
    const store = new BufferStoreNode(a, [idx('i')], new FloatImmNode(0));
    const block = new BlockNode('blk', [], [], [{ buffer: a }], store);
    const loop = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    const pf = makePrimFunc('test', ['p0'], loop, new Map([['p0', a]]));

    const lir = lowerToLIR(pf, WasmTarget());
    expect(lir.metadata.locals.has('i')).toBe(true);
    expect(lir.metadata.memoryLayout.totalBytes).toBeGreaterThan(0);
  });
});

describe('lowerToLIR — BufferStore/Load → LIRFlatStore/Load', () => {
  it('converts BufferStoreNode to LIRFlatStoreNode', () => {
    const a = buf('a', [4]);
    const b = buf('b', [4]);
    const load = new BufferLoadNode(a, [idx('i')]);
    const store = new BufferStoreNode(b, [idx('i')], load);
    const block = new BlockNode('blk', [], [{ buffer: a }], [{ buffer: b }], store);
    const loop = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    const pf = makePrimFunc('copy', ['p0', 'p1'], loop, new Map([['p0', a], ['p1', b]]));

    const lir = lowerToLIR(pf, WasmTarget());
    const forBody = lir.body.body;
    expect(forBody.type).toBe('LIRFlatStoreNode');
    expect(forBody.buffer.name).toBe('b');
  });

  it('converts BufferLoadNode to LIRFlatLoadNode in expressions', () => {
    const a = buf('a', [4]);
    const b = buf('b', [4]);
    const load = new BufferLoadNode(a, [idx('i')]);
    const store = new BufferStoreNode(b, [idx('i')], load);
    const block = new BlockNode('blk', [], [{ buffer: a }], [{ buffer: b }], store);
    const loop = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    const pf = makePrimFunc('copy', ['p0', 'p1'], loop, new Map([['p0', a], ['p1', b]]));

    const lir = lowerToLIR(pf, WasmTarget());
    const flatStore = lir.body.body;
    expect(flatStore.value.type).toBe('LIRFlatLoadNode');
    expect(flatStore.value.buffer.name).toBe('a');
    expect(flatStore.value.dtype).toBe('f32');
  });

  it('flattens 2D index to expression tree', () => {
    const m = buf('m', [3, 4]);
    const out = buf('out', [3, 4]);
    const load = new BufferLoadNode(m, [idx('i'), idx('j')]);
    const store = new BufferStoreNode(out, [idx('i'), idx('j')], load);
    const block = new BlockNode('blk', [], [{ buffer: m }], [{ buffer: out }], store);
    const inner = new ForNode(idx('j'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    const outer = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(3), ForKind.SERIAL, inner);
    const pf = makePrimFunc('copy2d', ['p0', 'p1'], outer, new Map([['p0', m], ['p1', out]]));

    const lir = lowerToLIR(pf, WasmTarget());
    const innerLoop = lir.body.body;
    const flatStore = innerLoop.body;
    expect(flatStore.type).toBe('LIRFlatStoreNode');
    expect(flatStore.offsetExpr.type).toBe('MathOpNode');
  });
});

describe('lowerToLIR — BlockNode → LIRBindingsNode', () => {
  it('converts block iterVars to LIRBindingsNode', () => {
    const b = buf('x', [4]);
    const store = new BufferStoreNode(b, [idx('bi')], new FloatImmNode(1));
    const block = new BlockNode('blk', [
      { iterVar: new VariableNode('bi', 'index'), binding: idx('outer') }
    ], [], [{ buffer: b }], store);
    const loop = new ForNode(idx('outer'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    const pf = makePrimFunc('test', ['p0'], loop, new Map([['p0', b]]));

    const lir = lowerToLIR(pf, WasmTarget());
    const body = lir.body.body;
    expect(body.type).toBe('LIRBindingsNode');
    expect(body.bindings.length).toBe(1);
    expect(body.bindings[0].name).toBe('bi');
  });

  it('passes through block with no iterVars', () => {
    const b = buf('x', [4]);
    const store = new BufferStoreNode(b, [idx('i')], new FloatImmNode(1));
    const block = new BlockNode('blk', [], [], [{ buffer: b }], store);
    const loop = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    const pf = makePrimFunc('test', ['p0'], loop, new Map([['p0', b]]));

    const lir = lowerToLIR(pf, WasmTarget());
    const body = lir.body.body;
    expect(body.type).toBe('LIRFlatStoreNode');
  });
});

describe('lowerToLIR — accumulator detection', () => {
  it('converts reduction pattern to LIRAccumulatorNode', () => {
    const accBuf = buf('acc', [4]);
    const inBuf = buf('inp', [4, 8]);
    const load = new BufferLoadNode(accBuf, [idx('i')]);
    const inLoad = new BufferLoadNode(inBuf, [idx('i'), idx('j')]);
    const add = new MathOpNode('+', load, inLoad);
    const store = new BufferStoreNode(accBuf, [idx('i')], add);
    const block = new BlockNode('red', [
      { iterVar: idx('i'), binding: idx('outer_i') }
    ], [{ buffer: inBuf }], [{ buffer: accBuf }], store);
    const innerLoop = new ForNode(idx('j'), new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, block);
    const outerLoop = new ForNode(idx('outer_i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, innerLoop);
    const pf = makePrimFunc('reduce', ['p0', 'p1'], outerLoop, new Map([['p0', inBuf], ['p1', accBuf]]));

    const lir = lowerToLIR(pf, WasmTarget());
    const innerBody = lir.body.body;
    expect(innerBody.type).toBe('LIRAccumulatorNode');
    expect(innerBody.localName).toMatch(/^_acc_/);
    expect(innerBody.dtype).toBe('f32');
    expect(innerBody.initLoad.type).toBe('LIRFlatLoadNode');
    expect(innerBody.flushStore.type).toBe('LIRFlatStoreNode');
  });

  it('does not convert a non-accumulator op (sub) to accumulator', () => {
    const accBuf = buf('acc', [4]);
    const load = new BufferLoadNode(accBuf, [idx('i')]);
    const sub = new MathOpNode('-', load, new FloatImmNode(2));
    const store = new BufferStoreNode(accBuf, [idx('i')], sub);
    const block = new BlockNode('blk', [], [{ buffer: accBuf }], [{ buffer: accBuf }], store);
    const loop = new ForNode(idx('j'), new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, block);
    const pf = makePrimFunc('test', ['p0'], loop, new Map([['p0', accBuf]]));

    const lir = lowerToLIR(pf, WasmTarget());
    expect(lir.body.type).toBe('ForNode');
  });

  it('converts a prod (mul) reduction pattern to LIRAccumulatorNode with op *', () => {
    const accBuf = buf('acc', [4]);
    const inBuf = buf('inp', [4, 8]);
    const load = new BufferLoadNode(accBuf, [idx('i')]);
    const inLoad = new BufferLoadNode(inBuf, [idx('i'), idx('j')]);
    const mul = new MathOpNode('*', load, inLoad);
    const store = new BufferStoreNode(accBuf, [idx('i')], mul);
    const block = new BlockNode('red', [
      { iterVar: idx('i'), binding: idx('outer_i') }
    ], [{ buffer: inBuf }], [{ buffer: accBuf }], store);
    const innerLoop = new ForNode(idx('j'), new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, block);
    const outerLoop = new ForNode(idx('outer_i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, innerLoop);
    const pf = makePrimFunc('reduce', ['p0', 'p1'], outerLoop, new Map([['p0', inBuf], ['p1', accBuf]]));

    const lir = lowerToLIR(pf, WasmTarget());
    const innerBody = lir.body.body;
    expect(innerBody.type).toBe('LIRAccumulatorNode');
    expect(innerBody.op).toBe('*');
  });

  it('handles swapped operands (val + acc)', () => {
    const accBuf = buf('acc', [4]);
    const inBuf = buf('inp', [4, 8]);
    const load = new BufferLoadNode(accBuf, [idx('i')]);
    const inLoad = new BufferLoadNode(inBuf, [idx('i'), idx('j')]);
    const add = new MathOpNode('+', inLoad, load);
    const store = new BufferStoreNode(accBuf, [idx('i')], add);
    const block = new BlockNode('red', [], [{ buffer: inBuf }], [{ buffer: accBuf }], store);
    const loop = new ForNode(idx('j'), new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, block);
    const pf = makePrimFunc('test', ['p0', 'p1'], loop, new Map([['p0', inBuf], ['p1', accBuf]]));

    const lir = lowerToLIR(pf, WasmTarget());
    expect(lir.body.type).toBe('LIRAccumulatorNode');
  });
});

describe('lowerToLIR — SeqNode', () => {
  it('lowers SeqNode children recursively', () => {
    const a = buf('a', [2]);
    const b = buf('b', [2]);
    const s1 = new BufferStoreNode(a, [new IntImmNode(0)], new FloatImmNode(1));
    const s2 = new BufferStoreNode(b, [new IntImmNode(0)], new FloatImmNode(2));
    const seq = new SeqNode([s1, s2]);
    const pf = makePrimFunc('test', ['p0', 'p1'], seq, new Map([['p0', a], ['p1', b]]));

    const lir = lowerToLIR(pf, WasmTarget());
    expect(lir.body.type).toBe('SeqNode');
    expect(lir.body.stmts[0].type).toBe('LIRFlatStoreNode');
    expect(lir.body.stmts[1].type).toBe('LIRFlatStoreNode');
  });
});

describe('lowerToLIR — LetStmtNode', () => {
  it('preserves LetStmtNode with lowered value and body', () => {
    const b = buf('x', [4]);
    const letVar = new VariableNode('tmp', 'f32');
    const store = new BufferStoreNode(b, [idx('i')], letVar);
    const letStmt = new LetStmtNode(letVar, new BufferLoadNode(b, [idx('i')]), store);
    const loop = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, letStmt);
    const pf = makePrimFunc('test', ['p0'], loop, new Map([['p0', b]]));

    const lir = lowerToLIR(pf, WasmTarget());
    const letBody = lir.body.body;
    expect(letBody.type).toBe('LetStmtNode');
    expect(letBody.value.type).toBe('LIRFlatLoadNode');
    expect(letBody.body.type).toBe('LIRFlatStoreNode');
  });
});

describe('lowerToLIR — AllocateNode', () => {
  it('preserves AllocateNode with lowered body', () => {
    const tmp = buf('tmp', [4]);
    const b = buf('b', [4]);
    const store = new BufferStoreNode(tmp, [idx('i')], new FloatImmNode(0));
    const block = new BlockNode('blk', [], [], [{ buffer: tmp }], store);
    const loop = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    const alloc = new AllocateNode(tmp, 'local', loop);
    const pf = makePrimFunc('test', ['p0'], alloc, new Map([['p0', b]]));

    const lir = lowerToLIR(pf, WasmTarget());
    expect(lir.body.type).toBe('AllocateNode');
    expect(lir.body.body.type).toBe('ForNode');
  });
});

describe('lowerToLIR — IfThenElseNode', () => {
  it('lowers condition and branches', () => {
    const inBuf = buf('x', [4]);
    const outBuf = buf('y', [4]);
    const cond = new CompareNode('gt', new BufferLoadNode(inBuf, [idx('i')]), new FloatImmNode(0));
    const thenS = new BufferStoreNode(outBuf, [idx('i')], new BufferLoadNode(inBuf, [idx('i')]));
    const elseS = new BufferStoreNode(outBuf, [idx('i')], new FloatImmNode(0));
    const ifNode = new IfThenElseNode(cond, thenS, elseS);
    const block = new BlockNode('relu', [], [{ buffer: inBuf }], [{ buffer: outBuf }], ifNode);
    const loop = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    const pf = makePrimFunc('relu', ['p0', 'p1'], loop, new Map([['p0', inBuf], ['p1', outBuf]]));

    const lir = lowerToLIR(pf, WasmTarget());
    const ifBody = lir.body.body;
    expect(ifBody.type).toBe('IfThenElseNode');
    expect(ifBody.condition.a.type).toBe('LIRFlatLoadNode');
    expect(ifBody.thenBody.type).toBe('LIRFlatStoreNode');
    expect(ifBody.elseBody.type).toBe('LIRFlatStoreNode');
  });
});

describe('lowerToLIR — expression lowering', () => {
  it('lowers MathOpNode operands recursively', () => {
    const a = buf('a', [4]);
    const b = buf('b', [4]);
    const out = buf('c', [4]);
    const add = new MathOpNode('+', new BufferLoadNode(a, [idx('i')]), new BufferLoadNode(b, [idx('i')]));
    const store = new BufferStoreNode(out, [idx('i')], add);
    const block = new BlockNode('add', [], [{ buffer: a }, { buffer: b }], [{ buffer: out }], store);
    const loop = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    const pf = makePrimFunc('add', ['p0', 'p1', 'p2'], loop, new Map([['p0', a], ['p1', b], ['p2', out]]));

    const lir = lowerToLIR(pf, WasmTarget());
    const flatStore = lir.body.body;
    const addExpr = flatStore.value;
    expect(addExpr.type).toBe('MathOpNode');
    expect(addExpr.a.type).toBe('LIRFlatLoadNode');
    expect(addExpr.b.type).toBe('LIRFlatLoadNode');
  });

  it('lowers CastNode expression', () => {
    const b = buf('x', [4]);
    const load = new BufferLoadNode(b, [idx('i')]);
    const cast = new CastNode(load, 'f32', 'i32');
    const store = new BufferStoreNode(b, [idx('i')], cast);
    const block = new BlockNode('blk', [], [{ buffer: b }], [{ buffer: b }], store);
    const loop = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    const pf = makePrimFunc('test', ['p0'], loop, new Map([['p0', b]]));

    const lir = lowerToLIR(pf, WasmTarget());
    const castExpr = lir.body.body.value;
    expect(castExpr.type).toBe('CastNode');
    expect(castExpr.expr.type).toBe('LIRFlatLoadNode');
  });

  it('lowers CallExternNode args', () => {
    const b = buf('x', [4]);
    const load = new BufferLoadNode(b, [idx('i')]);
    const call = new CallExternNode('exp', [load], 'f32');
    const store = new BufferStoreNode(b, [idx('i')], call);
    const block = new BlockNode('blk', [], [{ buffer: b }], [{ buffer: b }], store);
    const loop = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    const pf = makePrimFunc('test', ['p0'], loop, new Map([['p0', b]]));

    const lir = lowerToLIR(pf, WasmTarget());
    const callExpr = lir.body.body.value;
    expect(callExpr.type).toBe('CallExternNode');
    expect(callExpr.args[0].type).toBe('LIRFlatLoadNode');
  });

  it('lowers CompareNode operands', () => {
    const b = buf('x', [4]);
    const cmp = new CompareNode('gt', new BufferLoadNode(b, [idx('i')]), new FloatImmNode(0));
    const store = new BufferStoreNode(b, [idx('i')], cmp);
    const block = new BlockNode('blk', [], [{ buffer: b }], [{ buffer: b }], store);
    const loop = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    const pf = makePrimFunc('test', ['p0'], loop, new Map([['p0', b]]));

    const lir = lowerToLIR(pf, WasmTarget());
    const cmpExpr = lir.body.body.value;
    expect(cmpExpr.type).toBe('CompareNode');
    expect(cmpExpr.a.type).toBe('LIRFlatLoadNode');
  });

  it('annotates _dtype on lowered expressions', () => {
    const b = buf('x', [4]);
    const load = new BufferLoadNode(b, [idx('i')]);
    const store = new BufferStoreNode(b, [idx('i')], load);
    const block = new BlockNode('blk', [], [{ buffer: b }], [{ buffer: b }], store);
    const loop = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    const pf = makePrimFunc('test', ['p0'], loop, new Map([['p0', b]]));

    const lir = lowerToLIR(pf, WasmTarget());
    const flatLoad = lir.body.body.value;
    expect(flatLoad._dtype).toBe('f32');
  });
});

describe('lowerToLIR — ForNode passthrough', () => {
  it('preserves ForNode kind and threadTag', () => {
    const b = buf('x', [4]);
    const store = new BufferStoreNode(b, [idx('i')], new FloatImmNode(0));
    const block = new BlockNode('blk', [], [], [{ buffer: b }], store);
    const loop = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.PARALLEL, block);
    const pf = makePrimFunc('test', ['p0'], loop, new Map([['p0', b]]));

    const lir = lowerToLIR(pf, WasmTarget());
    expect(lir.body.type).toBe('ForNode');
    expect(lir.body.kind).toBe(ForKind.PARALLEL);
  });

  it('preserves nested loop structure', () => {
    const m = buf('m', [3, 4]);
    const store = new BufferStoreNode(m, [idx('i'), idx('j')], new FloatImmNode(0));
    const block = new BlockNode('blk', [], [], [{ buffer: m }], store);
    const inner = new ForNode(idx('j'), new IntImmNode(0), new IntImmNode(4), ForKind.VECTORIZED, block);
    const outer = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(3), ForKind.SERIAL, inner);
    const pf = makePrimFunc('test', ['p0'], outer, new Map([['p0', m]]));

    const lir = lowerToLIR(pf, WasmTarget());
    expect(lir.body.type).toBe('ForNode');
    expect(lir.body.kind).toBe(ForKind.SERIAL);
    expect(lir.body.body.type).toBe('ForNode');
    expect(lir.body.body.kind).toBe(ForKind.VECTORIZED);
  });
});
