import { describe, it, expect } from 'vitest';
import { CPUCodegen } from '../../../src/backend/cpu/codegen.js';
import { CPUTarget } from '../../../src/backend/target.js';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import {
  PrimFunc, ForNode, BlockNode, SeqNode,
  BufferStoreNode, BufferLoadNode, AllocateNode,
  MathOpNode, CompareNode, IfThenElseNode, CastNode,
  LetStmtNode, CallExternNode,
  VariableNode, IntImmNode, FloatImmNode,
  ForKind, mathOp
} from '../../../src/compiler/ir/tensor/nodes.js';

function makeCodegen() {
  return new CPUCodegen(CPUTarget());
}

function buf(name, shape, dtype = 'f32') {
  return new Buffer(name, shape, dtype, 'global');
}

function idx(name) {
  return new VariableNode(name, 'index');
}

function makePrimFunc(name, params, body, bufferMap, shapeParams = []) {
  return new PrimFunc(name, params, body, bufferMap, shapeParams);
}

describe('CPUCodegen.generate — extent-1 loop elimination', () => {
  it('eliminates batch=1 loop and aliases variable to 0', () => {
    const inBuf = buf('input', [1, 4], 'f32');
    const outBuf = buf('output', [1, 4], 'f32');

    const load = new BufferLoadNode(inBuf, [idx('i'), idx('j')]);
    const store = new BufferStoreNode(outBuf, [idx('i'), idx('j')], load);
    const block = new BlockNode('copy', [], [{ buffer: inBuf }], [{ buffer: outBuf }], store);
    const innerFor = new ForNode(idx('j'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    const outerFor = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(1), ForKind.SERIAL, innerFor);

    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('copy_b1', ['p0', 'p1'], outerFor, bufferMap);

    const cg = makeCodegen();
    const src = cg.generate(pf);
    expect(src).not.toMatch(/for.*i.*< 1/);
    expect(src).toMatch(/for.*j.*< 4/);
  });
});

describe('CPUCodegen.generate — unrolled loops', () => {
  it('unrolls small-extent loop with UNROLLED kind', () => {
    const inBuf = buf('x', [3], 'f32');
    const outBuf = buf('y', [3], 'f32');
    const load = new BufferLoadNode(inBuf, [idx('i')]);
    const store = new BufferStoreNode(outBuf, [idx('i')], load);
    const block = new BlockNode('cp', [], [{ buffer: inBuf }], [{ buffer: outBuf }], store);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(3), ForKind.UNROLLED, block);

    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('unroll_test', ['p0', 'p1'], forNode, bufferMap);

    const cg = makeCodegen();
    const src = cg.generate(pf);
    expect(src).not.toMatch(/for/);
    expect(src).toMatch(/const i = 0/);
    expect(src).toMatch(/const i = 1/);
    expect(src).toMatch(/const i = 2/);
  });

});

describe('CPUCodegen.generate — zero-fill skip', () => {
  it('skips redundant zero-fill loops in output', () => {
    const outBuf = buf('output', [4], 'f32');
    const store = new BufferStoreNode(outBuf, [idx('i')], new FloatImmNode(0));
    const block = new BlockNode('init', [], [], [{ buffer: outBuf }], store);
    const zeroFill = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);

    const store2 = new BufferStoreNode(outBuf, [idx('j')], new FloatImmNode(42));
    const block2 = new BlockNode('write', [], [], [{ buffer: outBuf }], store2);
    const writeLoop = new ForNode(idx('j'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block2);

    const seq = new SeqNode([zeroFill, writeLoop]);
    const bufferMap = new Map([['p0', outBuf]]);
    const pf = makePrimFunc('skip_zero', ['p0'], seq, bufferMap);

    const cg = makeCodegen();
    const src = cg.generate(pf);
    expect(src).toMatch(/42/);
    expect((src.match(/for/g) || []).length).toBe(1);
  });
});

describe('CPUCodegen.generate — constant buffer inlining', () => {
  it('inlines constant buffer reads as literals and skips allocation', () => {
    const constBuf = buf('cbuf', [4], 'f32');
    const inBuf = buf('input', [4], 'f32');
    const outBuf = buf('output', [4], 'f32');

    const constFill = new BufferStoreNode(constBuf, [idx('i')], new FloatImmNode(2.5));
    const constBlock = new BlockNode('cfill', [], [], [{ buffer: constBuf }], constFill);
    const constLoop = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, constBlock);

    const load = new BufferLoadNode(inBuf, [idx('j')]);
    const constLoad = new BufferLoadNode(constBuf, [idx('j')]);
    const mul = new MathOpNode('*', load, constLoad);
    const store = new BufferStoreNode(outBuf, [idx('j')], mul);
    const compBlock = new BlockNode('comp', [], [{ buffer: inBuf }, { buffer: constBuf }], [{ buffer: outBuf }], store);
    const compLoop = new ForNode(idx('j'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, compBlock);

    const seq = new SeqNode([constLoop, compLoop]);
    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('const_inline', ['p0', 'p1'], seq, bufferMap);

    const cg = makeCodegen();
    const src = cg.generate(pf);
    expect(src).not.toMatch(/cbuf/);
    expect(src).toMatch(/2\.5/);
    expect(src).not.toMatch(/new Float32Array\(4\)/);
  });
});

describe('CPUCodegen.generate — reduction accumulator', () => {
  it('hoists accumulation load/store into a local variable', () => {
    const inBuf = buf('input', [4, 8], 'f32');
    const outBuf = buf('output', [4], 'f32');

    const iVar = idx('i');
    const jVar = idx('j');
    const accLoad = new BufferLoadNode(outBuf, [iVar]);
    const inLoad = new BufferLoadNode(inBuf, [iVar, jVar]);
    const add = new MathOpNode('+', accLoad, inLoad);
    const store = new BufferStoreNode(outBuf, [iVar], add);
    const block = new BlockNode('sum', [], [{ buffer: inBuf }], [{ buffer: outBuf }], store);
    const innerFor = new ForNode(jVar, new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, block);
    const outerBlock = new BlockNode('outer', [], [], [], innerFor);
    const outerFor = new ForNode(iVar, new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, outerBlock);

    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('reduce_sum', ['p0', 'p1'], outerFor, bufferMap);

    const cg = makeCodegen();
    const src = cg.generate(pf);
    expect(src).toMatch(/_acc_\d+/);
    expect(src).toMatch(/let _acc_/);
  });
});

describe('CPUCodegen.generate — IfThenElseNode as statement', () => {
  it('generates if/else block for conditional store', () => {
    const inBuf = buf('x', [4], 'f32');
    const outBuf = buf('y', [4], 'f32');

    const cond = new CompareNode('gt', new BufferLoadNode(inBuf, [idx('i')]), new FloatImmNode(0));
    const storeThen = new BufferStoreNode(outBuf, [idx('i')], new BufferLoadNode(inBuf, [idx('i')]));
    const storeElse = new BufferStoreNode(outBuf, [idx('i')], new FloatImmNode(0));
    const ifStmt = new IfThenElseNode(cond, storeThen, storeElse);
    const block = new BlockNode('relu_blk', [], [{ buffer: inBuf }], [{ buffer: outBuf }], ifStmt);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);

    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('if_test', ['p0', 'p1'], forNode, bufferMap);

    const cg = makeCodegen();
    const src = cg.generate(pf);
    expect(src).toMatch(/if\s*\(/);
    expect(src).toMatch(/else/);
  });
});

describe('CPUCodegen.generate — LetStmtNode', () => {
  it('generates const binding for let statement', () => {
    const inBuf = buf('x', [4], 'f32');
    const outBuf = buf('y', [4], 'f32');

    const load = new BufferLoadNode(inBuf, [idx('i')]);
    const letVar = new VariableNode('tmp_val', 'f32');
    const store = new BufferStoreNode(outBuf, [idx('i')], letVar);
    const block = new BlockNode('let_blk', [], [{ buffer: inBuf }], [{ buffer: outBuf }], store);
    const letStmt = new LetStmtNode(letVar, load, block);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, letStmt);

    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('let_test', ['p0', 'p1'], forNode, bufferMap);

    const cg = makeCodegen();
    const src = cg.generate(pf);
    expect(src).toMatch(/const tmp_val = /);
  });
});

describe('CPUCodegen.generate — AllocateNode', () => {
  it('emits local buffer allocation inside function body', () => {
    const localBuf = buf('scratch', [16], 'f32');
    const inBuf = buf('input', [16], 'f32');
    const outBuf = buf('output', [16], 'f32');

    const load = new BufferLoadNode(inBuf, [idx('i')]);
    const storeScratch = new BufferStoreNode(localBuf, [idx('i')], load);
    const blk1 = new BlockNode('fill', [], [{ buffer: inBuf }], [{ buffer: localBuf }], storeScratch);
    const loop1 = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(16), ForKind.SERIAL, blk1);

    const loadScratch = new BufferLoadNode(localBuf, [idx('j')]);
    const storeOut = new BufferStoreNode(outBuf, [idx('j')], loadScratch);
    const blk2 = new BlockNode('copy', [], [{ buffer: localBuf }], [{ buffer: outBuf }], storeOut);
    const loop2 = new ForNode(idx('j'), new IntImmNode(0), new IntImmNode(16), ForKind.SERIAL, blk2);

    const seq = new SeqNode([loop1, loop2]);
    const alloc = new AllocateNode(localBuf, 'local', seq);

    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('alloc_test', ['p0', 'p1'], alloc, bufferMap);

    const cg = makeCodegen();
    const src = cg.generate(pf);
    expect(src).toMatch(/const scratch = new Float32Array\(16\)/);
  });
});

describe('CPUCodegen.generate — full function', () => {
  it('generates valid executable JS function', () => {
    const inBuf = buf('x', [4], 'f32');
    const outBuf = buf('y', [4], 'f32');

    const load = new BufferLoadNode(inBuf, [idx('i')]);
    const neg = new MathOpNode('-', load);
    const store = new BufferStoreNode(outBuf, [idx('i')], neg);
    const block = new BlockNode('neg_blk', [], [{ buffer: inBuf }], [{ buffer: outBuf }], store);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);

    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('negate', ['p0', 'p1'], forNode, bufferMap);

    const cg = makeCodegen();
    const src = cg.generate(pf);

    const fn = new Function('return ' + src)();
    const input = new Float32Array([1, -2, 3, -4]);
    const output = new Float32Array(4);
    fn(input, output);
    expect(Array.from(output)).toEqual([-1, 2, -3, 4]);
  });

  it('generates correct vector add', () => {
    const aBuf = buf('a', [4], 'f32');
    const bBuf = buf('b', [4], 'f32');
    const outBuf = buf('c', [4], 'f32');

    const loadA = new BufferLoadNode(aBuf, [idx('i')]);
    const loadB = new BufferLoadNode(bBuf, [idx('i')]);
    const add = new MathOpNode('+', loadA, loadB);
    const store = new BufferStoreNode(outBuf, [idx('i')], add);
    const block = new BlockNode('add_blk', [], [{ buffer: aBuf }, { buffer: bBuf }], [{ buffer: outBuf }], store);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);

    const bufferMap = new Map([['p0', aBuf], ['p1', bBuf], ['p2', outBuf]]);
    const pf = makePrimFunc('vec_add', ['p0', 'p1', 'p2'], forNode, bufferMap);

    const cg = makeCodegen();
    const src = cg.generate(pf);

    const fn = new Function('return ' + src)();
    const a = new Float32Array([1, 2, 3, 4]);
    const b = new Float32Array([10, 20, 30, 40]);
    const c = new Float32Array(4);
    fn(a, b, c);
    expect(Array.from(c)).toEqual([11, 22, 33, 44]);
  });

  it('generates correct 2D matrix copy with strides', () => {
    const inBuf = buf('src', [2, 3], 'f32');
    const outBuf = buf('dst', [2, 3], 'f32');

    const load = new BufferLoadNode(inBuf, [idx('i'), idx('j')]);
    const store = new BufferStoreNode(outBuf, [idx('i'), idx('j')], load);
    const block = new BlockNode('cp', [], [{ buffer: inBuf }], [{ buffer: outBuf }], store);
    const innerFor = new ForNode(idx('j'), new IntImmNode(0), new IntImmNode(3), ForKind.SERIAL, block);
    const outerFor = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(2), ForKind.SERIAL, innerFor);

    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('mat_copy', ['p0', 'p1'], outerFor, bufferMap);

    const cg = makeCodegen();
    const src = cg.generate(pf);

    const fn = new Function('return ' + src)();
    const input = new Float32Array([1, 2, 3, 4, 5, 6]);
    const output = new Float32Array(6);
    fn(input, output);
    expect(Array.from(output)).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it('generates correct conditional (relu) via IfThenElse expression', () => {
    const inBuf = buf('x', [4], 'f32');
    const outBuf = buf('y', [4], 'f32');

    const load = new BufferLoadNode(inBuf, [idx('i')]);
    const cond = new CompareNode('gt', load, new FloatImmNode(0));
    const loadAgain = new BufferLoadNode(inBuf, [idx('i')]);
    const ternary = new IfThenElseNode(cond, loadAgain, new FloatImmNode(0));
    const store = new BufferStoreNode(outBuf, [idx('i')], ternary);
    const block = new BlockNode('relu', [], [{ buffer: inBuf }], [{ buffer: outBuf }], store);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);

    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('relu_fn', ['p0', 'p1'], forNode, bufferMap);

    const cg = makeCodegen();
    const src = cg.generate(pf);

    const fn = new Function('return ' + src)();
    const input = new Float32Array([-3, -1, 0, 5]);
    const output = new Float32Array(4);
    fn(input, output);
    expect(Array.from(output)).toEqual([0, 0, 0, 5]);
  });

  it('generates correct CallExtern (exp)', () => {
    const inBuf = buf('x', [3], 'f32');
    const outBuf = buf('y', [3], 'f32');

    const load = new BufferLoadNode(inBuf, [idx('i')]);
    const expCall = new CallExternNode('exp', [load], 'f32');
    const store = new BufferStoreNode(outBuf, [idx('i')], expCall);
    const block = new BlockNode('exp_blk', [], [{ buffer: inBuf }], [{ buffer: outBuf }], store);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(3), ForKind.SERIAL, block);

    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('exp_fn', ['p0', 'p1'], forNode, bufferMap);

    const cg = makeCodegen();
    const src = cg.generate(pf);

    const fn = new Function('return ' + src)();
    const input = new Float32Array([0, 1, -1]);
    const output = new Float32Array(3);
    fn(input, output);
    expect(output[0]).toBeCloseTo(1.0, 5);
    expect(output[1]).toBeCloseTo(Math.E, 4);
    expect(output[2]).toBeCloseTo(1 / Math.E, 4);
  });
});

describe('CPUCodegen.generate — dtype support', () => {
  it('allocates Int32Array for i32 buffers', () => {
    const inBuf = buf('x', [4], 'i32');
    const tmpBuf = buf('tmp', [4], 'i32');
    const outBuf = buf('y', [4], 'i32');

    const load = new BufferLoadNode(inBuf, [idx('i')]);
    const storeTmp = new BufferStoreNode(tmpBuf, [idx('i')], load);
    const blk1 = new BlockNode('fill', [], [{ buffer: inBuf }], [{ buffer: tmpBuf }], storeTmp);
    const loop1 = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, blk1);

    const loadTmp = new BufferLoadNode(tmpBuf, [idx('j')]);
    const storeOut = new BufferStoreNode(outBuf, [idx('j')], loadTmp);
    const blk2 = new BlockNode('cp', [], [{ buffer: tmpBuf }], [{ buffer: outBuf }], storeOut);
    const loop2 = new ForNode(idx('j'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, blk2);

    const seq = new SeqNode([loop1, loop2]);
    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('i32_test', ['p0', 'p1'], seq, bufferMap);

    const cg = makeCodegen();
    const src = cg.generate(pf);
    expect(src).toMatch(/new Int32Array\(4\)/);
  });

  it('allocates Float64Array for f64 buffers', () => {
    const inBuf = buf('x', [4], 'f64');
    const tmpBuf = buf('tmp', [4], 'f64');
    const outBuf = buf('y', [4], 'f64');

    const load = new BufferLoadNode(inBuf, [idx('i')]);
    const storeTmp = new BufferStoreNode(tmpBuf, [idx('i')], load);
    const blk = new BlockNode('fill', [], [{ buffer: inBuf }], [{ buffer: tmpBuf }], storeTmp);
    const loop1 = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, blk);

    const loadTmp = new BufferLoadNode(tmpBuf, [idx('j')]);
    const storeOut = new BufferStoreNode(outBuf, [idx('j')], loadTmp);
    const blk2 = new BlockNode('cp', [], [{ buffer: tmpBuf }], [{ buffer: outBuf }], storeOut);
    const loop2 = new ForNode(idx('j'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, blk2);

    const seq = new SeqNode([loop1, loop2]);
    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('f64_test', ['p0', 'p1'], seq, bufferMap);

    const cg = makeCodegen();
    const src = cg.generate(pf);
    expect(src).toMatch(/new Float64Array\(4\)/);
  });
});

describe('CPUCodegen.generate — zero buffer read inlining', () => {
  it('inlines reads from zero-only buffer as literal 0', () => {
    const zeroBuf = buf('zbuf', [4], 'f32');
    const inBuf = buf('input', [4], 'f32');
    const outBuf = buf('output', [4], 'f32');

    const zeroStore = new BufferStoreNode(zeroBuf, [idx('i')], new FloatImmNode(0));
    const zblk = new BlockNode('zfill', [], [], [{ buffer: zeroBuf }], zeroStore);
    const zloop = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, zblk);

    const inLoad = new BufferLoadNode(inBuf, [idx('j')]);
    const zLoad = new BufferLoadNode(zeroBuf, [idx('j')]);
    const add = new MathOpNode('+', inLoad, zLoad);
    const store = new BufferStoreNode(outBuf, [idx('j')], add);
    const cblk = new BlockNode('comp', [], [{ buffer: inBuf }, { buffer: zeroBuf }], [{ buffer: outBuf }], store);
    const cloop = new ForNode(idx('j'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, cblk);

    const seq = new SeqNode([zloop, cloop]);
    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('zero_inline', ['p0', 'p1'], seq, bufferMap);

    const cg = makeCodegen();
    const src = cg.generate(pf);
    expect(src).not.toMatch(/zbuf/);

    const fn = new Function('return ' + src)();
    const input = new Float32Array([1, 2, 3, 4]);
    const output = new Float32Array(4);
    fn(input, output);
    expect(Array.from(output)).toEqual([1, 2, 3, 4]);
  });
});

describe('regression — _findZeroOnlyBuffers runs before allocation', () => {
  it('constant buffer is not allocated when detected before allocation loop', () => {
    const constBuf = buf('cbuf', [4], 'f32');
    const inBuf = buf('input', [4], 'f32');
    const outBuf = buf('output', [4], 'f32');

    const constStore = new BufferStoreNode(constBuf, [idx('i')], new FloatImmNode(2.5));
    const cblk = new BlockNode('cfill', [], [], [{ buffer: constBuf }], constStore);
    const cloop = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, cblk);

    const inLoad = new BufferLoadNode(inBuf, [idx('j')]);
    const cLoad = new BufferLoadNode(constBuf, [idx('j')]);
    const mul = new MathOpNode('*', inLoad, cLoad);
    const store = new BufferStoreNode(outBuf, [idx('j')], mul);
    const blk = new BlockNode('comp', [], [{ buffer: inBuf }, { buffer: constBuf }], [{ buffer: outBuf }], store);
    const loop = new ForNode(idx('j'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, blk);

    const seq = new SeqNode([cloop, loop]);
    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('const_alloc', ['p0', 'p1'], seq, bufferMap);

    const cg = makeCodegen();
    const src = cg.generate(pf);
    expect(src).not.toMatch(/new Float32Array\(4\)/);
    expect(src).toMatch(/2\.5/);
  });
});

describe('regression — _isRedundantZeroFill works for non-local buffers', () => {
  it('skips zero-fill for param buffer identified as zero-only', () => {
    const zeroBuf = buf('zbuf', [4], 'f32');
    const outBuf = buf('output', [4], 'f32');

    const zeroStore = new BufferStoreNode(zeroBuf, [idx('i')], new FloatImmNode(0));
    const zblk = new BlockNode('zfill', [], [], [{ buffer: zeroBuf }], zeroStore);
    const zloop = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, zblk);

    const load = new BufferLoadNode(zeroBuf, [idx('j')]);
    const store = new BufferStoreNode(outBuf, [idx('j')], load);
    const blk = new BlockNode('cp', [], [{ buffer: zeroBuf }], [{ buffer: outBuf }], store);
    const loop = new ForNode(idx('j'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, blk);

    const seq = new SeqNode([zloop, loop]);
    const bufferMap = new Map([['p0', outBuf]]);
    const pf = makePrimFunc('zero_skip', ['p0'], seq, bufferMap);

    const cg = makeCodegen();
    const src = cg.generate(pf);
    expect(src).not.toMatch(/zbuf/);
  });
});

describe('CPUCodegen — VECTORIZED loops emitted as regular loops (no unroll)', () => {
  it('VECTORIZED loop emits for-loop instead of unrolling', () => {
    const inBuf = buf('x', [8], 'f32');
    const outBuf = buf('y', [8], 'f32');
    const load = new BufferLoadNode(inBuf, [idx('i')]);
    const store = new BufferStoreNode(outBuf, [idx('i')], load);
    const block = new BlockNode('cp', [], [{ buffer: inBuf }], [{ buffer: outBuf }], store);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(8), ForKind.VECTORIZED, block);

    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('vec_loop', ['p0', 'p1'], forNode, bufferMap);

    const cg = makeCodegen();
    const src = cg.generate(pf);
    expect(src).toMatch(/for\s*\(/);
    expect(src).not.toMatch(/const i = 0/);
  });

  it('UNROLLED loop still unrolls normally', () => {
    const inBuf = buf('x', [4], 'f32');
    const outBuf = buf('y', [4], 'f32');
    const load = new BufferLoadNode(inBuf, [idx('i')]);
    const store = new BufferStoreNode(outBuf, [idx('i')], load);
    const block = new BlockNode('cp', [], [{ buffer: inBuf }], [{ buffer: outBuf }], store);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.UNROLLED, block);

    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('unroll_ok', ['p0', 'p1'], forNode, bufferMap);

    const cg = makeCodegen();
    const src = cg.generate(pf);
    expect(src).not.toMatch(/for\s*\(/);
    expect(src).toMatch(/const i = 0/);
    expect(src).toMatch(/const i = 3/);
  });

  it('UNROLLED zero-fill is not unrolled into scalar stores', () => {
    const b = buf('tmp', [8], 'f32');
    const store = new BufferStoreNode(b, [idx('i')], new FloatImmNode(0));
    const block = new BlockNode('blk', [], [], [{ buffer: b }], store);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(8), ForKind.UNROLLED, block);

    const bufferMap = new Map([['p0', b]]);
    const pf = makePrimFunc('zero_nounroll', ['p0'], forNode, bufferMap);

    const cg = makeCodegen();
    const src = cg.generate(pf);
    expect(src).not.toMatch(/const i = 0/);
  });
});
