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

describe('CPUCodegen._exprToJS', () => {
  function exprToJS(node) {
    const cg = makeCodegen();
    cg._zeroBuffers = new Set();
    cg._constantBuffers = new Map();
    cg._aliases = new Map();
    return cg._exprToJS(node);
  }

  it('renders IntImmNode', () => {
    expect(exprToJS(new IntImmNode(42))).toBe('42');
    expect(exprToJS(new IntImmNode(0))).toBe('0');
    expect(exprToJS(new IntImmNode(-7))).toBe('-7');
  });

  it('renders FloatImmNode', () => {
    expect(exprToJS(new FloatImmNode(3.14))).toBe('3.14');
    expect(exprToJS(new FloatImmNode(0))).toBe('0');
    expect(exprToJS(new FloatImmNode(-1.5))).toBe('-1.5');
  });

  it('renders VariableNode', () => {
    expect(exprToJS(new VariableNode('i0', 'index'))).toBe('i0');
  });

  it('resolves aliased VariableNode', () => {
    const cg = makeCodegen();
    cg._zeroBuffers = new Set();
    cg._constantBuffers = new Map();
    cg._aliases = new Map([['i0', '(x + 1)']]);
    expect(cg._exprToJS(new VariableNode('i0', 'index'))).toBe('(x + 1)');
  });

  it('renders MathOpNode add', () => {
    const node = new MathOpNode('+', new IntImmNode(3), new IntImmNode(4));
    expect(exprToJS(node)).toBe('(3 + 4)');
  });

  it('renders MathOpNode mul', () => {
    const node = new MathOpNode('*', new VariableNode('i', 'index'), new IntImmNode(8));
    expect(exprToJS(node)).toBe('(i * 8)');
  });

  it('renders MathOpNode sub', () => {
    const node = new MathOpNode('-', new IntImmNode(10), new IntImmNode(3));
    expect(exprToJS(node)).toBe('(10 - 3)');
  });

  it('renders unary MathOpNode (neg)', () => {
    const node = new MathOpNode('-', new VariableNode('x', 'f32'));
    expect(exprToJS(node)).toBe('(-x)');
  });

  it('renders modulo with Python semantics', () => {
    const node = new MathOpNode('%', new VariableNode('x', 'index'), new IntImmNode(4));
    expect(exprToJS(node)).toBe('((x % 4 + 4) % 4)');
  });

  it('renders integer division with truncation', () => {
    const node = new MathOpNode('//', new VariableNode('x', 'index'), new IntImmNode(4));
    expect(exprToJS(node)).toBe('((x / 4) | 0)');
  });

  it('simplifies x + 0 → x', () => {
    const node = new MathOpNode('+', new VariableNode('x', 'f32'), new IntImmNode(0));
    expect(exprToJS(node)).toBe('x');
  });

  it('simplifies 0 + x → x', () => {
    const node = new MathOpNode('+', new IntImmNode(0), new VariableNode('x', 'f32'));
    expect(exprToJS(node)).toBe('x');
  });

  it('simplifies x - 0 → x', () => {
    const node = new MathOpNode('-', new VariableNode('x', 'f32'), new IntImmNode(0));
    expect(exprToJS(node)).toBe('x');
  });

  it('simplifies x * 0 → 0', () => {
    const node = new MathOpNode('*', new VariableNode('x', 'f32'), new IntImmNode(0));
    expect(exprToJS(node)).toBe('0');
  });

  it('simplifies 0 * x → 0', () => {
    const node = new MathOpNode('*', new IntImmNode(0), new VariableNode('x', 'f32'));
    expect(exprToJS(node)).toBe('0');
  });

  it('simplifies x * 1 → x', () => {
    const node = new MathOpNode('*', new VariableNode('x', 'f32'), new IntImmNode(1));
    expect(exprToJS(node)).toBe('x');
  });

  it('simplifies 1 * x → x', () => {
    const node = new MathOpNode('*', new IntImmNode(1), new VariableNode('x', 'f32'));
    expect(exprToJS(node)).toBe('x');
  });

  it('renders nested MathOpNode', () => {
    const inner = new MathOpNode('+', new VariableNode('i', 'index'), new IntImmNode(1));
    const outer = new MathOpNode('*', inner, new IntImmNode(4));
    expect(exprToJS(outer)).toBe('((i + 1) * 4)');
  });

  it('renders CompareNode', () => {
    const node = new CompareNode('ge', new VariableNode('x', 'index'), new IntImmNode(0));
    expect(exprToJS(node)).toBe('(x >= 0)');
  });

  it('renders CompareNode eq', () => {
    const node = new CompareNode('eq', new VariableNode('a', 'index'), new VariableNode('b', 'index'));
    expect(exprToJS(node)).toBe('(a === b)');
  });

  it('renders IfThenElseNode as ternary', () => {
    const cond = new CompareNode('gt', new VariableNode('x', 'f32'), new IntImmNode(0));
    const node = new IfThenElseNode(cond, new FloatImmNode(1), new FloatImmNode(0));
    expect(exprToJS(node)).toBe('((x > 0) ? 1 : 0)');
  });

  it('renders CastNode to int', () => {
    const node = new CastNode(new FloatImmNode(3.7), 'f32', 'i32');
    expect(exprToJS(node)).toBe('(3.7 | 0)');
  });

  it('renders CastNode to float', () => {
    const node = new CastNode(new IntImmNode(5), 'i32', 'f32');
    expect(exprToJS(node)).toBe('(+5)');
  });

  it('renders CastNode to bool', () => {
    const node = new CastNode(new VariableNode('x', 'index'), 'index', 'bool');
    expect(exprToJS(node)).toBe('(x ? 1 : 0)');
  });

  it('renders CallExternNode for Math functions', () => {
    const node = new CallExternNode('exp', [new VariableNode('x', 'f32')], 'f32');
    expect(exprToJS(node)).toBe('Math.exp(x)');
  });

  it('renders CallExternNode for sqrt', () => {
    const node = new CallExternNode('sqrt', [new VariableNode('x', 'f32')], 'f32');
    expect(exprToJS(node)).toBe('Math.sqrt(x)');
  });

  it('renders CallExternNode for rsqrt', () => {
    const node = new CallExternNode('rsqrt', [new VariableNode('x', 'f32')], 'f32');
    expect(exprToJS(node)).toBe('(1.0 / Math.sqrt(x))');
  });

  it('renders CallExternNode with multiple args', () => {
    const node = new CallExternNode('max', [new VariableNode('a', 'f32'), new VariableNode('b', 'f32')], 'f32');
    expect(exprToJS(node)).toBe('Math.max(a, b)');
  });

  it('renders CallExternNode for fmod', () => {
    const node = new CallExternNode('fmod', [new VariableNode('a', 'f32'), new VariableNode('b', 'f32')], 'f32');
    expect(exprToJS(node)).toBe('((a % b + b) % b)');
  });

  it('throws on unknown extern function', () => {
    const node = new CallExternNode('fake_func', [new VariableNode('x', 'f32')], 'f32');
    expect(() => exprToJS(node)).toThrow('CPU codegen: unsupported extern function "fake_func"');
  });

  it('renders BufferLoadNode', () => {
    const b = buf('buf_1', [4], 'f32');
    const node = new BufferLoadNode(b, [new VariableNode('i', 'index')]);
    const cg = makeCodegen();
    cg._zeroBuffers = new Set();
    cg._constantBuffers = new Map();
    cg._aliases = new Map();
    expect(cg._exprToJS(node)).toBe('buf_1[i]');
  });

  it('renders BufferLoadNode from zero buffer as 0', () => {
    const b = buf('zbuf', [4], 'f32');
    const node = new BufferLoadNode(b, [new IntImmNode(0)]);
    const cg = makeCodegen();
    cg._zeroBuffers = new Set(['zbuf']);
    cg._constantBuffers = new Map();
    cg._aliases = new Map();
    expect(cg._exprToJS(node)).toBe('0');
  });

  it('renders BufferLoadNode from constant buffer as literal', () => {
    const b = buf('cbuf', [4], 'f32');
    const node = new BufferLoadNode(b, [new IntImmNode(0)]);
    const cg = makeCodegen();
    cg._zeroBuffers = new Set();
    cg._constantBuffers = new Map([['cbuf', '3.14']]);
    cg._aliases = new Map();
    expect(cg._exprToJS(node)).toBe('3.14');
  });

  it('returns 0 for null input', () => {
    expect(exprToJS(null)).toBe('0');
  });
});

describe('CPUCodegen._flatIndex', () => {
  function flatIndex(buffer, indices) {
    const cg = makeCodegen();
    cg._zeroBuffers = new Set();
    cg._constantBuffers = new Map();
    cg._aliases = new Map();
    cg._primFunc = null;
    return cg._flatIndex(buffer, indices);
  }

  it('scalar index returns 0', () => {
    const b = buf('s', [], 'f32');
    expect(flatIndex(b, [])).toBe('0');
  });

  it('1D index returns the index expression directly', () => {
    const b = buf('v', [8], 'f32');
    expect(flatIndex(b, [new VariableNode('i', 'index')])).toBe('i');
  });

  it('2D index computes row-major flat index', () => {
    const b = buf('m', [3, 4], 'f32');
    const result = flatIndex(b, [new VariableNode('i', 'index'), new VariableNode('j', 'index')]);
    expect(result).toBe('i * 4 + j');
  });

  it('3D index computes correct strides', () => {
    const b = buf('t', [2, 3, 4], 'f32');
    const result = flatIndex(b, [
      new VariableNode('i', 'index'),
      new VariableNode('j', 'index'),
      new VariableNode('k', 'index')
    ]);
    expect(result).toBe('i * 12 + j * 4 + k');
  });

  it('skips dimension with zero index', () => {
    const b = buf('m', [3, 4], 'f32');
    const result = flatIndex(b, [new IntImmNode(0), new VariableNode('j', 'index')]);
    expect(result).toBe('j');
  });

  it('all-zero indices return 0', () => {
    const b = buf('m', [3, 4], 'f32');
    expect(flatIndex(b, [new IntImmNode(0), new IntImmNode(0)])).toBe('0');
  });
});

describe('CPUCodegen._isRedundantZeroFill', () => {
  function isRedundantZeroFill(forNode) {
    const cg = makeCodegen();
    return cg._isRedundantZeroFill(forNode);
  }

  it('detects simple zero-fill loop', () => {
    const b = buf('tmp', [4], 'f32');
    const store = new BufferStoreNode(b, [idx('i')], new FloatImmNode(0));
    const block = new BlockNode('blk', [], [], [{ buffer: b }], store);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    expect(isRedundantZeroFill(forNode)).toBe(true);
  });

  it('detects zero-fill with IntImmNode(0)', () => {
    const b = buf('tmp', [4], 'i32');
    const store = new BufferStoreNode(b, [idx('i')], new IntImmNode(0));
    const block = new BlockNode('blk', [], [], [{ buffer: b }], store);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    expect(isRedundantZeroFill(forNode)).toBe(true);
  });

  it('rejects non-zero value store', () => {
    const b = buf('tmp', [4], 'f32');
    const store = new BufferStoreNode(b, [idx('i')], new FloatImmNode(1));
    const block = new BlockNode('blk', [], [], [{ buffer: b }], store);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    expect(isRedundantZeroFill(forNode)).toBe(false);
  });

  it('rejects non-constant value store', () => {
    const b = buf('tmp', [4], 'f32');
    const store = new BufferStoreNode(b, [idx('i')], new VariableNode('x', 'f32'));
    const block = new BlockNode('blk', [], [], [{ buffer: b }], store);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    expect(isRedundantZeroFill(forNode)).toBe(false);
  });

  it('traverses nested for loop with extent 1', () => {
    const b = buf('tmp', [1, 4], 'f32');
    const store = new BufferStoreNode(b, [idx('i'), idx('j')], new FloatImmNode(0));
    const block = new BlockNode('blk', [], [], [{ buffer: b }], store);
    const innerFor = new ForNode(idx('j'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    const outerFor = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(1), ForKind.SERIAL, innerFor);
    expect(isRedundantZeroFill(outerFor)).toBe(true);
  });

  it('detects nested zero-fill through multiple for loops', () => {
    const b = buf('tmp', [3, 4], 'f32');
    const store = new BufferStoreNode(b, [idx('i'), idx('j')], new FloatImmNode(0));
    const block = new BlockNode('blk', [], [], [{ buffer: b }], store);
    const innerFor = new ForNode(idx('j'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    const outerFor = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(3), ForKind.SERIAL, innerFor);
    expect(isRedundantZeroFill(outerFor)).toBe(true);
  });
});

describe('CPUCodegen._findZeroOnlyBuffers', () => {
  function findZero(root, paramNames = []) {
    const cg = makeCodegen();
    const paramBuffers = new Set(paramNames);
    const zeroSet = cg._findZeroOnlyBuffers(root, paramBuffers);
    return { zero: zeroSet, constant: cg._constantBuffers };
  }

  it('identifies buffer written only with zero', () => {
    const b = buf('tmp', [4], 'f32');
    const store = new BufferStoreNode(b, [idx('i')], new FloatImmNode(0));
    const { zero } = findZero(store);
    expect(zero.has('tmp')).toBe(true);
  });

  it('does not mark param buffers as zero-only', () => {
    const b = buf('input', [4], 'f32');
    const store = new BufferStoreNode(b, [idx('i')], new FloatImmNode(0));
    const { zero } = findZero(store, ['input']);
    expect(zero.has('input')).toBe(false);
  });

  it('identifies constant buffer with uniform non-zero value', () => {
    const b = buf('cbuf', [4], 'f32');
    const s1 = new BufferStoreNode(b, [new IntImmNode(0)], new FloatImmNode(3.14));
    const s2 = new BufferStoreNode(b, [new IntImmNode(1)], new FloatImmNode(3.14));
    const seq = new SeqNode([s1, s2]);
    const { zero, constant } = findZero(seq);
    expect(zero.has('cbuf')).toBe(false);
    expect(constant.get('cbuf')).toBe('3.14');
  });

  it('does not mark buffer with mixed values as constant', () => {
    const b = buf('mbuf', [4], 'f32');
    const s1 = new BufferStoreNode(b, [new IntImmNode(0)], new FloatImmNode(1));
    const s2 = new BufferStoreNode(b, [new IntImmNode(1)], new FloatImmNode(2));
    const seq = new SeqNode([s1, s2]);
    const { zero, constant } = findZero(seq);
    expect(zero.has('mbuf')).toBe(false);
    expect(constant.has('mbuf')).toBe(false);
  });

  it('traverses SeqNode children', () => {
    const b1 = buf('z1', [4], 'f32');
    const b2 = buf('z2', [4], 'f32');
    const s1 = new BufferStoreNode(b1, [idx('i')], new FloatImmNode(0));
    const s2 = new BufferStoreNode(b2, [idx('i')], new IntImmNode(0));
    const seq = new SeqNode([s1, s2]);
    const { zero } = findZero(seq);
    expect(zero.has('z1')).toBe(true);
    expect(zero.has('z2')).toBe(true);
  });

  it('traverses ForNode body', () => {
    const b = buf('fz', [4], 'f32');
    const store = new BufferStoreNode(b, [idx('i')], new FloatImmNode(0));
    const block = new BlockNode('blk', [], [], [{ buffer: b }], store);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    const { zero } = findZero(forNode);
    expect(zero.has('fz')).toBe(true);
  });

  it('identifies integer constant buffer', () => {
    const b = buf('icbuf', [4], 'i32');
    const s1 = new BufferStoreNode(b, [new IntImmNode(0)], new IntImmNode(42));
    const s2 = new BufferStoreNode(b, [new IntImmNode(1)], new IntImmNode(42));
    const seq = new SeqNode([s1, s2]);
    const { constant } = findZero(seq);
    expect(constant.get('icbuf')).toBe('42');
  });
});

describe('CPUCodegen._scanTree', () => {
  function scanTree(root) {
    const cg = makeCodegen();
    const usedBuffers = new Map();
    const allocatedBuffers = new Set();
    cg._scanTree(root, usedBuffers, allocatedBuffers);
    return { used: usedBuffers, allocated: allocatedBuffers };
  }

  it('finds buffers in BufferStoreNode', () => {
    const b = buf('out', [4], 'f32');
    const store = new BufferStoreNode(b, [idx('i')], new FloatImmNode(0));
    const { used } = scanTree(store);
    expect(used.has('out')).toBe(true);
  });

  it('finds buffers in BufferLoadNode via value', () => {
    const bIn = buf('in', [4], 'f32');
    const bOut = buf('out', [4], 'f32');
    const load = new BufferLoadNode(bIn, [idx('i')]);
    const store = new BufferStoreNode(bOut, [idx('i')], load);
    const { used } = scanTree(store);
    expect(used.has('in')).toBe(true);
    expect(used.has('out')).toBe(true);
  });

  it('finds allocated buffers in AllocateNode', () => {
    const b = buf('local', [16], 'f32');
    const store = new BufferStoreNode(b, [idx('i')], new FloatImmNode(0));
    const alloc = new AllocateNode(b, 'local', store);
    const { allocated } = scanTree(alloc);
    expect(allocated.has('local')).toBe(true);
  });

  it('finds buffers in block reads/writes', () => {
    const bR = buf('rbuf', [4], 'f32');
    const bW = buf('wbuf', [4], 'f32');
    const store = new BufferStoreNode(bW, [idx('i')], new FloatImmNode(0));
    const block = new BlockNode('blk', [], [{ buffer: bR }], [{ buffer: bW }], store);
    const { used } = scanTree(block);
    expect(used.has('rbuf')).toBe(true);
    expect(used.has('wbuf')).toBe(true);
  });

  it('traverses SeqNode children', () => {
    const b1 = buf('a', [4], 'f32');
    const b2 = buf('b', [4], 'f32');
    const s1 = new BufferStoreNode(b1, [idx('i')], new FloatImmNode(0));
    const s2 = new BufferStoreNode(b2, [idx('j')], new FloatImmNode(0));
    const seq = new SeqNode([s1, s2]);
    const { used } = scanTree(seq);
    expect(used.has('a')).toBe(true);
    expect(used.has('b')).toBe(true);
  });

  it('traverses IfThenElseNode branches', () => {
    const bT = buf('tbuf', [4], 'f32');
    const bE = buf('ebuf', [4], 'f32');
    const storeT = new BufferStoreNode(bT, [idx('i')], new FloatImmNode(1));
    const storeE = new BufferStoreNode(bE, [idx('i')], new FloatImmNode(0));
    const cond = new CompareNode('gt', new VariableNode('x', 'f32'), new IntImmNode(0));
    const ifNode = new IfThenElseNode(cond, storeT, storeE);
    const { used } = scanTree(ifNode);
    expect(used.has('tbuf')).toBe(true);
    expect(used.has('ebuf')).toBe(true);
  });
});

describe('CPUCodegen._cleanupSource', () => {
  function cleanup(src) {
    const cg = makeCodegen();
    return cg._cleanupSource(src);
  }

  it('removes empty for loops', () => {
    const src = [
      'function f() {',
      '  for (let i = 0; i < 4; i++) {',
      '  }',
      '}'
    ].join('\n');
    const result = cleanup(src);
    expect(result).not.toMatch(/for/);
    expect(result).toMatch(/function f\(\)/);
  });

  it('removes nested empty for loops', () => {
    const src = [
      'function f() {',
      '  for (let i = 0; i < 4; i++) {',
      '    for (let j = 0; j < 4; j++) {',
      '    }',
      '  }',
      '}'
    ].join('\n');
    const result = cleanup(src);
    expect(result).not.toMatch(/for/);
  });

  it('removes unused buffer allocations', () => {
    const src = [
      'function f(input, output) {',
      '  const unused = new Float32Array(16);',
      '  output[0] = input[0];',
      '}'
    ].join('\n');
    const result = cleanup(src);
    expect(result).not.toMatch(/unused/);
    expect(result).toMatch(/output\[0\] = input\[0\]/);
  });

  it('keeps used buffer allocations', () => {
    const src = [
      'function f(input, output) {',
      '  const tmp = new Float32Array(16);',
      '  tmp[0] = input[0];',
      '  output[0] = tmp[0];',
      '}'
    ].join('\n');
    const result = cleanup(src);
    expect(result).toMatch(/const tmp = new Float32Array\(16\)/);
  });

  it('preserves non-empty for loops', () => {
    const src = [
      'function f(x, y) {',
      '  for (let i = 0; i < 4; i++) {',
      '    y[i] = x[i];',
      '  }',
      '}'
    ].join('\n');
    const result = cleanup(src);
    expect(result).toMatch(/for/);
    expect(result).toMatch(/y\[i\] = x\[i\]/);
  });

  it('iterates until fixpoint — removes alloc after its only loop is emptied', () => {
    const src = [
      'function f() {',
      '  const unused = new Float32Array(4);',
      '  for (let i = 0; i < 4; i++) {',
      '  }',
      '}'
    ].join('\n');
    const result = cleanup(src);
    expect(result).not.toMatch(/unused/);
    expect(result).not.toMatch(/for/);
  });
});

describe('CPUCodegen._detectReductionAcc', () => {
  function detectAcc(forNode) {
    const cg = makeCodegen();
    cg._zeroBuffers = new Set();
    cg._constantBuffers = new Map();
    cg._aliases = new Map();
    cg._primFunc = null;
    return cg._detectReductionAcc(forNode);
  }

  it('detects simple accumulation pattern: acc[i] += val', () => {
    const accBuf = buf('acc', [4], 'f32');
    const inBuf = buf('inp', [4, 8], 'f32');
    const load = new BufferLoadNode(accBuf, [idx('i')]);
    const inLoad = new BufferLoadNode(inBuf, [idx('i'), idx('j')]);
    const add = new MathOpNode('+', load, inLoad);
    const store = new BufferStoreNode(accBuf, [idx('i')], add);
    const block = new BlockNode('red', [
      { iterVar: idx('i'), binding: idx('outer_i') },
      { iterVar: idx('j'), binding: idx('inner_j') }
    ], [{ buffer: inBuf }], [{ buffer: accBuf }], store);
    const forNode = new ForNode(idx('inner_j'), new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, block);
    const result = detectAcc(forNode);
    expect(result).not.toBeNull();
    expect(result.bufName).toBe('acc');
  });

  it('rejects non-add accumulation', () => {
    const accBuf = buf('acc', [4], 'f32');
    const load = new BufferLoadNode(accBuf, [idx('i')]);
    const mul = new MathOpNode('*', load, new FloatImmNode(2));
    const store = new BufferStoreNode(accBuf, [idx('i')], mul);
    const block = new BlockNode('red', [], [{ buffer: accBuf }], [{ buffer: accBuf }], store);
    const forNode = new ForNode(idx('j'), new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, block);
    expect(detectAcc(forNode)).toBeNull();
  });

  it('rejects when body is not BlockNode', () => {
    const accBuf = buf('acc', [4], 'f32');
    const store = new BufferStoreNode(accBuf, [idx('i')], new FloatImmNode(0));
    const forNode = new ForNode(idx('j'), new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, store);
    expect(detectAcc(forNode)).toBeNull();
  });
});

describe('CPUCodegen._isZeroFillBody', () => {
  function isZeroFillBody(body) {
    const cg = makeCodegen();
    return cg._isZeroFillBody(body);
  }

  it('detects direct zero-store body', () => {
    const b = buf('tmp', [4], 'f32');
    const store = new BufferStoreNode(b, [idx('i')], new FloatImmNode(0));
    expect(isZeroFillBody(store)).toBe(true);
  });

  it('detects zero-store through BlockNode', () => {
    const b = buf('tmp', [4], 'f32');
    const store = new BufferStoreNode(b, [idx('i')], new FloatImmNode(0));
    const block = new BlockNode('blk', [], [], [{ buffer: b }], store);
    expect(isZeroFillBody(block)).toBe(true);
  });

  it('detects zero-store through nested ForNode', () => {
    const b = buf('tmp', [4, 8], 'f32');
    const store = new BufferStoreNode(b, [idx('i'), idx('j')], new IntImmNode(0));
    const block = new BlockNode('blk', [], [], [{ buffer: b }], store);
    const innerFor = new ForNode(idx('j'), new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, block);
    expect(isZeroFillBody(innerFor)).toBe(true);
  });

  it('rejects non-zero store', () => {
    const b = buf('tmp', [4], 'f32');
    const store = new BufferStoreNode(b, [idx('i')], new FloatImmNode(1));
    expect(isZeroFillBody(store)).toBe(false);
  });

  it('rejects variable store', () => {
    const b = buf('tmp', [4], 'f32');
    const load = new BufferLoadNode(buf('src', [4], 'f32'), [idx('i')]);
    const store = new BufferStoreNode(b, [idx('i')], load);
    expect(isZeroFillBody(store)).toBe(false);
  });
});

