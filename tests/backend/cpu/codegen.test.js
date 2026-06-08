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
