import { describe, it, expect } from 'vitest';
import { WasmCodegen } from '../../../src/backend/wasm/codegen.js';
import { WasmTarget } from '../../../src/backend/target.js';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import {
  PrimFunc, ForNode, BlockNode, SeqNode,
  BufferStoreNode, BufferLoadNode, AllocateNode,
  MathOpNode, CompareNode, IfThenElseNode, CastNode,
  LetStmtNode, CallExternNode,
  VariableNode, IntImmNode, FloatImmNode,
  ForKind
} from '../../../src/compiler/ir/tensor/nodes.js';

function makeCodegen() {
  return new WasmCodegen(WasmTarget());
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

// ────────────────────────────────────────────────────────────────────
// _emitExpr — expression emission
// ────────────────────────────────────────────────────────────────────

describe('WasmCodegen._emitExpr', () => {
  function emitExpr(node) {
    const cg = makeCodegen();
    cg._lines = [];
    cg._indent = 0;
    cg._locals = new Map();
    cg._bufferOffsets = new Map();
    cg._wasmAcc = null;
    cg._emitExpr(node);
    return cg._lines.map(l => l.trim()).join('\n');
  }

  it('emits IntImmNode', () => {
    expect(emitExpr(new IntImmNode(42))).toBe('(i32.const 42)');
  });

  it('emits IntImmNode zero', () => {
    expect(emitExpr(new IntImmNode(0))).toBe('(i32.const 0)');
  });

  it('emits IntImmNode negative', () => {
    expect(emitExpr(new IntImmNode(-7))).toBe('(i32.const -7)');
  });

  it('emits FloatImmNode', () => {
    expect(emitExpr(new FloatImmNode(3.14))).toBe('(f32.const 3.14)');
  });

  it('emits FloatImmNode zero', () => {
    expect(emitExpr(new FloatImmNode(0))).toBe('(f32.const 0)');
  });

  it('emits VariableNode', () => {
    expect(emitExpr(new VariableNode('i0', 'index'))).toBe('(local.get $i0)');
  });
});

// ────────────────────────────────────────────────────────────────────
// _emitExpr — MathOpNode
// ────────────────────────────────────────────────────────────────────

describe('WasmCodegen._emitExpr — MathOpNode', () => {
  function emitExpr(node) {
    const cg = makeCodegen();
    cg._lines = [];
    cg._indent = 0;
    cg._locals = new Map();
    cg._bufferOffsets = new Map();
    cg._wasmAcc = null;
    cg._emitExpr(node);
    return cg._lines.map(l => l.trim()).join('\n');
  }

  it('emits i32 add', () => {
    const node = new MathOpNode('+', new IntImmNode(3), new IntImmNode(4));
    const out = emitExpr(node);
    expect(out).toContain('(i32.const 3)');
    expect(out).toContain('(i32.const 4)');
    expect(out).toContain('i32.add');
  });

  it('emits f32 add', () => {
    const node = new MathOpNode('+', new FloatImmNode(1.5), new FloatImmNode(2.5));
    const out = emitExpr(node);
    expect(out).toContain('(f32.const 1.5)');
    expect(out).toContain('(f32.const 2.5)');
    expect(out).toContain('f32.add');
  });

  it('emits f32 sub', () => {
    const node = new MathOpNode('-', new FloatImmNode(10), new FloatImmNode(3));
    const out = emitExpr(node);
    expect(out).toContain('f32.sub');
  });

  it('emits f32 mul', () => {
    const node = new MathOpNode('*', new FloatImmNode(2), new FloatImmNode(5));
    const out = emitExpr(node);
    expect(out).toContain('f32.mul');
  });

  it('emits f32 div', () => {
    const node = new MathOpNode('/', new FloatImmNode(10), new FloatImmNode(2));
    const out = emitExpr(node);
    expect(out).toContain('f32.div');
  });

  it('emits i32 div_s for integer division', () => {
    const node = new MathOpNode('/', new IntImmNode(10), new IntImmNode(3));
    const out = emitExpr(node);
    expect(out).toContain('i32.div_s');
  });

  it('emits i32 rem_s for modulo', () => {
    const node = new MathOpNode('%', new IntImmNode(10), new IntImmNode(3));
    const out = emitExpr(node);
    expect(out).toContain('i32.rem_s');
  });

  it('emits f32.neg for unary minus on float', () => {
    const node = new MathOpNode('-', new VariableNode('x', 'f32'));
    const out = emitExpr(node);
    expect(out).toContain('f32.neg');
  });

  it('emits i32.sub from 0 for unary minus on int', () => {
    const node = new MathOpNode('-', new VariableNode('x', 'index'));
    const out = emitExpr(node);
    expect(out).toContain('(i32.const 0)');
    expect(out).toContain('i32.sub');
  });

  it('promotes i32 to f32 in mixed arithmetic', () => {
    const node = new MathOpNode('+', new FloatImmNode(1.5), new IntImmNode(2));
    const out = emitExpr(node);
    expect(out).toContain('f32.convert_i32_s');
    expect(out).toContain('f32.add');
  });
});

// ────────────────────────────────────────────────────────────────────
// _emitExpr — CompareNode
// ────────────────────────────────────────────────────────────────────

describe('WasmCodegen._emitExpr — CompareNode', () => {
  function emitExpr(node) {
    const cg = makeCodegen();
    cg._lines = [];
    cg._indent = 0;
    cg._locals = new Map();
    cg._bufferOffsets = new Map();
    cg._wasmAcc = null;
    cg._emitExpr(node);
    return cg._lines.map(l => l.trim()).join('\n');
  }

  it('emits i32.lt_s for int <', () => {
    const node = new CompareNode('lt', new IntImmNode(1), new IntImmNode(2));
    expect(emitExpr(node)).toContain('i32.lt_s');
  });

  it('emits i32.gt_s for int >', () => {
    const node = new CompareNode('gt', new IntImmNode(5), new IntImmNode(3));
    expect(emitExpr(node)).toContain('i32.gt_s');
  });

  it('emits f32.lt for float <', () => {
    const node = new CompareNode('lt', new FloatImmNode(1.5), new FloatImmNode(2.5));
    expect(emitExpr(node)).toContain('f32.lt');
  });

  it('emits i32.eq for eq', () => {
    const node = new CompareNode('eq', new IntImmNode(1), new IntImmNode(1));
    expect(emitExpr(node)).toContain('i32.eq');
  });

  it('emits i32.ne for ne', () => {
    const node = new CompareNode('ne', new IntImmNode(1), new IntImmNode(2));
    expect(emitExpr(node)).toContain('i32.ne');
  });
});

// ────────────────────────────────────────────────────────────────────
// _emitExpr — CastNode
// ────────────────────────────────────────────────────────────────────

describe('WasmCodegen._emitExpr — CastNode', () => {
  function emitExpr(node) {
    const cg = makeCodegen();
    cg._lines = [];
    cg._indent = 0;
    cg._locals = new Map();
    cg._bufferOffsets = new Map();
    cg._wasmAcc = null;
    cg._emitExpr(node);
    return cg._lines.map(l => l.trim()).join('\n');
  }

  it('emits i32.trunc_f32_s for float to int', () => {
    const node = new CastNode(new FloatImmNode(3.7), 'f32', 'i32');
    expect(emitExpr(node)).toContain('i32.trunc_f32_s');
  });

  it('emits f32.convert_i32_s for int to float', () => {
    const node = new CastNode(new IntImmNode(5), 'i32', 'f32');
    expect(emitExpr(node)).toContain('f32.convert_i32_s');
  });
});

// ────────────────────────────────────────────────────────────────────
// _emitExpr — CallExternNode
// ────────────────────────────────────────────────────────────────────

describe('WasmCodegen._emitExpr — CallExternNode', () => {
  function emitExpr(node) {
    const cg = makeCodegen();
    cg._lines = [];
    cg._indent = 0;
    cg._locals = new Map();
    cg._bufferOffsets = new Map();
    cg._wasmAcc = null;
    cg._imports = new Map();
    if (node.externName && !['sqrt', 'abs', 'ceil', 'floor', 'min', 'max'].includes(node.externName)) {
      cg._imports.set(node.externName, '(param f32) (result f32)');
    }
    cg._emitExpr(node);
    return cg._lines.map(l => l.trim()).join('\n');
  }

  it('emits f32.sqrt for sqrt (native)', () => {
    const node = new CallExternNode('sqrt', [new FloatImmNode(4)], 'f32');
    expect(emitExpr(node)).toContain('f32.sqrt');
  });

  it('emits f32.abs for abs (native)', () => {
    const node = new CallExternNode('abs', [new FloatImmNode(-3)], 'f32');
    expect(emitExpr(node)).toContain('f32.abs');
  });

  it('emits f32.ceil for ceil (native)', () => {
    const node = new CallExternNode('ceil', [new FloatImmNode(1.5)], 'f32');
    expect(emitExpr(node)).toContain('f32.ceil');
  });

  it('emits f32.floor for floor (native)', () => {
    const node = new CallExternNode('floor', [new FloatImmNode(1.5)], 'f32');
    expect(emitExpr(node)).toContain('f32.floor');
  });

  it('emits f32.min for min (native)', () => {
    const node = new CallExternNode('min', [new FloatImmNode(1), new FloatImmNode(2)], 'f32');
    expect(emitExpr(node)).toContain('f32.min');
  });

  it('emits f32.max for max (native)', () => {
    const node = new CallExternNode('max', [new FloatImmNode(1), new FloatImmNode(2)], 'f32');
    expect(emitExpr(node)).toContain('f32.max');
  });

  it('emits rsqrt as sqrt + div', () => {
    const node = new CallExternNode('rsqrt', [new FloatImmNode(4)], 'f32');
    const out = emitExpr(node);
    expect(out).toContain('f32.sqrt');
    expect(out).toContain('(f32.const 1)');
    expect(out).toContain('f32.div');
  });

  it('emits call $math_exp for imported exp', () => {
    const node = new CallExternNode('exp', [new FloatImmNode(1)], 'f32');
    expect(emitExpr(node)).toContain('call $math_exp');
  });

  it('emits call $math_log for imported log', () => {
    const node = new CallExternNode('log', [new FloatImmNode(1)], 'f32');
    expect(emitExpr(node)).toContain('call $math_log');
  });

  it('emits call $math_tanh for imported tanh', () => {
    const node = new CallExternNode('tanh', [new FloatImmNode(0)], 'f32');
    expect(emitExpr(node)).toContain('call $math_tanh');
  });

  it('emits call $math_sin for imported sin', () => {
    const node = new CallExternNode('sin', [new FloatImmNode(0)], 'f32');
    expect(emitExpr(node)).toContain('call $math_sin');
  });
});

// ────────────────────────────────────────────────────────────────────
// _emitExpr — BufferLoadNode
// ────────────────────────────────────────────────────────────────────

describe('WasmCodegen._emitExpr — BufferLoadNode', () => {
  function emitExpr(node, offsets) {
    const cg = makeCodegen();
    cg._lines = [];
    cg._indent = 0;
    cg._locals = new Map();
    cg._bufferOffsets = offsets || new Map();
    cg._wasmAcc = null;
    cg._emitExpr(node);
    return cg._lines.map(l => l.trim()).join('\n');
  }

  it('emits f32.load for f32 buffer', () => {
    const b = buf('data', [4], 'f32');
    const node = new BufferLoadNode(b, [new IntImmNode(0)]);
    const out = emitExpr(node, new Map([['data', 0]]));
    expect(out).toContain('f32.load');
  });

  it('emits i32.load for i32 buffer', () => {
    const b = buf('data', [4], 'i32');
    const node = new BufferLoadNode(b, [new IntImmNode(0)]);
    const out = emitExpr(node, new Map([['data', 0]]));
    expect(out).toContain('i32.load');
  });

  it('loads from _wacc local when matching accumulator target', () => {
    const cg = makeCodegen();
    cg._lines = [];
    cg._indent = 0;
    cg._locals = new Map();
    cg._bufferOffsets = new Map([['acc', 0]]);
    const accBuf = buf('acc', [4], 'f32');
    cg._wasmAcc = { local: '_wacc_1', bufName: 'acc', indices: [idx('i')] };
    const node = new BufferLoadNode(accBuf, [idx('i')]);
    cg._emitExpr(node);
    const out = cg._lines.map(l => l.trim()).join('\n');
    expect(out).toContain('(local.get $_wacc_1)');
  });
});

// ────────────────────────────────────────────────────────────────────
// _emitAddr — address computation
// ────────────────────────────────────────────────────────────────────

describe('WasmCodegen._emitAddr', () => {
  function emitAddr(buffer, indices, offsets) {
    const cg = makeCodegen();
    cg._lines = [];
    cg._indent = 0;
    cg._bufferOffsets = offsets || new Map([[buffer.name, 0]]);
    cg._emitAddr(buffer, indices);
    return cg._lines.map(l => l.trim()).join('\n');
  }

  it('scalar (no indices): emits constant offset', () => {
    const b = buf('s', [], 'f32');
    const out = emitAddr(b, [], new Map([['s', 64]]));
    expect(out).toBe('(i32.const 64)');
  });

  it('1D: multiplies index by byte size', () => {
    const b = buf('v', [8], 'f32');
    const out = emitAddr(b, [idx('i')], new Map([['v', 0]]));
    expect(out).toContain('(i32.const 4)');
    expect(out).toContain('i32.mul');
  });

  it('1D with offset: adds base offset', () => {
    const b = buf('v', [8], 'f32');
    const out = emitAddr(b, [idx('i')], new Map([['v', 32]]));
    expect(out).toContain('(i32.const 32)');
    expect(out).toContain('i32.add');
  });

  it('2D: uses stride multiplication', () => {
    const b = buf('m', [3, 4], 'f32');
    const out = emitAddr(b, [idx('i'), idx('j')], new Map([['m', 0]]));
    expect(out).toContain('(i32.const 4)');
    expect(out).toContain('i32.mul');
    expect(out).toContain('i32.add');
  });
});

// ────────────────────────────────────────────────────────────────────
// _layoutBuffers — buffer memory layout
// ────────────────────────────────────────────────────────────────────

describe('WasmCodegen._layoutBuffers', () => {
  it('assigns sequential offsets to buffers', () => {
    const b1 = buf('a', [4], 'f32');
    const b2 = buf('b', [4], 'f32');
    const store = new BufferStoreNode(b2, [idx('i')], new BufferLoadNode(b1, [idx('i')]));
    const block = new BlockNode('cp', [], [{ buffer: b1 }], [{ buffer: b2 }], store);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    const bufferMap = new Map([['p0', b1], ['p1', b2]]);
    const pf = makePrimFunc('test', ['p0', 'p1'], forNode, bufferMap);

    const cg = makeCodegen();
    cg._layoutBuffers(pf);

    const offA = cg._bufferOffsets.get('a');
    const offB = cg._bufferOffsets.get('b');
    expect(offA).toBeDefined();
    expect(offB).toBeDefined();
    expect(offB).toBeGreaterThan(offA);
  });

  it('aligns buffers to 16-byte boundaries', () => {
    const b1 = buf('a', [1], 'f32');
    const b2 = buf('b', [4], 'f32');
    const store = new BufferStoreNode(b2, [idx('i')], new BufferLoadNode(b1, [new IntImmNode(0)]));
    const block = new BlockNode('cp', [], [{ buffer: b1 }], [{ buffer: b2 }], store);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    const bufferMap = new Map([['p0', b1], ['p1', b2]]);
    const pf = makePrimFunc('test', ['p0', 'p1'], forNode, bufferMap);

    const cg = makeCodegen();
    cg._layoutBuffers(pf);

    const offB = cg._bufferOffsets.get('b');
    expect(offB % 16).toBe(0);
  });
});

// ────────────────────────────────────────────────────────────────────
// _scanMathImports — import detection
// ────────────────────────────────────────────────────────────────────

describe('WasmCodegen._scanMathImports', () => {
  it('detects exp as import', () => {
    const cg = makeCodegen();
    cg._imports = new Map();
    const call = new CallExternNode('exp', [new FloatImmNode(1)], 'f32');
    cg._scanMathImports(call);
    expect(cg._imports.has('exp')).toBe(true);
  });

  it('detects tanh as import', () => {
    const cg = makeCodegen();
    cg._imports = new Map();
    const call = new CallExternNode('tanh', [new FloatImmNode(0)], 'f32');
    cg._scanMathImports(call);
    expect(cg._imports.has('tanh')).toBe(true);
  });

  it('does NOT import sqrt (native wasm op)', () => {
    const cg = makeCodegen();
    cg._imports = new Map();
    const call = new CallExternNode('sqrt', [new FloatImmNode(4)], 'f32');
    cg._scanMathImports(call);
    expect(cg._imports.has('sqrt')).toBe(false);
  });

  it('does NOT import max (native wasm op)', () => {
    const cg = makeCodegen();
    cg._imports = new Map();
    const call = new CallExternNode('max', [new FloatImmNode(1), new FloatImmNode(2)], 'f32');
    cg._scanMathImports(call);
    expect(cg._imports.has('max')).toBe(false);
  });

  it('does NOT import min (native wasm op)', () => {
    const cg = makeCodegen();
    cg._imports = new Map();
    const call = new CallExternNode('min', [new FloatImmNode(1), new FloatImmNode(2)], 'f32');
    cg._scanMathImports(call);
    expect(cg._imports.has('min')).toBe(false);
  });

  it('traverses into ForNode body', () => {
    const cg = makeCodegen();
    cg._imports = new Map();
    const call = new CallExternNode('log', [new FloatImmNode(1)], 'f32');
    const store = new BufferStoreNode(buf('o', [4], 'f32'), [idx('i')], call);
    const block = new BlockNode('b', [], [], [], store);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    cg._scanMathImports(forNode);
    expect(cg._imports.has('log')).toBe(true);
  });

  it('traverses into SeqNode children', () => {
    const cg = makeCodegen();
    cg._imports = new Map();
    const call1 = new CallExternNode('sin', [new FloatImmNode(0)], 'f32');
    const call2 = new CallExternNode('cos', [new FloatImmNode(0)], 'f32');
    const s1 = new BufferStoreNode(buf('o1', [1], 'f32'), [new IntImmNode(0)], call1);
    const s2 = new BufferStoreNode(buf('o2', [1], 'f32'), [new IntImmNode(0)], call2);
    const seq = new SeqNode([s1, s2]);
    cg._scanMathImports(seq);
    expect(cg._imports.has('sin')).toBe(true);
    expect(cg._imports.has('cos')).toBe(true);
  });
});

// ────────────────────────────────────────────────────────────────────
// _inferDtype — type inference
// ────────────────────────────────────────────────────────────────────

describe('WasmCodegen._inferDtype', () => {
  function inferDtype(node) {
    const cg = makeCodegen();
    return cg._inferDtype(node);
  }

  it('IntImmNode → i32', () => {
    expect(inferDtype(new IntImmNode(0))).toBe('i32');
  });

  it('FloatImmNode → f32', () => {
    expect(inferDtype(new FloatImmNode(0))).toBe('f32');
  });

  it('BufferLoadNode → buffer dtype', () => {
    const b = buf('x', [4], 'f32');
    expect(inferDtype(new BufferLoadNode(b, [new IntImmNode(0)]))).toBe('f32');
  });

  it('BufferLoadNode i32 → i32', () => {
    const b = buf('x', [4], 'i32');
    expect(inferDtype(new BufferLoadNode(b, [new IntImmNode(0)]))).toBe('i32');
  });

  it('CastNode → target dtype', () => {
    expect(inferDtype(new CastNode(new IntImmNode(5), 'i32', 'f32'))).toBe('f32');
  });

  it('CompareNode → i32', () => {
    const node = new CompareNode('gt', new IntImmNode(1), new IntImmNode(0));
    expect(inferDtype(node)).toBe('i32');
  });

  it('MathOpNode int+int → i32', () => {
    const node = new MathOpNode('+', new IntImmNode(1), new IntImmNode(2));
    expect(inferDtype(node)).toBe('i32');
  });

  it('MathOpNode float+int → f32', () => {
    const node = new MathOpNode('+', new FloatImmNode(1), new IntImmNode(2));
    expect(inferDtype(node)).toBe('f32');
  });

  it('null → default dtype', () => {
    expect(inferDtype(null)).toBe('f32');
  });
});

// ────────────────────────────────────────────────────────────────────
// _constExtent
// ────────────────────────────────────────────────────────────────────

describe('WasmCodegen._constExtent', () => {
  it('returns value for IntImmNode', () => {
    const cg = makeCodegen();
    expect(cg._constExtent(new IntImmNode(8))).toBe(8);
  });

  it('returns null for VariableNode', () => {
    const cg = makeCodegen();
    expect(cg._constExtent(new VariableNode('n', 'index'))).toBeNull();
  });
});

// ────────────────────────────────────────────────────────────────────
// _visitFor — unrolling
// ────────────────────────────────────────────────────────────────────

describe('WasmCodegen._visitFor — unrolling', () => {
  it('unrolls UNROLLED loops with small extent', () => {
    const b = buf('x', [3], 'f32');
    const store = new BufferStoreNode(b, [idx('i')], new FloatImmNode(1));
    const block = new BlockNode('blk', [], [], [{ buffer: b }], store);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(3), ForKind.UNROLLED, block);

    const bufferMap = new Map([['p0', b]]);
    const pf = makePrimFunc('unroll', ['p0'], forNode, bufferMap);

    const cg = makeCodegen();
    const result = cg.generate(pf);
    const wat = result.wat;
    expect(wat).not.toMatch(/\(loop\s/);
    expect(wat).toContain('(i32.const 0)');
    expect(wat).toContain('(i32.const 1)');
    expect(wat).toContain('(i32.const 2)');
  });

  it('VECTORIZED loops emit SIMD v128 instructions', () => {
    const b = buf('x', [4], 'f32');
    const store = new BufferStoreNode(b, [idx('i')], new FloatImmNode(1));
    const block = new BlockNode('blk', [], [], [{ buffer: b }], store);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.VECTORIZED, block);

    const bufferMap = new Map([['p0', b]]);
    const pf = makePrimFunc('vec_unroll', ['p0'], forNode, bufferMap);

    const cg = makeCodegen();
    const result = cg.generate(pf);
    expect(result.wat).toContain('v128.store');
    expect(result.wat).toContain('f32x4.splat');
  });

  it('does not unroll large-extent loops', () => {
    const b = buf('x', [64], 'f32');
    const store = new BufferStoreNode(b, [idx('i')], new FloatImmNode(1));
    const block = new BlockNode('blk', [], [], [{ buffer: b }], store);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(64), ForKind.UNROLLED, block);

    const bufferMap = new Map([['p0', b]]);
    const pf = makePrimFunc('no_unroll', ['p0'], forNode, bufferMap);

    const cg = makeCodegen();
    const result = cg.generate(pf);
    expect(result.wat).toMatch(/\(loop\s/);
  });

  it('does not unroll VECTORIZED zero-fill loops', () => {
    const b = buf('z', [8], 'f32');
    const store = new BufferStoreNode(b, [idx('i')], new FloatImmNode(0));
    const block = new BlockNode('blk', [], [], [{ buffer: b }], store);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(8), ForKind.VECTORIZED, block);

    const bufferMap = new Map([['p0', b]]);
    const pf = makePrimFunc('vec_zero', ['p0'], forNode, bufferMap);

    const cg = makeCodegen();
    const result = cg.generate(pf);
    expect(result.wat).toMatch(/\(loop\s/);
  });

  it('does not unroll UNROLLED zero-fill loops', () => {
    const b = buf('z', [4], 'f32');
    const store = new BufferStoreNode(b, [idx('i')], new IntImmNode(0));
    const block = new BlockNode('blk', [], [], [{ buffer: b }], store);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.UNROLLED, block);

    const bufferMap = new Map([['p0', b]]);
    const pf = makePrimFunc('unroll_zero', ['p0'], forNode, bufferMap);

    const cg = makeCodegen();
    const result = cg.generate(pf);
    expect(result.wat).toMatch(/\(loop\s/);
  });
});

describe('WasmCodegen._isZeroFillBody', () => {
  function isZeroFillBody(body) {
    const cg = makeCodegen();
    return cg._isZeroFillBody(body);
  }

  it('detects direct float zero-store', () => {
    const b = buf('tmp', [4], 'f32');
    const store = new BufferStoreNode(b, [idx('i')], new FloatImmNode(0));
    expect(isZeroFillBody(store)).toBe(true);
  });

  it('detects direct int zero-store', () => {
    const b = buf('tmp', [4], 'i32');
    const store = new BufferStoreNode(b, [idx('i')], new IntImmNode(0));
    expect(isZeroFillBody(store)).toBe(true);
  });

  it('detects zero-store nested in ForNode', () => {
    const b = buf('tmp', [4, 8], 'f32');
    const store = new BufferStoreNode(b, [idx('i'), idx('j')], new FloatImmNode(0));
    const block = new BlockNode('blk', [], [], [{ buffer: b }], store);
    const innerFor = new ForNode(idx('j'), new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, block);
    expect(isZeroFillBody(innerFor)).toBe(true);
  });

  it('rejects non-zero store', () => {
    const b = buf('tmp', [4], 'f32');
    const store = new BufferStoreNode(b, [idx('i')], new FloatImmNode(3.14));
    expect(isZeroFillBody(store)).toBe(false);
  });

  it('rejects load-store body', () => {
    const src = buf('src', [4], 'f32');
    const dst = buf('dst', [4], 'f32');
    const load = new BufferLoadNode(src, [idx('i')]);
    const store = new BufferStoreNode(dst, [idx('i')], load);
    expect(isZeroFillBody(store)).toBe(false);
  });
});

// ────────────────────────────────────────────────────────────────────
// _emitForLoop — loop structure
// ────────────────────────────────────────────────────────────────────

describe('WasmCodegen._emitForLoop', () => {
  it('generates block+loop+br_if+br structure', () => {
    const b = buf('x', [8], 'f32');
    const store = new BufferStoreNode(b, [idx('i')], new FloatImmNode(0));
    const block = new BlockNode('blk', [], [], [{ buffer: b }], store);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, block);

    const bufferMap = new Map([['p0', b]]);
    const pf = makePrimFunc('loop_test', ['p0'], forNode, bufferMap);

    const cg = makeCodegen();
    const result = cg.generate(pf);
    const wat = result.wat;
    expect(wat).toMatch(/\(block \$break_i/);
    expect(wat).toMatch(/\(loop \$loop_i/);
    expect(wat).toMatch(/br_if \$break_i/);
    expect(wat).toMatch(/br \$loop_i/);
    expect(wat).toContain('i32.ge_s');
  });
});

// ────────────────────────────────────────────────────────────────────
// _detectWasmAcc — accumulator pattern detection
// ────────────────────────────────────────────────────────────────────

describe('WasmCodegen._detectWasmAcc', () => {
  function detectAcc(forNode) {
    const cg = makeCodegen();
    return cg._detectWasmAcc(forNode);
  }

  it('detects simple sum accumulation pattern', () => {
    const accBuf = buf('acc', [4], 'f32');
    const inBuf = buf('inp', [4, 8], 'f32');
    const load = new BufferLoadNode(accBuf, [idx('i')]);
    const inLoad = new BufferLoadNode(inBuf, [idx('i'), idx('j')]);
    const add = new MathOpNode('+', load, inLoad);
    const store = new BufferStoreNode(accBuf, [idx('i')], add);
    const block = new BlockNode('red', [
      { iterVar: idx('i'), binding: idx('outer_i') }
    ], [{ buffer: inBuf }], [{ buffer: accBuf }], store);
    const forNode = new ForNode(idx('j'), new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, block);
    const result = detectAcc(forNode);
    expect(result).not.toBeNull();
    expect(result.buf.name).toBe('acc');
  });

  it('detects accumulation with operands swapped (val + acc)', () => {
    const accBuf = buf('acc', [4], 'f32');
    const inBuf = buf('inp', [4, 8], 'f32');
    const load = new BufferLoadNode(accBuf, [idx('i')]);
    const inLoad = new BufferLoadNode(inBuf, [idx('i'), idx('j')]);
    const add = new MathOpNode('+', inLoad, load);
    const store = new BufferStoreNode(accBuf, [idx('i')], add);
    const block = new BlockNode('red', [], [{ buffer: inBuf }], [{ buffer: accBuf }], store);
    const forNode = new ForNode(idx('j'), new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, block);
    const result = detectAcc(forNode);
    expect(result).not.toBeNull();
  });

  it('rejects non-add pattern', () => {
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

// ────────────────────────────────────────────────────────────────────
// generate — full WAT output
// ────────────────────────────────────────────────────────────────────

describe('WasmCodegen.generate — full function', () => {
  it('generates valid WAT module with exports', () => {
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
    const result = cg.generate(pf);

    expect(result.name).toBe('negate');
    expect(result.wat).toMatch(/^\(module/);
    expect(result.wat).toContain('(export "negate")');
    expect(result.wat).toContain('(memory (export "memory")');
    expect(result.wat).toContain('f32.neg');
    expect(result.memoryPages).toBeGreaterThanOrEqual(1);
    expect(result.bufferOffsets).toBeDefined();
    expect(result.params).toEqual(['x', 'y']);
  });

  it('generates correct vector add WAT', () => {
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
    const result = cg.generate(pf);
    expect(result.wat).toContain('f32.load');
    expect(result.wat).toContain('f32.add');
    expect(result.wat).toContain('f32.store');
    expect(result.params).toEqual(['a', 'b', 'c']);
  });

  it('includes math imports in result', () => {
    const inBuf = buf('x', [4], 'f32');
    const outBuf = buf('y', [4], 'f32');
    const load = new BufferLoadNode(inBuf, [idx('i')]);
    const expCall = new CallExternNode('exp', [load], 'f32');
    const store = new BufferStoreNode(outBuf, [idx('i')], expCall);
    const block = new BlockNode('exp_blk', [], [{ buffer: inBuf }], [{ buffer: outBuf }], store);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('exp_fn', ['p0', 'p1'], forNode, bufferMap);

    const cg = makeCodegen();
    const result = cg.generate(pf);
    expect(result.imports.has('exp')).toBe(true);
    expect(result.wat).toContain('(import "math" "exp"');
    expect(result.wat).toContain('call $math_exp');
  });

  it('2D loop: emits nested block/loop structure', () => {
    const inBuf = buf('src', [2, 3], 'f32');
    const outBuf = buf('dst', [2, 3], 'f32');
    const load = new BufferLoadNode(inBuf, [idx('i'), idx('j')]);
    const store = new BufferStoreNode(outBuf, [idx('i'), idx('j')], load);
    const block = new BlockNode('cp', [], [{ buffer: inBuf }], [{ buffer: outBuf }], store);
    const innerFor = new ForNode(idx('j'), new IntImmNode(0), new IntImmNode(3), ForKind.SERIAL, block);
    const outerFor = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(2), ForKind.SERIAL, innerFor);
    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('copy2d', ['p0', 'p1'], outerFor, bufferMap);

    const cg = makeCodegen();
    const result = cg.generate(pf);
    const loops = result.wat.match(/\(loop\s/g) || [];
    expect(loops.length).toBe(2);
  });
});

// ────────────────────────────────────────────────────────────────────
// generate — IfThenElseNode
// ────────────────────────────────────────────────────────────────────

describe('WasmCodegen.generate — IfThenElseNode', () => {
  it('generates wasm if/then/else for conditional store', () => {
    const inBuf = buf('x', [4], 'f32');
    const outBuf = buf('y', [4], 'f32');
    const cond = new CompareNode('gt', new BufferLoadNode(inBuf, [idx('i')]), new FloatImmNode(0));
    const storeThen = new BufferStoreNode(outBuf, [idx('i')], new BufferLoadNode(inBuf, [idx('i')]));
    const storeElse = new BufferStoreNode(outBuf, [idx('i')], new FloatImmNode(0));
    const ifStmt = new IfThenElseNode(cond, storeThen, storeElse);
    const block = new BlockNode('relu', [], [{ buffer: inBuf }], [{ buffer: outBuf }], ifStmt);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('if_test', ['p0', 'p1'], forNode, bufferMap);

    const cg = makeCodegen();
    const result = cg.generate(pf);
    expect(result.wat).toContain('(if');
    expect(result.wat).toContain('(then');
    expect(result.wat).toContain('(else');
  });
});

// ────────────────────────────────────────────────────────────────────
// generate — LetStmtNode
// ────────────────────────────────────────────────────────────────────

describe('WasmCodegen.generate — LetStmtNode', () => {
  it('generates local.set for let binding', () => {
    const inBuf = buf('x', [4], 'f32');
    const outBuf = buf('y', [4], 'f32');
    const load = new BufferLoadNode(inBuf, [idx('i')]);
    const letVar = new VariableNode('tmp', 'f32');
    const store = new BufferStoreNode(outBuf, [idx('i')], letVar);
    const block = new BlockNode('blk', [], [{ buffer: inBuf }], [{ buffer: outBuf }], store);
    const letStmt = new LetStmtNode(letVar, load, block);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, letStmt);
    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('let_test', ['p0', 'p1'], forNode, bufferMap);

    const cg = makeCodegen();
    const result = cg.generate(pf);
    expect(result.wat).toContain('local.set $tmp');
    expect(result.wat).toMatch(/\(local\s+\$tmp\s+f32\)/);
  });
});

// ────────────────────────────────────────────────────────────────────
// generate — AllocateNode
// ────────────────────────────────────────────────────────────────────

describe('WasmCodegen.generate — AllocateNode', () => {
  it('allocates temp buffer in memory layout', () => {
    const inBuf = buf('input', [4], 'f32');
    const tmpBuf = buf('tmp', [4], 'f32');
    const outBuf = buf('output', [4], 'f32');

    const loadIn = new BufferLoadNode(inBuf, [idx('i')]);
    const storeTmp = new BufferStoreNode(tmpBuf, [idx('i')], loadIn);
    const blk1 = new BlockNode('fill', [], [{ buffer: inBuf }], [{ buffer: tmpBuf }], storeTmp);
    const loop1 = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, blk1);

    const loadTmp = new BufferLoadNode(tmpBuf, [idx('j')]);
    const storeOut = new BufferStoreNode(outBuf, [idx('j')], loadTmp);
    const blk2 = new BlockNode('cp', [], [{ buffer: tmpBuf }], [{ buffer: outBuf }], storeOut);
    const loop2 = new ForNode(idx('j'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, blk2);

    const seq = new SeqNode([loop1, loop2]);
    const alloc = new AllocateNode(tmpBuf, 'local', seq);
    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('alloc_test', ['p0', 'p1'], alloc, bufferMap);

    const cg = makeCodegen();
    const result = cg.generate(pf);
    expect(result.bufferOffsets.has('tmp')).toBe(true);
  });
});

// ────────────────────────────────────────────────────────────────────
// generate — SeqNode
// ────────────────────────────────────────────────────────────────────

describe('WasmCodegen.generate — SeqNode', () => {
  it('emits both children of SeqNode', () => {
    const b1 = buf('a', [2], 'f32');
    const b2 = buf('b', [2], 'f32');
    const s1 = new BufferStoreNode(b1, [new IntImmNode(0)], new FloatImmNode(1));
    const s2 = new BufferStoreNode(b2, [new IntImmNode(0)], new FloatImmNode(2));
    const seq = new SeqNode([s1, s2]);
    const bufferMap = new Map([['p0', b1], ['p1', b2]]);
    const pf = makePrimFunc('seq_test', ['p0', 'p1'], seq, bufferMap);

    const cg = makeCodegen();
    const result = cg.generate(pf);
    expect(result.wat).toContain('(f32.const 1)');
    expect(result.wat).toContain('(f32.const 2)');
    expect(result.wat).toContain('f32.store');
  });
});

// ────────────────────────────────────────────────────────────────────
// generate — result metadata
// ────────────────────────────────────────────────────────────────────

describe('WasmCodegen.generate — result metadata', () => {
  it('returns correct name', () => {
    const b = buf('x', [4], 'f32');
    const store = new BufferStoreNode(b, [idx('i')], new FloatImmNode(0));
    const block = new BlockNode('blk', [], [], [{ buffer: b }], store);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    const bufferMap = new Map([['p0', b]]);
    const pf = makePrimFunc('my_kernel', ['p0'], forNode, bufferMap);

    const cg = makeCodegen();
    const result = cg.generate(pf);
    expect(result.name).toBe('my_kernel');
  });

  it('returns correct params list', () => {
    const a = buf('alpha', [4], 'f32');
    const b = buf('beta', [4], 'f32');
    const store = new BufferStoreNode(b, [idx('i')], new BufferLoadNode(a, [idx('i')]));
    const block = new BlockNode('cp', [], [{ buffer: a }], [{ buffer: b }], store);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    const bufferMap = new Map([['p0', a], ['p1', b]]);
    const pf = makePrimFunc('params_test', ['p0', 'p1'], forNode, bufferMap);

    const cg = makeCodegen();
    const result = cg.generate(pf);
    expect(result.params).toEqual(['alpha', 'beta']);
  });

  it('returns memoryPages >= 1', () => {
    const b = buf('x', [4], 'f32');
    const store = new BufferStoreNode(b, [idx('i')], new FloatImmNode(0));
    const block = new BlockNode('blk', [], [], [{ buffer: b }], store);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    const bufferMap = new Map([['p0', b]]);
    const pf = makePrimFunc('mem_test', ['p0'], forNode, bufferMap);

    const cg = makeCodegen();
    const result = cg.generate(pf);
    expect(result.memoryPages).toBeGreaterThanOrEqual(1);
  });

  it('scales memoryPages for large buffers', () => {
    const b = buf('big', [100000], 'f32');
    const store = new BufferStoreNode(b, [idx('i')], new FloatImmNode(0));
    const block = new BlockNode('blk', [], [], [{ buffer: b }], store);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(100000), ForKind.SERIAL, block);
    const bufferMap = new Map([['p0', b]]);
    const pf = makePrimFunc('big_test', ['p0'], forNode, bufferMap);

    const cg = makeCodegen();
    const result = cg.generate(pf);
    expect(result.memoryPages).toBeGreaterThan(1);
  });

  it('returns empty imports for native-only ops', () => {
    const b = buf('x', [4], 'f32');
    const load = new BufferLoadNode(b, [idx('i')]);
    const sqrtCall = new CallExternNode('sqrt', [load], 'f32');
    const store = new BufferStoreNode(b, [idx('i')], sqrtCall);
    const block = new BlockNode('blk', [], [{ buffer: b }], [{ buffer: b }], store);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    const bufferMap = new Map([['p0', b]]);
    const pf = makePrimFunc('sqrt_test', ['p0'], forNode, bufferMap);

    const cg = makeCodegen();
    const result = cg.generate(pf);
    expect(result.imports.size).toBe(0);
  });
});

// ────────────────────────────────────────────────────────────────────
// generate — balanced parens in output
// ────────────────────────────────────────────────────────────────────

describe('WasmCodegen.generate — balanced WAT', () => {
  function checkBalanced(wat) {
    expect((wat.match(/\(/g) || []).length).toBe((wat.match(/\)/g) || []).length);
  }

  it('simple for loop: balanced parens', () => {
    const b = buf('x', [8], 'f32');
    const store = new BufferStoreNode(b, [idx('i')], new FloatImmNode(0));
    const block = new BlockNode('blk', [], [], [{ buffer: b }], store);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, block);
    const bufferMap = new Map([['p0', b]]);
    const pf = makePrimFunc('bal', ['p0'], forNode, bufferMap);
    checkBalanced(makeCodegen().generate(pf).wat);
  });

  it('nested loops: balanced parens', () => {
    const b = buf('x', [4, 8], 'f32');
    const store = new BufferStoreNode(b, [idx('i'), idx('j')], new FloatImmNode(0));
    const block = new BlockNode('blk', [], [], [{ buffer: b }], store);
    const innerFor = new ForNode(idx('j'), new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, block);
    const outerFor = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, innerFor);
    const bufferMap = new Map([['p0', b]]);
    const pf = makePrimFunc('bal2', ['p0'], outerFor, bufferMap);
    checkBalanced(makeCodegen().generate(pf).wat);
  });

  it('if/then/else: balanced parens', () => {
    const inBuf = buf('x', [4], 'f32');
    const outBuf = buf('y', [4], 'f32');
    const cond = new CompareNode('gt', new BufferLoadNode(inBuf, [idx('i')]), new FloatImmNode(0));
    const sThen = new BufferStoreNode(outBuf, [idx('i')], new FloatImmNode(1));
    const sElse = new BufferStoreNode(outBuf, [idx('i')], new FloatImmNode(0));
    const ifStmt = new IfThenElseNode(cond, sThen, sElse);
    const block = new BlockNode('relu', [], [{ buffer: inBuf }], [{ buffer: outBuf }], ifStmt);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('bal3', ['p0', 'p1'], forNode, bufferMap);
    checkBalanced(makeCodegen().generate(pf).wat);
  });

  it('unrolled loop: balanced parens', () => {
    const b = buf('x', [3], 'f32');
    const store = new BufferStoreNode(b, [idx('i')], new FloatImmNode(1));
    const block = new BlockNode('blk', [], [], [{ buffer: b }], store);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(3), ForKind.UNROLLED, block);
    const bufferMap = new Map([['p0', b]]);
    const pf = makePrimFunc('bal4', ['p0'], forNode, bufferMap);
    checkBalanced(makeCodegen().generate(pf).wat);
  });
});

// ────────────────────────────────────────────────────────────────────
// _emitFlatIndex — multi-dim index flattening
// ────────────────────────────────────────────────────────────────────

describe('WasmCodegen._emitFlatIndex', () => {
  function emitFlatIndex(buffer, indices) {
    const cg = makeCodegen();
    cg._lines = [];
    cg._indent = 0;
    cg._emitFlatIndex(buffer, indices);
    return cg._lines.map(l => l.trim()).join('\n');
  }

  it('1D: emits just the index', () => {
    const b = buf('v', [8], 'f32');
    const out = emitFlatIndex(b, [idx('i')]);
    expect(out).toBe('(local.get $i)');
  });

  it('2D: emits stride multiplication and add', () => {
    const b = buf('m', [3, 4], 'f32');
    const out = emitFlatIndex(b, [idx('i'), idx('j')]);
    expect(out).toContain('(i32.const 4)');
    expect(out).toContain('i32.mul');
    expect(out).toContain('i32.add');
  });

  it('3D: emits two stride multiplications', () => {
    const b = buf('t', [2, 3, 4], 'f32');
    const out = emitFlatIndex(b, [idx('i'), idx('j'), idx('k')]);
    expect(out).toContain('(i32.const 12)');
    expect(out).toContain('(i32.const 4)');
  });
});

// ────────────────────────────────────────────────────────────────────
// _prescanLocals — local variable declaration
// ────────────────────────────────────────────────────────────────────

describe('WasmCodegen._prescanLocals', () => {
  it('registers loop variables as i32 locals', () => {
    const cg = makeCodegen();
    cg._locals = new Map();
    cg._waccCounter = 0;
    const b = buf('x', [4], 'f32');
    const store = new BufferStoreNode(b, [idx('i')], new FloatImmNode(0));
    const block = new BlockNode('blk', [], [], [{ buffer: b }], store);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    cg._prescanLocals(forNode);
    expect(cg._locals.get('i')).toBe('i32');
  });

  it('registers LetStmtNode variable as local', () => {
    const cg = makeCodegen();
    cg._locals = new Map();
    cg._waccCounter = 0;
    const letVar = new VariableNode('tmp', 'f32');
    const letStmt = new LetStmtNode(letVar, new FloatImmNode(0), null);
    cg._prescanLocals(letStmt);
    expect(cg._locals.has('tmp')).toBe(true);
  });
});

// ────────────────────────────────────────────────────────────────────
// _emitMathOp — comparison operators
// ────────────────────────────────────────────────────────────────────

describe('WasmCodegen._emitExpr — comparison in MathOpNode', () => {
  function emitExpr(node) {
    const cg = makeCodegen();
    cg._lines = [];
    cg._indent = 0;
    cg._locals = new Map();
    cg._bufferOffsets = new Map();
    cg._wasmAcc = null;
    cg._emitExpr(node);
    return cg._lines.map(l => l.trim()).join('\n');
  }

  it('emits f32.lt for float <', () => {
    const node = new MathOpNode('<', new FloatImmNode(1), new FloatImmNode(2));
    expect(emitExpr(node)).toContain('f32.lt');
  });

  it('emits f32.gt for float >', () => {
    const node = new MathOpNode('>', new FloatImmNode(5), new FloatImmNode(3));
    expect(emitExpr(node)).toContain('f32.gt');
  });

  it('emits i32.le_s for int <=', () => {
    const node = new MathOpNode('<=', new IntImmNode(1), new IntImmNode(2));
    expect(emitExpr(node)).toContain('i32.le_s');
  });

  it('emits i32.ge_s for int >=', () => {
    const node = new MathOpNode('>=', new IntImmNode(5), new IntImmNode(3));
    expect(emitExpr(node)).toContain('i32.ge_s');
  });
});

// ────────────────────────────────────────────────────────────────────
// boolean type handling — LetStmtNode local type + IfThenElseNode condition
// ────────────────────────────────────────────────────────────────────

describe('WasmCodegen — boolean type system', () => {
  it('_fixLetStmtLocals declares i32 for CompareNode value', () => {
    const cg = makeCodegen();
    cg._locals = new Map();
    cg._locals.set('cse0', 'f32');
    const letVar = new VariableNode('cse0', 'f32');
    const compare = new CompareNode('gt', new FloatImmNode(1), new FloatImmNode(2));
    const letStmt = new LetStmtNode(letVar, compare, null);
    cg._fixLetStmtLocals(letStmt);
    expect(cg._locals.get('cse0')).toBe('i32');
  });

  it('_fixLetStmtLocals declares f32 for float MathOpNode value', () => {
    const cg = makeCodegen();
    cg._locals = new Map();
    const letVar = new VariableNode('tmp', 'f32');
    const add = new MathOpNode('+', new FloatImmNode(1), new FloatImmNode(2));
    const letStmt = new LetStmtNode(letVar, add, null);
    cg._fixLetStmtLocals(letStmt);
    expect(cg._locals.get('tmp')).toBe('f32');
  });

  it('_fixLetStmtLocals declares i32 for logical AND value', () => {
    const cg = makeCodegen();
    cg._locals = new Map();
    const letVar = new VariableNode('cse1');
    const cmpA = new CompareNode('gt', new FloatImmNode(1), new FloatImmNode(0));
    const cmpB = new CompareNode('lt', new FloatImmNode(1), new FloatImmNode(2));
    const andOp = new MathOpNode('&&', cmpA, cmpB);
    const letStmt = new LetStmtNode(letVar, andOp, null);
    cg._fixLetStmtLocals(letStmt);
    expect(cg._locals.get('cse1')).toBe('i32');
  });

  it('_wasmExprDtype returns i32 for CompareNode', () => {
    const cg = makeCodegen();
    cg._locals = new Map();
    const node = new CompareNode('gt', new FloatImmNode(1), new FloatImmNode(2));
    expect(cg._wasmExprDtype(node)).toBe('i32');
  });

  it('_wasmExprDtype returns i32 for boolean MathOpNode (&&, ||, !)', () => {
    const cg = makeCodegen();
    cg._locals = new Map();
    const a = new CompareNode('gt', new FloatImmNode(1), new FloatImmNode(0));
    const b = new CompareNode('lt', new FloatImmNode(1), new FloatImmNode(2));
    expect(cg._wasmExprDtype(new MathOpNode('&&', a, b))).toBe('i32');
    expect(cg._wasmExprDtype(new MathOpNode('||', a, b))).toBe('i32');
    expect(cg._wasmExprDtype(new MathOpNode('!', a))).toBe('i32');
  });

  it('_wasmExprDtype returns i32 for comparison MathOpNode (<, >, <=, >=)', () => {
    const cg = makeCodegen();
    cg._locals = new Map();
    expect(cg._wasmExprDtype(new MathOpNode('<', new FloatImmNode(1), new FloatImmNode(2)))).toBe('i32');
    expect(cg._wasmExprDtype(new MathOpNode('>', new IntImmNode(5), new IntImmNode(3)))).toBe('i32');
    expect(cg._wasmExprDtype(new MathOpNode('<=', new IntImmNode(1), new IntImmNode(2)))).toBe('i32');
    expect(cg._wasmExprDtype(new MathOpNode('>=', new IntImmNode(5), new IntImmNode(3)))).toBe('i32');
  });

  it('_wasmExprDtype returns local type for VariableNode', () => {
    const cg = makeCodegen();
    cg._locals = new Map();
    cg._locals.set('cse0', 'i32');
    cg._locals.set('tmp', 'f32');
    expect(cg._wasmExprDtype(new VariableNode('cse0'))).toBe('i32');
    expect(cg._wasmExprDtype(new VariableNode('tmp'))).toBe('f32');
  });

  it('IfThenElseNode skips f32.ne for i32 condition (bool op)', () => {
    const cg = makeCodegen();
    cg._lines = [];
    cg._indent = 0;
    cg._locals = new Map();
    cg._locals.set('cond', 'i32');
    cg._bufferOffsets = new Map();
    cg._wasmAcc = null;
    const cond = new MathOpNode('||',
      new CompareNode('gt', new FloatImmNode(1), new FloatImmNode(0)),
      new CompareNode('lt', new FloatImmNode(1), new FloatImmNode(2))
    );
    const ite = new IfThenElseNode(cond, new FloatImmNode(1), new FloatImmNode(0));
    cg._emitExpr(ite);
    const wat = cg._lines.map(l => l.trim()).join('\n');
    expect(wat).not.toContain('f32.ne');
    expect(wat).toContain('i32.or');
    expect(wat).toContain('(if (result f32)');
  });

  it('IfThenElseNode adds f32.ne for float condition', () => {
    const cg = makeCodegen();
    cg._lines = [];
    cg._indent = 0;
    cg._locals = new Map();
    cg._bufferOffsets = new Map();
    cg._wasmAcc = null;
    const cond = new FloatImmNode(1.0);
    const ite = new IfThenElseNode(cond, new FloatImmNode(1), new FloatImmNode(0));
    cg._emitExpr(ite);
    const wat = cg._lines.map(l => l.trim()).join('\n');
    expect(wat).toContain('f32.ne');
  });
});
