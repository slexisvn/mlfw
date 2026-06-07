import { describe, it, expect } from 'vitest';
import {
  LIRNode, LIRFunc, LIRFlatLoadNode, LIRFlatStoreNode,
  LIRAccumulatorNode, LIRBindingsNode, LIRMetadata,
  inferDtype, annotateDtype, normalizeDtype, isWasmNativeOp,
} from '../../../../src/compiler/ir/lir/nodes.js';
import { Buffer } from '../../../../src/compiler/ir/tensor/buffer.js';
import {
  IntImmNode, FloatImmNode, VariableNode,
  MathOpNode, CompareNode, CastNode, CallExternNode,
  IfThenElseNode, BufferLoadNode, ForKind,
} from '../../../../src/compiler/ir/tensor/nodes.js';

function buf(name, shape, dtype = 'f32') {
  return new Buffer(name, shape, dtype, 'global');
}

describe('LIRNode — parent tracking', () => {
  it('sets parent on child assignment', () => {
    const store = new LIRFlatStoreNode(buf('x', [4]), new IntImmNode(0), new FloatImmNode(1), 'f32');
    expect(store.offsetExpr._parent).toBe(store);
    expect(store.value._parent).toBe(store);
  });

  it('replaceWith updates parent reference', () => {
    const store = new LIRFlatStoreNode(buf('x', [4]), new IntImmNode(0), new FloatImmNode(1), 'f32');
    const newVal = new FloatImmNode(2);
    store.value.replaceWith(newVal);
    expect(store.value).toBe(newVal);
    expect(newVal._parent).toBe(store);
  });
});

describe('LIRFunc', () => {
  it('stores all fields correctly', () => {
    const meta = new LIRMetadata();
    const body = new LIRFlatStoreNode(buf('x', [1]), new IntImmNode(0), new FloatImmNode(0), 'f32');
    const bm = new Map([['p0', buf('x', [1])]]);
    const func = new LIRFunc('test', ['p0'], body, bm, [], new Map(), meta);
    expect(func.name).toBe('test');
    expect(func.type).toBe('LIRFunc');
    expect(func.body).toBe(body);
    expect(func.metadata).toBe(meta);
    expect(body._parent).toBe(func);
  });
});

describe('LIRFlatLoadNode', () => {
  it('stores buffer, offset, dtype', () => {
    const b = buf('data', [8]);
    const offset = new MathOpNode('*', new VariableNode('i', 'index'), new IntImmNode(4));
    const load = new LIRFlatLoadNode(b, offset, 'f32');
    expect(load.type).toBe('LIRFlatLoadNode');
    expect(load.buffer.name).toBe('data');
    expect(load.dtype).toBe('f32');
    expect(load.offsetExpr).toBe(offset);
  });
});

describe('LIRFlatStoreNode', () => {
  it('stores buffer, offset, value, dtype', () => {
    const b = buf('out', [8]);
    const offset = new VariableNode('i', 'index');
    const value = new FloatImmNode(3.14);
    const store = new LIRFlatStoreNode(b, offset, value, 'f32');
    expect(store.type).toBe('LIRFlatStoreNode');
    expect(store.buffer.name).toBe('out');
    expect(store.value).toBe(value);
  });
});

describe('LIRAccumulatorNode', () => {
  it('stores all accumulator fields', () => {
    const b = buf('acc', [4]);
    const initLoad = new LIRFlatLoadNode(b, new VariableNode('i', 'index'), 'f32');
    const body = new FloatImmNode(1);
    const flushStore = new LIRFlatStoreNode(b, new VariableNode('i', 'index'), null, 'f32');
    const acc = new LIRAccumulatorNode({
      localName: '_acc_0',
      dtype: 'f32',
      initLoad,
      loopVar: new VariableNode('j', 'index'),
      extent: new IntImmNode(8),
      loopKind: ForKind.SERIAL,
      body,
      flushStore,
    });
    expect(acc.type).toBe('LIRAccumulatorNode');
    expect(acc.localName).toBe('_acc_0');
    expect(acc.dtype).toBe('f32');
    expect(acc.loopKind).toBe(ForKind.SERIAL);
  });
});

describe('LIRBindingsNode', () => {
  it('stores bindings and body', () => {
    const body = new LIRFlatStoreNode(buf('x', [4]), new IntImmNode(0), new FloatImmNode(0), 'f32');
    const bindings = [{ name: 'bi', dtype: 'index', expr: new VariableNode('i', 'index') }];
    const node = new LIRBindingsNode(bindings, body);
    expect(node.type).toBe('LIRBindingsNode');
    expect(node.bindings.length).toBe(1);
    expect(node.body).toBe(body);
  });
});

describe('LIRMetadata', () => {
  it('initializes with empty collections', () => {
    const meta = new LIRMetadata();
    expect(meta.locals.size).toBe(0);
    expect(meta.externCalls.size).toBe(0);
    expect(meta.memoryLayout.totalBytes).toBe(0);
    expect(meta.zeroBuffers.size).toBe(0);
    expect(meta.threadBindings.size).toBe(0);
    expect(meta.sharedBuffers.length).toBe(0);
  });
});

describe('inferDtype', () => {
  it('IntImmNode -> i32', () => {
    expect(inferDtype(new IntImmNode(0))).toBe('i32');
  });

  it('FloatImmNode -> f32', () => {
    expect(inferDtype(new FloatImmNode(0))).toBe('f32');
  });

  it('BufferLoadNode -> buffer dtype', () => {
    const b = buf('x', [4], 'f32');
    expect(inferDtype(new BufferLoadNode(b, [new IntImmNode(0)]))).toBe('f32');
  });

  it('LIRFlatLoadNode -> dtype', () => {
    const load = new LIRFlatLoadNode(buf('x', [4]), new IntImmNode(0), 'i32');
    expect(inferDtype(load)).toBe('i32');
  });

  it('CastNode -> toDtype', () => {
    expect(inferDtype(new CastNode(new IntImmNode(1), 'i32', 'f32'))).toBe('f32');
  });

  it('CompareNode -> i32', () => {
    expect(inferDtype(new CompareNode('gt', new IntImmNode(1), new IntImmNode(0)))).toBe('i32');
  });

  it('MathOpNode float+int -> f32', () => {
    const node = new MathOpNode('+', new FloatImmNode(1), new IntImmNode(2));
    expect(inferDtype(node)).toBe('f32');
  });

  it('MathOpNode int+int -> i32', () => {
    const node = new MathOpNode('+', new IntImmNode(1), new IntImmNode(2));
    expect(inferDtype(node)).toBe('i32');
  });

  it('CallExternNode -> node dtype', () => {
    const node = new CallExternNode('exp', [new FloatImmNode(1)], 'f32');
    expect(inferDtype(node)).toBe('f32');
  });

  it('IfThenElseNode -> thenBody dtype', () => {
    const node = new IfThenElseNode(new IntImmNode(1), new FloatImmNode(1), new FloatImmNode(0));
    expect(inferDtype(node)).toBe('f32');
  });

  it('VariableNode index -> i32', () => {
    expect(inferDtype(new VariableNode('i', 'index'))).toBe('i32');
  });

  it('VariableNode float32 -> f32', () => {
    expect(inferDtype(new VariableNode('x', 'float32'))).toBe('f32');
  });

  it('null -> f32 default', () => {
    expect(inferDtype(null)).toBe('f32');
  });

  it('uses cached _dtype if present', () => {
    const node = new IntImmNode(5);
    node._dtype = 'f64';
    expect(inferDtype(node)).toBe('f64');
  });
});

describe('annotateDtype', () => {
  it('sets _dtype on node', () => {
    const node = new IntImmNode(5);
    annotateDtype(node);
    expect(node._dtype).toBe('i32');
  });

  it('no-ops on null', () => {
    expect(() => annotateDtype(null)).not.toThrow();
  });
});

describe('normalizeDtype', () => {
  it('maps index to i32', () => expect(normalizeDtype('index')).toBe('i32'));
  it('maps int32 to i32', () => expect(normalizeDtype('int32')).toBe('i32'));
  it('maps float32 to f32', () => expect(normalizeDtype('float32')).toBe('f32'));
  it('maps f32 to f32', () => expect(normalizeDtype('f32')).toBe('f32'));
  it('maps i8 to i32', () => expect(normalizeDtype('i8')).toBe('i32'));
  it('maps bool to i32', () => expect(normalizeDtype('bool')).toBe('i32'));
  it('defaults unknown to f32', () => expect(normalizeDtype('unknown')).toBe('f32'));
});

describe('isWasmNativeOp', () => {
  it('sqrt is native', () => expect(isWasmNativeOp('sqrt')).toBe(true));
  it('abs is native', () => expect(isWasmNativeOp('abs')).toBe(true));
  it('ceil is native', () => expect(isWasmNativeOp('ceil')).toBe(true));
  it('floor is native', () => expect(isWasmNativeOp('floor')).toBe(true));
  it('min is native', () => expect(isWasmNativeOp('min')).toBe(true));
  it('max is native', () => expect(isWasmNativeOp('max')).toBe(true));
  it('exp is NOT native', () => expect(isWasmNativeOp('exp')).toBe(false));
  it('log is NOT native', () => expect(isWasmNativeOp('log')).toBe(false));
  it('tanh is NOT native', () => expect(isWasmNativeOp('tanh')).toBe(false));
  it('sin is NOT native', () => expect(isWasmNativeOp('sin')).toBe(false));
});
