import { isDtypeFloat } from '../../../backend/dtype_map.js';

export class LIRNode {
  constructor() {
    this.type = this.constructor.name.replace(/^_+/, '');
    this._parent = null;
    this._parentKey = null;
    this._parentIdx = -1;
  }

  _setChild(key, child, idx = -1) {
    if (child instanceof LIRNode || (child && child._parent !== undefined)) {
      child._parent = this;
      child._parentKey = key;
      child._parentIdx = idx;
    }
  }

  _setChildren(key, arr) {
    if (!arr) return;
    for (let i = 0; i < arr.length; i++) {
      const c = arr[i];
      if (c instanceof LIRNode || (c && c._parent !== undefined)) {
        c._parent = this;
        c._parentKey = key;
        c._parentIdx = i;
      }
    }
  }

  replaceWith(newNode) {
    const p = this._parent;
    if (!p) return false;
    if (this._parentIdx >= 0) {
      p[this._parentKey][this._parentIdx] = newNode;
    } else {
      p[this._parentKey] = newNode;
    }
    if (newNode && (newNode instanceof LIRNode || newNode._parent !== undefined)) {
      newNode._parent = p;
      newNode._parentKey = this._parentKey;
      newNode._parentIdx = this._parentIdx;
    }
    this._parent = null;
    this._parentKey = null;
    this._parentIdx = -1;
    return true;
  }
}

export class LIRFunc extends LIRNode {
  constructor(name, params, body, bufferMap, shapeParams, shapeParamMap, metadata) {
    super();
    this.name = name;
    this.params = params;
    this.body = body;
    this.bufferMap = bufferMap;
    this.shapeParams = shapeParams;
    this.shapeParamMap = shapeParamMap;
    this.metadata = metadata;
    this._setChild('body', body);
  }
}

export class LIRFlatLoadNode extends LIRNode {
  constructor(buffer, offsetExpr, dtype) {
    super();
    this.buffer = buffer;
    this.offsetExpr = offsetExpr;
    this.dtype = dtype;
    this._setChild('offsetExpr', offsetExpr);
  }
}

export class LIRFlatStoreNode extends LIRNode {
  constructor(buffer, offsetExpr, value, dtype) {
    super();
    this.buffer = buffer;
    this.offsetExpr = offsetExpr;
    this.value = value;
    this.dtype = dtype;
    this._setChild('offsetExpr', offsetExpr);
    this._setChild('value', value);
  }
}

export class LIRAccumulatorNode extends LIRNode {
  constructor(config) {
    super();
    this.localName = config.localName;
    this.dtype = config.dtype;
    this.op = config.op || '+';
    this.initLoad = config.initLoad;
    this.loopVar = config.loopVar;
    this.extent = config.extent;
    this.loopKind = config.loopKind;
    this.body = config.body;
    this.flushStore = config.flushStore;
    this.initBody = config.initBody || null;
    this._setChild('initLoad', config.initLoad);
    this._setChild('body', config.body);
    this._setChild('flushStore', config.flushStore);
    this._setChild('initBody', config.initBody || null);
  }
}

export class LIRBindingsNode extends LIRNode {
  constructor(bindings, body) {
    super();
    this.bindings = bindings;
    this.body = body;
    this._setChild('body', body);
  }
}

export class LIRMetadata {
  constructor() {
    this.locals = new Map();
    this.externCalls = new Map();
    this.memoryLayout = { bufferOffsets: new Map(), totalBytes: 0, alignment: 16 };
    this.threadBindings = new Map();
    this.sharedBuffers = [];
    this.zeroBuffers = new Set();
    this.constantBuffers = new Map();
    this.usedBuffers = new Map();
    this.allocatedBuffers = new Set();
    this.paramBuffers = new Set();
  }
}

const WASM_NATIVE_OPS = new Set(['sqrt', 'abs', 'ceil', 'floor', 'min', 'max']);

const DTYPE_NORMALIZE = {
  'int32': 'i32', 'index': 'i32', 'i32': 'i32',
  'float32': 'f32', 'f32': 'f32', 'f64': 'f64',
  'f16': 'f32', 'bf16': 'f32', 'i8': 'i32', 'i16': 'i32',
  'i64': 'i64', 'ui8': 'i32', 'bool': 'i32',
};

export function normalizeDtype(dtype) {
  return DTYPE_NORMALIZE[dtype] || 'f32';
}

export function inferDtype(node) {
  if (!node) return 'f32';
  if (node._dtype) return node._dtype;

  switch (node.type) {
    case 'IntImmNode': return 'i32';
    case 'FloatImmNode': return 'f32';
    case 'LIRFlatLoadNode': return normalizeDtype(node.dtype);
    case 'BufferLoadNode': return normalizeDtype(node.buffer.dtype);
    case 'CastNode': return normalizeDtype(node.toDtype);
    case 'CallExternNode': return normalizeDtype(node.dtype || 'f32');
    case 'CompareNode': return 'i32';
    case 'VariableNode': return normalizeDtype(node.dtype || 'i32');
    case 'MathOpNode': {
      const da = inferDtype(node.a);
      if (isDtypeFloat(da)) return da;
      if (node.b) {
        const db = inferDtype(node.b);
        if (isDtypeFloat(db)) return db;
      }
      return da;
    }
    case 'IfThenElseNode': return inferDtype(node.thenBody);
    default: return 'f32';
  }
}

export function annotateDtype(node) {
  if (!node || typeof node !== 'object' || !node.type) return;
  node._dtype = inferDtype(node);
}

export function isWasmNativeOp(name) {
  return WASM_NATIVE_OPS.has(name);
}
