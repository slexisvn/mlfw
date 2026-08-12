import { isDtypeFloat } from '../../../util/dtype_map.js';
import { TensorNode } from '../tensor/nodes.js';
import { withFuncAttrs } from '../func_attrs.js';
import type { Buffer } from '../tensor/buffer.js';
import type { TirNode, VariableNode } from '../tensor/nodes.js';

export type IRStmtNode = TirNode | LirNode;

export class LIRFunc extends withFuncAttrs(TensorNode) {
  declare type: 'LIRFunc';
  name: string;
  params: readonly VariableNode[];
  body: IRStmtNode;
  bufferMap: Map<VariableNode, Buffer>;
  shapeParams: readonly VariableNode[];
  shapeParamMap: Map<string, VariableNode>;
  metadata: LIRMetadata;

  constructor(
    name: string,
    params: readonly VariableNode[],
    body: IRStmtNode,
    bufferMap: Map<VariableNode, Buffer>,
    shapeParams: readonly VariableNode[],
    shapeParamMap: Map<string, VariableNode>,
    metadata: LIRMetadata,
  ) {
    super();
    this.name = name;
    this.params = params;
    this.body = body;
    this.bufferMap = bufferMap;
    this.shapeParams = shapeParams;
    this.shapeParamMap = shapeParamMap;
    this.metadata = metadata;
    this._setChild('body', body as TensorNode);
  }
}

export class LIRFlatLoadNode extends TensorNode {
  declare type: 'LIRFlatLoadNode';
  buffer: Buffer;
  offsetExpr: IRStmtNode;
  dtype: string;

  constructor(buffer: Buffer, offsetExpr: IRStmtNode, dtype: string) {
    super();
    this.buffer = buffer;
    this.offsetExpr = offsetExpr;
    this.dtype = dtype;
    this._setChild('offsetExpr', offsetExpr);
  }
}

export class LIRFlatStoreNode extends TensorNode {
  declare type: 'LIRFlatStoreNode';
  buffer: Buffer;
  offsetExpr: IRStmtNode;
  value: IRStmtNode | null;
  dtype: string;

  constructor(buffer: Buffer, offsetExpr: IRStmtNode, value: IRStmtNode | null, dtype: string) {
    super();
    this.buffer = buffer;
    this.offsetExpr = offsetExpr;
    this.value = value;
    this.dtype = dtype;
    this._setChild('offsetExpr', offsetExpr);
    this._setChild('value', value);
  }
}

export type LIRAccumulatorConfig = Readonly<{
  localName: string;
  dtype: string;
  op?: string;
  initLoad: LIRFlatLoadNode;
  loopVar: VariableNode;
  extent: IRStmtNode;
  loopKind: string;
  body: IRStmtNode;
  flushStore: LIRFlatStoreNode;
  initBody?: IRStmtNode | null;
}>;

export class LIRAccumulatorNode extends TensorNode {
  declare type: 'LIRAccumulatorNode';
  localName: string;
  dtype: string;
  op: string;
  initLoad: LIRFlatLoadNode;
  loopVar: VariableNode;
  extent: IRStmtNode;
  loopKind: string;
  body: IRStmtNode;
  flushStore: LIRFlatStoreNode;
  initBody: IRStmtNode | null;

  constructor(config: LIRAccumulatorConfig) {
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

export class LIRBindingsNode extends TensorNode {
  declare type: 'LIRBindingsNode';
  bindings: readonly { name: string; expr: IRStmtNode }[];
  body: IRStmtNode;

  constructor(bindings: readonly { name: string; expr: IRStmtNode }[], body: IRStmtNode) {
    super();
    this.bindings = bindings;
    this.body = body;
    this._setChild('body', body);
  }
}

export type LIRThreadBinding = { varName: string; extent: number; isDynamic: boolean; extentNode: IRStmtNode | null };
export type LIRExternCall = { argCount: number; dtype: string };

export type LIRMemoryLayout = {
  bufferOffsets: Map<string, number>;
  totalBytes: number;
  alignment: number;
};

export class LIRMetadata {
  locals: Map<string, string>;
  externCalls: Map<string, LIRExternCall>;
  memoryLayout: LIRMemoryLayout;
  threadBindings: Map<string, LIRThreadBinding[]>;
  sharedBuffers: Buffer[];
  zeroBuffers: Set<string>;
  constantBuffers: Map<string, number>;
  usedBuffers: Map<string, Buffer>;
  allocatedBuffers: Set<string>;
  paramBuffers: Set<string>;

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

const DTYPE_NORMALIZE: Readonly<Record<string, string>> = {
  'int32': 'i32', 'index': 'i32', 'i32': 'i32',
  'float32': 'f32', 'f32': 'f32', 'f64': 'f64',
  'f16': 'f32', 'bf16': 'f32', 'i8': 'i32', 'i16': 'i32',
  'i64': 'i64', 'ui8': 'i32', 'bool': 'i32',
};

export function normalizeDtype(dtype: string): string {
  return DTYPE_NORMALIZE[dtype] || 'f32';
}

export type DtypeInferable = {
  type?: string;
  _dtype?: string;
  dtype?: string;
  toDtype?: string;
  buffer?: { dtype: string };
  a?: DtypeInferable | null;
  b?: DtypeInferable | null;
  thenBody?: DtypeInferable | null;
};

export function inferDtype(node: DtypeInferable | null | undefined): string {
  if (!node) return 'f32';
  if (node._dtype) return node._dtype;

  switch (node.type) {
    case 'IntImmNode': return 'i32';
    case 'FloatImmNode': return 'f32';
    case 'LIRFlatLoadNode': return normalizeDtype(node.dtype as string);
    case 'BufferLoadNode': return normalizeDtype((node.buffer as { dtype: string }).dtype);
    case 'CastNode': return normalizeDtype(node.toDtype as string);
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

export function annotateDtype(node: DtypeInferable | null | undefined): void {
  if (!node || typeof node !== 'object' || !node.type) return;
  node._dtype = inferDtype(node);
}

export function isWasmNativeOp(name: string): boolean {
  return WASM_NATIVE_OPS.has(name);
}

export type LirNode =
  LIRFunc
  | LIRFlatLoadNode
  | LIRFlatStoreNode
  | LIRAccumulatorNode
  | LIRBindingsNode;
