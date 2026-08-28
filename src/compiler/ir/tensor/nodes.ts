import { withFuncAttrs } from '../func_attrs.js';
import { floorDiv, floorMod } from '../../../util/divmod.js';
import type { Location } from '../location.js';
import type { Buffer, BufferRegionLike } from './buffer.js';

export const ForKind = Object.freeze({
  SERIAL: 'serial',
  PARALLEL: 'parallel',
  VECTORIZED: 'vectorized',
  UNROLLED: 'unrolled',
  THREAD_BINDING: 'thread_binding',
  RECURRENCE: 'recurrence'
});

export type ForKindValue = (typeof ForKind)[keyof typeof ForKind];

export class TensorNode {
  type: string;
  declare _dtype?: string;
  _parent: TensorNode | null;
  _parentKey: string | null;
  _parentIdx: number;

  constructor() {
    this.type = this.constructor.name.replace(/^_+/, '');
    this._parent = null;
    this._parentKey = null;
    this._parentIdx = -1;
  }

  _setChild(key: string, child: TensorNode | null, idx = -1): void {
    if (child instanceof TensorNode) {
      child._parent = this;
      child._parentKey = key;
      child._parentIdx = idx;
    }
  }

  _setChildren(key: string, arr: readonly (TensorNode | null)[] | null): void {
    if (!arr) return;
    for (let i = 0; i < arr.length; i++) {
      const child = arr[i];
      if (child instanceof TensorNode) {
        child._parent = this;
        child._parentKey = key;
        child._parentIdx = i;
      }
    }
  }

  replaceWith(newNode: TensorNode | null): boolean {
    const p = this._parent;
    if (!p) return false;
    const slots = p as unknown as Record<string, TensorNode | null | (TensorNode | null)[]>;
    const key = this._parentKey as string;
    if (this._parentIdx >= 0) {
      (slots[key] as (TensorNode | null)[])[this._parentIdx] = newNode;
    } else {
      slots[key] = newNode;
    }
    if (newNode instanceof TensorNode) {
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

export type PrimFuncParam = VariableNode;

export class PrimFunc extends withFuncAttrs(TensorNode) {
  declare type: 'PrimFunc';
  declare _module?: import('./module.js').TirModule;
  name: string;
  params: PrimFuncParam[];
  body: TirNode;
  bufferMap: Map<VariableNode, Buffer>;
  shapeParams: VariableNode[];
  shapeParamMap: Map<string, VariableNode>;

  constructor(
    name: string,
    params: PrimFuncParam[],
    body: TirNode,
    bufferMap: Map<VariableNode, Buffer> = new Map(),
    shapeParams: VariableNode[] = [],
    shapeParamMap: Map<string, VariableNode> | null = null,
  ) {
    super();
    this.name = name;
    this.params = params;
    this.body = body;
    this.bufferMap = bufferMap;
    this.shapeParams = shapeParams;
    this.shapeParamMap = shapeParamMap || new Map();
    this._setChild('body', body);
  }
}

export class ForNode extends TensorNode {
  declare type: 'ForNode';
  loopVar: VariableNode;
  min: TirNode;
  extent: TirNode;
  kind: ForKindValue;
  body: TirNode;
  threadTag: string | null;
  declare accumulator?: unknown;
  declare annotations?: Record<string, unknown>;

  constructor(loopVar: VariableNode, min: TirNode, extent: TirNode, kind: ForKindValue, body: TirNode, threadTag: string | null = null) {
    super();
    this.loopVar = loopVar;
    this.min = min;
    this.extent = extent;
    this.kind = kind;
    this.body = body;
    this.threadTag = threadTag;
    this._setChild('body', body);
  }
}

export type BlockSourceOp = { name: string; id: number; loc: Location | null };

export class BlockNode extends TensorNode {
  declare type: 'BlockNode';
  declare sourceOp?: BlockSourceOp;
  name: string;
  iterVars: BlockRealizeNode[];
  reads: BufferRegionLike[];
  writes: BufferRegionLike[];
  body: TirNode;
  initBody: TirNode | null;

  constructor(name: string, iterVars: BlockRealizeNode[], reads: BufferRegionLike[], writes: BufferRegionLike[], body: TirNode, initBody: TirNode | null = null) {
    super();
    this.name = name;
    this.iterVars = iterVars;
    this.reads = reads;
    this.writes = writes;
    this.body = body;
    this.initBody = initBody;
    this._setChild('body', body);
    this._setChild('initBody', initBody);
  }
}

export const IterVarKind = Object.freeze({ DATA_PAR: 'DataPar', COMM_REDUCE: 'CommReduce' });

export type IterVarKindValue = (typeof IterVarKind)[keyof typeof IterVarKind];

export type NodeSlots = Record<string, TirNode | TirNode[] | undefined>;

export class BlockRealizeNode extends TensorNode {
  declare type: 'BlockRealizeNode';
  iterVar: VariableNode;
  binding: TirNode;
  kind: IterVarKindValue;

  constructor(iterVar: VariableNode, binding: TirNode, kind: IterVarKindValue = IterVarKind.DATA_PAR) {
    super();
    this.iterVar = iterVar;
    this.binding = binding;
    this.kind = kind;
  }
}

export class BufferStoreNode extends TensorNode {
  declare type: 'BufferStoreNode';
  buffer: Buffer;
  indices: TirNode[];
  value: TirNode;

  constructor(buffer: Buffer, indices: TirNode[], value: TirNode) {
    super();
    this.buffer = buffer;
    this.indices = indices;
    this.value = value;
  }
}

export class BufferLoadNode extends TensorNode {
  declare type: 'BufferLoadNode';
  buffer: Buffer;
  indices: TirNode[];

  constructor(buffer: Buffer, indices: TirNode[]) {
    super();
    this.buffer = buffer;
    this.indices = indices;
  }
}

export class IfThenElseNode extends TensorNode {
  declare type: 'IfThenElseNode';
  condition: TirNode;
  thenBody: TirNode;
  elseBody: TirNode | null;

  constructor(condition: TirNode, thenBody: TirNode, elseBody: TirNode | null = null) {
    super();
    this.condition = condition;
    this.thenBody = thenBody;
    this.elseBody = elseBody;
    this._setChild('thenBody', thenBody);
    this._setChild('elseBody', elseBody);
  }
}

export class LetStmtNode extends TensorNode {
  declare type: 'LetStmtNode';
  variable: VariableNode;
  value: TirNode;
  body: TirNode;

  constructor(variable: VariableNode, value: TirNode, body: TirNode) {
    super();
    this.variable = variable;
    this.value = value;
    this.body = body;
    this._setChild('body', body);
  }
}

export class AllocateNode extends TensorNode {
  declare type: 'AllocateNode';
  buffer: Buffer;
  scope: string;
  body: TirNode;

  constructor(buffer: Buffer, scope: string, body: TirNode) {
    super();
    this.buffer = buffer;
    this.scope = scope;
    this.body = body;
    this._setChild('body', body);
  }
}

export class SeqNode extends TensorNode {
  declare type: 'SeqNode';
  stmts: TirNode[];

  constructor(stmts: TirNode[]) {
    super();
    this.stmts = stmts;
    this._setChildren('stmts', stmts);
  }
}

export class WhileNode extends TensorNode {
  declare type: 'WhileNode';
  condVar: VariableNode | Buffer;
  condBody: TirNode;
  loopBody: TirNode;

  constructor(condVar: VariableNode | Buffer, condBody: TirNode, loopBody: TirNode) {
    super();
    this.condVar = condVar;
    this.condBody = condBody;
    this.loopBody = loopBody;
    this._setChild('condBody', condBody);
    this._setChild('loopBody', loopBody);
  }
}

export class EvaluateNode extends TensorNode {
  declare type: 'EvaluateNode';
  value: TirNode;

  constructor(value: TirNode) {
    super();
    this.value = value;
  }
}

export class SyncThreadsNode extends TensorNode {
  declare type: 'SyncThreadsNode';
  constructor() {
    super();
  }
}

export class VecCopyNode extends TensorNode {
  declare type: 'VecCopyNode';
  dstBuffer: Buffer;
  dstIndex: TirNode;
  srcBuffer: Buffer;
  srcIndex: TirNode;
  width: number;

  constructor(dstBuffer: Buffer, dstIndex: TirNode, srcBuffer: Buffer, srcIndex: TirNode, width: number) {
    super();
    this.dstBuffer = dstBuffer;
    this.dstIndex = dstIndex;
    this.srcBuffer = srcBuffer;
    this.srcIndex = srcIndex;
    this.width = width;
  }
}

export class CallExternNode extends TensorNode {
  declare type: 'CallExternNode';
  externName: string;
  args: TirNode[];
  dtype: string;

  constructor(name: string, args: TirNode[], dtype: string) {
    super();
    this.externName = name;
    this.args = args;
    this.dtype = dtype;
  }
}

export class MathOpNode extends TensorNode {
  declare type: 'MathOpNode';
  op: string;
  a: TirNode;
  b: TirNode | null;

  constructor(op: string, a: TirNode, b: TirNode | null = null) {
    super();
    this.op = op;
    this.a = a;
    this.b = b;
  }
}

export class CompareNode extends TensorNode {
  declare type: 'CompareNode';
  direction: string;
  a: TirNode;
  b: TirNode;

  constructor(direction: string, a: TirNode, b: TirNode) {
    super();
    this.direction = direction;
    this.a = a;
    this.b = b;
  }
}

export class CastNode extends TensorNode {
  declare type: 'CastNode';
  expr: TirNode;
  fromDtype: string;
  toDtype: string;

  constructor(expr: TirNode, fromDtype: string, toDtype: string) {
    super();
    this.expr = expr;
    this.fromDtype = fromDtype;
    this.toDtype = toDtype;
  }
}

export class VariableNode extends TensorNode {
  declare type: 'VariableNode';
  name: string;
  dtype: string;

  constructor(name: string, dtype: string) {
    super();
    this.name = name;
    this.dtype = dtype;
  }
}

export class IntImmNode extends TensorNode {
  declare type: 'IntImmNode';
  value: number;

  constructor(value: number) {
    super();
    this.value = value;
  }
}

export class FloatImmNode extends TensorNode {
  declare type: 'FloatImmNode';
  value: number;

  constructor(value: number) {
    super();
    this.value = value;
  }
}

export function mathOp(op: string, a: TirNode, b: TirNode | null): TirNode {
  if (b === null || b === undefined) return new MathOpNode(op, a, b);

  const aIsInt = a instanceof IntImmNode;
  const bIsInt = b instanceof IntImmNode;

  if (aIsInt && bIsInt) {
    const av = a.value, bv = b.value;
    switch (op) {
      case '+': return new IntImmNode(av + bv);
      case '-': return new IntImmNode(av - bv);
      case '*': return new IntImmNode(av * bv);
      case '//': if (bv !== 0) return new IntImmNode(floorDiv(av, bv)); break;
      case '%': if (bv !== 0) return new IntImmNode(floorMod(av, bv)); break;
    }
  }

  if (bIsInt) {
    const bv = b.value;
    if ((op === '+' || op === '-') && bv === 0) return a;
    if (op === '*' && bv === 1) return a;
    if (op === '*' && bv === 0) return new IntImmNode(0);
    if (op === '//' && bv === 1) return a;
    if (op === '%' && bv === 1) return new IntImmNode(0);
  }

  if (aIsInt) {
    const av = a.value;
    if (op === '+' && av === 0) return b;
    if (op === '*' && av === 1) return b;
    if (op === '*' && av === 0) return new IntImmNode(0);
  }

  return new MathOpNode(op, a, b);
}

export type TirNode =
  PrimFunc
  | ForNode
  | BlockNode
  | BlockRealizeNode
  | BufferStoreNode
  | BufferLoadNode
  | IfThenElseNode
  | LetStmtNode
  | AllocateNode
  | SeqNode
  | WhileNode
  | EvaluateNode
  | SyncThreadsNode
  | VecCopyNode
  | CallExternNode
  | MathOpNode
  | CompareNode
  | CastNode
  | VariableNode
  | IntImmNode
  | FloatImmNode;
