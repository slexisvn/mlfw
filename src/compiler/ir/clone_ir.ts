export type CloneableIRNode = Record<string, unknown> & { type?: string };
export type CloneRecurse = (node: unknown) => unknown;
export type CloneFallback = (node: CloneableIRNode, copy: CloneableIRNode, rec: CloneRecurse) => unknown;

export function cloneIRShared(node: unknown, rec: CloneRecurse, handleOther: CloneFallback): unknown {
  if (!node || typeof node !== 'object') return node;
  if (Array.isArray(node)) return node.map(rec);
  const src = node as CloneableIRNode & { attrs?: unknown; _setChild?: (k: string, c: unknown) => void; _setChildren?: (k: string, c: unknown) => void };
  const copy = Object.create(Object.getPrototypeOf(node)) as typeof src;
  copy.type = src.type;
  copy._parent = null;
  copy._parentKey = null;
  copy._parentIdx = -1;
  if (src.attrs instanceof Map) copy.attrs = new Map(src.attrs);
  switch (src.type) {
    case 'ForNode':
      copy.loopVar = src.loopVar;
      copy.min = rec(src.min);
      copy.extent = rec(src.extent);
      copy.kind = src.kind;
      copy.body = rec(src.body);
      copy.threadTag = src.threadTag;
      (copy._setChild as (k: string, c: unknown) => void)('body', copy.body);
      return copy;
    case 'BlockNode':
      copy.name = src.name;
      copy.iterVars = (src.iterVars as unknown[]).map(rec);
      copy.reads = src.reads;
      copy.writes = src.writes;
      copy.body = rec(src.body);
      copy.initBody = src.initBody ? rec(src.initBody) : null;
      (copy._setChild as (k: string, c: unknown) => void)('body', copy.body);
      (copy._setChild as (k: string, c: unknown) => void)('initBody', copy.initBody);
      return copy;
    case 'SeqNode':
      copy.stmts = (src.stmts as unknown[]).map(rec);
      (copy._setChildren as (k: string, c: unknown) => void)('stmts', copy.stmts);
      return copy;
    case 'IfThenElseNode':
      copy.condition = rec(src.condition);
      copy.thenBody = rec(src.thenBody);
      copy.elseBody = src.elseBody ? rec(src.elseBody) : null;
      (copy._setChild as (k: string, c: unknown) => void)('thenBody', copy.thenBody);
      (copy._setChild as (k: string, c: unknown) => void)('elseBody', copy.elseBody);
      return copy;
    case 'BufferStoreNode':
      copy.buffer = src.buffer;
      copy.indices = (src.indices as unknown[]).map(rec);
      copy.value = rec(src.value);
      return copy;
    case 'BufferLoadNode':
      copy.buffer = src.buffer;
      copy.indices = (src.indices as unknown[]).map(rec);
      return copy;
    case 'MathOpNode':
      copy.op = src.op;
      copy.a = rec(src.a);
      copy.b = rec(src.b);
      return copy;
    case 'CompareNode':
      copy.direction = src.direction;
      copy.a = rec(src.a);
      copy.b = rec(src.b);
      return copy;
    case 'CastNode':
      copy.expr = rec(src.expr);
      copy.fromDtype = src.fromDtype;
      copy.toDtype = src.toDtype;
      return copy;
    case 'CallExternNode':
      copy.externName = src.externName;
      copy.args = (src.args as unknown[]).map(rec);
      copy.dtype = src.dtype;
      return copy;
    default:
      return handleOther(src, copy, rec);
  }
}
