export function cloneIRShared(node, rec, handleOther) {
  if (!node || typeof node !== 'object') return node;
  if (Array.isArray(node)) return node.map(rec);
  const copy = Object.create(Object.getPrototypeOf(node));
  copy.type = node.type;
  copy._parent = null;
  copy._parentKey = null;
  copy._parentIdx = -1;
  switch (node.type) {
    case 'ForNode':
      copy.loopVar = node.loopVar;
      copy.min = rec(node.min);
      copy.extent = rec(node.extent);
      copy.kind = node.kind;
      copy.body = rec(node.body);
      copy.threadTag = node.threadTag;
      copy._setChild('body', copy.body);
      return copy;
    case 'BlockNode':
      copy.name = node.name;
      copy.iterVars = node.iterVars.map(rec);
      copy.reads = node.reads;
      copy.writes = node.writes;
      copy.body = rec(node.body);
      copy.initBody = node.initBody ? rec(node.initBody) : null;
      copy._setChild('body', copy.body);
      copy._setChild('initBody', copy.initBody);
      return copy;
    case 'SeqNode':
      copy.stmts = node.stmts.map(rec);
      copy._setChildren('stmts', copy.stmts);
      return copy;
    case 'IfThenElseNode':
      copy.condition = rec(node.condition);
      copy.thenBody = rec(node.thenBody);
      copy.elseBody = node.elseBody ? rec(node.elseBody) : null;
      copy._setChild('thenBody', copy.thenBody);
      copy._setChild('elseBody', copy.elseBody);
      return copy;
    case 'BufferStoreNode':
      copy.buffer = node.buffer;
      copy.indices = node.indices.map(rec);
      copy.value = rec(node.value);
      return copy;
    case 'BufferLoadNode':
      copy.buffer = node.buffer;
      copy.indices = node.indices.map(rec);
      return copy;
    case 'MathOpNode':
      copy.op = node.op;
      copy.a = rec(node.a);
      copy.b = rec(node.b);
      return copy;
    case 'CompareNode':
      copy.direction = node.direction;
      copy.a = rec(node.a);
      copy.b = rec(node.b);
      return copy;
    case 'CastNode':
      copy.expr = rec(node.expr);
      copy.fromDtype = node.fromDtype;
      copy.toDtype = node.toDtype;
      return copy;
    case 'CallExternNode':
      copy.externName = node.externName;
      copy.args = node.args.map(rec);
      copy.dtype = node.dtype;
      return copy;
    default:
      return handleOther(node, copy, rec);
  }
}
