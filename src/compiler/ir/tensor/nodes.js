export const ForKind = Object.freeze({
  SERIAL: 'serial',
  PARALLEL: 'parallel',
  VECTORIZED: 'vectorized',
  UNROLLED: 'unrolled',
  THREAD_BINDING: 'thread_binding'
});

export class TensorNode {
  constructor() {
    this.type = this.constructor.name;
  }
}

export class PrimFunc extends TensorNode {
  constructor(name, params, body, bufferMap = new Map(), shapeParams = []) {
    super();
    this.name = name;
    this.params = params;
    this.body = body;
    this.bufferMap = bufferMap;
    this.shapeParams = shapeParams;
  }
}

export class ForNode extends TensorNode {
  constructor(loopVar, min, extent, kind, body, threadTag = null) {
    super();
    this.loopVar = loopVar;
    this.min = min;
    this.extent = extent;
    this.kind = kind;
    this.body = body;
    this.threadTag = threadTag;
  }
}

export class BlockNode extends TensorNode {
  constructor(name, iterVars, reads, writes, body, initBody = null) {
    super();
    this.name = name;
    this.iterVars = iterVars;
    this.reads = reads;
    this.writes = writes;
    this.body = body;
    this.initBody = initBody;
  }
}

export class BlockRealizeNode extends TensorNode {
  constructor(iterVar, binding) {
    super();
    this.iterVar = iterVar;
    this.binding = binding;
  }
}

export class BufferStoreNode extends TensorNode {
  constructor(buffer, indices, value) {
    super();
    this.buffer = buffer;
    this.indices = indices;
    this.value = value;
  }
}

export class BufferLoadNode extends TensorNode {
  constructor(buffer, indices) {
    super();
    this.buffer = buffer;
    this.indices = indices;
  }
}

export class IfThenElseNode extends TensorNode {
  constructor(condition, thenBody, elseBody = null) {
    super();
    this.condition = condition;
    this.thenBody = thenBody;
    this.elseBody = elseBody;
  }
}

export class LetStmtNode extends TensorNode {
  constructor(variable, value, body) {
    super();
    this.variable = variable;
    this.value = value;
    this.body = body;
  }
}

export class AllocateNode extends TensorNode {
  constructor(buffer, scope, body) {
    super();
    this.buffer = buffer;
    this.scope = scope;
    this.body = body;
  }
}

export class SeqNode extends TensorNode {
  constructor(stmts) {
    super();
    this.stmts = stmts;
  }
}

export class WhileNode extends TensorNode {
  constructor(condVar, condBody, loopBody) {
    super();
    this.condVar = condVar;
    this.condBody = condBody;
    this.loopBody = loopBody;
  }
}

export class EvaluateNode extends TensorNode {
  constructor(value) {
    super();
    this.value = value;
  }
}

export class CallExternNode extends TensorNode {
  constructor(name, args, dtype) {
    super();
    this.externName = name;
    this.args = args;
    this.dtype = dtype;
  }
}

export class MathOpNode extends TensorNode {
  constructor(op, a, b = null) {
    super();
    this.op = op;
    this.a = a;
    this.b = b;
  }
}

export class CompareNode extends TensorNode {
  static JS_OPS = { eq: '===', ne: '!==', lt: '<', le: '<=', gt: '>', ge: '>=' };
  static C_OPS = { eq: '==', ne: '!=', lt: '<', le: '<=', gt: '>', ge: '>=' };

  constructor(direction, a, b) {
    super();
    this.direction = direction;
    this.a = a;
    this.b = b;
  }

  toJS() {
    const op = CompareNode.JS_OPS[this.direction];
    if (!op) throw new Error(`CompareNode: unsupported direction '${this.direction}'`);
    return op;
  }

  toC() {
    const op = CompareNode.C_OPS[this.direction];
    if (!op) throw new Error(`CompareNode: unsupported direction '${this.direction}'`);
    return op;
  }
}

export class CastNode extends TensorNode {
  constructor(expr, fromDtype, toDtype) {
    super();
    this.expr = expr;
    this.fromDtype = fromDtype;
    this.toDtype = toDtype;
  }
}

export class VariableNode extends TensorNode {
  constructor(name, dtype) {
    super();
    this.name = name;
    this.dtype = dtype;
  }
}

export class IntImmNode extends TensorNode {
  constructor(value) {
    super();
    this.value = value;
  }
}

export class FloatImmNode extends TensorNode {
  constructor(value) {
    super();
    this.value = value;
  }
}
