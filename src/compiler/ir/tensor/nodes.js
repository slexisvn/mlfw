export const ForKind = Object.freeze({
  SERIAL: 'serial',
  PARALLEL: 'parallel',
  VECTORIZED: 'vectorized',
  UNROLLED: 'unrolled',
  THREAD_BINDING: 'thread_binding'
});

export class TensorNode {
  constructor() {
    this.type = this.constructor.name.replace(/^_+/, '');
    this._parent = null;
    this._parentKey = null;
    this._parentIdx = -1;
  }

  _setChild(key, child, idx = -1) {
    if (child instanceof TensorNode) {
      child._parent = this;
      child._parentKey = key;
      child._parentIdx = idx;
    }
  }

  _setChildren(key, arr) {
    if (!arr) return;
    for (let i = 0; i < arr.length; i++) {
      if (arr[i] instanceof TensorNode) {
        arr[i]._parent = this;
        arr[i]._parentKey = key;
        arr[i]._parentIdx = i;
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

export class PrimFunc extends TensorNode {
  constructor(name, params, body, bufferMap = new Map(), shapeParams = [], shapeParamMap = null) {
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
  constructor(loopVar, min, extent, kind, body, threadTag = null) {
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

export class BlockNode extends TensorNode {
  constructor(name, iterVars, reads, writes, body, initBody = null) {
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
    this._setChild('thenBody', thenBody);
    this._setChild('elseBody', elseBody);
  }
}

export class LetStmtNode extends TensorNode {
  constructor(variable, value, body) {
    super();
    this.variable = variable;
    this.value = value;
    this.body = body;
    this._setChild('body', body);
  }
}

export class AllocateNode extends TensorNode {
  constructor(buffer, scope, body) {
    super();
    this.buffer = buffer;
    this.scope = scope;
    this.body = body;
    this._setChild('body', body);
  }
}

export class SeqNode extends TensorNode {
  constructor(stmts) {
    super();
    this.stmts = stmts;
    this._setChildren('stmts', stmts);
  }
}

export class WhileNode extends TensorNode {
  constructor(condVar, condBody, loopBody) {
    super();
    this.condVar = condVar;
    this.condBody = condBody;
    this.loopBody = loopBody;
    this._setChild('condBody', condBody);
    this._setChild('loopBody', loopBody);
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

export function mathOp(op, a, b) {
  if (b === null || b === undefined) return new MathOpNode(op, a, b);

  const aIsInt = a instanceof IntImmNode;
  const bIsInt = b instanceof IntImmNode;

  if (aIsInt && bIsInt) {
    const av = a.value, bv = b.value;
    switch (op) {
      case '+': return new IntImmNode(av + bv);
      case '-': return new IntImmNode(av - bv);
      case '*': return new IntImmNode(av * bv);
      case '//': return new IntImmNode(Math.trunc(av / bv));
      case '%': return new IntImmNode(((av % bv) + bv) % bv);
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
