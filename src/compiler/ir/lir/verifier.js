import { inferDtype } from './nodes.js';

export class LIRVerificationError {
  constructor(message, nodePath) {
    this.message = message;
    this.nodePath = nodePath || [];
  }

  toString() {
    const path = this.nodePath.length > 0 ? ` at ${this.nodePath.join(' > ')}` : '';
    return `LIR verification: ${this.message}${path}`;
  }
}

export function verifyLIR(lirFunc) {
  const errors = [];
  const ctx = {
    errors,
    boundVars: new Set(),
    path: [],
    bufferNames: new Set(),
  };

  if (!lirFunc || lirFunc.type !== 'LIRFunc') {
    errors.push(new LIRVerificationError('root must be LIRFunc'));
    return errors;
  }

  for (const [, buf] of lirFunc.bufferMap) {
    ctx.bufferNames.add(buf.name);
  }

  for (const sp of lirFunc.shapeParams) {
    ctx.boundVars.add(sp.name);
  }

  if (lirFunc.metadata) {
    for (const [name] of lirFunc.metadata.locals) {
      ctx.boundVars.add(name);
    }
  }

  verifyStmt(lirFunc.body, ctx);
  return errors;
}

function verifyStmt(node, ctx) {
  if (!node || typeof node !== 'object') return;
  ctx.path.push(node.type);

  switch (node.type) {
    case 'ForNode':
      verifyForNode(node, ctx);
      break;
    case 'SeqNode':
      for (const s of node.stmts) verifyStmt(s, ctx);
      break;
    case 'LIRFlatStoreNode':
      verifyFlatStore(node, ctx);
      break;
    case 'LIRAccumulatorNode':
      verifyAccumulator(node, ctx);
      break;
    case 'LIRBindingsNode':
      verifyBindings(node, ctx);
      break;
    case 'LetStmtNode':
      verifyExpr(node.value, ctx);
      ctx.boundVars.add(node.variable.name);
      verifyStmt(node.body, ctx);
      break;
    case 'AllocateNode':
      if (node.buffer) ctx.bufferNames.add(node.buffer.name);
      verifyStmt(node.body, ctx);
      break;
    case 'IfThenElseNode':
      verifyExpr(node.condition, ctx);
      verifyStmt(node.thenBody, ctx);
      if (node.elseBody) verifyStmt(node.elseBody, ctx);
      break;
    case 'WhileNode':
      verifyStmt(node.condBody, ctx);
      verifyStmt(node.loopBody, ctx);
      break;
    case 'EvaluateNode':
      verifyExpr(node.value, ctx);
      break;
    default:
      break;
  }

  ctx.path.pop();
}

function verifyForNode(node, ctx) {
  if (!node.loopVar) {
    ctx.errors.push(new LIRVerificationError('ForNode missing loopVar', [...ctx.path]));
  } else {
    ctx.boundVars.add(node.loopVar.name);
  }
  if (!node.extent) {
    ctx.errors.push(new LIRVerificationError('ForNode missing extent', [...ctx.path]));
  }
  verifyStmt(node.body, ctx);
}

function verifyFlatStore(node, ctx) {
  if (!node.buffer) {
    ctx.errors.push(new LIRVerificationError('LIRFlatStoreNode missing buffer', [...ctx.path]));
  }
  verifyExpr(node.offsetExpr, ctx);
  verifyExpr(node.value, ctx);
}

function verifyAccumulator(node, ctx) {
  if (!node.localName) {
    ctx.errors.push(new LIRVerificationError('LIRAccumulatorNode missing localName', [...ctx.path]));
  }
  if (!node.dtype) {
    ctx.errors.push(new LIRVerificationError('LIRAccumulatorNode missing dtype', [...ctx.path]));
  }
  ctx.boundVars.add(node.localName);
  verifyExpr(node.initLoad, ctx);
  verifyExpr(node.body, ctx);
  verifyStmt(node.flushStore, ctx);
  if (node.initBody) verifyStmt(node.initBody, ctx);
}

function verifyBindings(node, ctx) {
  for (const bind of node.bindings) {
    verifyExpr(bind.expr, ctx);
    ctx.boundVars.add(bind.name);
  }
  verifyStmt(node.body, ctx);
}

function verifyExpr(node, ctx) {
  if (!node || typeof node !== 'object' || !node.type) return;

  switch (node.type) {
    case 'LIRFlatLoadNode':
      if (!node.buffer) {
        ctx.errors.push(new LIRVerificationError('LIRFlatLoadNode missing buffer', [...ctx.path]));
      }
      verifyExpr(node.offsetExpr, ctx);
      break;
    case 'MathOpNode':
      verifyExpr(node.a, ctx);
      if (node.b) verifyExpr(node.b, ctx);
      break;
    case 'CompareNode':
      verifyExpr(node.a, ctx);
      verifyExpr(node.b, ctx);
      break;
    case 'CastNode':
      verifyExpr(node.expr, ctx);
      break;
    case 'CallExternNode':
      if (node.args) {
        for (const a of node.args) verifyExpr(a, ctx);
      }
      break;
    case 'IfThenElseNode':
      verifyExpr(node.condition, ctx);
      verifyExpr(node.thenBody, ctx);
      if (node.elseBody) verifyExpr(node.elseBody, ctx);
      break;
    case 'VariableNode':
    case 'IntImmNode':
    case 'FloatImmNode':
      break;
    default:
      break;
  }
}
