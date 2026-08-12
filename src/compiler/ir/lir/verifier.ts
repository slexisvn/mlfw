

import type { LIRFunc, LirNode, LIRFlatStoreNode, LIRAccumulatorNode, LIRBindingsNode } from './nodes.js';
import type { TirNode, ForNode } from '../tensor/nodes.js';

export type LIRVerifyNode = LirNode | TirNode;

export type LIRVerifyContext = {
  errors: LIRVerificationError[];
  boundVars: Set<string>;
  path: string[];
  bufferNames: Set<string>;
};

export class LIRVerificationError {
  message: string;
  nodePath: readonly string[];

  constructor(message: string, nodePath?: readonly string[]) {
    this.message = message;
    this.nodePath = nodePath || [];
  }

  toString(): string {
    const path = this.nodePath.length > 0 ? ` at ${this.nodePath.join(' > ')}` : '';
    return `LIR verification: ${this.message}${path}`;
  }
}

export function verifyLIR(lirFunc: LIRFunc): LIRVerificationError[] {
  const errors: LIRVerificationError[] = [];
  const ctx: LIRVerifyContext = {
    errors,
    boundVars: new Set<string>(),
    path: [],
    bufferNames: new Set<string>(),
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

function verifyStmt(node: LIRVerifyNode | null | undefined, ctx: LIRVerifyContext): void {
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
    case 'LetStmtNode': {
      verifyExpr(node.value, ctx);
      const had = ctx.boundVars.has(node.variable.name);
      ctx.boundVars.add(node.variable.name);
      verifyStmt(node.body, ctx);
      if (!had) ctx.boundVars.delete(node.variable.name);
      break;
    }
    case 'AllocateNode': {
      const had = node.buffer ? ctx.bufferNames.has(node.buffer.name) : true;
      if (node.buffer) ctx.bufferNames.add(node.buffer.name);
      verifyStmt(node.body, ctx);
      if (node.buffer && !had) ctx.bufferNames.delete(node.buffer.name);
      break;
    }
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

function verifyForNode(node: ForNode, ctx: LIRVerifyContext): void {
  if (!node.extent) {
    ctx.errors.push(new LIRVerificationError('ForNode missing extent', [...ctx.path]));
  } else {
    verifyExpr(node.extent, ctx);
  }
  let had = true;
  if (!node.loopVar) {
    ctx.errors.push(new LIRVerificationError('ForNode missing loopVar', [...ctx.path]));
  } else {
    had = ctx.boundVars.has(node.loopVar.name);
    ctx.boundVars.add(node.loopVar.name);
  }
  verifyStmt(node.body, ctx);
  if (node.loopVar && !had) ctx.boundVars.delete(node.loopVar.name);
}

function verifyFlatStore(node: LIRFlatStoreNode, ctx: LIRVerifyContext): void {
  if (!node.buffer) {
    ctx.errors.push(new LIRVerificationError('LIRFlatStoreNode missing buffer', [...ctx.path]));
  }
  verifyExpr(node.offsetExpr, ctx);
  verifyExpr(node.value, ctx);
}

function verifyAccumulator(node: LIRAccumulatorNode, ctx: LIRVerifyContext): void {
  if (!node.localName) {
    ctx.errors.push(new LIRVerificationError('LIRAccumulatorNode missing localName', [...ctx.path]));
  }
  if (!node.dtype) {
    ctx.errors.push(new LIRVerificationError('LIRAccumulatorNode missing dtype', [...ctx.path]));
  }
  const had = node.localName ? ctx.boundVars.has(node.localName) : true;
  if (node.localName) ctx.boundVars.add(node.localName);
  verifyExpr(node.initLoad, ctx);
  verifyExpr(node.body, ctx);
  verifyStmt(node.flushStore, ctx);
  if (node.initBody) verifyStmt(node.initBody, ctx);
  if (node.localName && !had) ctx.boundVars.delete(node.localName);
}

function verifyBindings(node: LIRBindingsNode, ctx: LIRVerifyContext): void {
  const added = [];
  for (const bind of node.bindings) {
    verifyExpr(bind.expr, ctx);
    if (!ctx.boundVars.has(bind.name)) added.push(bind.name);
    ctx.boundVars.add(bind.name);
  }
  verifyStmt(node.body, ctx);
  for (const name of added) ctx.boundVars.delete(name);
}

function verifyExpr(node: LIRVerifyNode | null | undefined, ctx: LIRVerifyContext): void {
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
      if (node.name !== undefined && !ctx.boundVars.has(node.name)) {
        ctx.errors.push(new LIRVerificationError(`unbound variable '${node.name}'`, [...ctx.path]));
      }
      break;
    case 'IntImmNode':
    case 'FloatImmNode':
      break;
    default:
      break;
  }
}
