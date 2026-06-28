import { Analyzer } from '../../analysis/analyzer.js';
import { RewriteSimplify, proveTrue, proveFalse, irBound } from '../../analysis/ir_arith.js';
import {
  ForNode, BlockNode, SeqNode, BufferStoreNode, BufferLoadNode,
  IfThenElseNode, LetStmtNode, AllocateNode, WhileNode, EvaluateNode,
  MathOpNode, CompareNode, CastNode, CallExternNode, BlockRealizeNode,
} from '../../ir/tensor/nodes.js';

export function simplifyPrimFunc(primFunc) {
  const ctx = { analyzer: new Analyzer(), simp: null };
  ctx.simp = new RewriteSimplify(ctx.analyzer);
  const body = simplifyStmt(primFunc.body, ctx);
  primFunc.body = body;
  primFunc._setChild('body', body);
  return primFunc;
}

function bindLoopVar(ctx, name, extentNode) {
  const prev = ctx.analyzer.getVarBound(name);
  if (extentNode && extentNode.type === 'IntImmNode' && extentNode.value > 0) {
    ctx.analyzer.bind(name, 0, extentNode.value - 1);
  } else {
    ctx.analyzer.setVarBound(name, null);
  }
  return prev;
}

function simplifyStmt(node, ctx) {
  if (!node || typeof node !== 'object') return node;
  switch (node.type) {
    case 'ForNode': {
      const prev = bindLoopVar(ctx, node.loopVar.name, node.extent);
      const body = simplifyStmt(node.body, ctx);
      ctx.analyzer.setVarBound(node.loopVar.name, prev);
      const out = new ForNode(node.loopVar, node.min, node.extent, node.kind, body, node.threadTag);
      if (node.annotations) out.annotations = node.annotations;
      return out;
    }
    case 'BlockNode': {
      const saved = [];
      for (const r of node.iterVars) {
        if (r.iterVar) {
          saved.push([r.iterVar.name, ctx.analyzer.getVarBound(r.iterVar.name)]);
          ctx.analyzer.setVarBound(r.iterVar.name, r.binding ? irBound(ctx.analyzer, r.binding) : null);
        }
      }
      const iterVars = node.iterVars.map(simplifyIterVar(ctx));
      const body = simplifyStmt(node.body, ctx);
      const initBody = node.initBody ? simplifyStmt(node.initBody, ctx) : null;
      for (const [name, b] of saved) ctx.analyzer.setVarBound(name, b);
      return new BlockNode(node.name, iterVars, node.reads, node.writes, body, initBody);
    }
    case 'SeqNode':
      return new SeqNode(node.stmts.map(s => simplifyStmt(s, ctx)));
    case 'IfThenElseNode': {
      const cond = simplifyExpr(node.condition, ctx);
      if (proveTrue(ctx.analyzer, cond)) return simplifyStmt(node.thenBody, ctx);
      if (proveFalse(ctx.analyzer, cond)) return node.elseBody ? simplifyStmt(node.elseBody, ctx) : new SeqNode([]);
      return new IfThenElseNode(cond, simplifyStmt(node.thenBody, ctx), node.elseBody ? simplifyStmt(node.elseBody, ctx) : null);
    }
    case 'BufferStoreNode':
      return new BufferStoreNode(node.buffer, node.indices.map(i => simplifyExpr(i, ctx)), simplifyExpr(node.value, ctx));
    case 'LetStmtNode':
      return new LetStmtNode(node.variable, simplifyExpr(node.value, ctx), simplifyStmt(node.body, ctx));
    case 'AllocateNode':
      return new AllocateNode(node.buffer, node.scope, simplifyStmt(node.body, ctx));
    case 'WhileNode':
      return new WhileNode(node.condVar, simplifyStmt(node.condBody, ctx), simplifyStmt(node.loopBody, ctx));
    case 'EvaluateNode':
      return new EvaluateNode(simplifyExpr(node.value, ctx));
    default:
      return node;
  }
}

function simplifyIterVar(ctx) {
  return (r) => {
    if (!r.iterVar || !r.binding) return r;
    const binding = simplifyExpr(r.binding, ctx);
    return new BlockRealizeNode(r.iterVar, binding, r.kind);
  };
}

function simplifyExpr(node, ctx) {
  if (!node || typeof node !== 'object' || !node.type) return node;
  switch (node.type) {
    case 'IntImmNode':
    case 'FloatImmNode':
    case 'VariableNode':
      return node;
    case 'BufferLoadNode':
      return new BufferLoadNode(node.buffer, node.indices.map(i => simplifyExpr(i, ctx)));
    case 'MathOpNode': {
      const a = simplifyExpr(node.a, ctx);
      const b = node.b ? simplifyExpr(node.b, ctx) : null;
      return ctx.simp.simplify(new MathOpNode(node.op, a, b));
    }
    case 'CompareNode': {
      const a = simplifyExpr(node.a, ctx);
      const b = simplifyExpr(node.b, ctx);
      return ctx.simp.simplify(new CompareNode(node.direction, a, b));
    }
    case 'CastNode':
      return new CastNode(simplifyExpr(node.expr, ctx), node.fromDtype, node.toDtype);
    case 'CallExternNode':
      return new CallExternNode(node.externName, node.args.map(a => simplifyExpr(a, ctx)), node.dtype);
    case 'IfThenElseNode': {
      const cond = simplifyExpr(node.condition, ctx);
      const thenE = simplifyExpr(node.thenBody, ctx);
      const elseE = node.elseBody ? simplifyExpr(node.elseBody, ctx) : null;
      if (proveTrue(ctx.analyzer, cond)) return thenE;
      if (elseE !== null && proveFalse(ctx.analyzer, cond)) return elseE;
      return new IfThenElseNode(cond, thenE, elseE);
    }
    default:
      return node;
  }
}
