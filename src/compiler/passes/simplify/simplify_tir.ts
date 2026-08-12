import { Analyzer } from '../../analysis/analyzer.js';
import { RewriteSimplify, proveTrue, proveFalse, irBound } from '../../analysis/ir_arith.js';
import {
  ForNode, BlockNode, SeqNode, BufferStoreNode, BufferLoadNode,
  IfThenElseNode, LetStmtNode, AllocateNode, WhileNode, EvaluateNode,
  MathOpNode, CompareNode, CastNode, CallExternNode, BlockRealizeNode,
} from '../../ir/tensor/nodes.js';
import type { IntImmNode, PrimFunc, TirNode } from '../../ir/tensor/nodes.js';

type SimplifyCtx = { analyzer: Analyzer; simp: RewriteSimplify };
type VarBound = ReturnType<Analyzer['getVarBound']>;

export function simplifyPrimFunc(primFunc: PrimFunc): PrimFunc {
  const ctx = { analyzer: new Analyzer(), simp: null } as unknown as SimplifyCtx;
  ctx.simp = new RewriteSimplify(ctx.analyzer);
  const body = simplifyStmt(primFunc.body, ctx);
  primFunc.body = body;
  primFunc._setChild('body', body);
  return primFunc;
}

function bindLoopVar(ctx: SimplifyCtx, name: string, extentNode: TirNode): VarBound {
  const prev = ctx.analyzer.getVarBound(name);
  const imm = extentNode as IntImmNode;
  if (extentNode && extentNode.type === 'IntImmNode' && imm.value > 0) {
    ctx.analyzer.bind(name, 0, imm.value - 1);
  } else {
    ctx.analyzer.setVarBound(name, null);
  }
  return prev;
}

function simplifyStmt(node: TirNode, ctx: SimplifyCtx): TirNode {
  if (!node || typeof node !== 'object') return node;
  switch (node.type) {
    case 'ForNode': {
      const f = node as ForNode;
      const prev = bindLoopVar(ctx, f.loopVar.name, f.extent);
      const body = simplifyStmt(f.body, ctx);
      ctx.analyzer.setVarBound(f.loopVar.name, prev);
      const out = new ForNode(f.loopVar, f.min, f.extent, f.kind, body, f.threadTag);
      if (f.annotations) out.annotations = f.annotations;
      return out;
    }
    case 'BlockNode': {
      const blk = node as BlockNode;
      const saved: [string, VarBound][] = [];
      for (const r of blk.iterVars) {
        if (r.iterVar) {
          saved.push([r.iterVar.name, ctx.analyzer.getVarBound(r.iterVar.name)]);
          ctx.analyzer.setVarBound(r.iterVar.name, r.binding ? irBound(ctx.analyzer, r.binding) : null);
        }
      }
      const iterVars = blk.iterVars.map(simplifyIterVar(ctx));
      const body = simplifyStmt(blk.body, ctx);
      const initBody = blk.initBody ? simplifyStmt(blk.initBody, ctx) : null;
      for (const [name, b] of saved) ctx.analyzer.setVarBound(name, b);
      return new BlockNode(blk.name, iterVars, blk.reads, blk.writes, body, initBody);
    }
    case 'SeqNode':
      return new SeqNode((node as SeqNode).stmts.map((s: TirNode) => simplifyStmt(s, ctx)));
    case 'IfThenElseNode': {
      const ite = node as IfThenElseNode;
      const cond = simplifyExpr(ite.condition, ctx);
      if (proveTrue(ctx.analyzer, cond)) return simplifyStmt(ite.thenBody, ctx);
      if (proveFalse(ctx.analyzer, cond)) return ite.elseBody ? simplifyStmt(ite.elseBody, ctx) : new SeqNode([]);
      return new IfThenElseNode(cond, simplifyStmt(ite.thenBody, ctx), ite.elseBody ? simplifyStmt(ite.elseBody, ctx) : null);
    }
    case 'BufferStoreNode': {
      const st = node as BufferStoreNode;
      return new BufferStoreNode(st.buffer, st.indices.map((i: TirNode) => simplifyExpr(i, ctx)), simplifyExpr(st.value, ctx));
    }
    case 'LetStmtNode': {
      const let_ = node as LetStmtNode;
      return new LetStmtNode(let_.variable, simplifyExpr(let_.value, ctx), simplifyStmt(let_.body, ctx));
    }
    case 'AllocateNode': {
      const al = node as AllocateNode;
      return new AllocateNode(al.buffer, al.scope, simplifyStmt(al.body, ctx));
    }
    case 'WhileNode': {
      const w = node as WhileNode;
      return new WhileNode(w.condVar, simplifyStmt(w.condBody, ctx), simplifyStmt(w.loopBody, ctx));
    }
    case 'EvaluateNode':
      return new EvaluateNode(simplifyExpr((node as EvaluateNode).value, ctx));
    default:
      return node;
  }
}

function simplifyIterVar(ctx: SimplifyCtx): (r: BlockRealizeNode) => BlockRealizeNode {
  return (r: BlockRealizeNode): BlockRealizeNode => {
    if (!r.iterVar || !r.binding) return r;
    const binding = simplifyExpr(r.binding, ctx);
    return new BlockRealizeNode(r.iterVar, binding, r.kind);
  };
}

function simplifyExpr(node: TirNode, ctx: SimplifyCtx): TirNode {
  if (!node || typeof node !== 'object' || !node.type) return node;
  switch (node.type) {
    case 'IntImmNode':
    case 'FloatImmNode':
    case 'VariableNode':
      return node;
    case 'BufferLoadNode': {
      const ld = node as BufferLoadNode;
      return new BufferLoadNode(ld.buffer, ld.indices.map((i: TirNode) => simplifyExpr(i, ctx)));
    }
    case 'MathOpNode': {
      const m = node as MathOpNode;
      const a = simplifyExpr(m.a, ctx);
      const b = m.b ? simplifyExpr(m.b, ctx) : null;
      return ctx.simp.simplify(new MathOpNode(m.op, a, b)) as TirNode;
    }
    case 'CompareNode': {
      const c = node as CompareNode;
      const a = simplifyExpr(c.a, ctx);
      const b = simplifyExpr(c.b, ctx);
      return ctx.simp.simplify(new CompareNode(c.direction, a, b)) as TirNode;
    }
    case 'CastNode': {
      const cast = node as CastNode;
      return new CastNode(simplifyExpr(cast.expr, ctx), cast.fromDtype, cast.toDtype);
    }
    case 'CallExternNode': {
      const call = node as CallExternNode;
      return new CallExternNode(call.externName, call.args.map((a: TirNode) => simplifyExpr(a, ctx)), call.dtype);
    }
    case 'IfThenElseNode': {
      const ite = node as IfThenElseNode;
      const cond = simplifyExpr(ite.condition, ctx);
      const thenE = simplifyExpr(ite.thenBody, ctx);
      const elseE = ite.elseBody ? simplifyExpr(ite.elseBody, ctx) : null;
      if (proveTrue(ctx.analyzer, cond)) return thenE;
      if (elseE !== null && proveFalse(ctx.analyzer, cond)) return elseE;
      return new IfThenElseNode(cond, thenE, elseE);
    }
    default:
      return node;
  }
}
