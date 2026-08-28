import { Analyzer } from '../../analysis/analyzer.js';
import { RewriteSimplify, proveTrue, proveFalse, irBound, assumeCondition, assumeLoopVar, symOfExtern } from '../../analysis/ir_arith.js';
import { SymInt } from '../../analysis/sym_int.js';
import {
  ForNode, BlockNode, SeqNode, BufferStoreNode, BufferLoadNode,
  IfThenElseNode, LetStmtNode, AllocateNode, WhileNode, EvaluateNode,
  MathOpNode, CompareNode, CastNode, CallExternNode, BlockRealizeNode,
} from '../../ir/tensor/nodes.js';
import { LIRFlatLoadNode, LIRFlatStoreNode, LIRAccumulatorNode, LIRBindingsNode } from '../../ir/lir/nodes.js';
import { walk } from '../../ir/ir_visitor.js';
import { TraceLevel } from '../../pipeline/trace.js';
import type { IntImmNode, PrimFunc, TirNode, VariableNode } from '../../ir/tensor/nodes.js';
import type { SymNode } from '../../analysis/ir_arith.js';
import type { IRNode } from '../../ir/ir_visitor.js';
import type { LIRFunc } from '../../ir/lir/nodes.js';
import type { TraceLog } from '../../pipeline/trace.js';

export type SimplifyStats = { branchesFolded: number };
export type SimplifyPassLike = { name: string; phase: string };

type SimplifyCtx = { analyzer: Analyzer; simp: RewriteSimplify; stats: SimplifyStats };
type VarBound = ReturnType<Analyzer['getVarBound']>;

function newCtx(stats: SimplifyStats): SimplifyCtx {
  const ctx = { analyzer: new Analyzer(), simp: null, stats } as unknown as SimplifyCtx;
  ctx.simp = new RewriteSimplify(ctx.analyzer);
  return ctx;
}

function countIRNodes(root: object): number {
  let total = 0;
  walk(root as IRNode, () => { total++; });
  return total;
}

function report(trace: TraceLog, pass: SimplifyPassLike, funcName: string, removed: number, stats: SimplifyStats): void {
  trace.emit({
    type: 'pass_detail', passName: pass.name,
    nodesRemoved: removed, branchesFolded: stats.branchesFolded,
    level: TraceLevel.DEBUG,
  });
  if (!trace.explainsEnabled) return;
  if (removed === 0 && stats.branchesFolded === 0) {
    trace.explain(pass.phase, funcName, 'unchanged',
      'nothing here reduced further under the bounds the loops give', {});
    return;
  }
  trace.explain(pass.phase, funcName, 'folded',
    'loop extents bound every index expression, so the analyzer can evaluate the arithmetic and decide the guards',
    { nodesRemoved: removed, branchesFolded: stats.branchesFolded });
}

export function simplifyAndReport<T extends object>(
  pass: SimplifyPassLike,
  func: T,
  simplify: (f: T, stats: SimplifyStats) => T,
  trace: TraceLog,
): T {
  const t0 = performance.now();
  const measure = trace.level >= TraceLevel.DEBUG;
  const nodesBefore = measure ? countIRNodes(func) : 0;
  const stats: SimplifyStats = { branchesFolded: 0 };
  const out = simplify(func, stats);
  const funcName = String((out as { name?: unknown }).name ?? '');
  trace.functionEvent(pass.phase, funcName, { durationMs: performance.now() - t0 });
  if (measure) report(trace, pass, funcName, nodesBefore - countIRNodes(out), stats);
  return out;
}

export function simplifyPrimFunc(primFunc: PrimFunc, stats: SimplifyStats = { branchesFolded: 0 }): PrimFunc {
  const ctx = newCtx(stats);
  const body = simplifyStmt(primFunc.body, ctx);
  primFunc.body = body;
  primFunc._setChild('body', body);
  return primFunc;
}

export function simplifyLirFunc(lirFunc: LIRFunc, stats: SimplifyStats = { branchesFolded: 0 }): LIRFunc {
  const ctx = newCtx(stats);
  const body = simplifyStmt(lirFunc.body as TirNode, ctx);
  lirFunc.body = body;
  lirFunc._setChild('body', body);
  return lirFunc;
}

function bindLoopVar(ctx: SimplifyCtx, name: string, extentNode: TirNode): () => void {
  const prev = ctx.analyzer.getVarBound(name);
  const imm = extentNode as IntImmNode;
  if (extentNode && extentNode.type === 'IntImmNode' && imm.value > 0) {
    ctx.analyzer.bind(name, 0, imm.value - 1);
    return () => ctx.analyzer.setVarBound(name, prev);
  }
  ctx.analyzer.setVarBound(name, null);
  const release = assumeLoopVar(ctx.analyzer, name, extentNode);
  return () => { release(); ctx.analyzer.setVarBound(name, prev); };
}

function withAssumption<T>(ctx: SimplifyCtx, condition: TirNode, truth: boolean, body: () => T): T {
  const release = assumeCondition(ctx.analyzer, condition, truth);
  try {
    return body();
  } finally {
    release();
  }
}

function simplifyStmt(node: TirNode, ctx: SimplifyCtx): TirNode {
  if (!node || typeof node !== 'object') return node;
  switch (node.type as string) {
    case 'ForNode': {
      const f = node as ForNode;
      const release = bindLoopVar(ctx, f.loopVar.name, f.extent);
      const body = simplifyStmt(f.body, ctx);
      release();
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
      if (proveTrue(ctx.analyzer, cond)) {
        ctx.stats.branchesFolded++;
        return simplifyStmt(ite.thenBody, ctx);
      }
      if (proveFalse(ctx.analyzer, cond)) {
        ctx.stats.branchesFolded++;
        return ite.elseBody ? simplifyStmt(ite.elseBody, ctx) : new SeqNode([]);
      }
      const thenBody = withAssumption(ctx, cond, true, () => simplifyStmt(ite.thenBody, ctx));
      const elseBody = ite.elseBody ? withAssumption(ctx, cond, false, () => simplifyStmt(ite.elseBody as TirNode, ctx)) : null;
      return new IfThenElseNode(cond, thenBody, elseBody);
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
    case 'LIRFlatStoreNode': {
      const st = node as unknown as LIRFlatStoreNode;
      return new LIRFlatStoreNode(
        st.buffer,
        simplifyExpr(st.offsetExpr as TirNode, ctx),
        st.value ? simplifyExpr(st.value as TirNode, ctx) : null,
        st.dtype,
      ) as unknown as TirNode;
    }
    case 'LIRBindingsNode': {
      const b = node as unknown as LIRBindingsNode;
      const bindings = b.bindings.map((bind) => ({ ...bind, expr: simplifyExpr(bind.expr as TirNode, ctx) }));
      return new LIRBindingsNode(bindings, simplifyStmt(b.body as TirNode, ctx)) as unknown as TirNode;
    }
    case 'LIRAccumulatorNode': {
      const acc = node as unknown as LIRAccumulatorNode;
      const release = bindLoopVar(ctx, acc.loopVar.name, acc.extent as TirNode);
      const body = simplifyExpr(acc.body as TirNode, ctx);
      const initBody = acc.initBody ? simplifyStmt(acc.initBody as TirNode, ctx) : null;
      const prologue = acc.prologue ? simplifyStmt(acc.prologue as TirNode, ctx) : null;
      const initLoad = simplifyExpr(acc.initLoad as unknown as TirNode, ctx);
      const flushStore = simplifyStmt(acc.flushStore as unknown as TirNode, ctx);
      release();
      return new LIRAccumulatorNode({
        localName: acc.localName,
        dtype: acc.dtype,
        op: acc.op,
        initLoad: initLoad as unknown as LIRFlatLoadNode,
        loopVar: acc.loopVar,
        extent: acc.extent,
        loopKind: acc.loopKind,
        body,
        flushStore: flushStore as unknown as LIRFlatStoreNode,
        initBody,
        prologue,
      }) as unknown as TirNode;
    }
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
  return simplifySym(node, ctx).node;
}

function opaque(node: TirNode): SymNode {
  return { node, sym: null };
}

function simplifySym(node: TirNode, ctx: SimplifyCtx): SymNode {
  if (!node || typeof node !== 'object' || !node.type) return opaque(node);
  switch (node.type as string) {
    case 'IntImmNode':
      return { node, sym: (node as IntImmNode).value };
    case 'VariableNode':
      return { node, sym: SymInt.var((node as VariableNode).name) };
    case 'FloatImmNode':
      return opaque(node);
    case 'BufferLoadNode': {
      const ld = node as BufferLoadNode;
      return opaque(new BufferLoadNode(ld.buffer, ld.indices.map((i: TirNode) => simplifyExpr(i, ctx))));
    }
    case 'MathOpNode': {
      const m = node as MathOpNode;
      const a = simplifySym(m.a, ctx);
      const b = m.b ? simplifySym(m.b, ctx) : null;
      return ctx.simp.mathOp(m.op, a, b);
    }
    case 'CompareNode': {
      const c = node as CompareNode;
      return ctx.simp.compare(c.direction, simplifySym(c.a, ctx), simplifySym(c.b, ctx));
    }
    case 'CastNode': {
      const cast = node as CastNode;
      return opaque(new CastNode(simplifyExpr(cast.expr, ctx), cast.fromDtype, cast.toDtype));
    }
    case 'CallExternNode': {
      const call = node as CallExternNode;
      const args = call.args.map((a: TirNode) => simplifySym(a, ctx));
      return {
        node: new CallExternNode(call.externName, args.map(a => a.node), call.dtype),
        sym: symOfExtern(call.externName, args.map(a => a.sym)),
      };
    }
    case 'IfThenElseNode': {
      const ite = node as IfThenElseNode;
      const cond = simplifyExpr(ite.condition, ctx);
      const thenE = simplifySym(ite.thenBody, ctx);
      const elseE = ite.elseBody ? simplifySym(ite.elseBody, ctx) : null;
      if (proveTrue(ctx.analyzer, cond)) {
        ctx.stats.branchesFolded++;
        return thenE;
      }
      if (elseE !== null && proveFalse(ctx.analyzer, cond)) {
        ctx.stats.branchesFolded++;
        return elseE;
      }
      return opaque(new IfThenElseNode(cond, thenE.node, elseE === null ? null : elseE.node));
    }
    case 'LIRFlatLoadNode': {
      const ld = node as unknown as LIRFlatLoadNode;
      return opaque(new LIRFlatLoadNode(ld.buffer, simplifyExpr(ld.offsetExpr as TirNode, ctx), ld.dtype) as unknown as TirNode);
    }
    default:
      return opaque(node);
  }
}
