import { PrimFuncPass } from '../tir_pass.js';
import { explainer } from '../explain.js';
import { ForNode, SeqNode, IntImmNode, VariableNode, ForKind } from '../../ir/tensor/nodes.js';
import { transform } from '../../ir/ir_visitor.js';
import { cloneTensorIR } from '../../ir/tensor/clone_tir.js';
import { Analyzer } from '../../analysis/analyzer.js';
import { proveTrue } from '../../analysis/ir_arith.js';
import type { CompareNode, ForKindValue, IfThenElseNode, IntImmNode as IntImmNodeType, MathOpNode, PrimFunc, TirNode, VariableNode as VariableNodeType } from '../../ir/tensor/nodes.js';
import type { IRNode } from '../../ir/ir_visitor.js';
import type { TirPassCtx } from '../tir_pass.js';
import type { Explain } from '../explain.js';

const PARTITIONABLE_KINDS = new Set<ForKindValue>([
  ForKind.SERIAL, ForKind.PARALLEL, ForKind.VECTORIZED, ForKind.UNROLLED,
]);

export class LoopPartitionPass extends PrimFuncPass {
  constructor() {
    super('LoopPartitionPass', 'loopPartition');
  }

  override run(pf: PrimFunc, ctx: TirPassCtx): PrimFunc {
    const explain = explainer(ctx.trace, this.name);
    const newBody = transform(pf.body as unknown as IRNode, ((n: TirNode) => partitionLoop(n, explain) || n) as never) as unknown as TirNode;
    if (newBody && newBody !== pf.body) {
      pf.body = newBody;
      pf._setChild('body', newBody);
    }
    return pf;
  }
}

function intImm(node: TirNode | null | undefined): number | null {
  return node && node.type === 'IntImmNode' ? (node as IntImmNodeType).value : null;
}

function partitionLoop(node: TirNode, explain: Explain | null): TirNode | null {
  if (!node || node.type !== 'ForNode') return null;
  const outer = node as ForNode;
  if (!PARTITIONABLE_KINDS.has(outer.kind)) return null;
  if (intImm(outer.min) !== 0) return null;
  const No = intImm(outer.extent);
  if (No === null) return null;

  const inner = outer.body as ForNode;
  if (!inner || inner.type !== 'ForNode') return null;
  if (!PARTITIONABLE_KINDS.has(inner.kind)) return null;
  if (intImm(inner.min) !== 0) return null;
  const F = intImm(inner.extent);
  if (F === null || F <= 0) return null;

  const guarded = inner.body as IfThenElseNode;
  if (!guarded || guarded.type !== 'IfThenElseNode' || guarded.elseBody) return null;

  const E = matchFlatGuard(guarded.condition, outer.loopVar.name, inner.loopVar.name, F);
  if (E === null || E <= 0) return null;

  const q = Math.floor(E / F);
  const r = E - q * F;
  if (r === 0) return null;
  if (q < 1) return null;
  if (No !== q + 1) return null;

  const analyzer = new Analyzer();
  analyzer.bind(outer.loopVar.name, 0, q - 1);
  analyzer.bind(inner.loopVar.name, 0, F - 1);
  if (!proveTrue(analyzer, guarded.condition)) return null;

  const thenBody = guarded.thenBody;

  const mainInner = new ForNode(inner.loopVar, new IntImmNode(0), new IntImmNode(F), inner.kind, thenBody, inner.threadTag);
  const mainOuter = new ForNode(outer.loopVar, new IntImmNode(0), new IntImmNode(q), outer.kind, mainInner, outer.threadTag);

  const epiBody = transform(cloneTensorIR(thenBody) as unknown as IRNode, ((n: TirNode) =>
    (n.type === 'VariableNode' && (n as VariableNodeType).name === outer.loopVar.name) ? new IntImmNode(q) : n) as never) as unknown as TirNode;
  const epiInnerVar = new VariableNode(inner.loopVar.name, inner.loopVar.dtype);
  const epilogue = new ForNode(epiInnerVar, new IntImmNode(0), new IntImmNode(r), inner.kind, epiBody, inner.threadTag);

  if (explain) {
    explain(outer.loopVar.name, `split into ${q} full steps plus a ${r}-iteration tail`,
      'inside the full steps the guard is provably true, so it is dropped there and only the tail still checks the bound',
      { fullIterations: q, tailIterations: r, innerExtent: F });
  }

  return new SeqNode([mainOuter, epilogue]);
}

function matchFlatGuard(cond: TirNode | null | undefined, outerName: string, innerName: string, F: number): number | null {
  if (!cond) return null;
  let lhs: TirNode | null | undefined, rhs: TirNode | null | undefined;
  if (cond.type === 'MathOpNode' && (cond as MathOpNode).op === '<') { lhs = (cond as MathOpNode).a; rhs = (cond as MathOpNode).b; }
  else if (cond.type === 'CompareNode' && (cond as CompareNode).direction === 'lt') { lhs = (cond as CompareNode).a; rhs = (cond as CompareNode).b; }
  else return null;

  const E = intImm(rhs);
  if (E === null) return null;
  if (!lhs || lhs.type !== 'MathOpNode' || (lhs as MathOpNode).op !== '+') return null;

  const prod = (lhs as MathOpNode).a as MathOpNode;
  const addend = (lhs as MathOpNode).b as VariableNodeType;
  if (!addend || addend.type !== 'VariableNode' || addend.name !== innerName) return null;
  if (!prod || prod.type !== 'MathOpNode' || prod.op !== '*') return null;
  if (!prod.a || prod.a.type !== 'VariableNode' || (prod.a as VariableNodeType).name !== outerName) return null;
  if (intImm(prod.b) !== F) return null;

  return E;
}
