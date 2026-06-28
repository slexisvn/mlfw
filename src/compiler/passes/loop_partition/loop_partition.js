import { PrimFuncPass } from '../tir_pass.js';
import { ForNode, SeqNode, IntImmNode, VariableNode, ForKind } from '../../ir/tensor/nodes.js';
import { transform } from '../../ir/ir_visitor.js';
import { cloneTensorIR } from '../../autotune/tune_ir.js';
import { Analyzer } from '../../analysis/analyzer.js';
import { proveTrue } from '../../analysis/ir_arith.js';

const PARTITIONABLE_KINDS = new Set([
  ForKind.SERIAL, ForKind.PARALLEL, ForKind.VECTORIZED, ForKind.UNROLLED,
]);

export class LoopPartitionPass extends PrimFuncPass {
  constructor() {
    super('LoopPartitionPass', 'loopPartition');
  }

  run(pf, ctx) {
    const newBody = transform(pf.body, (n) => partitionLoop(n) || n);
    if (newBody && newBody !== pf.body) {
      pf.body = newBody;
      pf._setChild('body', newBody);
    }
    return pf;
  }
}

function intImm(node) {
  return node && node.type === 'IntImmNode' ? node.value : null;
}

function partitionLoop(outer) {
  if (!outer || outer.type !== 'ForNode') return null;
  if (!PARTITIONABLE_KINDS.has(outer.kind)) return null;
  if (intImm(outer.min) !== 0) return null;
  const No = intImm(outer.extent);
  if (No === null) return null;

  const inner = outer.body;
  if (!inner || inner.type !== 'ForNode') return null;
  if (!PARTITIONABLE_KINDS.has(inner.kind)) return null;
  if (intImm(inner.min) !== 0) return null;
  const F = intImm(inner.extent);
  if (F === null || F <= 0) return null;

  const guarded = inner.body;
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

  const epiBody = transform(cloneTensorIR(thenBody), (n) =>
    (n.type === 'VariableNode' && n.name === outer.loopVar.name) ? new IntImmNode(q) : n);
  const epiInnerVar = new VariableNode(inner.loopVar.name, inner.loopVar.dtype);
  const epilogue = new ForNode(epiInnerVar, new IntImmNode(0), new IntImmNode(r), inner.kind, epiBody, inner.threadTag);

  return new SeqNode([mainOuter, epilogue]);
}

function matchFlatGuard(cond, outerName, innerName, F) {
  if (!cond) return null;
  let lhs, rhs;
  if (cond.type === 'MathOpNode' && cond.op === '<') { lhs = cond.a; rhs = cond.b; }
  else if (cond.type === 'CompareNode' && cond.direction === 'lt') { lhs = cond.a; rhs = cond.b; }
  else return null;

  const E = intImm(rhs);
  if (E === null) return null;
  if (!lhs || lhs.type !== 'MathOpNode' || lhs.op !== '+') return null;

  const prod = lhs.a;
  const addend = lhs.b;
  if (!addend || addend.type !== 'VariableNode' || addend.name !== innerName) return null;
  if (!prod || prod.type !== 'MathOpNode' || prod.op !== '*') return null;
  if (!prod.a || prod.a.type !== 'VariableNode' || prod.a.name !== outerName) return null;
  if (intImm(prod.b) !== F) return null;

  return E;
}
