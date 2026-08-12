import { PrimFuncPass } from '../tir_pass.js';
import { detectAccumulator } from './accumulator.js';
import type { PrimFunc, TirNode, ForNode, BlockNode, SeqNode, IfThenElseNode, WhileNode, AllocateNode, LetStmtNode } from '../../ir/tensor/nodes.js';
import type { TirPassCtx } from '../tir_pass.js';

export class AccumulatorDetectionPass extends PrimFuncPass {
  constructor() {
    super('AccumulatorDetectionPass', 'accumulatorDetect');
  }

  override run(pf: PrimFunc, ctx: TirPassCtx): PrimFunc {
    annotateStmt(pf.body);
    return pf;
  }
}

function annotateStmt(node: TirNode | null | undefined): void {
  if (!node || typeof node !== 'object' || !node.type) return;
  switch (node.type) {
    case 'ForNode': {
      const f = node as ForNode;
      const acc = detectAccumulator(f);
      f.accumulator = acc;
      if (!acc) annotateStmt(f.body);
      return;
    }
    case 'BlockNode': {
      const b = node as BlockNode;
      annotateStmt(b.body);
      if (b.initBody) annotateStmt(b.initBody);
      return;
    }
    case 'SeqNode':
      for (const st of (node as SeqNode).stmts) annotateStmt(st);
      return;
    case 'LetStmtNode':
    case 'AllocateNode':
      annotateStmt((node as LetStmtNode | AllocateNode).body);
      return;
    case 'IfThenElseNode': {
      const ite = node as IfThenElseNode;
      annotateStmt(ite.thenBody);
      if (ite.elseBody) annotateStmt(ite.elseBody);
      return;
    }
    case 'WhileNode': {
      const w = node as WhileNode;
      annotateStmt(w.condBody);
      annotateStmt(w.loopBody);
      return;
    }
    default:
      return;
  }
}
