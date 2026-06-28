import { PrimFuncPass } from '../tir_pass.js';
import { detectAccumulator } from './accumulator.js';

export class AccumulatorDetectionPass extends PrimFuncPass {
  constructor() {
    super('AccumulatorDetectionPass', 'accumulatorDetect');
  }

  run(pf, ctx) {
    annotateStmt(pf.body);
    return pf;
  }
}

function annotateStmt(node) {
  if (!node || typeof node !== 'object' || !node.type) return;
  switch (node.type) {
    case 'ForNode': {
      const acc = detectAccumulator(node);
      node.accumulator = acc;
      if (!acc) annotateStmt(node.body);
      return;
    }
    case 'BlockNode':
      annotateStmt(node.body);
      if (node.initBody) annotateStmt(node.initBody);
      return;
    case 'SeqNode':
      for (const s of node.stmts) annotateStmt(s);
      return;
    case 'LetStmtNode':
    case 'AllocateNode':
      annotateStmt(node.body);
      return;
    case 'IfThenElseNode':
      annotateStmt(node.thenBody);
      if (node.elseBody) annotateStmt(node.elseBody);
      return;
    case 'WhileNode':
      annotateStmt(node.condBody);
      annotateStmt(node.loopBody);
      return;
    default:
      return;
  }
}
