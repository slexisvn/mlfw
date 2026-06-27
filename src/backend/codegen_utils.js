export function parseThreadAxis(tag) {
  const idx = tag.indexOf('.');
  if (idx < 0) return null;
  const axis = tag.charCodeAt(idx + 1) - 120;
  if (axis < 0 || axis > 2) return null;
  const prefix = tag.substring(0, idx);
  if (prefix === 'threadIdx') return { space: 'thread', axis };
  if (prefix === 'blockIdx') return { space: 'block', axis };
  return null;
}

export function visitStatements(cg, start) {
  const stack = [start];
  while (stack.length > 0) {
    const cur = stack.pop();
    if (!cur) continue;
    switch (cur.type) {
      case 'SeqNode':
        for (let i = cur.stmts.length - 1; i >= 0; i--) stack.push(cur.stmts[i]);
        continue;
      case 'AllocateNode':
        cg._visitAllocateNode(cur);
        stack.push(cur.body);
        continue;
      case 'ForNode': cg._visitForNode(cur); continue;
      case 'BlockNode': cg._visitBlockNode(cur); continue;
      case 'IfThenElseNode': cg._visitIfStmt(cur); continue;
      case 'LetStmtNode': cg._visitLetStmtNode(cur); continue;
      case 'BufferStoreNode': cg._visitBufferStoreNode(cur); continue;
      case 'LIRFlatStoreNode': cg._visitLIRFlatStore(cur); continue;
      case 'LIRBindingsNode': cg._visitLIRBindings(cur); continue;
      case 'LIRAccumulatorNode': cg._visitLIRAccumulator(cur); continue;
      case 'WhileNode': cg._visitWhileNode(cur); continue;
      case 'SyncThreadsNode': cg._emitSync(); continue;
      case 'EvaluateNode': continue;
      default: throw new Error(`${cg.constructor.name}: unhandled statement node '${cur.type}'`);
    }
  }
}

export function maxBindingExtent(threadBindings, tag) {
  const entries = threadBindings.get(tag);
  if (!entries) return 0;
  let max = 0;
  for (const e of entries) if (e.extent > max) max = e.extent;
  return max;
}
