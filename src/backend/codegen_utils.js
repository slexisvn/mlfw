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

const STMT_CHILD_FIELDS = ['body', 'stmts', 'thenBody', 'elseBody', 'loopBody', 'condBody', 'initBody'];

export function walkStmtTree(root, visit) {
  const stack = [root];
  while (stack.length > 0) {
    const n = stack.pop();
    if (!n || typeof n !== 'object') continue;
    if (visit(n) === false) continue;
    for (const k of STMT_CHILD_FIELDS) {
      const v = n[k];
      if (v == null) continue;
      if (Array.isArray(v)) { for (let i = v.length - 1; i >= 0; i--) stack.push(v[i]); }
      else stack.push(v);
    }
  }
}

export function isZeroFillBody(body) {
  let cur = body;
  while (cur) {
    if (cur.type === 'ForNode' || cur.type === 'BlockNode') { cur = cur.body; continue; }
    if (cur.type === 'BufferStoreNode' || cur.type === 'LIRFlatStoreNode') {
      const val = cur.value;
      return (val.type === 'FloatImmNode' && val.value === 0) || (val.type === 'IntImmNode' && val.value === 0);
    }
    return false;
  }
  return false;
}

export function resolveShapeParam(primFunc, buffer, dimIdx, format, label) {
  if (primFunc && primFunc.shapeParamMap) {
    const v = primFunc.shapeParamMap.get(`${buffer.name}:${dimIdx}`);
    if (v) return format(v);
  }
  throw new Error(`${label} codegen: missing shape param for ${buffer.name}:${dimIdx}`);
}

export function estimateBufferSize(buffer) {
  let n = 1;
  for (const d of buffer.shape) {
    if (typeof d === 'number' && d > 0) n *= d;
  }
  return n;
}

export function dynamicDimProduct(buffer, startDim, resolveShapeParam) {
  const parts = [];
  for (let j = startDim; j < buffer.shape.length; j++) {
    const d = buffer.shape[j];
    if (typeof d === 'number' && d >= 0) parts.push(String(d));
    else parts.push(resolveShapeParam(buffer, j));
  }
  return parts.length === 0 ? '1' : parts.join(' * ');
}

export function maxBindingExtent(threadBindings, tag) {
  const entries = threadBindings.get(tag);
  if (!entries) return 0;
  let max = 0;
  for (const e of entries) if (e.extent > max) max = e.extent;
  return max;
}
