import {
  LIRFunc, LIRFlatLoadNode, LIRFlatStoreNode,
  LIRAccumulatorNode, LIRBindingsNode,
  inferDtype, annotateDtype,
} from '../../ir/lir/nodes.js';
import { scanMetadata } from '../../ir/lir/scanner.js';
import { flattenIndex } from '../../ir/lir/flatten.js';
import {
  ForNode, SeqNode, LetStmtNode, AllocateNode,
  IfThenElseNode, WhileNode, EvaluateNode, BlockNode,
} from '../../ir/tensor/nodes.js';

export function lowerToLIR(primFunc, target) {
  const metadata = scanMetadata(primFunc, target);
  const ctx = {
    shapeParamMap: primFunc.shapeParamMap,
    accCounter: 0,
    metadata,
  };

  const body = lowerStmt(primFunc.body, ctx);

  return new LIRFunc(
    primFunc.name,
    primFunc.params,
    body,
    primFunc.bufferMap,
    primFunc.shapeParams,
    primFunc.shapeParamMap,
    metadata,
  );
}

function lowerStmt(node, ctx) {
  if (!node || typeof node !== 'object') return node;

  switch (node.type) {
    case 'ForNode': return lowerForNode(node, ctx);
    case 'BlockNode': return lowerBlockNode(node, ctx);
    case 'SeqNode': return lowerSeqNode(node, ctx);
    case 'BufferStoreNode': return lowerBufferStore(node, ctx);
    case 'LetStmtNode': return lowerLetStmt(node, ctx);
    case 'AllocateNode': return lowerAllocate(node, ctx);
    case 'IfThenElseNode': return lowerIfThenElse(node, ctx);
    case 'WhileNode': return lowerWhile(node, ctx);
    case 'EvaluateNode': return new EvaluateNode(lowerExpr(node.value, ctx));
    default: return node;
  }
}

function lowerForNode(node, ctx) {
  const acc = detectAccumulator(node);
  if (acc) return lowerAccumulator(node, acc, ctx);

  const loweredBody = lowerStmt(node.body, ctx);
  return new ForNode(
    node.loopVar,
    node.min,
    node.extent,
    node.kind,
    loweredBody,
    node.threadTag,
  );
}

function lowerBlockNode(node, ctx) {
  const bindings = [];
  for (const bind of node.iterVars) {
    if (bind.iterVar && bind.binding) {
      bindings.push({
        name: bind.iterVar.name,
        dtype: bind.iterVar.dtype,
        expr: lowerExpr(bind.binding, ctx),
      });
    }
  }

  let body = lowerStmt(node.body, ctx);
  let initBody = node.initBody ? lowerStmt(node.initBody, ctx) : null;

  if (bindings.length > 0) {
    if (initBody) {
      body = new SeqNode([
        new LIRBindingsNode(bindings, initBody),
        new LIRBindingsNode(bindings, body),
      ]);
      initBody = null;
    } else {
      body = new LIRBindingsNode(bindings, body);
    }
  } else if (initBody) {
    body = new SeqNode([initBody, body]);
  }

  return body;
}

function lowerSeqNode(node, ctx) {
  const stmts = [];
  for (const s of node.stmts) {
    stmts.push(lowerStmt(s, ctx));
  }
  return new SeqNode(stmts);
}

function lowerBufferStore(node, ctx) {
  const offsetExpr = flattenIndex(node.buffer, node.indices, ctx.shapeParamMap);
  const value = lowerExpr(node.value, ctx);
  const dtype = node.buffer.dtype || inferDtype(node.value);
  return new LIRFlatStoreNode(node.buffer, offsetExpr, value, dtype);
}

function lowerLetStmt(node, ctx) {
  return new LetStmtNode(
    node.variable,
    lowerExpr(node.value, ctx),
    lowerStmt(node.body, ctx),
  );
}

function lowerAllocate(node, ctx) {
  return new AllocateNode(
    node.buffer,
    node.scope,
    lowerStmt(node.body, ctx),
  );
}

function lowerIfThenElse(node, ctx) {
  return new IfThenElseNode(
    lowerExpr(node.condition, ctx),
    lowerStmt(node.thenBody, ctx),
    node.elseBody ? lowerStmt(node.elseBody, ctx) : null,
  );
}

function lowerWhile(node, ctx) {
  return new WhileNode(
    node.condVar,
    lowerStmt(node.condBody, ctx),
    lowerStmt(node.loopBody, ctx),
  );
}

function lowerExpr(node, ctx) {
  if (!node || typeof node !== 'object' || !node.type) return node;

  switch (node.type) {
    case 'BufferLoadNode': {
      const offsetExpr = flattenIndex(node.buffer, node.indices, ctx.shapeParamMap);
      const result = new LIRFlatLoadNode(node.buffer, offsetExpr, node.buffer.dtype);
      annotateDtype(result);
      return result;
    }
    case 'MathOpNode': {
      const a = lowerExpr(node.a, ctx);
      const b = node.b ? lowerExpr(node.b, ctx) : null;
      const result = { ...node, a, b };
      Object.setPrototypeOf(result, Object.getPrototypeOf(node));
      annotateDtype(result);
      return result;
    }
    case 'CompareNode': {
      const a = lowerExpr(node.a, ctx);
      const b = lowerExpr(node.b, ctx);
      const result = { ...node, a, b };
      Object.setPrototypeOf(result, Object.getPrototypeOf(node));
      annotateDtype(result);
      return result;
    }
    case 'CastNode': {
      const expr = lowerExpr(node.expr, ctx);
      const result = { ...node, expr };
      Object.setPrototypeOf(result, Object.getPrototypeOf(node));
      annotateDtype(result);
      return result;
    }
    case 'CallExternNode': {
      const args = node.args.map(a => lowerExpr(a, ctx));
      const result = { ...node, args };
      Object.setPrototypeOf(result, Object.getPrototypeOf(node));
      annotateDtype(result);
      return result;
    }
    case 'IfThenElseNode': {
      const condition = lowerExpr(node.condition, ctx);
      const thenBody = lowerExpr(node.thenBody, ctx);
      const elseBody = node.elseBody ? lowerExpr(node.elseBody, ctx) : null;
      const result = { ...node, condition, thenBody, elseBody };
      Object.setPrototypeOf(result, Object.getPrototypeOf(node));
      annotateDtype(result);
      return result;
    }
    default:
      annotateDtype(node);
      return node;
  }
}

function detectAccumulator(forNode) {
  const block = forNode.body;
  if (!block || block.type !== 'BlockNode') return null;

  const inner = block.body;
  if (!inner || inner.type !== 'BufferStoreNode') return null;

  const store = inner;
  const val = store.value;
  if (!val || val.type !== 'MathOpNode' || val.op !== '+') return null;

  let loadSide = null;
  let valueSide = null;

  if (val.a && val.a.type === 'BufferLoadNode' && val.a.buffer.name === store.buffer.name) {
    loadSide = val.a;
    valueSide = val.b;
  } else if (val.b && val.b.type === 'BufferLoadNode' && val.b.buffer.name === store.buffer.name) {
    loadSide = val.b;
    valueSide = val.a;
  }
  if (!loadSide) return null;

  const storeKey = indicesKey(store.indices);
  const loadKey = indicesKey(loadSide.indices);
  if (storeKey !== loadKey) return null;
  if (storeKey.includes('?')) return null;

  const outerIndices = store.indices.map(idx => {
    if (idx.type !== 'VariableNode') return idx;
    for (const bind of block.iterVars) {
      if (bind.iterVar && bind.iterVar.name === idx.name && bind.binding) {
        return bind.binding;
      }
    }
    return idx;
  });

  const loopVarName = forNode.loopVar.name;
  const resolvedKey = indicesKey(outerIndices);
  if (resolvedKey.includes('?')) return null;
  if (resolvedKey.includes('$' + loopVarName)) return null;

  return {
    store,
    loadSide,
    valueSide,
    outerIndices,
    block,
  };
}

function lowerAccumulator(forNode, acc, ctx) {
  const localName = `_acc_${ctx.accCounter++}`;
  const dtype = inferDtype(acc.loadSide);

  ctx.metadata.locals.set(localName, dtype);

  const bindingSubs = new Map();
  for (const bind of acc.block.iterVars) {
    if (bind.iterVar && bind.binding) {
      bindingSubs.set(bind.iterVar.name, bind.binding);
    }
  }

  const resolvedValue = bindingSubs.size > 0
    ? substituteVars(acc.valueSide, bindingSubs)
    : acc.valueSide;

  const initOffsetExpr = flattenIndex(acc.store.buffer, acc.outerIndices, ctx.shapeParamMap);
  const initLoad = new LIRFlatLoadNode(acc.store.buffer, initOffsetExpr, acc.store.buffer.dtype);
  annotateDtype(initLoad);

  const loweredValue = lowerExpr(resolvedValue, ctx);

  const flushOffsetExpr = flattenIndex(acc.store.buffer, acc.outerIndices, ctx.shapeParamMap);
  const flushStore = new LIRFlatStoreNode(
    acc.store.buffer,
    flushOffsetExpr,
    null,
    acc.store.buffer.dtype,
  );

  const resolvedInitBody = acc.block.initBody
    ? (bindingSubs.size > 0 ? substituteVarsStmt(acc.block.initBody, bindingSubs) : acc.block.initBody)
    : null;

  return new LIRAccumulatorNode({
    localName,
    dtype,
    initLoad,
    loopVar: forNode.loopVar,
    extent: forNode.extent,
    loopKind: forNode.kind,
    body: loweredValue,
    flushStore,
    initBody: resolvedInitBody ? lowerStmt(resolvedInitBody, ctx) : null,
  });
}

function substituteVars(node, subs) {
  if (!node || typeof node !== 'object' || !node.type) return node;

  if (node.type === 'VariableNode' && subs.has(node.name)) {
    return subs.get(node.name);
  }

  if (node.type === 'BufferLoadNode') {
    const newIndices = node.indices.map(idx => substituteVars(idx, subs));
    const changed = newIndices.some((idx, i) => idx !== node.indices[i]);
    if (!changed) return node;
    const result = { ...node, indices: newIndices };
    Object.setPrototypeOf(result, Object.getPrototypeOf(node));
    return result;
  }

  if (node.type === 'MathOpNode') {
    const a = substituteVars(node.a, subs);
    const b = node.b ? substituteVars(node.b, subs) : null;
    if (a === node.a && b === node.b) return node;
    const result = { ...node, a, b };
    Object.setPrototypeOf(result, Object.getPrototypeOf(node));
    return result;
  }

  if (node.type === 'CompareNode') {
    const a = substituteVars(node.a, subs);
    const b = substituteVars(node.b, subs);
    if (a === node.a && b === node.b) return node;
    const result = { ...node, a, b };
    Object.setPrototypeOf(result, Object.getPrototypeOf(node));
    return result;
  }

  if (node.type === 'CastNode') {
    const expr = substituteVars(node.expr, subs);
    if (expr === node.expr) return node;
    const result = { ...node, expr };
    Object.setPrototypeOf(result, Object.getPrototypeOf(node));
    return result;
  }

  if (node.type === 'CallExternNode') {
    const args = node.args.map(a => substituteVars(a, subs));
    const changed = args.some((a, i) => a !== node.args[i]);
    if (!changed) return node;
    const result = { ...node, args };
    Object.setPrototypeOf(result, Object.getPrototypeOf(node));
    return result;
  }

  if (node.type === 'IfThenElseNode') {
    const condition = substituteVars(node.condition, subs);
    const thenBody = substituteVars(node.thenBody, subs);
    const elseBody = node.elseBody ? substituteVars(node.elseBody, subs) : null;
    if (condition === node.condition && thenBody === node.thenBody && elseBody === node.elseBody) return node;
    const result = { ...node, condition, thenBody, elseBody };
    Object.setPrototypeOf(result, Object.getPrototypeOf(node));
    return result;
  }

  return node;
}

function substituteVarsStmt(node, subs) {
  if (!node || typeof node !== 'object' || !node.type) return node;

  switch (node.type) {
    case 'BufferStoreNode': {
      const newIndices = node.indices.map(idx => substituteVars(idx, subs));
      const newValue = substituteVars(node.value, subs);
      if (newIndices.every((idx, i) => idx === node.indices[i]) && newValue === node.value) return node;
      const result = { ...node, indices: newIndices, value: newValue };
      Object.setPrototypeOf(result, Object.getPrototypeOf(node));
      return result;
    }
    case 'SeqNode': {
      const stmts = node.stmts.map(s => substituteVarsStmt(s, subs));
      const result = { ...node, stmts };
      Object.setPrototypeOf(result, Object.getPrototypeOf(node));
      return result;
    }
    case 'ForNode': {
      const body = substituteVarsStmt(node.body, subs);
      if (body === node.body) return node;
      const result = { ...node, body };
      Object.setPrototypeOf(result, Object.getPrototypeOf(node));
      return result;
    }
    case 'IfThenElseNode': {
      const condition = substituteVars(node.condition, subs);
      const thenBody = substituteVarsStmt(node.thenBody, subs);
      const elseBody = node.elseBody ? substituteVarsStmt(node.elseBody, subs) : null;
      const result = { ...node, condition, thenBody, elseBody };
      Object.setPrototypeOf(result, Object.getPrototypeOf(node));
      return result;
    }
    default:
      return node;
  }
}

function indicesKey(indices) {
  return indices.map(exprKey).join(',');
}

function exprKey(node) {
  if (!node) return '?';
  if (node.type === 'VariableNode') return '$' + node.name;
  if (node.type === 'IntImmNode') return String(node.value);
  if (node.type === 'MathOpNode') {
    return '(' + exprKey(node.a) + node.op + (node.b ? exprKey(node.b) : '') + ')';
  }
  return '?';
}
