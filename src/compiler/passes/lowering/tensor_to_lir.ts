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
import { detectAccumulator } from './accumulator.js';
import type { AccumulatorInfo } from './accumulator.js';
import type { PrimFunc, TirNode, BufferLoadNode, BufferStoreNode, CallExternNode, CastNode, CompareNode, MathOpNode, VariableNode } from '../../ir/tensor/nodes.js';
import type { LIRMetadata as LIRMetadataType } from '../../ir/lir/nodes.js';
import type { CompileTarget } from '../../pipeline/pipeline_types.js';

type LowerCtx = {
  shapeParamMap: PrimFunc['shapeParamMap'];
  accCounter: number;
  metadata: LIRMetadataType;
};
type VarSubs = ReadonlyMap<string, TirNode>;

export function lowerToLIR(primFunc: PrimFunc, target: CompileTarget): LIRFunc {
  const metadata = scanMetadata(primFunc, target);
  const ctx: LowerCtx = {
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
  ).copyAttrsFrom(primFunc);
}

function lowerStmt(node: TirNode, ctx: LowerCtx): TirNode {
  if (!node || typeof node !== 'object') return node;

  switch (node.type) {
    case 'ForNode': return lowerForNode(node as ForNode, ctx);
    case 'BlockNode': return lowerBlockNode(node as BlockNode, ctx);
    case 'SeqNode': return lowerSeqNode(node as SeqNode, ctx);
    case 'BufferStoreNode': return lowerBufferStore(node as BufferStoreNode, ctx);
    case 'LetStmtNode': return lowerLetStmt(node as LetStmtNode, ctx);
    case 'AllocateNode': return lowerAllocate(node as AllocateNode, ctx);
    case 'IfThenElseNode': return lowerIfThenElse(node as IfThenElseNode, ctx);
    case 'WhileNode': return lowerWhile(node as WhileNode, ctx);
    case 'EvaluateNode': return new EvaluateNode(lowerExpr((node as EvaluateNode).value, ctx));
    case 'SyncThreadsNode': return node;
    default: return node;
  }
}

function lowerForNode(node: ForNode, ctx: LowerCtx): TirNode {
  const acc = node.accumulator !== undefined ? node.accumulator as AccumulatorInfo | null : detectAccumulator(node);
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

function lowerBlockNode(node: BlockNode, ctx: LowerCtx): TirNode {
  const bindings: { name: string; dtype: string; expr: TirNode }[] = [];
  for (const bind of node.iterVars) {
    if (bind.iterVar && bind.binding) {
      bindings.push({
        name: bind.iterVar.name,
        dtype: bind.iterVar.dtype,
        expr: lowerExpr(bind.binding, ctx),
      });
    }
  }

  let body: TirNode = lowerStmt(node.body, ctx);
  let initBody: TirNode | null = node.initBody ? lowerStmt(node.initBody, ctx) : null;

  if (bindings.length > 0) {
    if (initBody) {
      body = new SeqNode([
        new LIRBindingsNode(bindings, initBody) as unknown as TirNode,
        new LIRBindingsNode(bindings, body) as unknown as TirNode,
      ]);
      initBody = null;
    } else {
      body = new LIRBindingsNode(bindings, body) as unknown as TirNode;
    }
  } else if (initBody) {
    body = new SeqNode([initBody, body]);
  }

  return body;
}

function lowerSeqNode(node: SeqNode, ctx: LowerCtx): TirNode {
  const stmts: TirNode[] = [];
  for (const s of node.stmts) {
    stmts.push(lowerStmt(s, ctx));
  }
  return new SeqNode(stmts);
}

function lowerBufferStore(node: BufferStoreNode, ctx: LowerCtx): TirNode {
  const offsetExpr = flattenIndex(node.buffer, node.indices, ctx.shapeParamMap);
  const value = lowerExpr(node.value, ctx);
  const dtype = node.buffer.dtype || inferDtype(node.value);
  return new LIRFlatStoreNode(node.buffer, offsetExpr, value, dtype) as unknown as TirNode;
}

function lowerLetStmt(node: LetStmtNode, ctx: LowerCtx): TirNode {
  return new LetStmtNode(
    node.variable,
    lowerExpr(node.value, ctx),
    lowerStmt(node.body, ctx),
  );
}

function lowerAllocate(node: AllocateNode, ctx: LowerCtx): TirNode {
  return new AllocateNode(
    node.buffer,
    node.scope,
    lowerStmt(node.body, ctx),
  );
}

function lowerIfThenElse(node: IfThenElseNode, ctx: LowerCtx): TirNode {
  return new IfThenElseNode(
    lowerExpr(node.condition, ctx),
    lowerStmt(node.thenBody, ctx),
    node.elseBody ? lowerStmt(node.elseBody, ctx) : null,
  );
}

function lowerWhile(node: WhileNode, ctx: LowerCtx): TirNode {
  return new WhileNode(
    node.condVar,
    lowerStmt(node.condBody, ctx),
    lowerStmt(node.loopBody, ctx),
  );
}

function lowerExpr(node: TirNode, ctx: LowerCtx): TirNode {
  if (!node || typeof node !== 'object' || !node.type) return node;

  switch (node.type) {
    case 'BufferLoadNode': {
      const load = node as BufferLoadNode;
      const offsetExpr = flattenIndex(load.buffer, load.indices, ctx.shapeParamMap);
      const result = new LIRFlatLoadNode(load.buffer, offsetExpr, load.buffer.dtype);
      annotateDtype(result);
      return result as unknown as TirNode;
    }
    case 'MathOpNode': {
      const m = node as MathOpNode;
      const a = lowerExpr(m.a, ctx);
      const b = m.b ? lowerExpr(m.b, ctx) : null;
      const result = { ...node, a, b };
      Object.setPrototypeOf(result, Object.getPrototypeOf(node));
      annotateDtype(result as TirNode);
      return result as TirNode;
    }
    case 'CompareNode': {
      const c = node as CompareNode;
      const a = lowerExpr(c.a, ctx);
      const b = lowerExpr(c.b, ctx);
      const result = { ...node, a, b };
      Object.setPrototypeOf(result, Object.getPrototypeOf(node));
      annotateDtype(result as TirNode);
      return result as TirNode;
    }
    case 'CastNode': {
      const expr = lowerExpr((node as CastNode).expr, ctx);
      const result = { ...node, expr };
      Object.setPrototypeOf(result, Object.getPrototypeOf(node));
      annotateDtype(result as TirNode);
      return result as TirNode;
    }
    case 'CallExternNode': {
      const args = (node as CallExternNode).args.map((a: TirNode) => lowerExpr(a, ctx));
      const result = { ...node, args };
      Object.setPrototypeOf(result, Object.getPrototypeOf(node));
      annotateDtype(result as TirNode);
      return result as TirNode;
    }
    case 'IfThenElseNode': {
      const ite = node as IfThenElseNode;
      const condition = lowerExpr(ite.condition, ctx);
      const thenBody = lowerExpr(ite.thenBody, ctx);
      const elseBody = ite.elseBody ? lowerExpr(ite.elseBody, ctx) : null;
      const result = { ...node, condition, thenBody, elseBody };
      Object.setPrototypeOf(result, Object.getPrototypeOf(node));
      annotateDtype(result as TirNode);
      return result as TirNode;
    }
    default:
      annotateDtype(node);
      return node;
  }
}

function lowerAccumulator(forNode: ForNode, acc: AccumulatorInfo, ctx: LowerCtx): TirNode {
  const localName = `_acc_${ctx.accCounter++}`;
  const dtype = inferDtype(acc.loadSide);

  ctx.metadata.locals.set(localName, dtype);

  const bindingSubs = new Map<string, TirNode>();
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
    null!,
    acc.store.buffer.dtype,
  );

  const resolvedInitBody = acc.block.initBody
    ? (bindingSubs.size > 0 ? substituteVarsStmt(acc.block.initBody, bindingSubs) : acc.block.initBody)
    : null;

  return new LIRAccumulatorNode({
    localName,
    dtype,
    op: acc.op,
    initLoad,
    loopVar: forNode.loopVar,
    extent: forNode.extent,
    loopKind: forNode.kind,
    body: loweredValue,
    flushStore,
    initBody: resolvedInitBody ? lowerStmt(resolvedInitBody, ctx) : null,
  }) as unknown as TirNode;
}

function substituteVars(node: TirNode, subs: VarSubs): TirNode {
  if (!node || typeof node !== 'object' || !node.type) return node;

  if (node.type === 'VariableNode' && subs.has((node as VariableNode).name)) {
    return subs.get((node as VariableNode).name) as TirNode;
  }

  if (node.type === 'BufferLoadNode') {
    const load = node as BufferLoadNode;
    const newIndices = load.indices.map((idx: TirNode) => substituteVars(idx, subs));
    const changed = newIndices.some((idx: TirNode, i: number) => idx !== load.indices[i]);
    if (!changed) return node;
    const result = { ...node, indices: newIndices };
    Object.setPrototypeOf(result, Object.getPrototypeOf(node));
    return result as TirNode;
  }

  if (node.type === 'MathOpNode') {
    const m = node as MathOpNode;
    const a = substituteVars(m.a, subs);
    const b = m.b ? substituteVars(m.b, subs) : null;
    if (a === m.a && b === m.b) return node;
    const result = { ...node, a, b };
    Object.setPrototypeOf(result, Object.getPrototypeOf(node));
    return result as TirNode;
  }

  if (node.type === 'CompareNode') {
    const c = node as CompareNode;
    const a = substituteVars(c.a, subs);
    const b = substituteVars(c.b, subs);
    if (a === c.a && b === c.b) return node;
    const result = { ...node, a, b };
    Object.setPrototypeOf(result, Object.getPrototypeOf(node));
    return result as TirNode;
  }

  if (node.type === 'CastNode') {
    const cast = node as CastNode;
    const expr = substituteVars(cast.expr, subs);
    if (expr === cast.expr) return node;
    const result = { ...node, expr };
    Object.setPrototypeOf(result, Object.getPrototypeOf(node));
    return result as TirNode;
  }

  if (node.type === 'CallExternNode') {
    const call = node as CallExternNode;
    const args = call.args.map((a: TirNode) => substituteVars(a, subs));
    const changed = args.some((a: TirNode, i: number) => a !== call.args[i]);
    if (!changed) return node;
    const result = { ...node, args };
    Object.setPrototypeOf(result, Object.getPrototypeOf(node));
    return result as TirNode;
  }

  if (node.type === 'IfThenElseNode') {
    const ite = node as IfThenElseNode;
    const condition = substituteVars(ite.condition, subs);
    const thenBody = substituteVars(ite.thenBody, subs);
    const elseBody = ite.elseBody ? substituteVars(ite.elseBody, subs) : null;
    if (condition === ite.condition && thenBody === ite.thenBody && elseBody === ite.elseBody) return node;
    const result = { ...node, condition, thenBody, elseBody };
    Object.setPrototypeOf(result, Object.getPrototypeOf(node));
    return result as TirNode;
  }

  return node;
}

function substituteVarsStmt(node: TirNode, subs: VarSubs): TirNode {
  if (!node || typeof node !== 'object' || !node.type) return node;

  switch (node.type) {
    case 'BufferStoreNode': {
      const store = node as BufferStoreNode;
      const newIndices = store.indices.map((idx: TirNode) => substituteVars(idx, subs));
      const newValue = substituteVars(store.value, subs);
      if (newIndices.every((idx: TirNode, i: number) => idx === store.indices[i]) && newValue === store.value) return node;
      const result = { ...node, indices: newIndices, value: newValue };
      Object.setPrototypeOf(result, Object.getPrototypeOf(node));
      return result as TirNode;
    }
    case 'SeqNode': {
      const stmts = (node as SeqNode).stmts.map((s: TirNode) => substituteVarsStmt(s, subs));
      const result = { ...node, stmts };
      Object.setPrototypeOf(result, Object.getPrototypeOf(node));
      return result as TirNode;
    }
    case 'ForNode': {
      const body = substituteVarsStmt((node as ForNode).body, subs);
      if (body === (node as ForNode).body) return node;
      const result = { ...node, body };
      Object.setPrototypeOf(result, Object.getPrototypeOf(node));
      return result as TirNode;
    }
    case 'IfThenElseNode': {
      const ite = node as IfThenElseNode;
      const condition = substituteVars(ite.condition, subs);
      const thenBody = substituteVarsStmt(ite.thenBody, subs);
      const elseBody = ite.elseBody ? substituteVarsStmt(ite.elseBody, subs) : null;
      const result = { ...node, condition, thenBody, elseBody };
      Object.setPrototypeOf(result, Object.getPrototypeOf(node));
      return result as TirNode;
    }
    default:
      return node;
  }
}
