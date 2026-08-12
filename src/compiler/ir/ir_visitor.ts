import { TensorNode } from './tensor/nodes.js';
import type { TirNode } from './tensor/nodes.js';
import type { LirNode } from './lir/nodes.js';

export const STOP = Symbol('ir-visitor-stop');

export type IRNode = TirNode | LirNode;
type NodeFields = Record<string, unknown>;

export type FieldKind = 'expr' | 'stmt';
export type FieldOpts = Readonly<{ array?: boolean; poly?: boolean; bind?: boolean; param?: boolean; region?: boolean; iterVarBinding?: boolean; bindingsExpr?: boolean }>;
export type FieldSpec = Readonly<{ key: string; kind: FieldKind; array: boolean; poly: boolean; bind: boolean; param: boolean; region: boolean; iterVarBinding: boolean; bindingsExpr: boolean }>;

export type ChildAccessor = { read(): IRNode[]; write(vals: readonly IRNode[]): void };
export type WalkOpts = Readonly<{ kinds?: FieldKind | 'both'; descendParams?: boolean; bindVars?: boolean }>;
export type WalkContext = Readonly<{ parent: IRNode | null; depth: number }>;
export type VisitResult = void | false | typeof STOP;
export type PreVisitor = (node: IRNode, ctx: WalkContext) => VisitResult;
export type Visitor = PreVisitor | Readonly<{ pre?: PreVisitor; post?: (node: IRNode, ctx: WalkContext) => void }>;

export function isIRNode(x: unknown): x is IRNode {
  return x instanceof TensorNode;
}

function field(key: string, kind: FieldKind, opts: FieldOpts = {}): FieldSpec {
  return { key, kind, array: !!opts.array, poly: !!opts.poly, bind: !!opts.bind, param: !!opts.param, region: !!opts.region, iterVarBinding: !!opts.iterVarBinding, bindingsExpr: !!opts.bindingsExpr };
}

const SCHEMA: Readonly<Record<string, readonly FieldSpec[]>> = {
  VariableNode: [],
  IntImmNode: [],
  FloatImmNode: [],
  SyncThreadsNode: [],

  MathOpNode: [field('a', 'expr'), field('b', 'expr')],
  CompareNode: [field('a', 'expr'), field('b', 'expr')],
  CastNode: [field('expr', 'expr')],
  CallExternNode: [field('args', 'expr', { array: true })],
  BufferLoadNode: [field('indices', 'expr', { array: true })],
  IfThenElseNode: [field('condition', 'expr'), field('thenBody', 'stmt'), field('elseBody', 'stmt')],
  BlockRealizeNode: [field('binding', 'expr', { poly: true }), field('iterVar', 'expr', { bind: true })],
  LIRFlatLoadNode: [field('offsetExpr', 'expr')],

  PrimFunc: [field('params', 'expr', { array: true, param: true }), field('body', 'stmt')],
  ForNode: [field('min', 'expr'), field('extent', 'expr'), field('loopVar', 'expr', { bind: true }), field('body', 'stmt')],
  BlockNode: [field('iterVars', 'expr', { iterVarBinding: true }), field('reads', 'expr', { region: true, array: true }), field('writes', 'expr', { region: true, array: true }), field('initBody', 'stmt'), field('body', 'stmt')],
  SeqNode: [field('stmts', 'stmt', { array: true })],
  VecCopyNode: [field('dstIndex', 'expr'), field('srcIndex', 'expr')],
  LetStmtNode: [field('value', 'expr'), field('variable', 'expr', { bind: true }), field('body', 'stmt')],
  AllocateNode: [field('body', 'stmt')],
  WhileNode: [field('condBody', 'stmt'), field('condVar', 'expr', { bind: true }), field('loopBody', 'stmt')],
  EvaluateNode: [field('value', 'expr')],
  BufferStoreNode: [field('indices', 'expr', { array: true }), field('value', 'expr')],

  LIRFunc: [field('body', 'stmt')],
  LIRFlatStoreNode: [field('offsetExpr', 'expr'), field('value', 'expr')],
  LIRAccumulatorNode: [field('loopVar', 'expr', { bind: true }), field('extent', 'expr'), field('initLoad', 'stmt'), field('initBody', 'stmt'), field('body', 'expr'), field('flushStore', 'stmt')],
  LIRBindingsNode: [field('bindings', 'expr', { bindingsExpr: true }), field('body', 'stmt')],
};

function normOpts(opts: WalkOpts): Required<WalkOpts> {
  return {
    kinds: opts.kinds || 'both',
    descendParams: opts.descendParams === true,
    bindVars: opts.bindVars !== false,
  };
}

function buildAccessor(node: IRNode, f: FieldSpec): ChildAccessor {
  const fields = node as unknown as NodeFields;
  if (f.iterVarBinding) {
    return {
      read() { return ((fields.iterVars as { binding?: unknown }[] | undefined) || []).map((iv) => iv && iv.binding as IRNode).filter(isIRNode); },
      write(vals) { let i = 0; for (const iv of ((fields.iterVars as { binding?: unknown }[] | undefined) || [])) { if (iv && isIRNode(iv.binding)) iv.binding = vals[i++]; } },
    };
  }
  if (f.bindingsExpr) {
    return {
      read() { return ((fields.bindings as { expr?: unknown }[] | undefined) || []).map((b) => b && b.expr as IRNode).filter(isIRNode); },
      write(vals) { let i = 0; for (const b of ((fields.bindings as { expr?: unknown }[] | undefined) || [])) { if (b && isIRNode(b.expr)) b.expr = vals[i++]; } },
    };
  }
  if (f.poly) {
    return {
      read() { const v = fields[f.key]; return Array.isArray(v) ? v.filter(isIRNode) : (isIRNode(v) ? [v] : []); },
      write(vals) { const v = fields[f.key]; if (Array.isArray(v)) { let i = 0; fields[f.key] = v.map((e) => (isIRNode(e) ? vals[i++] : e)); } else if (vals.length) { fields[f.key] = vals[0]; } },
    };
  }
  if (f.array) {
    return {
      read() { const v = fields[f.key]; return Array.isArray(v) ? v.filter(isIRNode) : []; },
      write(vals) { const v = fields[f.key]; if (!Array.isArray(v)) return; let i = 0; fields[f.key] = v.map((e) => (isIRNode(e) ? vals[i++] : e)); },
    };
  }
  return {
    read() { return isIRNode(fields[f.key]) ? [fields[f.key] as IRNode] : []; },
    write(vals) { if (vals.length && isIRNode(fields[f.key])) fields[f.key] = vals[0]; },
  };
}

export function childAccessors(node: IRNode, opts: WalkOpts = {}): ChildAccessor[] {
  const o = normOpts(opts);
  const fields = SCHEMA[node.type];
  if (fields === undefined) throw new Error(`ir_visitor: no child schema for node type '${node.type}'`);
  const out: ChildAccessor[] = [];
  for (const f of fields) {
    if (f.region) continue;
    if (f.param && !o.descendParams) continue;
    if (f.bind && !o.bindVars) continue;
    if (o.kinds !== 'both' && f.kind !== o.kinds) continue;
    out.push(buildAccessor(node, f));
  }
  return out;
}

export function irChildNodes(node: IRNode, opts: WalkOpts = {}): IRNode[] {
  if (!isIRNode(node) || SCHEMA[node.type] === undefined) return [];
  const out: IRNode[] = [];
  for (const acc of childAccessors(node, opts)) {
    const kids = acc.read();
    for (let i = 0; i < kids.length; i++) out.push(kids[i]);
  }
  return out;
}

function setParent(child: IRNode, parent: IRNode, key: string, idx: number): void {
  (child as unknown as NodeFields)._parent = parent;
  (child as unknown as NodeFields)._parentKey = key;
  (child as unknown as NodeFields)._parentIdx = idx;
}

function relink(node: IRNode): void {
  const slots = node as unknown as NodeFields;
  const fields = SCHEMA[node.type];
  if (!fields) return;
  for (const f of fields) {
    if (f.iterVarBinding || f.bindingsExpr || f.region) continue;
    const v = slots[f.key];
    if (Array.isArray(v)) {
      for (let i = 0; i < v.length; i++) if (isIRNode(v[i])) setParent(v[i], node, f.key, i);
    } else if (isIRNode(v)) {
      setParent(v, node, f.key, -1);
    }
  }
}

export function walk(node: IRNode, visitor: Visitor, opts: WalkOpts = {}): void {
  const v: { pre?: PreVisitor; post?: (n: IRNode, c: WalkContext) => void } = typeof visitor === 'function' ? { pre: visitor } : (visitor || {});
  const o = normOpts(opts);
  if (!isIRNode(node)) return;

  const stack: { node: IRNode; parent: IRNode | null; depth: number; entered: boolean }[] = [{ node, parent: null, depth: 0, entered: false }];
  while (stack.length > 0) {
    const frame = stack[stack.length - 1];
    if (frame.entered) {
      stack.pop();
      if (v.post) v.post(frame.node, { parent: frame.parent, depth: frame.depth });
      continue;
    }
    frame.entered = true;
    let descend = true;
    if (v.pre) {
      const r = v.pre(frame.node, { parent: frame.parent, depth: frame.depth });
      if (r === STOP) return;
      if (r === false) descend = false;
    }
    if (!descend) continue;
    const kids: IRNode[] = [];
    for (const acc of childAccessors(frame.node, o)) {
      for (const kid of acc.read()) kids.push(kid);
    }
    for (let i = kids.length - 1; i >= 0; i--) {
      stack.push({ node: kids[i], parent: frame.node, depth: frame.depth + 1, entered: false });
    }
  }
}

export type ScopeVisitor<C> = (node: IRNode, scope: C) => C | false;

export function walkScoped<C>(node: IRNode, initial: C, visit: ScopeVisitor<C>, opts: WalkOpts = {}): void {
  if (!isIRNode(node)) return;
  const stack: { node: IRNode; scope: C }[] = [{ node, scope: initial }];
  while (stack.length > 0) {
    const frame = stack.pop() as { node: IRNode; scope: C };
    const inner = visit(frame.node, frame.scope);
    if (inner === false) continue;
    const kids = irChildNodes(frame.node, opts);
    for (let i = kids.length - 1; i >= 0; i--) stack.push({ node: kids[i], scope: inner });
  }
}

export function collect(node: IRNode, pred: (n: IRNode) => boolean, opts: WalkOpts = {}): IRNode[] {
  const out: IRNode[] = [];
  walk(node, (n) => { if (pred(n)) out.push(n); }, opts);
  return out;
}

export function some(node: IRNode, pred: (n: IRNode) => boolean, opts: WalkOpts = {}): boolean {
  let found = false;
  walk(node, (n) => { if (pred(n)) { found = true; return STOP; } }, opts);
  return found;
}

export function find(node: IRNode, pred: (n: IRNode) => boolean, opts: WalkOpts = {}): IRNode | null {
  let result: IRNode | null = null;
  walk(node, (n) => { if (pred(n)) { result = n; return STOP; } }, opts);
  return result;
}

export function transform(node: IRNode, fn: (n: IRNode) => IRNode | null | undefined, opts: WalkOpts = {}): IRNode {
  const o = normOpts(opts);
  if (!isIRNode(node)) return node;

  const replacements = new Map<IRNode, IRNode>();
  const stack: { node: IRNode; entered: boolean }[] = [{ node, entered: false }];
  while (stack.length > 0) {
    const frame = stack[stack.length - 1];
    if (!frame.entered) {
      frame.entered = true;
      const kids: IRNode[] = [];
      for (const acc of childAccessors(frame.node, o)) {
        for (const kid of acc.read()) kids.push(kid);
      }
      for (let i = kids.length - 1; i >= 0; i--) stack.push({ node: kids[i], entered: false });
      continue;
    }
    stack.pop();

    let anyChanged = false;
    for (const acc of childAccessors(frame.node, o)) {
      const kids = acc.read();
      if (kids.length === 0) continue;
      let changed = false;
      const next = new Array<IRNode>(kids.length);
      for (let i = 0; i < kids.length; i++) {
        const nk = replacements.has(kids[i]) ? replacements.get(kids[i]) as IRNode : kids[i];
        if (nk !== kids[i]) changed = true;
        next[i] = nk;
      }
      if (changed) { acc.write(next); anyChanged = true; }
    }
    if (anyChanged) relink(frame.node);

    const replaced = fn(frame.node);
    if (replaced !== undefined && replaced !== null && replaced !== frame.node) {
      replacements.set(frame.node, replaced);
    }
  }
  return replacements.has(node) ? replacements.get(node) as IRNode : node;
}
