import { TensorNode } from './tensor/nodes.js';
import { LIRNode } from './lir/nodes.js';

export const STOP = Symbol('ir-visitor-stop');

export function isIRNode(x) {
  return x instanceof TensorNode || x instanceof LIRNode;
}

function field(key, kind, opts = {}) {
  return { key, kind, array: !!opts.array, poly: !!opts.poly, bind: !!opts.bind, param: !!opts.param, region: !!opts.region, iterVarBinding: !!opts.iterVarBinding, bindingsExpr: !!opts.bindingsExpr };
}

const SCHEMA = {
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

function normOpts(opts) {
  return {
    kinds: opts.kinds || 'both',
    descendParams: opts.descendParams === true,
    bindVars: opts.bindVars !== false,
  };
}

function buildAccessor(node, f) {
  if (f.iterVarBinding) {
    return {
      read() { return (node.iterVars || []).map((iv) => iv && iv.binding).filter(isIRNode); },
      write(vals) { let i = 0; for (const iv of (node.iterVars || [])) { if (iv && isIRNode(iv.binding)) iv.binding = vals[i++]; } },
    };
  }
  if (f.bindingsExpr) {
    return {
      read() { return (node.bindings || []).map((b) => b && b.expr).filter(isIRNode); },
      write(vals) { let i = 0; for (const b of (node.bindings || [])) { if (b && isIRNode(b.expr)) b.expr = vals[i++]; } },
    };
  }
  if (f.poly) {
    return {
      read() { const v = node[f.key]; return Array.isArray(v) ? v.filter(isIRNode) : (isIRNode(v) ? [v] : []); },
      write(vals) { const v = node[f.key]; if (Array.isArray(v)) { let i = 0; node[f.key] = v.map((e) => (isIRNode(e) ? vals[i++] : e)); } else if (vals.length) { node[f.key] = vals[0]; } },
    };
  }
  if (f.array) {
    return {
      read() { const v = node[f.key]; return Array.isArray(v) ? v.filter(isIRNode) : []; },
      write(vals) { const v = node[f.key]; if (!Array.isArray(v)) return; let i = 0; node[f.key] = v.map((e) => (isIRNode(e) ? vals[i++] : e)); },
    };
  }
  return {
    read() { return isIRNode(node[f.key]) ? [node[f.key]] : []; },
    write(vals) { if (vals.length && isIRNode(node[f.key])) node[f.key] = vals[0]; },
  };
}

export function childAccessors(node, opts = {}) {
  const o = normOpts(opts);
  const fields = SCHEMA[node.type];
  if (fields === undefined) throw new Error(`ir_visitor: no child schema for node type '${node.type}'`);
  const out = [];
  for (const f of fields) {
    if (f.region) continue;
    if (f.param && !o.descendParams) continue;
    if (f.bind && !o.bindVars) continue;
    if (o.kinds !== 'both' && f.kind !== o.kinds) continue;
    out.push(buildAccessor(node, f));
  }
  return out;
}

export function irChildNodes(node, opts = {}) {
  if (!isIRNode(node) || SCHEMA[node.type] === undefined) return [];
  const out = [];
  for (const acc of childAccessors(node, opts)) {
    const kids = acc.read();
    for (let i = 0; i < kids.length; i++) out.push(kids[i]);
  }
  return out;
}

function setParent(child, parent, key, idx) {
  child._parent = parent;
  child._parentKey = key;
  child._parentIdx = idx;
}

function relink(node) {
  const fields = SCHEMA[node.type];
  if (!fields) return;
  for (const f of fields) {
    if (f.iterVarBinding || f.bindingsExpr || f.region) continue;
    const v = node[f.key];
    if (Array.isArray(v)) {
      for (let i = 0; i < v.length; i++) if (isIRNode(v[i])) setParent(v[i], node, f.key, i);
    } else if (isIRNode(v)) {
      setParent(v, node, f.key, -1);
    }
  }
}

function walkInner(node, v, o, parent, depth, state) {
  if (state.stop || !isIRNode(node)) return;
  const ctx = { parent, depth };
  let descend = true;
  if (v.pre) {
    const r = v.pre(node, ctx);
    if (r === STOP) { state.stop = true; return; }
    if (r === false) descend = false;
  }
  if (descend) {
    for (const acc of childAccessors(node, o)) {
      const kids = acc.read();
      for (let i = 0; i < kids.length; i++) {
        walkInner(kids[i], v, o, node, depth + 1, state);
        if (state.stop) return;
      }
    }
  }
  if (v.post) v.post(node, ctx);
}

export function walk(node, visitor, opts = {}) {
  const v = typeof visitor === 'function' ? { pre: visitor } : (visitor || {});
  walkInner(node, v, normOpts(opts), null, 0, { stop: false });
}

export function collect(node, pred, opts = {}) {
  const out = [];
  walk(node, (n) => { if (pred(n)) out.push(n); }, opts);
  return out;
}

export function some(node, pred, opts = {}) {
  let found = false;
  walk(node, (n) => { if (pred(n)) { found = true; return STOP; } }, opts);
  return found;
}

export function find(node, pred, opts = {}) {
  let result = null;
  walk(node, (n) => { if (pred(n)) { result = n; return STOP; } }, opts);
  return result;
}

export function transform(node, fn, opts = {}) {
  const o = normOpts(opts);
  return transformInner(node, fn, o);
}

function transformInner(node, fn, o) {
  if (!isIRNode(node)) return node;
  let anyChanged = false;
  for (const acc of childAccessors(node, o)) {
    const kids = acc.read();
    if (kids.length === 0) continue;
    let changed = false;
    const next = new Array(kids.length);
    for (let i = 0; i < kids.length; i++) {
      const nk = transformInner(kids[i], fn, o);
      if (nk !== kids[i]) changed = true;
      next[i] = nk;
    }
    if (changed) { acc.write(next); anyChanged = true; }
  }
  if (anyChanged) relink(node);
  const replaced = fn(node);
  return replaced === undefined || replaced === null ? node : replaced;
}
