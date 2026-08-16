import { walk, some, find, irChildNodes } from '../ir/ir_visitor.js';
import type { IRNode } from '../ir/ir_visitor.js';
import type { Buffer } from '../ir/tensor/buffer.js';
import type { ForNode, TirNode } from '../ir/tensor/nodes.js';

export type ScopeBinding = { name: string; expr: IRNode };

const STORE_TYPES = new Set(['BufferStoreNode', 'LIRFlatStoreNode']);
const ACCESS_TYPES = new Set(['BufferLoadNode', 'BufferStoreNode', 'LIRFlatLoadNode', 'LIRFlatStoreNode']);

export function collectVarNames(node: IRNode | null | undefined, out: Set<string>): Set<string> {
  if (node) walk(node, (n) => { if (n.type === 'VariableNode') out.add((n as unknown as { name: string }).name); });
  return out;
}

export function varNamesOf(node: IRNode | null | undefined): Set<string> {
  return collectVarNames(node, new Set<string>());
}

export function usesAnyVar(node: IRNode | null | undefined, names: Iterable<string>): boolean {
  const wanted = names instanceof Set ? names as ReadonlySet<string> : new Set(names);
  if (!node || wanted.size === 0) return false;
  return some(node, (n) => n.type === 'VariableNode' && wanted.has((n as unknown as { name: string }).name));
}

export function usesAnyVarIn(nodes: Iterable<IRNode | null | undefined>, names: ReadonlySet<string>): boolean {
  for (const node of nodes) {
    if (usesAnyVar(node, names)) return true;
  }
  return false;
}

export function bufferAccessCount(root: IRNode): number {
  let count = 0;
  walk(root, (node) => { if (ACCESS_TYPES.has(node.type)) count++; });
  return count;
}

function accessDtype(node: IRNode, fallback: string): string {
  const n = node as unknown as { buffer?: { dtype: string }; dtype?: string };
  if (node.type === 'BufferLoadNode' || node.type === 'BufferStoreNode') {
    return (n.buffer && n.buffer.dtype) || fallback;
  }
  return n.dtype || fallback;
}

function accessesInIndexPosition(root: IRNode): Set<IRNode> {
  const marked = new Set<IRNode>();
  walk(root, (node) => {
    if (!ACCESS_TYPES.has(node.type)) return;
    const n = node as unknown as { indices?: (IRNode | null)[]; offsetExpr?: IRNode | null };
    const subtrees = n.indices || (n.offsetExpr ? [n.offsetExpr] : []);
    for (const sub of subtrees) {
      if (sub) walk(sub, (inner) => { if (ACCESS_TYPES.has(inner.type)) marked.add(inner); });
    }
  });
  return marked;
}

export function vectorValueDtype(root: IRNode, fallback: string): string | null {
  const indexPosition = accessesInIndexPosition(root);
  let dtype: string | null = null;
  let mixed = false;
  walk(root, (node) => {
    if (mixed || !ACCESS_TYPES.has(node.type) || indexPosition.has(node)) return;
    const found = accessDtype(node, fallback);
    if (dtype === null) dtype = found;
    else if (dtype !== found) mixed = true;
  });
  return mixed ? null : (dtype ?? fallback);
}

export function storedBufferNames(root: IRNode): Set<string> {
  const names = new Set<string>();
  walk(root, (node) => {
    const n = node as unknown as { buffer?: { name: string }; dstBuffer?: { name: string }; flushStore?: { buffer?: { name: string } } };
    if (STORE_TYPES.has(node.type) && n.buffer) names.add(n.buffer.name);
    if (node.type === 'VecCopyNode' && n.dstBuffer) names.add(n.dstBuffer.name);
    if (node.type === 'LIRAccumulatorNode' && n.flushStore && n.flushStore.buffer) names.add(n.flushStore.buffer.name);
  });
  return names;
}

export function allocatedBufferNames(root: IRNode, into: Set<string> = new Set()): Set<string> {
  walk(root, (node) => {
    if (node.type !== 'AllocateNode') return;
    const buffer = (node as unknown as { buffer?: Buffer }).buffer;
    if (buffer) into.add(buffer.name);
  });
  return into;
}

export function referencedBuffers(root: IRNode, into: Map<string, Buffer> = new Map()): Map<string, Buffer> {
  const add = (buffer: Buffer | null | undefined) => { if (buffer) into.set(buffer.name, buffer); };
  walk(root, (node) => {
    const n = node as unknown as {
      buffer?: Buffer; dstBuffer?: Buffer; srcBuffer?: Buffer;
      flushStore?: { buffer?: Buffer }; initLoad?: { buffer?: Buffer };
    };
    if (ACCESS_TYPES.has(node.type)) add(n.buffer);
    else if (node.type === 'VecCopyNode') { add(n.dstBuffer); add(n.srcBuffer); }
    else if (node.type === 'LIRAccumulatorNode') {
      add(n.flushStore && n.flushStore.buffer);
      add(n.initLoad && n.initLoad.buffer);
    }
  });
  return into;
}

export function collectScopeBindings(root: IRNode): ScopeBinding[] {
  const bindings: ScopeBinding[] = [];
  walk(root, (node) => {
    if (node.type === 'BlockNode') {
      const iterVars = (node as unknown as { iterVars?: readonly { iterVar?: { name: string }; binding?: IRNode }[] }).iterVars || [];
      for (const iv of iterVars) {
        if (iv.iterVar && iv.binding) bindings.push({ name: iv.iterVar.name, expr: iv.binding });
      }
    } else if (node.type === 'LIRBindingsNode') {
      const declared = (node as unknown as { bindings?: readonly ScopeBinding[] }).bindings || [];
      for (const b of declared) bindings.push(b);
    }
  });
  return bindings;
}

export function findLoopOfKind(root: IRNode, kind: string): ForNode | null {
  const node = find(root, (n) => n.type === 'ForNode' && (n as unknown as { kind: string }).kind === kind);
  return node as ForNode | null;
}

export function staticExtentOf(node: { extent?: TirNode | null } | null | undefined): number {
  const e = node && node.extent as unknown as { type?: string; value?: number } | null | undefined;
  return e && e.type === 'IntImmNode' ? (e.value as number) : 0;
}

export function loadsAreUnitStrideIn(root: IRNode, vecVars: ReadonlySet<string>): boolean {
  if (vecVars.size === 0) return true;
  const stridedMul = (expr: IRNode) => some(expr, (m) => {
    const math = m as unknown as { op?: string; a?: IRNode; b?: IRNode };
    return m.type === 'MathOpNode' && math.op === '*'
      && (usesAnyVar(math.a, vecVars) || usesAnyVar(math.b, vecVars));
  });

  return !some(root, (node) => {
    if (node.type === 'BufferLoadNode') {
      const indices = (node as unknown as { indices?: readonly IRNode[] }).indices || [];
      for (let i = 0; i < indices.length - 1; i++) {
        if (usesAnyVar(indices[i], vecVars)) return true;
      }
      return false;
    }
    if (node.type === 'LIRFlatLoadNode') {
      const offsetExpr = (node as unknown as { offsetExpr?: IRNode | null }).offsetExpr;
      return !!offsetExpr && stridedMul(offsetExpr);
    }
    return false;
  });
}

function laneAffineExpr(node: IRNode | null | undefined, laneVars: ReadonlySet<string>): boolean {
  if (!node || !usesAnyVar(node, laneVars)) return true;
  if (node.type === 'VariableNode') return true;
  if (node.type === 'CastNode') return irChildNodes(node).every((k) => laneAffineExpr(k, laneVars));
  if (node.type === 'MathOpNode') {
    const op = (node as unknown as { op?: string }).op;
    if (op !== '+' && op !== '-') return false;
    return irChildNodes(node).every((k) => laneAffineExpr(k, laneVars));
  }
  return false;
}

export function indicesAreLaneAffine(root: IRNode, laneVars: ReadonlySet<string>): boolean {
  if (laneVars.size === 0) return true;
  return !some(root, (node) => {
    if (!ACCESS_TYPES.has(node.type)) return false;
    const n = node as unknown as { indices?: (IRNode | null)[]; offsetExpr?: IRNode | null };
    const parts = n.indices || (n.offsetExpr !== undefined ? [n.offsetExpr] : []);
    for (const part of parts) if (!laneAffineExpr(part, laneVars)) return true;
    return false;
  });
}

const KEY_FIELDS = ['name', 'value', 'op', 'externName', 'dtype', 'toDtype', 'fromDtype'] as const;

export function exprKey(node: IRNode | null | undefined): string {
  if (!node) return '_';
  const n = node as unknown as Record<string, unknown>;
  let key = node.type;
  for (const field of KEY_FIELDS) {
    const v = n[field];
    if (typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean') key += `|${field}=${v}`;
  }
  const buffer = n.buffer as { name?: string } | undefined;
  if (buffer && buffer.name) key += `|buf=${buffer.name}`;
  const kids = irChildNodes(node);
  key += '(';
  for (let i = 0; i < kids.length; i++) key += (i ? ',' : '') + exprKey(kids[i]);
  return key + ')';
}

function indexKey(node: IRNode): string {
  const n = node as unknown as { indices?: (IRNode | null)[]; offsetExpr?: IRNode | null };
  const parts = n.indices || (n.offsetExpr !== undefined ? [n.offsetExpr] : []);
  return parts.map((p) => exprKey(p)).join(',');
}

export function guardedLoadsAreInRange(root: IRNode, laneVars: ReadonlySet<string>): boolean {
  if (laneVars.size === 0) return true;

  const storeKeys = new Set<string>();
  walk(root, (node) => { if (STORE_TYPES.has(node.type)) storeKeys.add(indexKey(node)); });

  let safe = true;
  walk(root, (node) => {
    if (!safe || node.type !== 'IfThenElseNode') return;
    const select = node as unknown as { condition?: IRNode; thenBody?: IRNode; elseBody?: IRNode };
    for (const branch of [select.thenBody, select.elseBody]) {
      if (!branch) continue;
      walk(branch, (inner) => {
        if (inner.type !== 'BufferLoadNode' && inner.type !== 'LIRFlatLoadNode') return;
        if (!storeKeys.has(indexKey(inner))) safe = false;
      });
    }
  });

  return safe;
}

function accessLocationKey(node: IRNode): string {
  const n = node as unknown as { buffer?: { name?: string }; indices?: (IRNode | null)[]; offsetExpr?: IRNode | null };
  const buffer = (n.buffer && n.buffer.name) || '?';
  const parts = n.indices || (n.offsetExpr !== undefined ? [n.offsetExpr] : []);
  return buffer + '@' + parts.map((p) => exprKey(p)).join(',');
}

export function loopCarriedDependenceIn(root: IRNode): boolean {
  const written = new Set<string>();
  const writtenLocations = new Set<string>();
  walk(root, (node) => {
    if (!STORE_TYPES.has(node.type)) return;
    const buffer = (node as unknown as { buffer?: { name?: string } }).buffer;
    if (buffer && buffer.name) written.add(buffer.name);
    writtenLocations.add(accessLocationKey(node));
  });
  if (written.size === 0) return false;

  return some(root, (node) => {
    if (node.type !== 'BufferLoadNode' && node.type !== 'LIRFlatLoadNode') return false;
    const buffer = (node as unknown as { buffer?: { name?: string } }).buffer;
    if (!buffer || !buffer.name || !written.has(buffer.name)) return false;
    return !writtenLocations.has(accessLocationKey(node));
  });
}

export function hasBufferAccessMatching(root: IRNode, pred: (dtype: string, node: IRNode) => boolean): boolean {
  return some(root, (node) => {
    const n = node as unknown as { buffer?: { dtype: string }; dtype?: string };
    if ((node.type === 'BufferLoadNode' || node.type === 'BufferStoreNode') && n.buffer) return pred(n.buffer.dtype, node);
    if ((node.type === 'LIRFlatLoadNode' || node.type === 'LIRFlatStoreNode') && n.dtype) return pred(n.dtype, node);
    return false;
  });
}
