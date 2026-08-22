import {
  ForNode, BlockNode, SeqNode, BufferStoreNode, BufferLoadNode, BlockRealizeNode,
  VariableNode, IntImmNode, FloatImmNode, MathOpNode, ForKind, IterVarKind, IfThenElseNode, AllocateNode, LetStmtNode
} from '../ir/tensor/nodes.js';
import { Buffer } from '../ir/tensor/buffer.js';
import type { BufferRegionLike } from '../ir/tensor/buffer.js';
import { ScheduleTrace } from './trace.js';
import { ScheduleValidator } from './validator.js';
import { ScheduleState } from './schedule_state.js';
import { ScheduleMutator } from './mutator.js';
import { loopCarriedDependence, reorderLegality, collectVarsUsed, IterVarPolicy } from './legality.js';
import { cloneIRShared } from '../ir/clone_ir.js';
import { transform as irTransform, walk as irWalk } from '../ir/ir_visitor.js';
import { FuncAttr } from '../ir/func_attrs.js';
import { isDtypeInt, reduceInitValue } from '../../util/dtype_map.js';
import { AccessKind, loadedBuffers, isStaticLevel } from '../analysis/buffer_access.js';
import { LinearForm, mixedRadixDecomposition, linearFormToNode, toLinearForm } from '../analysis/iter_map.js';
import type { IRNode } from '../ir/ir_visitor.js';
import type { TirNode, PrimFunc } from '../ir/tensor/nodes.js';
import type { RadixDecomposition, RadixFactor, VarRange } from '../analysis/iter_map.js';
import type { BufferAccess } from '../analysis/buffer_access.js';

import type { SRef } from './sref.js';
import type { CloneableIRNode, CloneRecurse } from '../ir/clone_ir.js';

export type LoopRef = ForNode | string;
export type LoadCounter = { n: number };
export type VarForms = Map<string, LinearForm>;
export type LoopVarExprMap = Map<string, () => TirNode>;
export type NodeSlots = Record<string, TirNode | TirNode[] | undefined>;
export type ReorderChain = { links: TirNode[]; innermostBody: TirNode | null };
export type DeferredChainEntry = { node: TirNode; needs: Set<string> };
export type InlineStorePlan = {
  store: BufferStoreNode;
  decompositions: RadixDecomposition[];
  varForms: VarForms;
};
export type InlinePlan = {
  sref: SRef;
  prod: BlockNode;
  plans: InlineStorePlan[];
};
export type IterBindingInfo = { name: string; form: LinearForm | null; kind: string | undefined };
export type BlockBindingInfo = { bindings: readonly IterBindingInfo[] };

const RFACTOR_REDUCE_TYPE: Record<string, string> = { '+': 'sum', '*': 'prod', 'min': 'min', 'max': 'max' };

function rfactorIdentity(op: string, dtype: string): TirNode {
  const value = reduceInitValue(RFACTOR_REDUCE_TYPE[op], dtype);
  return isDtypeInt(dtype) ? new IntImmNode(value) : new FloatImmNode(value);
}

function sameIndices(a: readonly TirNode[], b: readonly TirNode[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    const fa = toLinearForm(a[i]);
    const fb = toLinearForm(b[i]);
    if (!fa || !fb || fa.offset !== fb.offset || fa.terms.size !== fb.terms.size) return false;
    for (const [name, coeff] of fa.terms) {
      if (fb.terms.get(name) !== coeff) return false;
    }
  }
  return true;
}

function readsBuffer(node: TirNode | null | undefined, buffer: Buffer): boolean {
  let found = false;
  irWalk(node as IRNode, (n: IRNode) => {
    if ((n as TirNode).type === 'BufferLoadNode' && (n as unknown as BufferLoadNode).buffer === buffer) found = true;
  });
  return found;
}

function substituteVar(node: TirNode, oldName: string, exprFactory: () => TirNode): TirNode {
  return irTransform(node, (n: IRNode) => {
    if (n.type === 'VariableNode' && (n as VariableNode).name === oldName) return exprFactory();
    return n;
  }, { bindVars: false }) as TirNode;
}

function replaceBufferLoads(node: TirNode, bufName: string, makeReplacement: (load: BufferLoadNode) => TirNode, counter: LoadCounter): TirNode {
  return irTransform(node, (n: IRNode) => {
    const ld = n as BufferLoadNode;
    if (n.type === 'BufferLoadNode' && ld.buffer && ld.buffer.name === bufName) {
      counter.n++;
      return makeReplacement(ld) as IRNode;
    }
    return n;
  }, { bindVars: false }) as TirNode;
}

function cloneExprTree(node: unknown): TirNode {
  return cloneIRShared(node, cloneExprTree as CloneRecurse, (n: CloneableIRNode, copy: CloneableIRNode, rec: CloneRecurse) => {
    if (n.type === 'BlockRealizeNode') {
      copy.iterVar = n.iterVar;
      copy.binding = rec(n.binding);
      copy.kind = n.kind;
      return copy;
    }
    for (const key of Object.keys(n)) {
      if (key === '_parent' || key === '_parentKey' || key === '_parentIdx') continue;
      copy[key] = n[key];
    }
    return copy;
  }) as TirNode;
}

function collectStores(node: TirNode | null | undefined, out: BufferStoreNode[]): void {
  if (!node) return;
  if (node.type === 'BufferStoreNode') { out.push(node as BufferStoreNode); return; }
  if (node.type === 'SeqNode') { for (const st of (node as SeqNode).stmts) collectStores(st, out); return; }
  if (node.type === 'ForNode' || node.type === 'BlockNode') collectStores((node as ForNode | BlockNode).body, out);
}

function usedVarNames(expr: TirNode): Set<string> {
  const names = new Set<string>();
  irWalk(expr, (n: IRNode) => { if (n.type === 'VariableNode') names.add((n as VariableNode).name); });
  return names;
}

function recoverIterVar(indexExpr: TirNode, decomposition: RadixDecomposition, factor: RadixFactor): TirNode {
  let expr = decomposition.offset === 0
    ? indexExpr
    : new MathOpNode('-', indexExpr, new IntImmNode(decomposition.offset));
  if (factor.coeff !== 1) expr = new MathOpNode('//', expr, new IntImmNode(factor.coeff));
  if (factor.coeff * factor.extent !== decomposition.extent) {
    expr = new MathOpNode('%', expr, new IntImmNode(factor.extent));
  }
  return factor.min === 0 ? expr : new MathOpNode('+', expr, new IntImmNode(factor.min));
}

function invertWriteIndices(access: BufferAccess): RadixDecomposition[] | null {
  const varRanges = new Map<string, VarRange>();
  for (const level of access.iterSpace) {
    if (isStaticLevel(level)) varRanges.set(level.name, [level.min, level.extent]);
  }
  const decompositions: RadixDecomposition[] = [];
  for (const form of access.forms) {
    const decomposition = mixedRadixDecomposition(form, varRanges);
    if (!decomposition) return null;
    decompositions.push(decomposition);
  }
  return decompositions;
}

function producerVarForms(access: BufferAccess, blockInfo: BlockBindingInfo | null): VarForms | null {
  const forms = new Map<string, LinearForm>();
  for (const level of access.iterSpace) {
    if (isStaticLevel(level)) forms.set(level.name, LinearForm.variable(level.name));
  }
  if (blockInfo) {
    for (const binding of blockInfo.bindings) {
      if (binding.form) forms.set(binding.name, binding.form);
    }
  }
  return forms;
}

function substituteAffineVars(expr: TirNode, varForms: VarForms, loopVarExpr: LoopVarExprMap): TirNode {
  return irTransform(expr, (n: IRNode) => {
    if (n.type !== 'VariableNode') return n;
    const form = varForms.get((n as VariableNode).name);
    if (!form) return n;
    return linearFormToNode<TirNode>(
      form,
      (name) => (loopVarExpr.get(name) as () => TirNode)(),
      (value) => new IntImmNode(value),
      (op, a, b) => new MathOpNode(op, a, b)
    ) as IRNode;
  }, { bindVars: false }) as TirNode;
}

function inlineStoreValue(funcBody: TirNode, store: BufferStoreNode, decompositions: readonly RadixDecomposition[], varForms: VarForms): number {
  const counter = { n: 0 };
  replaceBufferLoads(funcBody, store.buffer.name, (load: BufferLoadNode) => {
    const loopVarExpr: LoopVarExprMap = new Map();
    for (let d = 0; d < decompositions.length; d++) {
      const decomposition = decompositions[d];
      for (const factor of decomposition.factors) {
        loopVarExpr.set(factor.name, () => recoverIterVar(cloneExprTree(load.indices[d]), decomposition, factor));
      }
    }
    return substituteAffineVars(cloneExprTree(store.value), varForms, loopVarExpr);
  }, counter);
  return counter.n;
}

function retargetBufferReads(funcBody: TirNode, bufName: string, inherited: readonly BufferRegionLike[]): void {
  const stack: (TirNode | null | undefined)[] = [funcBody];
  while (stack.length > 0) {
    const node = stack.pop();
    if (!node || typeof node !== 'object') continue;
    const blk = node as BlockNode;
    if (node.type === 'BlockNode' && blk.reads) {
      const kept = blk.reads.filter((r) => !(r.buffer && r.buffer.name === bufName));
      if (kept.length !== blk.reads.length) {
        const present = new Set(kept.map((r) => r.buffer && r.buffer.name));
        for (const r of inherited) {
          if (!r.buffer || present.has(r.buffer.name)) continue;
          present.add(r.buffer.name);
          kept.push({ ...r });
        }
        blk.reads = kept;
      }
    }
    const slots = node as unknown as NodeSlots;
    if (slots.body) stack.push(slots.body as TirNode);
    if (slots.stmts) for (const st of slots.stmts as TirNode[]) stack.push(st);
    if (slots.thenBody) stack.push(slots.thenBody as TirNode);
    if (slots.elseBody) stack.push(slots.elseBody as TirNode);
    if (slots.initBody) stack.push(slots.initBody as TirNode);
  }
}

function getConstExtent(node: TirNode | null | undefined): number | null {
  if (node && node.type === 'IntImmNode') return (node as IntImmNode).value;
  return null;
}

let _varId = 0;
function freshVar(hint: string, dtype = 'int32'): VariableNode {
  return new VariableNode(`${hint}_${_varId++}`, dtype);
}

export function resetVarCounter(): void {
  _varId = 0;
}

export class Schedule {
  func: PrimFunc;
  trace: ScheduleTrace;
  state: ScheduleState;
  mutator: ScheduleMutator;
  _replaying: boolean;

  constructor(primFunc: PrimFunc) {
    this.func = primFunc;
    this.trace = new ScheduleTrace();
    this.state = new ScheduleState(primFunc);
    this._replaying = false;
    this.mutator = new ScheduleMutator(primFunc);
  }

  _replaceInTree(oldNode: TirNode, newNode: TirNode): void {
    this.state.replaceNode(oldNode, newNode);
  }

  _removeFromTree(node: TirNode): void {
    this.state.removeNode(node);
  }

  getBlockSRef(name: string): SRef {
    const sref = this.state.tree.getBlockSRef(name);
    if (!sref) throw new Error(`Block '${name}' not found`);
    return sref;
  }

  getBlock(name: string): BlockNode {
    return this.getBlockSRef(name).node as BlockNode;
  }

  getLoops(blockName: string): ForNode[] {
    return this.state.tree.loopsOf(blockName).map(sref => sref.node as ForNode);
  }

  _resolveLoop(ref: LoopRef): ForNode {
    if (typeof ref !== 'string') return ref;
    let found: ForNode | null = null;
    const walk = (node: TirNode | null | undefined): void => {
      if (!node || typeof node !== 'object' || found) return;
      const f = node as ForNode;
      if (node.type === 'ForNode' && f.loopVar && f.loopVar.name === ref) { found = f; return; }
      const slots = node as unknown as NodeSlots;
      if (slots.body) walk(slots.body as TirNode);
      if (slots.initBody) walk(slots.initBody as TirNode);
      if (slots.stmts) for (const st of slots.stmts as TirNode[]) walk(st);
      if (slots.thenBody) walk(slots.thenBody as TirNode);
      if (slots.elseBody) walk(slots.elseBody as TirNode);
    };
    walk(this.func.body);
    return (found ?? ref) as unknown as ForNode;
  }

  split(loop: LoopRef, factor: number): ForNode[] {
    loop = this._resolveLoop(loop);
    const extent = getConstExtent(loop.extent);
    if (extent === null) {
      throw new Error(`Cannot split loop '${loop.loopVar.name}' with non-constant extent`);
    }
    if (factor <= 0 || !Number.isInteger(factor)) {
      throw new Error(`Split factor must be a positive integer, got ${factor}`);
    }
    if (loop.threadTag) {
      throw new Error(`Cannot split loop '${loop.loopVar.name}': it is bound to '${loop.threadTag}', and a split has no single thread-axis meaning; split before binding`);
    }

    const outerExtent = Math.ceil(extent / factor);
    const minValue = getConstExtent(loop.min);
    const shift = (node: TirNode): TirNode =>
      minValue === 0 ? node : new MathOpNode('+', cloneExprTree(loop.min), node);
    const outerVar = freshVar(`${loop.loopVar.name}_o`);
    const innerVar = freshVar(`${loop.loopVar.name}_i`);
    const oldVarName = loop.loopVar.name;

    const clonedBody = cloneExprTree(loop.body);
    const innerLoop = new ForNode(
      innerVar,
      new IntImmNode(0),
      new IntImmNode(factor),
      loop.kind,
      clonedBody,
      null
    );

    const needsGuard = extent % factor !== 0;
    if (needsGuard) {
      const flatIdx = new MathOpNode('+',
        new MathOpNode('*', outerVar, new IntImmNode(factor)),
        innerVar
      );
      const guard = new MathOpNode('<', flatIdx, new IntImmNode(extent));
      const guarded = new IfThenElseNode(guard, innerLoop.body);
      innerLoop.body = guarded;
      innerLoop._setChild('body', guarded);
    }

    const outerLoop = new ForNode(
      outerVar,
      new IntImmNode(0),
      new IntImmNode(outerExtent),
      loop.kind,
      innerLoop,
      null
    );

    substituteVar(innerLoop.body, oldVarName, () =>
      shift(new MathOpNode('+',
        new MathOpNode('*', outerVar, new IntImmNode(factor)),
        innerVar
      ))
    );

    this.mutator.replaceNode(loop, outerLoop);
    this._replaceInTree(loop, outerLoop);

    if (!this._replaying) {
      this.trace.record('split', [loop.loopVar.name, factor]);
    }

    return [outerLoop, innerLoop];
  }

  reorder(...args: (LoopRef | readonly LoopRef[])[]): void {
    const raw: readonly LoopRef[] = (args.length === 1 && Array.isArray(args[0])) ? args[0] as readonly LoopRef[] : args as LoopRef[];
    const newOrder: ForNode[] = raw.map((l) => this._resolveLoop(l));
    if (newOrder.length < 2) return;

    for (const loop of newOrder) {
      if (loop.type !== 'ForNode') {
        throw new Error('reorder expects ForNode arguments');
      }
    }

    const loopSet = new Set<ForNode>(newOrder);
    if (loopSet.size !== newOrder.length) {
      throw new Error('reorder: duplicate loop in requested order');
    }

    const topmostLoop = this._topmostOf(loopSet) as ForNode;
    const { links, innermostBody } = this._collectReorderChain(loopSet, topmostLoop);

    const after: TirNode[] = links.slice();
    let slot = 0;
    for (let i = 0; i < links.length; i++) {
      if (loopSet.has(links[i] as ForNode)) after[i] = newOrder[slot++];
    }

    const isLoop = (n: TirNode): n is ForNode => n.type === 'ForNode';
    const reason = reorderLegality(this.state, links.filter(isLoop), after.filter(isLoop));
    if (reason) throw new Error(`reorder: ${reason}`);

    const chain = this._arrangeChain(after);
    this.mutator.replaceNode(topmostLoop, chain[0]);
    for (const node of chain) {
      if (node === chain[0]) continue;
      node._parent = null;
      node._parentKey = null;
      node._parentIdx = -1;
    }
    for (let i = 0; i < chain.length; i++) {
      this._setChainChild(chain[i], (i + 1 < chain.length ? chain[i + 1] : innermostBody) as TirNode);
    }

    this._replaceInTree(topmostLoop, chain[0]);

    if (!this._replaying) {
      this.trace.record('reorder', [newOrder.map(l => l.loopVar.name)]);
    }
  }

  _topmostOf(loopSet: ReadonlySet<ForNode>): ForNode | null {
    let topmost: ForNode | null = null;
    let found = 0;
    const stack: { node: TirNode | null | undefined; depth: number }[] = [{ node: this.func.body, depth: 0 }];
    let topmostDepth = Infinity;
    while (stack.length > 0) {
      const { node, depth } = stack.pop() as { node: TirNode | null | undefined; depth: number };
      if (!node) continue;
      if (node.type === 'ForNode') {
        if (loopSet.has(node as ForNode)) {
          found++;
          if (depth < topmostDepth) { topmostDepth = depth; topmost = node; }
        }
        stack.push({ node: node.body, depth: depth + 1 });
        continue;
      }
      if (node.type === 'SeqNode') {
        for (const s of node.stmts) stack.push({ node: s, depth });
      } else if (node.type === 'IfThenElseNode') {
        stack.push({ node: node.thenBody, depth });
        if (node.elseBody) stack.push({ node: node.elseBody, depth });
      } else if (node.type === 'BlockNode' || node.type === 'AllocateNode' || node.type === 'LetStmtNode') {
        const slots = node as unknown as NodeSlots;
        stack.push({ node: slots.body as TirNode, depth });
        if (slots.initBody) stack.push({ node: slots.initBody as TirNode, depth });
      }
    }
    if (found !== loopSet.size) {
      throw new Error('reorder: not all requested loops were found in the function nest');
    }
    return topmost;
  }

  _arrangeChain(after: readonly TirNode[]): TirNode[] {
    const loopNames = new Set<string>(after.filter((n): n is ForNode => n.type === 'ForNode').map((n) => n.loopVar.name));
    const guardOf = (node: TirNode): TirNode | null => {
      if (node.type === 'IfThenElseNode') return (node as IfThenElseNode).condition;
      if (node.type === 'LetStmtNode') return (node as LetStmtNode).value;
      return null;
    };
    const dependsOn = (node: TirNode): Set<string> => {
      const guard = guardOf(node);
      if (!guard) return new Set<string>();
      const needs = new Set<string>();
      for (const name of usedVarNames(guard)) if (loopNames.has(name)) needs.add(name);
      return needs;
    };

    const chain: TirNode[] = [];
    const bound = new Set<string>();
    const deferred: DeferredChainEntry[] = [];
    const release = () => {
      for (let i = 0; i < deferred.length;) {
        if ([...deferred[i].needs].every((n) => bound.has(n))) {
          chain.push(deferred[i].node);
          deferred.splice(i, 1);
        } else i++;
      }
    };
    for (const node of after) {
      if (node.type === 'ForNode') {
        chain.push(node);
        bound.add((node as ForNode).loopVar.name);
        release();
        continue;
      }
      const needs = dependsOn(node);
      if ([...needs].every((n) => bound.has(n))) chain.push(node);
      else deferred.push({ node, needs });
    }
    for (const entry of deferred) chain.push(entry.node);
    return chain;
  }

  _setChainChild(node: TirNode, child: TirNode): void {
    const key = node.type === 'IfThenElseNode' ? 'thenBody' : 'body';
    (node as unknown as NodeSlots)[key] = child;
    node._setChild(key, child);
  }

  _collectReorderChain(loopSet: ReadonlySet<ForNode>, topmostLoop: ForNode): ReorderChain {
    const remaining = new Set<ForNode>(loopSet);
    const links: TirNode[] = [];
    let node: TirNode | null | undefined = topmostLoop;
    let innermostBody: TirNode | null = null;
    while (node) {
      if (node.type === 'ForNode') {
        const f = node as ForNode;
        links.push(f);
        remaining.delete(f);
        if (remaining.size === 0) { innermostBody = f.body; break; }
        node = f.body;
      } else if (node.type === 'IfThenElseNode') {
        const ite = node as IfThenElseNode;
        if (ite.elseBody) {
          throw new Error('reorder: a two-way conditional separates the reordered loops, so they do not form a single chain');
        }
        links.push(ite);
        node = ite.thenBody;
      } else if (node.type === 'AllocateNode' || node.type === 'LetStmtNode') {
        links.push(node);
        node = (node as AllocateNode | LetStmtNode).body;
      } else if (node.type === 'BlockNode') {
        throw new Error(`reorder: block '${(node as BlockNode).name}' separates the reordered loops, which therefore belong to different block scopes`);
      } else if (node.type === 'SeqNode') {
        const seq = node as SeqNode;
        if (seq.stmts.length !== 1) {
          throw new Error('reorder: multiple statements separate the reordered loops, so they do not form a single chain');
        }
        node = seq.stmts[0];
      } else {
        break;
      }
    }
    if (remaining.size > 0) {
      throw new Error('reorder: loops do not form a single chain');
    }
    return { links, innermostBody };
  }

  fuseLoops(outer: LoopRef, inner: LoopRef): ForNode {
    outer = this._resolveLoop(outer);
    inner = this._resolveLoop(inner);
    if (outer.type !== 'ForNode' || inner.type !== 'ForNode') {
      throw new Error('fuseLoops expects two ForNode arguments');
    }
    if (outer.body !== inner) {
      throw new Error('fuseLoops requires inner loop to be direct child of outer loop');
    }

    const outerExtent = getConstExtent(outer.extent);
    const innerExtent = getConstExtent(inner.extent);
    if (outerExtent === null || innerExtent === null) {
      throw new Error('Cannot fuse loops with non-constant extents');
    }

    const fusedExtent = outerExtent * innerExtent;
    const fusedVar = freshVar(`${outer.loopVar.name}_${inner.loopVar.name}_fused`);
    const outerName = outer.loopVar.name;
    const innerName = inner.loopVar.name;

    const fusedLoop = new ForNode(
      fusedVar,
      new IntImmNode(0),
      new IntImmNode(fusedExtent),
      outer.kind,
      inner.body
    );

    substituteVar(fusedLoop.body, outerName, () =>
      new MathOpNode('//', fusedVar, new IntImmNode(innerExtent))
    );
    substituteVar(fusedLoop.body, innerName, () =>
      new MathOpNode('%', fusedVar, new IntImmNode(innerExtent))
    );

    this.mutator.replaceNode(outer, fusedLoop);
    this._replaceInTree(outer, fusedLoop);

    if (!this._replaying) {
      this.trace.record('fuseLoops', [outerName, innerName]);
    }

    return fusedLoop;
  }

  tile(blockName: string, loopIndices: readonly number[], tileSizes: readonly number[]): { outerLoops: ForNode[]; innerLoops: ForNode[] } {
    if (loopIndices.length !== tileSizes.length) {
      throw new Error('tile: loopIndices and tileSizes must have same length');
    }

    const loops = this.getLoops(blockName);
    const targetLoops = loopIndices.map(idx => {
      if (idx >= loops.length) throw new Error(`tile: loop index ${idx} out of range`);
      return loops[idx];
    });

    const outerLoops = [];
    const innerLoops = [];

    for (let i = 0; i < targetLoops.length; i++) {
      const currentLoops = this.getLoops(blockName);
      const loop = currentLoops.find(l =>
        l.loopVar.name === targetLoops[i].loopVar.name ||
        l === targetLoops[i]
      );
      if (!loop) throw new Error(`tile: lost track of loop at index ${i}`);
      const [outer, inner] = this.split(loop, tileSizes[i]);
      outerLoops.push(outer);
      innerLoops.push(inner);
    }

    const allLoops = this.getLoops(blockName);
    const outers = [];
    const inners = [];
    for (const l of allLoops) {
      if (outerLoops.some(o => o.loopVar.name === l.loopVar.name)) outers.push(l);
      else if (innerLoops.some(o => o.loopVar.name === l.loopVar.name)) inners.push(l);
    }

    if (outers.length > 0 && inners.length > 0) {
      this.reorder(...outers, ...inners);
    }

    return { outerLoops: outers, innerLoops: inners };
  }

  vectorize(loop: LoopRef): void {
    loop = this._resolveLoop(loop);
    if (loop.type !== 'ForNode') throw new Error('vectorize expects ForNode');
    const extent = getConstExtent(loop.extent);
    if (extent === null) throw new Error('Cannot vectorize loop with non-constant extent');
    const carried = loopCarriedDependence(this.state, loop, IterVarPolicy.ACCUMULABLE);
    if (carried !== null) throw new Error(`Cannot vectorize: ${carried}`);
    loop.kind = ForKind.VECTORIZED;
    this.state.invalidate();
    if (!this._replaying) {
      this.trace.record('vectorize', [loop.loopVar.name]);
    }
  }

  unroll(loop: LoopRef): void {
    loop = this._resolveLoop(loop);
    if (loop.type !== 'ForNode') throw new Error('unroll expects ForNode');
    loop.kind = ForKind.UNROLLED;
    this.state.invalidate();
    if (!this._replaying) {
      this.trace.record('unroll', [loop.loopVar.name]);
    }
  }

  parallelize(loop: LoopRef): void {
    loop = this._resolveLoop(loop);
    if (loop.type !== 'ForNode') throw new Error('parallelize expects ForNode');
    const carried = loopCarriedDependence(this.state, loop, IterVarPolicy.SPATIAL);
    if (carried !== null) throw new Error(`Cannot parallelize: ${carried}`);
    loop.kind = ForKind.PARALLEL;
    this.state.invalidate();
    if (!this._replaying) {
      this.trace.record('parallelize', [loop.loopVar.name]);
    }
  }

  bindThread(loop: LoopRef, threadTag: string): void {
    loop = this._resolveLoop(loop);
    if (loop.type !== 'ForNode') throw new Error('bindThread expects ForNode');
    const validTags = [
      'blockIdx.x', 'blockIdx.y', 'blockIdx.z',
      'threadIdx.x', 'threadIdx.y', 'threadIdx.z'
    ];
    if (!validTags.includes(threadTag)) {
      throw new Error(`Invalid thread tag: ${threadTag}. Must be one of: ${validTags.join(', ')}`);
    }
    loop.kind = ForKind.THREAD_BINDING;
    loop.threadTag = threadTag;
    this.state.invalidate();
    if (!this._replaying) {
      this.trace.record('bindThread', [loop.loopVar.name, threadTag]);
    }
  }

  rfactor(blockName: string, reductionVarName: string, factor: number): Buffer {
    const block = this.getBlock(blockName);
    const loops = this.getLoops(blockName);
    const kLoop = loops.find(l => l.loopVar.name === reductionVarName);
    if (!kLoop) throw new Error(`rfactor: reduction loop '${reductionVarName}' not found for block '${blockName}'`);
    const K = getConstExtent(kLoop.extent);
    if (K === null) throw new Error(`rfactor: reduction loop '${reductionVarName}' has non-constant extent`);
    if (!Number.isInteger(factor) || factor <= 1 || factor >= K || K % factor !== 0) {
      throw new Error(`rfactor: factor ${factor} must divide reduction extent ${K} with 1 < factor < ${K}`);
    }

    const store = block.body;
    if (!store || store.type !== 'BufferStoreNode' || !store.value || store.value.type !== 'MathOpNode') {
      throw new Error(`rfactor: block '${blockName}' body is not a single accumulating store`);
    }
    const acc = store.buffer;
    const spatialIdx = store.indices;
    const storeMath = store.value as MathOpNode;
    const op = storeMath.op;
    const isAccLoad = (node: TirNode | null | undefined): boolean =>
      !!node && node.type === 'BufferLoadNode'
      && (node as BufferLoadNode).buffer === acc
      && sameIndices((node as BufferLoadNode).indices, spatialIdx);
    let update: TirNode | undefined;
    if (isAccLoad(storeMath.a)) update = storeMath.b as TirNode;
    else if (isAccLoad(storeMath.b)) update = storeMath.a;
    else throw new Error(`rfactor: block '${blockName}' body is not an accumulation into '${acc.name}' at the stored subscript`);
    if (readsBuffer(update, acc)) {
      throw new Error(`rfactor: update expression in block '${blockName}' reads accumulator '${acc.name}'; cannot factor reduction`);
    }
    if (!RFACTOR_REDUCE_TYPE[op]) {
      throw new Error(`rfactor: op '${op}' is not associative+commutative; cannot factor reduction`);
    }
    const initVal = (block.initBody && block.initBody.type === 'BufferStoreNode' && block.initBody.value)
      ? block.initBody.value : rfactorIdentity(op, acc.dtype);

    const spatialLoops = loops.filter(l => l.loopVar.name !== reductionVarName);
    const spatialLoopVars = new Set(spatialLoops.map(l => l.loopVar.name));
    const KO = K / factor;
    const partialBuf = new Buffer(`${acc.name}_rf`, [factor, ...acc.shape], acc.dtype, acc.scope);

    const kiVar = freshVar(`${reductionVarName}_rfi`);
    const koVar = freshVar(`${reductionVarName}_rfo`);
    const piVar = freshVar(`${reductionVarName}_rfp`);
    const kiIter = new BlockRealizeNode(freshVar(`${reductionVarName}_rfvi`), kiVar, IterVarKind.DATA_PAR);
    const piIter = new BlockRealizeNode(freshVar(`${reductionVarName}_rfvp`), piVar, IterVarKind.COMM_REDUCE);

    const splitK = () => new MathOpNode('+', new MathOpNode('*', koVar, new IntImmNode(factor)), kiVar);
    const cfIdx = (iterVar: TirNode): TirNode[] => [iterVar, ...spatialIdx.map(cloneExprTree)];
    const partialUpdate = substituteVar(cloneExprTree(update as TirNode), reductionVarName, splitK);
    const partialIterVars = block.iterVars.map((iv) => {
      const copy = cloneExprTree(iv) as BlockRealizeNode;
      copy.binding = substituteVar(copy.binding, reductionVarName, splitK);
      return copy;
    });
    partialIterVars.push(kiIter);

    const partialStore = new BufferStoreNode(partialBuf, cfIdx(kiIter.iterVar),
      new MathOpNode(op, new BufferLoadNode(partialBuf, cfIdx(kiIter.iterVar)), partialUpdate));
    const partialInit = new BufferStoreNode(partialBuf, cfIdx(kiIter.iterVar), cloneExprTree(initVal));
    const partialBlock = new BlockNode(`${blockName}_rf_p`, partialIterVars,
      block.reads.map(r => ({ buffer: r.buffer })), [{ buffer: partialBuf }], partialStore, partialInit);

    let partialNest = new ForNode(koVar, new IntImmNode(0), new IntImmNode(KO), ForKind.SERIAL, partialBlock);
    partialNest = new ForNode(kiVar, new IntImmNode(0), new IntImmNode(factor), ForKind.SERIAL, partialNest);
    for (let i = spatialLoops.length - 1; i >= 0; i--) {
      partialNest = new ForNode(spatialLoops[i].loopVar, new IntImmNode(0),
        cloneExprTree(spatialLoops[i].extent), ForKind.SERIAL, partialNest);
    }

    const combineStore = new BufferStoreNode(acc, spatialIdx.map(cloneExprTree),
      new MathOpNode(op, new BufferLoadNode(acc, spatialIdx.map(cloneExprTree)),
        new BufferLoadNode(partialBuf, cfIdx(piIter.iterVar))));
    const combineInit = new BufferStoreNode(acc, spatialIdx.map(cloneExprTree), cloneExprTree(initVal));
    const combineBlock = new BlockNode(`${blockName}_rf_c`,
      [...this._iterVarsOver(block, spatialLoopVars), piIter],
      [{ buffer: partialBuf }], [{ buffer: acc }], combineStore, combineInit);

    let combineNest = new ForNode(piVar, new IntImmNode(0), new IntImmNode(factor), ForKind.SERIAL, combineBlock);
    for (let i = spatialLoops.length - 1; i >= 0; i--) {
      combineNest = new ForNode(spatialLoops[i].loopVar, new IntImmNode(0),
        cloneExprTree(spatialLoops[i].extent), ForKind.SERIAL, combineNest);
    }

    const rfReplacement = new SeqNode([partialNest, combineNest]);
    this.mutator.replaceNode(loops[0], rfReplacement);
    this._replaceInTree(loops[0], rfReplacement);
    if (!this._replaying) {
      this.trace.record('rfactor', [blockName, reductionVarName, factor]);
    }
    return partialBuf;
  }

  decomposeReduction(blockName: string): void {
    const block = this.getBlock(blockName);
    if (!block.initBody) throw new Error(`decomposeReduction: block '${blockName}' has no initBody`);
    const loops = this.getLoops(blockName);
    const store = block.body;
    if (!store || store.type !== 'BufferStoreNode') throw new Error(`decomposeReduction: block '${blockName}' body is not a store`);
    const acc = store.buffer;

    const spatialLoopVars = this._writeIndexLoopVars('decomposeReduction', block, store);
    const spatialLoops = loops.filter(l => spatialLoopVars.has(l.loopVar.name));
    const reductionLoops = loops.filter(l => !spatialLoopVars.has(l.loopVar.name));
    if (reductionLoops.length === 0) throw new Error(`decomposeReduction: block '${blockName}' has no reduction loop`);

    const initStore = new BufferStoreNode(acc, store.indices.map(cloneExprTree), cloneExprTree((block.initBody as BufferStoreNode).value));
    const initBlock = new BlockNode(`${blockName}_init`,
      this._iterVarsOver(block, spatialLoopVars), [], [{ buffer: acc }], initStore);
    let initNest: TirNode = initBlock;
    for (let i = spatialLoops.length - 1; i >= 0; i--) {
      initNest = new ForNode(spatialLoops[i].loopVar, new IntImmNode(0), cloneExprTree(spatialLoops[i].extent), ForKind.SERIAL, initNest);
    }

    const updBlock = new BlockNode(`${blockName}_upd`,
      block.iterVars.map(cloneExprTree) as BlockRealizeNode[], block.reads.map(r => ({ buffer: r.buffer })), [{ buffer: acc }], cloneExprTree(store));
    let updNest: TirNode = updBlock;
    for (let i = loops.length - 1; i >= 0; i--) {
      updNest = new ForNode(loops[i].loopVar, new IntImmNode(0), cloneExprTree(loops[i].extent), ForKind.SERIAL, updNest);
    }

    const decompReplacement = new SeqNode([initNest, updNest]);
    this.mutator.replaceNode(loops[0], decompReplacement);
    this._replaceInTree(loops[0], decompReplacement);
    if (!this._replaying) {
      this.trace.record('decomposeReduction', [blockName]);
    }
  }

  cacheWrite(blockName: string, bufferName: string, scope = 'local'): Buffer | void {
    const block = this.getBlock(blockName);
    const loops = this.getLoops(blockName);
    if (loops.length === 0) throw new Error('cacheWrite: block has no enclosing loops');
    const writeEntry = (block.writes || []).find(w => w.buffer && w.buffer.name === bufferName);
    if (!writeEntry) throw new Error(`cacheWrite: block '${blockName}' does not write '${bufferName}'`);
    const buf = writeEntry.buffer;
    const cache = new Buffer(`${bufferName}_${blockName}_cachew`, [...buf.shape], buf.dtype, scope);

    this.mutator.redirectBuffer(block.body, buf, cache);
    if (block.initBody) this.mutator.redirectBuffer(block.initBody, buf, cache);
    writeEntry.buffer = cache;

    const idxVars = buf.shape.map((_, d) => new VariableNode(`${cache.name}_o${d}`, 'int32'));
    const backStore = new BufferStoreNode(buf, idxVars, new BufferLoadNode(cache, idxVars));
    const backBlock = new BlockNode(`${cache.name}_flush`, idxVars.map(v => new BlockRealizeNode(v, v)),
      [{ buffer: cache }], [{ buffer: buf }], backStore);
    let backNest: TirNode = backBlock;
    for (let d = buf.shape.length - 1; d >= 0; d--) {
      backNest = new ForNode(idxVars[d], new IntImmNode(0), new IntImmNode(buf.shape[d] as number), ForKind.SERIAL, backNest);
    }

    const blockNest = loops[0];
    const seq = new SeqNode([]);
    const alloc = new AllocateNode(cache, scope, seq);
    this.mutator.replaceNode(blockNest, alloc);
    seq.stmts.push(blockNest, backNest);
    this._replaceInTree(blockNest, alloc);
    if (!this._replaying) this.trace.record('cacheWrite', [blockName, bufferName, scope]);
  }

  setScope(blockName: string, bufferName: string, scope: string): void {
    const block = this.getBlock(blockName);
    const writeEntry = (block.writes || []).find(w => w.buffer && w.buffer.name === bufferName);
    if (!writeEntry) throw new Error(`setScope: block '${blockName}' does not write '${bufferName}'`);
    writeEntry.buffer.scope = scope;
    this.state.invalidate();
    if (!this._replaying) this.trace.record('setScope', [blockName, bufferName, scope]);
  }

  storageAlign(blockName: string, bufferName: string, axis: number, factor: number, offset: number): void {
    const block = this.getBlock(blockName);
    const entry = [...(block.writes || []), ...(block.reads || [])].find(e => e.buffer && e.buffer.name === bufferName);
    if (!entry) throw new Error(`storageAlign: block '${blockName}' does not access '${bufferName}'`);
    if (!Number.isInteger(factor) || factor <= 0) throw new Error('storageAlign: factor must be a positive integer');
    entry.buffer.storageAlign = { axis, factor, offset: offset || 0 };
    this.state.invalidate();
    if (!this._replaying) this.trace.record('storageAlign', [blockName, bufferName, axis, factor, offset || 0]);
  }

  _writeIndexLoopVars(primitive: string, block: BlockNode, store: BufferStoreNode): Set<string> {
    const info = this.state.blockAccessInfo(block);
    const access = (info ? info.accesses : []).find((a) => a.kind === AccessKind.WRITE && a.node === store);
    if (!access) throw new Error(`${primitive}: block '${block.name}' write is not reachable from the function body`);
    const names = new Set<string>();
    for (const form of access.forms) {
      if (!form) throw new Error(`${primitive}: write index of '${access.buffer.name}' is not an affine map of its loop variables`);
      for (const name of form.terms.keys()) names.add(name);
    }
    return names;
  }

  _iterVarsOver(block: BlockNode, loopVarNames: ReadonlySet<string> | readonly string[]): BlockRealizeNode[] {
    const info = this.state.blockAccessInfo(block);
    if (!info) return [];
    const formOf = new Map<string, LinearForm | null>(info.bindings.map((b) => [b.name, b.form]));
    const kept: BlockRealizeNode[] = [];
    for (const iv of block.iterVars) {
      if (!iv || !iv.iterVar) continue;
      const form = formOf.get(iv.iterVar.name);
      const nameSet = loopVarNames instanceof Set ? loopVarNames : new Set(loopVarNames as readonly string[]);
      if (form && [...form.terms.keys()].every((name) => nameSet.has(name))) kept.push(cloneExprTree(iv) as BlockRealizeNode);
    }
    return kept;
  }

  _removeBlockNest(producerName: string, prod: SRef): void {
    const loops = this.getLoops(producerName);
    const root = (loops.length > 0 ? loops[0] : prod.node) as TirNode;
    const empty = new SeqNode([]);
    this.mutator.replaceNode(root, empty);
    this._replaceInTree(root, empty);
  }

  _inlinePlan(primitive: string, producerName: string): InlinePlan {
    const sref = this.getBlockSRef(producerName);
    const prod = sref.node as BlockNode;
    if (prod.initBody) throw new Error(`${primitive}: cannot inline a reduction block (has init)`);

    const stores: BufferStoreNode[] = [];
    collectStores(prod.body, stores);
    if (stores.length === 0) throw new Error(`${primitive}: producer has no store to inline`);

    const blockInfo = this.state.blockAccessInfo(prod);
    const byNode = new Map<TirNode, BufferAccess>();
    for (const access of (blockInfo ? blockInfo.accesses : [])) {
      if (access.kind === AccessKind.WRITE) byNode.set(access.node, access);
    }

    const produced = new Set(stores.map((s) => s.buffer));
    if (produced.size !== stores.length) throw new Error(`${primitive}: buffer written more than once in block`);

    const plans: InlineStorePlan[] = [];
    for (const store of stores) {
      const access = byNode.get(store);
      if (!access) throw new Error(`${primitive}: producer store is not reachable from the analyzed function body`);
      if (access.selfReferential) {
        throw new Error(`${primitive}: producer is self-referential (recurrence), cannot inline`);
      }
      for (const other of loadedBuffers([store.value])) {
        if (produced.has(other)) throw new Error(`${primitive}: producer store depends on a co-produced buffer`);
      }
      if (this.state.accessInfo.indexLoaded.has(store.buffer)) {
        throw new Error(`${primitive}: buffer '${store.buffer.name}' is used inside an index expression (indirect), cannot safely inline`);
      }
      const decompositions = invertWriteIndices(access);
      if (!decompositions) {
        throw new Error(`${primitive}: producer write index of '${store.buffer.name}' is not an invertible affine map of its loop variables`);
      }
      const varForms = producerVarForms(access, blockInfo) as VarForms;
      for (const name of usedVarNames(store.value)) {
        if (!varForms.has(name)) {
          throw new Error(`${primitive}: producer value depends on '${name}', which is not an iteration variable of the block`);
        }
      }
      plans.push({ store, decompositions, varForms });
    }
    return { sref, prod, plans };
  }

  _applyInline(primitive: string, producerName: string, plan: InlinePlan): void {
    let total = 0;
    for (const { store, decompositions, varForms } of plan.plans) {
      total += inlineStoreValue(this.func.body, store, decompositions, varForms);
    }
    if (total === 0) throw new Error(`${primitive}: block '${producerName}' has no consumers to inline into`);
    for (const { store } of plan.plans) retargetBufferReads(this.func.body, store.buffer.name, plan.prod.reads);
    this._removeBlockNest(producerName, plan.sref);
    if (!this._replaying) this.trace.record(primitive, [producerName]);
  }

  computeInline(producerName: string): void {
    const plan = this._inlinePlan('computeInline', producerName);
    if (plan.plans.length !== 1) {
      throw new Error('computeInline: block writes more than one buffer; use computeInlineBlock');
    }
    this._applyInline('computeInline', producerName, plan);
  }

  computeInlineBlock(producerName: string): void {
    this._applyInline('computeInlineBlock', producerName, this._inlinePlan('computeInlineBlock', producerName));
  }

  _checkRelocationDependences(primitive: string, blockSRef: SRef, targetLoop: ForNode, atStart: boolean): void {
    const scope = this.state.scopeOf(blockSRef);
    const self = scope && scope.memberOf(blockSRef);
    if (!self) throw new Error(`${primitive}: block '${(blockSRef.node as BlockNode).name}' is not part of any block scope`);
    const targetSRef = this.state.tree.getSRef(targetLoop);
    if (!targetSRef) throw new Error(`${primitive}: target loop is not part of the schedule tree`);

    const inside = targetSRef.childBlocks().map((s) => scope.memberOf(s)).filter((m) => m !== null);
    if (inside.length === 0) {
      throw new Error(`${primitive}: the target loop contains no block of the same block scope`);
    }
    const positions = inside.map((m) => m.position);
    const destination = atStart ? Math.min(...positions) : Math.max(...positions);
    const lo = Math.min(self.position, destination);
    const hi = Math.max(self.position, destination);

    for (const dep of [...scope.depsBySrc(blockSRef), ...scope.depsByDst(blockSRef)]) {
      const other = dep.src === blockSRef ? dep.dst : dep.src;
      if (other === blockSRef) continue;
      const member = scope.memberOf(other);
      if (!member || member.position <= lo || member.position >= hi) continue;
      throw new Error(
        `${primitive}: moving '${(blockSRef.node as BlockNode).name}' across '${(other.node as BlockNode).name}' would violate a ` +
        `${dep.kind} dependence on buffer '${dep.buffer.name}'`
      );
    }
  }

  _alignedLoopPairs(primitive: string, blockLoops: readonly ForNode[], targetLoop: ForNode): [ForNode, ForNode][] {
    const targetChain = [];
    for (let sref = this.state.tree.getSRef(targetLoop); sref; sref = sref.parent) {
      if (sref.isLoop) targetChain.unshift(sref.node);
    }
    if (blockLoops.length > targetChain.length) {
      throw new Error(`${primitive}: the moved block is nested deeper than the target loop`);
    }
    const aligned = targetChain.slice(targetChain.length - blockLoops.length);
    const pairs: [ForNode, ForNode][] = [];
    for (let i = 0; i < blockLoops.length; i++) {
      const from = blockLoops[i];
      const to = aligned[i];
      const fromExtent = getConstExtent((from as ForNode).extent);
      const toExtent = getConstExtent((to as ForNode).extent);
      const fromMin = getConstExtent((from as ForNode).min);
      const toMin = getConstExtent((to as ForNode).min);
      if (fromExtent === null || toExtent === null || fromMin === null || toMin === null
          || fromExtent !== toExtent || fromMin !== toMin) {
        throw new Error(
          `${primitive}: iteration domain of '${(from as ForNode).loopVar.name}' does not match '${(to as ForNode).loopVar.name}'; ` +
          `relocation would change the block's iteration space`
        );
      }
      pairs.push([from as ForNode, to as ForNode]);
    }
    return pairs;
  }

  _relocateBlockToLoop(primitive: string, blockName: string, targetLoopRef: LoopRef, atStart: boolean): string {
    const blockSRef = this.getBlockSRef(blockName);
    const targetLoop = this._resolveLoop(targetLoopRef);
    if (!targetLoop || targetLoop.type !== 'ForNode') throw new Error(`${primitive}: target must be a loop`);

    const blkLoops = this.getLoops(blockName);
    if (blkLoops.length === 0) throw new Error(`${primitive}: the moved block has no enclosing loop`);
    const blkLoop = blkLoops[0];
    if (blkLoops.includes(targetLoop)) throw new Error(`${primitive}: block already sits under the target loop`);

    const pairs = this._alignedLoopPairs(primitive, blkLoops, targetLoop);
    this._checkRelocationDependences(primitive, blockSRef, targetLoop, atStart);

    const moved = cloneExprTree(blkLoops[blkLoops.length - 1].body);
    const rename = new Map(pairs.map(([from, to]) => [from.loopVar.name, to.loopVar]));
    irTransform(moved, (n) => (n.type === 'VariableNode' && rename.has(n.name) ? rename.get(n.name) : n), { bindVars: false });
    const relocEmpty = new SeqNode([]);
    this.mutator.replaceNode(blkLoop, relocEmpty);

    const tbody = targetLoop.body;
    if (tbody && tbody.type === 'SeqNode') {
      if (atStart) tbody.stmts.unshift(moved); else tbody.stmts.push(moved);
      tbody._setChildren('stmts', tbody.stmts);
    } else {
      const seq = atStart ? new SeqNode([moved, tbody]) : new SeqNode([tbody, moved]);
      targetLoop.body = seq;
      targetLoop._setChild('body', seq);
    }
    this._replaceInTree(blkLoop, relocEmpty);
    this._replaceInTree(targetLoop, targetLoop);
    return targetLoop.loopVar.name;
  }

  computeAt(producerName: string, targetLoopRef: LoopRef): void {
    const targetVar = this._relocateBlockToLoop('computeAt', producerName, targetLoopRef, true);
    if (!this._replaying) this.trace.record('computeAt', [producerName, targetVar]);
  }

  reverseComputeAt(consumerName: string, targetLoopRef: LoopRef): void {
    const targetVar = this._relocateBlockToLoop('reverseComputeAt', consumerName, targetLoopRef, false);
    if (!this._replaying) this.trace.record('reverseComputeAt', [consumerName, targetVar]);
  }

  cacheRead(blockName: string, bufferName: string, scope = 'local'): void {
    const block = this.getBlock(blockName);
    const loops = this.getLoops(blockName);
    if (loops.length === 0) throw new Error('cacheRead: block has no enclosing loops');
    const readEntry = (block.reads || []).find(r => r.buffer && r.buffer.name === bufferName);
    if (!readEntry) throw new Error(`cacheRead: block '${blockName}' does not read '${bufferName}'`);
    const buf = readEntry.buffer;
    const cache = new Buffer(`${bufferName}_${blockName}_cache`, [...buf.shape], buf.dtype, scope);

    const idxVars = buf.shape.map((_, d) => new VariableNode(`${cache.name}_i${d}`, 'int32'));
    const copyStore = new BufferStoreNode(cache, idxVars, new BufferLoadNode(buf, idxVars));
    const copyBlock = new BlockNode(`${cache.name}_fill`, idxVars.map(v => new BlockRealizeNode(v, v)),
      [{ buffer: buf }], [{ buffer: cache }], copyStore);
    let copyNest: TirNode = copyBlock;
    for (let d = buf.shape.length - 1; d >= 0; d--) {
      copyNest = new ForNode(idxVars[d], new IntImmNode(0), new IntImmNode(buf.shape[d] as number), ForKind.SERIAL, copyNest);
    }

    this.mutator.redirectReads(block.body, buf, cache);
    if (block.initBody) this.mutator.redirectReads(block.initBody, buf, cache);
    readEntry.buffer = cache;

    const blockNest = loops[0];
    const seq = new SeqNode([copyNest]);
    const alloc = new AllocateNode(cache, scope, seq);
    this.mutator.replaceNode(blockNest, alloc);
    seq.stmts.push(blockNest);
    this._replaceInTree(blockNest, alloc);
    if (!this._replaying) this.trace.record('cacheRead', [blockName, bufferName, scope]);
  }

  fuseConsumer(producerBlockName: string, consumerBlockName: string): void {
    const pBlock = this.getBlock(producerBlockName);
    const cBlock = this.getBlock(consumerBlockName);
    const pLoops = this.getLoops(producerBlockName);
    const cLoops = this.getLoops(consumerBlockName);
    if (!pBlock.body || pBlock.body.type !== 'BufferStoreNode') {
      throw new Error(`fuseConsumer: producer '${producerBlockName}' body is not a store`);
    }
    const pSpatialVars = new Set<string>();
    for (const idx of (pBlock.body as BufferStoreNode).indices) collectVarsUsed(idx, pSpatialVars);
    const pSpatialLoops = pLoops.filter(l => pSpatialVars.has(l.loopVar.name));
    if (pSpatialLoops.length === 0 || cLoops.length !== pSpatialLoops.length) {
      throw new Error(`fuseConsumer: producer/consumer spatial rank mismatch (${pSpatialLoops.length} vs ${cLoops.length})`);
    }
    if (!cLoops[0]._parent || cLoops[0]._parent.type !== 'SeqNode') {
      throw new Error('fuseConsumer: consumer loop nest is not a direct SeqNode sibling; cannot fuse without duplicating it');
    }

    const innermostSpatial = pSpatialLoops[pSpatialLoops.length - 1];
    const reductionSubNest = cloneExprTree(innermostSpatial.body);

    let cBody = cloneExprTree(cBlock.body);
    for (let i = 0; i < cLoops.length; i++) {
      const targetName = pSpatialLoops[i].loopVar.name;
      cBody = substituteVar(cBody, cLoops[i].loopVar.name, () => new VariableNode(targetName, 'int32'));
    }
    const cFusedBlock = new BlockNode(`${consumerBlockName}_fused`, [],
      cBlock.reads.map(r => ({ buffer: r.buffer })), cBlock.writes.map(w => ({ buffer: w.buffer })), cBody);

    let fused: TirNode = new SeqNode([reductionSubNest, cFusedBlock]);
    for (let i = pSpatialLoops.length - 1; i >= 0; i--) {
      const sl = pSpatialLoops[i];
      fused = new ForNode(sl.loopVar, new IntImmNode(0), cloneExprTree(sl.extent), sl.kind, fused, sl.threadTag);
    }

    const consumerNest = cLoops[0];
    this.mutator.replaceNode(pLoops[0], fused);
    this.mutator.removeNode(consumerNest);
    this._replaceInTree(pLoops[0], fused);
    this._removeFromTree(consumerNest);
    if (!this._replaying) {
      this.trace.record('fuseConsumer', [producerBlockName, consumerBlockName]);
    }
  }

  annotate(loop: LoopRef, key: string, value: unknown): void {
    loop = this._resolveLoop(loop);
    if (loop.type !== 'ForNode') throw new Error('annotate expects ForNode');
    const lp = loop as ForNode;
    if (!lp.annotations) lp.annotations = {};
    lp.annotations[key] = value;
    this.state.invalidate();
    if (!this._replaying) {
      this.trace.record('annotate', [loop.loopVar.name, key, value]);
    }
  }

  tensorize(intrinName: string, info: unknown): void {
    if (typeof intrinName !== 'string') throw new Error('tensorize expects an intrinsic name');
    const intrin = info as { M?: unknown; N?: unknown; K?: unknown } | null;
    if (!intrin || typeof intrin.M !== 'number' || typeof intrin.N !== 'number' || typeof intrin.K !== 'number') {
      throw new Error('tensorize expects info { M, N, K, a, b, c }');
    }
    this.func.setAttr(FuncAttr.TENSOR_INTRIN, { name: intrinName, info });
    this.state.invalidate();
  }

  blockize(loopRef: LoopRef, name: string | null = null): BlockNode {
    const loop = this._resolveLoop(loopRef);
    if (!loop || loop.type !== 'ForNode') throw new Error('blockize expects a loop');
    const reads = new Map(), writes = new Map();
    collectBufferAccess(loop, reads, writes);
    const blockName = name || `blockized_${loop.loopVar.name}`;
    const wrapper = new BlockNode(
      blockName,
      [],
      [...reads.values()].map(b => ({ buffer: b })),
      [...writes.values()].map(b => ({ buffer: b })),
      new SeqNode([])
    );
    this.mutator.replaceNode(loop, wrapper);
    wrapper.body = loop;
    wrapper._setChild('body', loop);
    this._replaceInTree(loop, wrapper);
    if (!this._replaying) this.trace.record('blockize', [loop.loopVar.name]);
    return wrapper;
  }

  getTrace(): ScheduleTrace {
    return this.trace;
  }

  verify(): string[] {
    return ScheduleValidator.validate(this.func);
  }
}

function collectBufferAccess(node: TirNode | null | undefined, reads: Map<string, Buffer>, writes: Map<string, Buffer>): void {
  if (!node || typeof node !== 'object') return;
  const ld = node as BufferLoadNode;
  const st = node as BufferStoreNode;
  if (node.type === 'BufferLoadNode' && ld.buffer) reads.set(ld.buffer.name, ld.buffer);
  if (node.type === 'BufferStoreNode' && st.buffer) writes.set(st.buffer.name, st.buffer);
  const slots = node as unknown as NodeSlots;
  for (const k of ['a', 'b', 'value', 'expr', 'condition', 'thenBody', 'elseBody', 'body', 'initBody']) {
    if (slots[k]) collectBufferAccess(slots[k] as TirNode, reads, writes);
  }
  if (slots.indices) for (const ix of slots.indices as TirNode[]) collectBufferAccess(ix, reads, writes);
  if (slots.args) for (const a of slots.args as TirNode[]) collectBufferAccess(a, reads, writes);
  if (slots.stmts) for (const st2 of slots.stmts as TirNode[]) collectBufferAccess(st2, reads, writes);
}
