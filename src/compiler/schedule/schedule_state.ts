import { SRefTree } from './sref.js';
import { ForKind } from '../ir/tensor/nodes.js';
import { collectBufferAccesses } from '../analysis/buffer_access.js';
import { dependences } from '../analysis/dependence.js';
import { buildBlockScopes, scopeRootSRef } from './block_scope.js';
import { LinearForm, toLinearForm, composeForm, coverRangeOfForm } from '../analysis/iter_map.js';
import type { VarRange } from '../analysis/iter_map.js';
import type { TirNode, PrimFunc, ForNode, BlockNode, IntImmNode, ForKindValue } from '../ir/tensor/nodes.js';
import type { Buffer } from '../ir/tensor/buffer.js';
import { opaqueLevel } from '../analysis/buffer_access.js';
import type { BufferAccessResult, BufferAccessEnv, IterLevel, BlockAccessInfo } from '../analysis/buffer_access.js';
import type { Dependence } from '../analysis/dependence.js';
import type { BlockScope, BlockInfo } from './block_scope.js';
import type { SRef } from './sref.js';

export type NestAnalysis = { info: BufferAccessResult; deps: Dependence[] };
export type ThreadBindingEntry = { varName: string; extent: number | null };
export type ThreadBindingSummary = Record<string, ThreadBindingEntry>;
export type ScheduleSummary = {
  blocks: string[];
  loops: string[];
  threadBindings: ThreadBindingSummary;
  blockDim: number[];
  gridDim: number[];
};

function constExtent(node: TirNode | null | undefined): number | null {
  return node && node.type === 'IntImmNode' && Number.isInteger((node as IntImmNode).value) ? (node as IntImmNode).value : null;
}

export class LoopBinding {
  loopSRef: SRef;
  iterVar: string;
  extent: number | null;
  kind: ForKindValue;
  threadTag: string | null;

  constructor(loopSRef: SRef, iterVar: string, extent: number | null, kind: ForKindValue, threadTag: string | null) {
    this.loopSRef = loopSRef;
    this.iterVar = iterVar;
    this.extent = extent;
    this.kind = kind;
    this.threadTag = threadTag;
  }
}

export class BlockBinding {
  blockSRef: SRef;
  iterVars: string[];
  readBuffers: string[];
  writeBuffers: string[];

  constructor(blockSRef: SRef, iterVars: string[], readBuffers: string[], writeBuffers: string[]) {
    this.blockSRef = blockSRef;
    this.iterVars = iterVars;
    this.readBuffers = readBuffers;
    this.writeBuffers = writeBuffers;
  }
}

export class ScheduleState {
  primFunc: PrimFunc;
  tree: SRefTree;
  private _loopBindings!: Map<string, LoopBinding> | null;
  private _blockBindings!: Map<string, BlockBinding> | null;
  private _accessInfo!: BufferAccessResult | null;
  private _scopes!: Map<SRef | null, BlockScope> | null;
  private _dependences!: Dependence[] | null;
  private _nestAnalyses!: Map<TirNode, NestAnalysis> | null;

  constructor(primFunc: PrimFunc) {
    this.primFunc = primFunc;
    this.tree = new SRefTree(primFunc);
    this.invalidate();
  }

  invalidate(): void {
    this._loopBindings = null;
    this._blockBindings = null;
    this._accessInfo = null;
    this._scopes = null;
    this._dependences = null;
    this._nestAnalyses = null;
  }

  _enclosingEnv(node: TirNode): BufferAccessEnv {
    const loopRanges = new Map<string, VarRange>();
    const varForms = new Map<string, LinearForm>();
    const iterSpace: IterLevel[] = [];
    const sref = this.tree.getSRef(node);
    const chain: SRef[] = [];
    for (let s = sref ? sref.parent : null; s; s = s.parent) chain.push(s);

    for (let i = chain.length - 1; i >= 0; i--) {
      const s = chain[i];
      if (s.isLoop) {
        const loop = s.node as ForNode;
        const min = constExtent(loop.min);
        const extent = constExtent(loop.extent);
        if (!loop.loopVar) {
          iterSpace.push(opaqueLevel(loop));
          continue;
        }
        const name = loop.loopVar.name;
        varForms.set(name, LinearForm.variable(name));
        if (min !== null && extent !== null) {
          loopRanges.set(name, [min, extent]);
          iterSpace.push({ name, min, extent, node: loop });
        } else {
          iterSpace.push({ name, min: null, extent: null, node: loop });
        }
        continue;
      }
      for (const iv of (s.node as BlockNode).iterVars || []) {
        if (!iv || !iv.iterVar) continue;
        const raw = iv.binding ? toLinearForm(iv.binding) : null;
        const range = coverRangeOfForm(raw, loopRanges);
        const composed = composeForm(raw, varForms);
        if (range === null) loopRanges.delete(iv.iterVar.name);
        else loopRanges.set(iv.iterVar.name, range);
        if (composed === null) varForms.delete(iv.iterVar.name);
        else varForms.set(iv.iterVar.name, composed);
      }
    }
    return { loopRanges, varForms, iterSpace };
  }

  nestAnalysis(node: TirNode): NestAnalysis {
    if (!this._nestAnalyses) this._nestAnalyses = new Map();
    let analysis = this._nestAnalyses.get(node);
    if (!analysis) {
      const info = collectBufferAccesses(node, this._enclosingEnv(node));
      analysis = { info, deps: dependences(info.byBuffer) };
      this._nestAnalyses.set(node, analysis);
    }
    return analysis;
  }

  replaceNode(oldNode: TirNode, newNode: TirNode): void {
    if (!this.tree.replaceNode(oldNode, newNode)) this.tree.rebuildFrom(this.primFunc.body);
    this.invalidate();
  }

  removeNode(node: TirNode): void {
    if (!this.tree.removeNode(node)) this.tree.rebuildFrom(this.primFunc.body);
    this.invalidate();
  }

  _ensureBindings(): void {
    if (this._loopBindings) return;
    const loopBindings = new Map<string, LoopBinding>();
    const blockBindings = new Map<string, BlockBinding>();
    this._loopBindings = loopBindings;
    this._blockBindings = blockBindings;

    for (const loopSRef of this.tree.allLoops()) {
      const node = loopSRef.node as ForNode;
      const extent = node.extent && node.extent.type === 'IntImmNode' ? (node.extent as IntImmNode).value : null;
      loopBindings.set(node.loopVar.name, new LoopBinding(
        loopSRef, node.loopVar.name, extent, node.kind, node.threadTag
      ));
    }

    for (const blockSRef of this.tree.allBlocks()) {
      const node = blockSRef.node as BlockNode;
      const iterVars: string[] = [];
      for (const b of node.iterVars) {
        if (b.iterVar) iterVars.push(b.iterVar.name);
      }
      const readBufs: string[] = [];
      for (const r of node.reads) readBufs.push(r.buffer.name);
      const writeBufs: string[] = [];
      for (const w of node.writes) writeBufs.push(w.buffer.name);
      blockBindings.set(node.name, new BlockBinding(blockSRef, iterVars, readBufs, writeBufs));
    }
  }

  get accessInfo(): BufferAccessResult {
    if (!this._accessInfo) this._accessInfo = collectBufferAccesses(this.primFunc.body);
    return this._accessInfo;
  }

  get scopes(): Map<SRef | null, BlockScope> {
    if (!this._scopes) this._scopes = buildBlockScopes(this.tree.allBlocks(), this.accessInfo);
    return this._scopes;
  }

  get dependences(): Dependence[] {
    if (!this._dependences) this._dependences = dependences(this.accessInfo.byBuffer);
    return this._dependences;
  }

  scopeOf(blockSRef: SRef): BlockScope | null {
    return this.scopes.get(scopeRootSRef(blockSRef)) || null;
  }

  blockInfo(blockSRef: SRef): BlockInfo | null {
    const scope = this.scopeOf(blockSRef);
    return scope ? scope.blockInfo(blockSRef) : null;
  }

  blockAccessInfo(blockNode: BlockNode): BlockAccessInfo | null {
    return this.accessInfo.byBlock.get(blockNode) || null;
  }

  getLoopBinding(varName: string): LoopBinding | null {
    this._ensureBindings();
    return (this._loopBindings as Map<string, LoopBinding>).get(varName) || null;
  }

  getBlockBinding(blockName: string): BlockBinding | null {
    this._ensureBindings();
    return (this._blockBindings as Map<string, BlockBinding>).get(blockName) || null;
  }

  getBlock(name: string): SRef {
    const sref = this.tree.getBlockSRef(name);
    if (!sref) throw new Error(`Block '${name}' not found in schedule state`);
    return sref;
  }

  getLoopsOf(blockName: string): SRef[] {
    return this.tree.loopsOf(blockName);
  }

  threadBindingSummary(): ThreadBindingSummary {
    this._ensureBindings();
    const bindings: ThreadBindingSummary = {};
    for (const [, lb] of this._loopBindings as Map<string, LoopBinding>) {
      if (lb.kind === ForKind.THREAD_BINDING && lb.threadTag) {
        bindings[lb.threadTag] = { varName: lb.iterVar, extent: lb.extent };
      }
    }
    return bindings;
  }

  blockDim(): number[] {
    const summary = this.threadBindingSummary();
    return [
      summary['threadIdx.x']?.extent || 1,
      summary['threadIdx.y']?.extent || 1,
      summary['threadIdx.z']?.extent || 1,
    ];
  }

  gridDim(): number[] {
    const summary = this.threadBindingSummary();
    return [
      summary['blockIdx.x']?.extent || 1,
      summary['blockIdx.y']?.extent || 1,
      summary['blockIdx.z']?.extent || 1,
    ];
  }

  allBlockNames(): string[] {
    this._ensureBindings();
    return [...(this._blockBindings as Map<string, BlockBinding>).keys()];
  }

  allLoopVarNames(): string[] {
    this._ensureBindings();
    return [...(this._loopBindings as Map<string, LoopBinding>).keys()];
  }

  summary(): ScheduleSummary {
    return {
      blocks: this.allBlockNames(),
      loops: this.allLoopVarNames(),
      threadBindings: this.threadBindingSummary(),
      blockDim: this.blockDim(),
      gridDim: this.gridDim(),
    };
  }
}
