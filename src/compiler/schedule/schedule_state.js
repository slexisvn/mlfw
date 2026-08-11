import { SRefTree } from './sref.js';
import { ForKind } from '../ir/tensor/nodes.js';
import { collectBufferAccesses } from '../analysis/buffer_access.js';
import { dependences } from '../analysis/dependence.js';
import { buildBlockScopes, scopeRootSRef } from './block_scope.js';
import { LinearForm, toLinearForm, composeForm, coverRangeOfForm } from '../analysis/iter_map.js';

function constExtent(node) {
  return node && node.type === 'IntImmNode' && Number.isInteger(node.value) ? node.value : null;
}

export class LoopBinding {
  constructor(loopSRef, iterVar, extent, kind, threadTag) {
    this.loopSRef = loopSRef;
    this.iterVar = iterVar;
    this.extent = extent;
    this.kind = kind;
    this.threadTag = threadTag;
  }
}

export class BlockBinding {
  constructor(blockSRef, iterVars, readBuffers, writeBuffers) {
    this.blockSRef = blockSRef;
    this.iterVars = iterVars;
    this.readBuffers = readBuffers;
    this.writeBuffers = writeBuffers;
  }
}

export class ScheduleState {
  constructor(primFunc) {
    this.primFunc = primFunc;
    this.tree = new SRefTree(primFunc);
    this.invalidate();
  }

  invalidate() {
    this._loopBindings = null;
    this._blockBindings = null;
    this._accessInfo = null;
    this._scopes = null;
    this._dependences = null;
    this._nestAnalyses = null;
  }

  _enclosingEnv(node) {
    const loopRanges = new Map();
    const varForms = new Map();
    const iterSpace = [];
    const sref = this.tree.getSRef(node);
    const chain = [];
    for (let s = sref ? sref.parent : null; s; s = s.parent) chain.push(s);

    for (let i = chain.length - 1; i >= 0; i--) {
      const s = chain[i];
      if (s.isLoop) {
        const loop = s.node;
        const min = constExtent(loop.min);
        const extent = constExtent(loop.extent);
        if (loop.loopVar && min !== null && extent !== null) {
          loopRanges.set(loop.loopVar.name, [min, extent]);
          varForms.set(loop.loopVar.name, LinearForm.variable(loop.loopVar.name));
          iterSpace.push({ name: loop.loopVar.name, min, extent, node: loop });
        } else {
          iterSpace.push(null);
        }
        continue;
      }
      for (const iv of s.node.iterVars || []) {
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

  nestAnalysis(node) {
    if (!this._nestAnalyses) this._nestAnalyses = new Map();
    let analysis = this._nestAnalyses.get(node);
    if (!analysis) {
      const info = collectBufferAccesses(node, this._enclosingEnv(node));
      analysis = { info, deps: dependences(info.byBuffer) };
      this._nestAnalyses.set(node, analysis);
    }
    return analysis;
  }

  replaceNode(oldNode, newNode) {
    if (!this.tree.replaceNode(oldNode, newNode)) this.tree.rebuildFrom(this.primFunc.body);
    this.invalidate();
  }

  removeNode(node) {
    if (!this.tree.removeNode(node)) this.tree.rebuildFrom(this.primFunc.body);
    this.invalidate();
  }

  _ensureBindings() {
    if (this._loopBindings) return;
    this._loopBindings = new Map();
    this._blockBindings = new Map();

    for (const loopSRef of this.tree.allLoops()) {
      const node = loopSRef.node;
      const extent = node.extent && node.extent.type === 'IntImmNode' ? node.extent.value : null;
      this._loopBindings.set(node.loopVar.name, new LoopBinding(
        loopSRef, node.loopVar.name, extent, node.kind, node.threadTag
      ));
    }

    for (const blockSRef of this.tree.allBlocks()) {
      const node = blockSRef.node;
      const iterVars = [];
      for (const b of node.iterVars) {
        if (b.iterVar) iterVars.push(b.iterVar.name);
      }
      const readBufs = [];
      for (const r of node.reads) readBufs.push(r.buffer.name);
      const writeBufs = [];
      for (const w of node.writes) writeBufs.push(w.buffer.name);
      this._blockBindings.set(node.name, new BlockBinding(blockSRef, iterVars, readBufs, writeBufs));
    }
  }

  get accessInfo() {
    if (!this._accessInfo) this._accessInfo = collectBufferAccesses(this.primFunc.body);
    return this._accessInfo;
  }

  get scopes() {
    if (!this._scopes) this._scopes = buildBlockScopes(this.tree.allBlocks(), this.accessInfo);
    return this._scopes;
  }

  get dependences() {
    if (!this._dependences) this._dependences = dependences(this.accessInfo.byBuffer);
    return this._dependences;
  }

  scopeOf(blockSRef) {
    return this.scopes.get(scopeRootSRef(blockSRef)) || null;
  }

  blockInfo(blockSRef) {
    const scope = this.scopeOf(blockSRef);
    return scope ? scope.blockInfo(blockSRef) : null;
  }

  blockAccessInfo(blockNode) {
    return this.accessInfo.byBlock.get(blockNode) || null;
  }

  getLoopBinding(varName) {
    this._ensureBindings();
    return this._loopBindings.get(varName) || null;
  }

  getBlockBinding(blockName) {
    this._ensureBindings();
    return this._blockBindings.get(blockName) || null;
  }

  getBlock(name) {
    const sref = this.tree.getBlockSRef(name);
    if (!sref) throw new Error(`Block '${name}' not found in schedule state`);
    return sref;
  }

  getLoopsOf(blockName) {
    return this.tree.loopsOf(blockName);
  }

  threadBindingSummary() {
    this._ensureBindings();
    const bindings = {};
    for (const [, lb] of this._loopBindings) {
      if (lb.kind === ForKind.THREAD_BINDING && lb.threadTag) {
        bindings[lb.threadTag] = { varName: lb.iterVar, extent: lb.extent };
      }
    }
    return bindings;
  }

  blockDim() {
    const summary = this.threadBindingSummary();
    return [
      summary['threadIdx.x']?.extent || 1,
      summary['threadIdx.y']?.extent || 1,
      summary['threadIdx.z']?.extent || 1,
    ];
  }

  gridDim() {
    const summary = this.threadBindingSummary();
    return [
      summary['blockIdx.x']?.extent || 1,
      summary['blockIdx.y']?.extent || 1,
      summary['blockIdx.z']?.extent || 1,
    ];
  }

  allBlockNames() {
    this._ensureBindings();
    return [...this._blockBindings.keys()];
  }

  allLoopVarNames() {
    this._ensureBindings();
    return [...this._loopBindings.keys()];
  }

  summary() {
    return {
      blocks: this.allBlockNames(),
      loops: this.allLoopVarNames(),
      threadBindings: this.threadBindingSummary(),
      blockDim: this.blockDim(),
      gridDim: this.gridDim(),
    };
  }
}
