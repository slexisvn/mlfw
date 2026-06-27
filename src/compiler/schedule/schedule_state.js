import { SRefTree } from './sref.js';
import { ForKind } from '../ir/tensor/nodes.js';

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
    this._dirty = true;
    this._tree = null;
    this._loopBindings = null;
    this._blockBindings = null;
  }

  _ensureBuilt() {
    if (!this._dirty) return;
    this._tree = new SRefTree(this.primFunc);
    this._loopBindings = new Map();
    this._blockBindings = new Map();

    for (const loopSRef of this._tree.allLoops()) {
      const node = loopSRef.node;
      const extent = node.extent && node.extent.type === 'IntImmNode' ? node.extent.value : null;
      this._loopBindings.set(node.loopVar.name, new LoopBinding(
        loopSRef, node.loopVar.name, extent, node.kind, node.threadTag
      ));
    }

    for (const blockSRef of this._tree.allBlocks()) {
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

    this._dirty = false;
  }

  get tree() {
    this._ensureBuilt();
    return this._tree;
  }

  invalidate() {
    this._dirty = true;
    this._tree = null;
    this._loopBindings = null;
    this._blockBindings = null;
  }

  getLoopBinding(varName) {
    this._ensureBuilt();
    return this._loopBindings.get(varName) || null;
  }

  getBlockBinding(blockName) {
    this._ensureBuilt();
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
    this._ensureBuilt();
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
    this._ensureBuilt();
    return [...this._blockBindings.keys()];
  }

  allLoopVarNames() {
    this._ensureBuilt();
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
