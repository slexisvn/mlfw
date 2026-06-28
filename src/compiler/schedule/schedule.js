import {
  ForNode, BlockNode, SeqNode, BufferStoreNode, BufferLoadNode,
  VariableNode, IntImmNode, MathOpNode, ForKind, IfThenElseNode, AllocateNode
} from '../ir/tensor/nodes.js';
import { Buffer } from '../ir/tensor/buffer.js';
import { ScheduleTrace } from './trace.js';
import { ScheduleValidator } from './validator.js';
import { ScheduleState } from './schedule_state.js';
import { ScheduleMutator } from './mutator.js';
import { SRefTree } from './sref.js';
import { loopCarriesReduction, collectVarsUsed } from './legality.js';
import { cloneIRShared } from '../ir/clone_ir.js';
import { transform as irTransform, some as irSome } from '../ir/ir_visitor.js';

const RFACTOR_ASSOCIATIVE_OPS = new Set(['+', '*', 'min', 'max']);

function substituteVar(node, oldName, exprFactory) {
  return irTransform(node, (n) => {
    if (n.type === 'VariableNode' && n.name === oldName) return exprFactory();
    return n;
  }, { bindVars: false });
}

function replaceBufferLoads(node, bufName, makeReplacement, counter) {
  return irTransform(node, (n) => {
    if (n.type === 'BufferLoadNode' && n.buffer && n.buffer.name === bufName) {
      counter.n++;
      return makeReplacement(n);
    }
    return n;
  }, { bindVars: false });
}

function loadsBuffer(node, bufName) {
  return irSome(node, (n) => n.type === 'BufferLoadNode' && n.buffer && n.buffer.name === bufName);
}

function cloneExprTree(node) {
  return cloneIRShared(node, cloneExprTree, (n, copy, rec) => {
    if (n.type === 'BlockRealizeNode') {
      copy.iterVar = n.iterVar;
      copy.binding = rec(n.binding);
      return copy;
    }
    for (const key of Object.keys(n)) {
      if (key === '_parent' || key === '_parentKey' || key === '_parentIdx') continue;
      copy[key] = n[key];
    }
    return copy;
  });
}

function getConstExtent(node) {
  if (node.type === 'IntImmNode') return node.value;
  return null;
}

let _varId = 0;
function freshVar(hint, dtype = 'int32') {
  return new VariableNode(`${hint}_${_varId++}`, dtype);
}

export function resetVarCounter() {
  _varId = 0;
}

export class Schedule {
  constructor(primFunc) {
    this.func = primFunc;
    this.trace = new ScheduleTrace();
    this.state = new ScheduleState(primFunc);
    this._replaying = false;
    this._srefTree = new SRefTree(primFunc);
    this.mutator = new ScheduleMutator(primFunc);
  }

  _replaceInTree(oldNode, newNode) {
    if (!this._srefTree.replaceNode(oldNode, newNode)) {
      this._srefTree.rebuildFrom(this.func.body);
    }
    this.state.invalidate();
  }

  _removeFromTree(node) {
    if (!this._srefTree.removeNode(node)) {
      this._srefTree.rebuildFrom(this.func.body);
    }
    this.state.invalidate();
  }

  getBlock(name) {
    const sref = this._srefTree.getBlockSRef(name);
    if (!sref) throw new Error(`Block '${name}' not found`);
    return sref.node;
  }

  getLoops(blockName) {
    return this._srefTree.loopsOf(blockName).map(sref => sref.node);
  }

  _resolveLoop(ref) {
    if (typeof ref !== 'string') return ref;
    let found = null;
    const walk = (node) => {
      if (!node || typeof node !== 'object' || found) return;
      if (node.type === 'ForNode' && node.loopVar && node.loopVar.name === ref) { found = node; return; }
      if (node.body) walk(node.body);
      if (node.initBody) walk(node.initBody);
      if (node.stmts) for (const s of node.stmts) walk(s);
      if (node.thenBody) walk(node.thenBody);
      if (node.elseBody) walk(node.elseBody);
    };
    walk(this.func.body);
    return found || ref;
  }

  split(loop, factor) {
    loop = this._resolveLoop(loop);
    const extent = getConstExtent(loop.extent);
    if (extent === null) {
      throw new Error(`Cannot split loop '${loop.loopVar.name}' with non-constant extent`);
    }
    if (factor <= 0 || !Number.isInteger(factor)) {
      throw new Error(`Split factor must be a positive integer, got ${factor}`);
    }

    const outerExtent = Math.ceil(extent / factor);
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
      loop.threadTag
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
      loop.threadTag
    );

    substituteVar(innerLoop.body, oldVarName, () =>
      new MathOpNode('+',
        new MathOpNode('*', outerVar, new IntImmNode(factor)),
        innerVar
      )
    );

    this.mutator.replaceNode(loop, outerLoop);
    this._replaceInTree(loop, outerLoop);

    if (!this._replaying) {
      this.trace.record('split', [loop.loopVar.name, factor]);
    }

    return [outerLoop, innerLoop];
  }

  reorder(...newOrder) {
    if (newOrder.length === 1 && Array.isArray(newOrder[0])) newOrder = newOrder[0];
    newOrder = newOrder.map((l) => this._resolveLoop(l));
    if (newOrder.length < 2) return;

    for (const loop of newOrder) {
      if (loop.type !== 'ForNode') {
        throw new Error('reorder expects ForNode arguments');
      }
    }

    const loopSet = new Set(newOrder);
    if (loopSet.size !== newOrder.length) {
      throw new Error('reorder: duplicate loop in requested order');
    }
    let topmostLoop = null;
    let topmostDepth = Infinity;
    const depthMap = new Map();

    const findDepths = (node, depth) => {
      if (!node) return;
      if (node.type === 'ForNode') {
        if (loopSet.has(node)) {
          depthMap.set(node, depth);
          if (depth < topmostDepth) {
            topmostDepth = depth;
            topmostLoop = node;
          }
        }
        findDepths(node.body, depth + 1);
        return;
      }
      if (node.type === 'SeqNode') {
        for (const s of node.stmts) findDepths(s, depth);
      } else if (node.type === 'IfThenElseNode') {
        findDepths(node.thenBody, depth);
      } else if (node.type === 'BlockNode' || node.type === 'AllocateNode' || node.type === 'LetStmtNode') {
        findDepths(node.body, depth);
      }
    };
    findDepths(this.func.body, 0);

    if (depthMap.size !== newOrder.length) {
      throw new Error('reorder: not all requested loops were found in the function nest');
    }

    const { wrappers, innermostBody } = this._collectReorderNest(loopSet, topmostLoop);

    this.mutator.replaceNode(topmostLoop, newOrder[0]);

    for (const loop of newOrder) {
      if (loop === newOrder[0]) continue;
      loop._parent = null;
      loop._parentKey = null;
      loop._parentIdx = -1;
    }

    let inner = innermostBody;
    for (let i = wrappers.length - 1; i >= 0; i--) {
      this._setWrapperChild(wrappers[i], inner);
      inner = wrappers[i];
    }

    for (let i = 0; i < newOrder.length; i++) {
      const child = i < newOrder.length - 1 ? newOrder[i + 1] : inner;
      newOrder[i].body = child;
      newOrder[i]._setChild('body', child);
    }

    this._replaceInTree(topmostLoop, newOrder[0]);

    if (!this._replaying) {
      this.trace.record('reorder', [newOrder.map(l => l.loopVar.name)]);
    }
  }

  _setWrapperChild(wrapper, child) {
    if (wrapper.type === 'IfThenElseNode') {
      wrapper._parent = null; wrapper._parentKey = null; wrapper._parentIdx = -1;
      wrapper.thenBody = child;
      wrapper._setChild('thenBody', child);
    } else {
      wrapper._parent = null; wrapper._parentKey = null; wrapper._parentIdx = -1;
      wrapper.body = child;
      wrapper._setChild('body', child);
    }
  }

  _collectReorderNest(loopSet, topmostLoop) {
    const remaining = new Set(loopSet);
    const wrappers = [];
    let node = topmostLoop;
    let started = false;
    let innermostBody = null;
    while (node) {
      if (node.type === 'ForNode') {
        if (loopSet.has(node)) {
          remaining.delete(node);
          started = true;
          if (remaining.size === 0) { innermostBody = node.body; break; }
          node = node.body;
          continue;
        }
        if (started) {
          throw new Error(
            `reorder: loops are not a perfect nest — non-reordered loop '${node.loopVar.name}' ` +
            `is interleaved between reordered loops`
          );
        }
        node = node.body;
      } else if (node.type === 'IfThenElseNode') {
        if (node.elseBody) {
          throw new Error('reorder: cannot reorder across a conditional with an else-branch');
        }
        if (started) wrappers.push(node);
        node = node.thenBody;
      } else if (node.type === 'AllocateNode' || node.type === 'LetStmtNode') {
        if (started) wrappers.push(node);
        node = node.body;
      } else if (node.type === 'BlockNode') {
        if (started) {
          throw new Error('reorder: a compute block separates the reordered loops');
        }
        node = node.body;
      } else if (node.type === 'SeqNode') {
        if (node.stmts.length !== 1) {
          throw new Error('reorder: loops are not a perfect nest — multiple statements separate the reordered loops');
        }
        node = node.stmts[0];
      } else {
        break;
      }
    }
    if (remaining.size > 0) {
      throw new Error('reorder: loops do not form a single perfect nest');
    }
    return { wrappers, innermostBody };
  }

  fuseLoops(outer, inner) {
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

  tile(blockName, loopIndices, tileSizes) {
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

  vectorize(loop) {
    loop = this._resolveLoop(loop);
    if (loop.type !== 'ForNode') throw new Error('vectorize expects ForNode');
    const extent = getConstExtent(loop.extent);
    if (extent === null) throw new Error('Cannot vectorize loop with non-constant extent');
    const reductionBlock = loopCarriesReduction(loop);
    if (reductionBlock !== null) {
      throw new Error(`Cannot vectorize reduction loop '${loop.loopVar.name}' (loop-carried dependency in block '${reductionBlock}')`);
    }
    loop.kind = ForKind.VECTORIZED;
    this.state.invalidate();
    if (!this._replaying) {
      this.trace.record('vectorize', [loop.loopVar.name]);
    }
  }

  unroll(loop) {
    loop = this._resolveLoop(loop);
    if (loop.type !== 'ForNode') throw new Error('unroll expects ForNode');
    loop.kind = ForKind.UNROLLED;
    this.state.invalidate();
    if (!this._replaying) {
      this.trace.record('unroll', [loop.loopVar.name]);
    }
  }

  parallelize(loop) {
    loop = this._resolveLoop(loop);
    if (loop.type !== 'ForNode') throw new Error('parallelize expects ForNode');
    const reductionBlock = loopCarriesReduction(loop);
    if (reductionBlock !== null) {
      throw new Error(`Cannot parallelize reduction loop '${loop.loopVar.name}' (loop-carried dependency in block '${reductionBlock}')`);
    }
    loop.kind = ForKind.PARALLEL;
    this.state.invalidate();
    if (!this._replaying) {
      this.trace.record('parallelize', [loop.loopVar.name]);
    }
  }

  bindThread(loop, threadTag) {
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

  rfactor(blockName, reductionVarName, factor) {
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
    const op = store.value.op;
    const isAccLoad = (node) => node && node.type === 'BufferLoadNode' && node.buffer === acc;
    let update;
    if (isAccLoad(store.value.a)) update = store.value.b;
    else if (isAccLoad(store.value.b)) update = store.value.a;
    else throw new Error(`rfactor: accumulator load not found in block '${blockName}' body`);
    if (!RFACTOR_ASSOCIATIVE_OPS.has(op)) {
      throw new Error(`rfactor: op '${op}' is not associative+commutative; cannot factor reduction`);
    }
    const initVal = (block.initBody && block.initBody.type === 'BufferStoreNode' && block.initBody.value)
      ? block.initBody.value : new IntImmNode(0);

    const spatialLoops = loops.filter(l => l.loopVar.name !== reductionVarName);
    const KO = K / factor;
    const partialBuf = new Buffer(`${acc.name}_rf`, [factor, ...acc.shape], acc.dtype, acc.scope);

    const kiVar = freshVar(`${reductionVarName}_rfi`);
    const koVar = freshVar(`${reductionVarName}_rfo`);
    const piVar = freshVar(`${reductionVarName}_rfp`);

    const cfIdx = (kvar) => [kvar, ...spatialIdx.map(cloneExprTree)];
    const partialUpdate = substituteVar(cloneExprTree(update), reductionVarName,
      () => new MathOpNode('+', new MathOpNode('*', koVar, new IntImmNode(factor)), kiVar));

    const partialStore = new BufferStoreNode(partialBuf, cfIdx(kiVar),
      new MathOpNode(op, new BufferLoadNode(partialBuf, cfIdx(kiVar)), partialUpdate));
    const partialInit = new BufferStoreNode(partialBuf, cfIdx(kiVar), cloneExprTree(initVal));
    const partialBlock = new BlockNode(`${blockName}_rf_p`, [],
      block.reads.map(r => ({ buffer: r.buffer })), [{ buffer: partialBuf }], partialStore, partialInit);

    let partialNest = new ForNode(koVar, new IntImmNode(0), new IntImmNode(KO), ForKind.SERIAL, partialBlock);
    partialNest = new ForNode(kiVar, new IntImmNode(0), new IntImmNode(factor), ForKind.SERIAL, partialNest);
    for (let i = spatialLoops.length - 1; i >= 0; i--) {
      partialNest = new ForNode(spatialLoops[i].loopVar, new IntImmNode(0),
        cloneExprTree(spatialLoops[i].extent), ForKind.SERIAL, partialNest);
    }

    const combineStore = new BufferStoreNode(acc, spatialIdx.map(cloneExprTree),
      new MathOpNode(op, new BufferLoadNode(acc, spatialIdx.map(cloneExprTree)),
        new BufferLoadNode(partialBuf, cfIdx(piVar))));
    const combineInit = new BufferStoreNode(acc, spatialIdx.map(cloneExprTree), cloneExprTree(initVal));
    const combineBlock = new BlockNode(`${blockName}_rf_c`, [],
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

  decomposeReduction(blockName) {
    const block = this.getBlock(blockName);
    if (!block.initBody) throw new Error(`decomposeReduction: block '${blockName}' has no initBody`);
    const loops = this.getLoops(blockName);
    const store = block.body;
    if (!store || store.type !== 'BufferStoreNode') throw new Error(`decomposeReduction: block '${blockName}' body is not a store`);
    const acc = store.buffer;

    const spatialVars = new Set();
    for (const idx of store.indices) collectVarsUsed(idx, spatialVars);
    const spatialLoops = loops.filter(l => spatialVars.has(l.loopVar.name));
    const reductionLoops = loops.filter(l => !spatialVars.has(l.loopVar.name));
    if (reductionLoops.length === 0) throw new Error(`decomposeReduction: block '${blockName}' has no reduction loop`);

    const initStore = new BufferStoreNode(acc, store.indices.map(cloneExprTree), cloneExprTree(block.initBody.value));
    const initBlock = new BlockNode(`${blockName}_init`, [], [], [{ buffer: acc }], initStore);
    let initNest = initBlock;
    for (let i = spatialLoops.length - 1; i >= 0; i--) {
      initNest = new ForNode(spatialLoops[i].loopVar, new IntImmNode(0), cloneExprTree(spatialLoops[i].extent), ForKind.SERIAL, initNest);
    }

    const updBlock = new BlockNode(`${blockName}_upd`, [], block.reads.map(r => ({ buffer: r.buffer })), [{ buffer: acc }], cloneExprTree(store));
    let updNest = updBlock;
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

  cacheWrite(blockName, bufferName, scope = 'local') {
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
    const backBlock = new BlockNode(`${cache.name}_flush`, idxVars.map(v => ({ iterVar: v, binding: v })),
      [{ buffer: cache }], [{ buffer: buf }], backStore);
    let backNest = backBlock;
    for (let d = buf.shape.length - 1; d >= 0; d--) {
      backNest = new ForNode(idxVars[d], new IntImmNode(0), new IntImmNode(buf.shape[d]), ForKind.SERIAL, backNest);
    }

    const blockNest = loops[0];
    const seq = new SeqNode([]);
    const alloc = new AllocateNode(cache, scope, seq);
    this.mutator.replaceNode(blockNest, alloc);
    seq.stmts.push(blockNest, backNest);
    this._replaceInTree(blockNest, alloc);
    if (!this._replaying) this.trace.record('cacheWrite', [blockName, bufferName, scope]);
  }

  setScope(blockName, bufferName, scope) {
    const block = this.getBlock(blockName);
    const writeEntry = (block.writes || []).find(w => w.buffer && w.buffer.name === bufferName);
    if (!writeEntry) throw new Error(`setScope: block '${blockName}' does not write '${bufferName}'`);
    writeEntry.buffer.scope = scope;
    this.state.invalidate();
    if (!this._replaying) this.trace.record('setScope', [blockName, bufferName, scope]);
  }

  storageAlign(blockName, bufferName, axis, factor, offset) {
    const block = this.getBlock(blockName);
    const entry = [...(block.writes || []), ...(block.reads || [])].find(e => e.buffer && e.buffer.name === bufferName);
    if (!entry) throw new Error(`storageAlign: block '${blockName}' does not access '${bufferName}'`);
    if (!Number.isInteger(factor) || factor <= 0) throw new Error('storageAlign: factor must be a positive integer');
    entry.buffer.storageAlign = { axis, factor, offset: offset || 0 };
    this.state.invalidate();
    if (!this._replaying) this.trace.record('storageAlign', [blockName, bufferName, axis, factor, offset || 0]);
  }

  computeInline(producerName) {
    const prod = this.getBlock(producerName);
    if (!prod) throw new Error(`computeInline: block '${producerName}' not found`);
    if (prod.initBody) throw new Error('computeInline: cannot inline a reduction block (has init)');

    let store = null;
    const findStore = (n) => {
      if (!n || store) return;
      if (n.type === 'BufferStoreNode') { store = n; return; }
      if (n.type === 'SeqNode') { for (const s of n.stmts) findStore(s); return; }
      if (n.type === 'BlockNode' || n.type === 'ForNode') findStore(n.body);
    };
    findStore(prod.body);
    if (!store) throw new Error('computeInline: producer has no single store to inline');

    const B = store.buffer;
    const pIdxNames = store.indices.map((ix) => (ix && ix.type === 'VariableNode') ? ix.name : null);
    if (pIdxNames.some((n) => n === null)) throw new Error('computeInline: producer indices must be simple loop variables');
    if (loadsBuffer(store.value, B.name)) throw new Error('computeInline: producer is self-referential (recurrence), cannot inline');

    const usesBInIndex = (node) => {
      if (!node || typeof node !== 'object') return false;
      if ((node.type === 'BufferLoadNode' || node.type === 'BufferStoreNode') && node.indices) {
        for (const ix of node.indices) if (loadsBuffer(ix, B.name)) return true;
      }
      for (const k of ['a', 'b', 'expr', 'value', 'condition', 'thenBody', 'elseBody', 'body', 'initBody']) {
        if (node[k] && usesBInIndex(node[k])) return true;
      }
      if (node.args) for (const a of node.args) if (usesBInIndex(a)) return true;
      if (node.indices) for (const ix of node.indices) if (usesBInIndex(ix)) return true;
      if (node.stmts) for (const s of node.stmts) if (usesBInIndex(s)) return true;
      return false;
    };
    if (usesBInIndex(this.func.body)) throw new Error(`computeInline: buffer '${B.name}' is used inside an index expression (indirect), cannot safely inline`);

    const valExpr = store.value;
    const counter = { n: 0 };
    const tmpNames = pIdxNames.map((_, k) => `__inl_${producerName}_${k}`);
    const makeRepl = (load) => {
      let expr = cloneExprTree(valExpr);
      for (let k = 0; k < pIdxNames.length; k++) {
        const tn = tmpNames[k];
        expr = substituteVar(expr, pIdxNames[k], () => new VariableNode(tn, 'int32'));
      }
      for (let k = 0; k < pIdxNames.length; k++) {
        const idx = load.indices[k];
        expr = substituteVar(expr, tmpNames[k], () => cloneExprTree(idx));
      }
      return expr;
    };
    replaceBufferLoads(this.func.body, B.name, makeRepl, counter);
    if (counter.n === 0) throw new Error(`computeInline: buffer '${B.name}' has no consumers to inline into`);

    const loops = this.getLoops(producerName);
    const root = loops.length > 0 ? loops[0] : prod;
    const inlineEmpty = new SeqNode([]);
    this.mutator.replaceNode(root, inlineEmpty);
    this._replaceInTree(root, inlineEmpty);
    if (!this._replaying) this.trace.record('computeInline', [producerName]);
  }

  _relocateBlockToLoop(blockName, targetLoopRef, atStart) {
    const blk = this.getBlock(blockName);
    if (!blk) throw new Error(`computeAt: block '${blockName}' not found`);
    const targetLoop = this._resolveLoop(targetLoopRef);
    if (!targetLoop || targetLoop.type !== 'ForNode') throw new Error('computeAt: target must be a loop');

    const blkLoops = this.getLoops(blockName);
    if (blkLoops.length !== 1) throw new Error('computeAt: aligned case requires exactly one enclosing loop on the moved block');
    const blkLoop = blkLoops[0];
    if (blkLoop === targetLoop) throw new Error('computeAt: block already at target loop');

    const bExt = getConstExtent(blkLoop.extent);
    const tExt = getConstExtent(targetLoop.extent);
    if (bExt === null || tExt === null || bExt !== tExt) {
      throw new Error('computeAt: aligned case requires equal static extent on the block and target loops');
    }

    const moved = cloneExprTree(blkLoop.body);
    substituteVar(moved, blkLoop.loopVar.name, () => targetLoop.loopVar);
    const relocEmpty = new SeqNode([]);
    this.mutator.replaceNode(blkLoop, relocEmpty);

    const tbody = targetLoop.body;
    if (tbody && tbody.type === 'SeqNode') {
      if (atStart) tbody.stmts.unshift(moved); else tbody.stmts.push(moved);
    } else {
      const seq = atStart ? new SeqNode([moved, tbody]) : new SeqNode([tbody, moved]);
      targetLoop.body = seq;
      targetLoop._setChild('body', seq);
    }
    this._replaceInTree(blkLoop, relocEmpty);
    this._replaceInTree(targetLoop, targetLoop);
    return targetLoop.loopVar.name;
  }

  computeAt(producerName, targetLoopRef) {
    const targetVar = this._relocateBlockToLoop(producerName, targetLoopRef, true);
    if (!this._replaying) this.trace.record('computeAt', [producerName, targetVar]);
  }

  reverseComputeAt(consumerName, targetLoopRef) {
    const targetVar = this._relocateBlockToLoop(consumerName, targetLoopRef, false);
    if (!this._replaying) this.trace.record('reverseComputeAt', [consumerName, targetVar]);
  }

  cacheRead(blockName, bufferName, scope = 'local') {
    const block = this.getBlock(blockName);
    const loops = this.getLoops(blockName);
    if (loops.length === 0) throw new Error('cacheRead: block has no enclosing loops');
    const readEntry = (block.reads || []).find(r => r.buffer && r.buffer.name === bufferName);
    if (!readEntry) throw new Error(`cacheRead: block '${blockName}' does not read '${bufferName}'`);
    const buf = readEntry.buffer;
    const cache = new Buffer(`${bufferName}_${blockName}_cache`, [...buf.shape], buf.dtype, scope);

    const idxVars = buf.shape.map((_, d) => new VariableNode(`${cache.name}_i${d}`, 'int32'));
    const copyStore = new BufferStoreNode(cache, idxVars, new BufferLoadNode(buf, idxVars));
    const copyBlock = new BlockNode(`${cache.name}_fill`, idxVars.map(v => ({ iterVar: v, binding: v })),
      [{ buffer: buf }], [{ buffer: cache }], copyStore);
    let copyNest = copyBlock;
    for (let d = buf.shape.length - 1; d >= 0; d--) {
      copyNest = new ForNode(idxVars[d], new IntImmNode(0), new IntImmNode(buf.shape[d]), ForKind.SERIAL, copyNest);
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

  fuseConsumer(producerBlockName, consumerBlockName) {
    const pBlock = this.getBlock(producerBlockName);
    const cBlock = this.getBlock(consumerBlockName);
    const pLoops = this.getLoops(producerBlockName);
    const cLoops = this.getLoops(consumerBlockName);
    if (!pBlock.body || pBlock.body.type !== 'BufferStoreNode') {
      throw new Error(`fuseConsumer: producer '${producerBlockName}' body is not a store`);
    }
    const pSpatialVars = new Set();
    for (const idx of pBlock.body.indices) collectVarsUsed(idx, pSpatialVars);
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

    let fused = new SeqNode([reductionSubNest, cFusedBlock]);
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

  annotate(loop, key, value) {
    loop = this._resolveLoop(loop);
    if (loop.type !== 'ForNode') throw new Error('annotate expects ForNode');
    if (!loop.annotations) loop.annotations = {};
    loop.annotations[key] = value;
    this.state.invalidate();
    if (!this._replaying) {
      this.trace.record('annotate', [loop.loopVar.name, key, value]);
    }
  }

  tensorize(intrinName, info) {
    if (typeof intrinName !== 'string') throw new Error('tensorize expects an intrinsic name');
    if (!info || typeof info.M !== 'number' || typeof info.N !== 'number' || typeof info.K !== 'number') {
      throw new Error('tensorize expects info { M, N, K, a, b, c }');
    }
    this.func._tensorIntrin = { name: intrinName, info };
    this.state.invalidate();
  }

  blockize(loopRef, name = null) {
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

  getTrace() {
    return this.trace;
  }

  verify() {
    return ScheduleValidator.validate(this.func);
  }
}

function collectBufferAccess(node, reads, writes) {
  if (!node || typeof node !== 'object') return;
  if (node.type === 'BufferLoadNode' && node.buffer) reads.set(node.buffer.name, node.buffer);
  if (node.type === 'BufferStoreNode' && node.buffer) writes.set(node.buffer.name, node.buffer);
  for (const k of ['a', 'b', 'value', 'expr', 'condition', 'thenBody', 'elseBody', 'body', 'initBody']) {
    if (node[k]) collectBufferAccess(node[k], reads, writes);
  }
  if (node.indices) for (const ix of node.indices) collectBufferAccess(ix, reads, writes);
  if (node.args) for (const a of node.args) collectBufferAccess(a, reads, writes);
  if (node.stmts) for (const s of node.stmts) collectBufferAccess(s, reads, writes);
}
