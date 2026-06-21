import {
  ForNode, BlockNode, SeqNode, BufferStoreNode, BufferLoadNode,
  VariableNode, IntImmNode, MathOpNode, ForKind, IfThenElseNode, AllocateNode
} from '../ir/tensor/nodes.js';
import { Buffer } from '../ir/tensor/buffer.js';
import { ScheduleTrace } from './trace.js';
import { ScheduleValidator } from './validator.js';
import { ScheduleState } from './schedule_state.js';
import { SRefTree } from './sref.js';
import { loopCarriesReduction, collectVarsUsed } from './legality.js';

const RFACTOR_ASSOCIATIVE_OPS = new Set(['+', '*', 'min', 'max']);

function substituteVar(node, oldName, exprFactory) {
  if (!node || typeof node !== 'object') return node;
  if (node.type === 'VariableNode' && node.name === oldName) return exprFactory();

  switch (node.type) {
    case 'MathOpNode':
      node.a = substituteVar(node.a, oldName, exprFactory);
      if (node.b) node.b = substituteVar(node.b, oldName, exprFactory);
      break;
    case 'CompareNode':
      node.a = substituteVar(node.a, oldName, exprFactory);
      node.b = substituteVar(node.b, oldName, exprFactory);
      break;
    case 'BufferLoadNode':
      for (let i = 0; i < node.indices.length; i++)
        node.indices[i] = substituteVar(node.indices[i], oldName, exprFactory);
      break;
    case 'BufferStoreNode':
      for (let i = 0; i < node.indices.length; i++)
        node.indices[i] = substituteVar(node.indices[i], oldName, exprFactory);
      node.value = substituteVar(node.value, oldName, exprFactory);
      break;
    case 'CallExternNode':
      for (let i = 0; i < node.args.length; i++)
        node.args[i] = substituteVar(node.args[i], oldName, exprFactory);
      break;
    case 'CastNode':
      node.expr = substituteVar(node.expr, oldName, exprFactory);
      break;
    case 'IfThenElseNode':
      node.condition = substituteVar(node.condition, oldName, exprFactory);
      node.thenBody = substituteVar(node.thenBody, oldName, exprFactory);
      if (node.elseBody) node.elseBody = substituteVar(node.elseBody, oldName, exprFactory);
      break;
    case 'ForNode':
      node.body = substituteVar(node.body, oldName, exprFactory);
      break;
    case 'BlockNode':
      node.body = substituteVar(node.body, oldName, exprFactory);
      if (node.initBody) node.initBody = substituteVar(node.initBody, oldName, exprFactory);
      for (let i = 0; i < node.iterVars.length; i++) {
        if (node.iterVars[i].binding)
          node.iterVars[i].binding = substituteVar(node.iterVars[i].binding, oldName, exprFactory);
      }
      break;
    case 'SeqNode':
      for (let i = 0; i < node.stmts.length; i++)
        node.stmts[i] = substituteVar(node.stmts[i], oldName, exprFactory);
      break;
    case 'LetStmtNode':
      node.value = substituteVar(node.value, oldName, exprFactory);
      node.body = substituteVar(node.body, oldName, exprFactory);
      break;
    case 'BlockRealizeNode':
      if (node.binding) node.binding = substituteVar(node.binding, oldName, exprFactory);
      break;
  }
  return node;
}

function cloneExprTree(node) {
  if (!node || typeof node !== 'object') return node;
  if (Array.isArray(node)) return node.map(cloneExprTree);
  const copy = Object.create(Object.getPrototypeOf(node));
  copy.type = node.type;
  copy._parent = null;
  copy._parentKey = null;
  copy._parentIdx = -1;
  switch (node.type) {
    case 'ForNode':
      copy.loopVar = node.loopVar; copy.min = cloneExprTree(node.min);
      copy.extent = cloneExprTree(node.extent); copy.kind = node.kind;
      copy.body = cloneExprTree(node.body); copy.threadTag = node.threadTag;
      copy._setChild('body', copy.body);
      break;
    case 'BlockNode':
      copy.name = node.name; copy.iterVars = node.iterVars.map(cloneExprTree);
      copy.reads = node.reads; copy.writes = node.writes;
      copy.body = cloneExprTree(node.body);
      copy.initBody = node.initBody ? cloneExprTree(node.initBody) : null;
      copy._setChild('body', copy.body);
      copy._setChild('initBody', copy.initBody);
      break;
    case 'SeqNode':
      copy.stmts = node.stmts.map(cloneExprTree);
      copy._setChildren('stmts', copy.stmts);
      break;
    case 'IfThenElseNode':
      copy.condition = cloneExprTree(node.condition);
      copy.thenBody = cloneExprTree(node.thenBody);
      copy.elseBody = node.elseBody ? cloneExprTree(node.elseBody) : null;
      copy._setChild('thenBody', copy.thenBody);
      copy._setChild('elseBody', copy.elseBody);
      break;
    case 'BufferStoreNode':
      copy.buffer = node.buffer;
      copy.indices = node.indices.map(cloneExprTree);
      copy.value = cloneExprTree(node.value);
      break;
    case 'BufferLoadNode':
      copy.buffer = node.buffer;
      copy.indices = node.indices.map(cloneExprTree);
      break;
    case 'BlockRealizeNode':
      copy.iterVar = node.iterVar;
      copy.binding = cloneExprTree(node.binding);
      break;
    case 'MathOpNode':
      copy.op = node.op; copy.a = cloneExprTree(node.a); copy.b = cloneExprTree(node.b);
      break;
    case 'CompareNode':
      copy.direction = node.direction; copy.a = cloneExprTree(node.a); copy.b = cloneExprTree(node.b);
      break;
    case 'CastNode':
      copy.expr = cloneExprTree(node.expr); copy.fromDtype = node.fromDtype; copy.toDtype = node.toDtype;
      break;
    case 'CallExternNode':
      copy.externName = node.externName; copy.args = node.args.map(cloneExprTree); copy.dtype = node.dtype;
      break;
    default:
      for (const key of Object.keys(node)) {
        if (key === '_parent' || key === '_parentKey' || key === '_parentIdx') continue;
        copy[key] = node[key];
      }
      break;
  }
  return copy;
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
  }

  _rebuildSRefTree() {
    this._srefTree.rebuildFrom(this.func.body);
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

  split(loop, factor) {
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

    this._replaceNode(loop, outerLoop);
    this._rebuildSRefTree();

    if (!this._replaying) {
      this.trace.record('split', [loop.loopVar.name, factor]);
    }

    return [outerLoop, innerLoop];
  }

  reorder(...newOrder) {
    if (newOrder.length < 2) return;

    for (const loop of newOrder) {
      if (loop.type !== 'ForNode') {
        throw new Error('reorder expects ForNode arguments');
      }
    }

    const loopSet = new Set(newOrder);
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
      }
      if (node.type === 'BlockNode') findDepths(node.body, depth);
    };
    findDepths(this.func.body, 0);

    const sorted = [...newOrder].sort((a, b) => depthMap.get(a) - depthMap.get(b));
    const innermostOriginal = sorted[sorted.length - 1];
    const innermostBody = innermostOriginal.body;

    const topmostSRef = this._srefTree.getSRef(topmostLoop);
    const topmostParent = topmostSRef ? topmostSRef.parent : null;

    this._replaceNode(topmostLoop, newOrder[0]);

    for (const loop of newOrder) {
      if (loop === newOrder[0]) continue;
      loop._parent = null;
      loop._parentKey = null;
      loop._parentIdx = -1;
    }

    for (let i = 0; i < newOrder.length; i++) {
      const child = i < newOrder.length - 1 ? newOrder[i + 1] : innermostBody;
      newOrder[i].body = child;
      newOrder[i]._setChild('body', child);
    }

    this._rebuildSRefTree();

    if (!this._replaying) {
      this.trace.record('reorder', [newOrder.map(l => l.loopVar.name)]);
    }
  }

  fuseLoops(outer, inner) {
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
      new MathOpNode('/', fusedVar, new IntImmNode(innerExtent))
    );
    substituteVar(fusedLoop.body, innerName, () =>
      new MathOpNode('%', fusedVar, new IntImmNode(innerExtent))
    );

    this._replaceNode(outer, fusedLoop);
    this._srefTree.replaceNode(outer, fusedLoop);
    this.state.invalidate();

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
    if (loop.type !== 'ForNode') throw new Error('vectorize expects ForNode');
    const extent = getConstExtent(loop.extent);
    if (extent === null) throw new Error('Cannot vectorize loop with non-constant extent');
    const reductionBlock = loopCarriesReduction(loop);
    if (reductionBlock !== null) {
      throw new Error(`Cannot vectorize reduction loop '${loop.loopVar.name}' (loop-carried dependency in block '${reductionBlock}')`);
    }
    loop.kind = ForKind.VECTORIZED;
    if (!this._replaying) {
      this.trace.record('vectorize', [loop.loopVar.name]);
    }
  }

  unroll(loop) {
    if (loop.type !== 'ForNode') throw new Error('unroll expects ForNode');
    loop.kind = ForKind.UNROLLED;
    if (!this._replaying) {
      this.trace.record('unroll', [loop.loopVar.name]);
    }
  }

  parallelize(loop) {
    if (loop.type !== 'ForNode') throw new Error('parallelize expects ForNode');
    const reductionBlock = loopCarriesReduction(loop);
    if (reductionBlock !== null) {
      throw new Error(`Cannot parallelize reduction loop '${loop.loopVar.name}' (loop-carried dependency in block '${reductionBlock}')`);
    }
    loop.kind = ForKind.PARALLEL;
    if (!this._replaying) {
      this.trace.record('parallelize', [loop.loopVar.name]);
    }
  }

  bindThread(loop, threadTag) {
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

    this._replaceNode(loops[0], new SeqNode([partialNest, combineNest]));
    this._rebuildSRefTree();
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

    this._replaceNode(loops[0], new SeqNode([initNest, updNest]));
    this._rebuildSRefTree();
    if (!this._replaying) {
      this.trace.record('decomposeReduction', [blockName]);
    }
  }

  _redirectReads(node, fromBuf, toBuf) {
    if (!node || typeof node !== 'object') return;
    if (node.type === 'BufferLoadNode' && node.buffer === fromBuf) node.buffer = toBuf;
    for (const key of ['a', 'b', 'expr', 'value', 'condition', 'thenBody', 'elseBody', 'body', 'initBody', 'min', 'extent']) {
      if (node[key]) this._redirectReads(node[key], fromBuf, toBuf);
    }
    if (Array.isArray(node.stmts)) for (const s of node.stmts) this._redirectReads(s, fromBuf, toBuf);
    if (Array.isArray(node.indices)) for (const i of node.indices) this._redirectReads(i, fromBuf, toBuf);
    if (Array.isArray(node.args)) for (const a of node.args) this._redirectReads(a, fromBuf, toBuf);
  }

  _redirectBuffer(node, fromBuf, toBuf) {
    if (!node || typeof node !== 'object') return;
    if ((node.type === 'BufferLoadNode' || node.type === 'BufferStoreNode') && node.buffer === fromBuf) node.buffer = toBuf;
    for (const key of ['a', 'b', 'expr', 'value', 'condition', 'thenBody', 'elseBody', 'body', 'initBody', 'min', 'extent']) {
      if (node[key]) this._redirectBuffer(node[key], fromBuf, toBuf);
    }
    if (Array.isArray(node.stmts)) for (const s of node.stmts) this._redirectBuffer(s, fromBuf, toBuf);
    if (Array.isArray(node.indices)) for (const i of node.indices) this._redirectBuffer(i, fromBuf, toBuf);
    if (Array.isArray(node.args)) for (const a of node.args) this._redirectBuffer(a, fromBuf, toBuf);
  }

  cacheWrite(blockName, bufferName, scope = 'local') {
    const block = this.getBlock(blockName);
    const loops = this.getLoops(blockName);
    if (loops.length === 0) throw new Error('cacheWrite: block has no enclosing loops');
    const writeEntry = (block.writes || []).find(w => w.buffer && w.buffer.name === bufferName);
    if (!writeEntry) throw new Error(`cacheWrite: block '${blockName}' does not write '${bufferName}'`);
    const buf = writeEntry.buffer;
    const cache = new Buffer(`${bufferName}_${blockName}_cachew`, [...buf.shape], buf.dtype, scope);

    this._redirectBuffer(block.body, buf, cache);
    if (block.initBody) this._redirectBuffer(block.initBody, buf, cache);
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
    this._replaceNode(blockNest, alloc);
    seq.stmts.push(blockNest, backNest);
    this._rebuildSRefTree();
    if (!this._replaying) this.trace.record('cacheWrite', [blockName, bufferName, scope]);
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

    this._redirectReads(block.body, buf, cache);
    if (block.initBody) this._redirectReads(block.initBody, buf, cache);
    readEntry.buffer = cache;

    const blockNest = loops[0];
    const seq = new SeqNode([copyNest]);
    const alloc = new AllocateNode(cache, scope, seq);
    this._replaceNode(blockNest, alloc);
    seq.stmts.push(blockNest);
    this._rebuildSRefTree();
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

    this._replaceNode(pLoops[0], fused);
    this._removeNode(cLoops[0]);
    this._rebuildSRefTree();
    if (!this._replaying) {
      this.trace.record('fuseConsumer', [producerBlockName, consumerBlockName]);
    }
  }

  annotate(loop, key, value) {
    if (loop.type !== 'ForNode') throw new Error('annotate expects ForNode');
    if (!loop.annotations) loop.annotations = {};
    loop.annotations[key] = value;
    if (!this._replaying) {
      this.trace.record('annotate', [loop.loopVar.name, key, value]);
    }
  }

  getTrace() {
    return this.trace;
  }

  verify() {
    return ScheduleValidator.validate(this.func);
  }

  _replaceNode(oldNode, newNode) {
    if (oldNode._parent) {
      oldNode.replaceWith(newNode);
      return;
    }
    if (this.func.body === oldNode || this.func.body === undefined) {
      this.func.body = newNode;
      if (this.func._setChild) this.func._setChild('body', newNode);
    }
  }

  _removeNode(node) {
    const parent = node._parent;
    if (parent && parent.type === 'SeqNode' && Array.isArray(parent.stmts)) {
      const i = parent.stmts.indexOf(node);
      if (i >= 0) {
        parent.stmts.splice(i, 1);
        if (parent._setChildren) parent._setChildren('stmts', parent.stmts);
        return;
      }
    }
    throw new Error('_removeNode: node parent is not a SeqNode; cannot remove without duplicating it');
  }
}
