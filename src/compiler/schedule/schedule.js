import {
  ForNode, BlockNode, SeqNode, BufferStoreNode, BufferLoadNode,
  VariableNode, IntImmNode, MathOpNode, ForKind, IfThenElseNode
} from '../ir/tensor/nodes.js';
import { Buffer } from '../ir/tensor/buffer.js';
import { ScheduleTrace } from './trace.js';
import { ScheduleValidator } from './validator.js';
import { ScheduleState } from './schedule_state.js';
import { SRefTree } from './sref.js';
import { loopCarriesReduction } from './legality.js';

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

  cacheRead(blockName, bufferName, cacheScope) {
    const block = this.getBlock(blockName);
    const readEntry = block.reads.find(r => r.buffer.name === bufferName);
    if (!readEntry) {
      throw new Error(`Block '${blockName}' does not read buffer '${bufferName}'`);
    }

    const origBuf = readEntry.buffer;
    const cacheBuf = new Buffer(
      `${origBuf.name}_${cacheScope}`,
      [...origBuf.shape],
      origBuf.dtype,
      cacheScope
    );

    const loops = this.getLoops(blockName);
    const loopVars = [];
    for (let i = 0; i < origBuf.shape.length; i++) {
      loopVars.push(freshVar(`cache_i${i}`));
    }

    const indices = loopVars.map(v => v);
    const load = new BufferLoadNode(origBuf, indices);
    const store = new BufferStoreNode(cacheBuf, indices, load);
    const cacheBlock = new BlockNode(
      `${bufferName}_${cacheScope}_cache`,
      [],
      [{ buffer: origBuf }],
      [{ buffer: cacheBuf }],
      store
    );

    let cacheBody = cacheBlock;
    for (let i = origBuf.shape.length - 1; i >= 0; i--) {
      cacheBody = new ForNode(
        loopVars[i],
        new IntImmNode(0),
        new IntImmNode(origBuf.shape[i]),
        ForKind.SERIAL,
        cacheBody
      );
    }

    substituteBufferInBlock(block, origBuf, cacheBuf);
    readEntry.buffer = cacheBuf;

    const outerLoop = loops.length > 0 ? loops[0] : null;
    if (outerLoop) {
      const seq = new SeqNode([cacheBody, outerLoop]);
      this._replaceNode(outerLoop, seq);
    }

    this._rebuildSRefTree();
    if (!this._replaying) {
      this.trace.record('cacheRead', [blockName, bufferName, cacheScope]);
    }

    return cacheBuf;
  }

  cacheWrite(blockName, bufferName, cacheScope) {
    const block = this.getBlock(blockName);
    const writeEntry = block.writes.find(r => r.buffer.name === bufferName);
    if (!writeEntry) {
      throw new Error(`Block '${blockName}' does not write buffer '${bufferName}'`);
    }

    const origBuf = writeEntry.buffer;
    const cacheBuf = new Buffer(
      `${origBuf.name}_${cacheScope}`,
      [...origBuf.shape],
      origBuf.dtype,
      cacheScope
    );

    const loops = this.getLoops(blockName);
    const loopVars = [];
    for (let i = 0; i < origBuf.shape.length; i++) {
      loopVars.push(freshVar(`wb_i${i}`));
    }

    const indices = loopVars.map(v => v);
    const load = new BufferLoadNode(cacheBuf, indices);
    const store = new BufferStoreNode(origBuf, indices, load);
    const writebackBlock = new BlockNode(
      `${bufferName}_${cacheScope}_writeback`,
      [],
      [{ buffer: cacheBuf }],
      [{ buffer: origBuf }],
      store
    );

    let writebackBody = writebackBlock;
    for (let i = origBuf.shape.length - 1; i >= 0; i--) {
      writebackBody = new ForNode(
        loopVars[i],
        new IntImmNode(0),
        new IntImmNode(origBuf.shape[i]),
        ForKind.SERIAL,
        writebackBody
      );
    }

    substituteBufferInBlock(block, origBuf, cacheBuf);
    writeEntry.buffer = cacheBuf;

    const outerLoop = loops.length > 0 ? loops[0] : null;
    if (outerLoop) {
      const seq = new SeqNode([outerLoop, writebackBody]);
      this._replaceNode(outerLoop, seq);
    }

    this._rebuildSRefTree();
    if (!this._replaying) {
      this.trace.record('cacheWrite', [blockName, bufferName, cacheScope]);
    }

    return cacheBuf;
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
}

function substituteBufferInBlock(block, oldBuf, newBuf) {
  const visit = (node) => {
    if (!node || typeof node !== 'object') return;
    if (node.type === 'BufferLoadNode' && node.buffer === oldBuf) node.buffer = newBuf;
    if (node.type === 'BufferStoreNode' && node.buffer === oldBuf) node.buffer = newBuf;
    if (node.type === 'BufferLoadNode' || node.type === 'BufferStoreNode') {
      for (const idx of node.indices) visit(idx);
      if (node.value) visit(node.value);
      return;
    }
    if (node.a) visit(node.a);
    if (node.b) visit(node.b);
    if (node.expr) visit(node.expr);
    if (node.args) for (const a of node.args) visit(a);
    if (node.body) visit(node.body);
    if (node.initBody) visit(node.initBody);
    if (node.stmts) for (const s of node.stmts) visit(s);
    if (node.condition) visit(node.condition);
    if (node.thenBody) visit(node.thenBody);
    if (node.elseBody) visit(node.elseBody);
  };
  visit(block.body);
  if (block.initBody) visit(block.initBody);
}
