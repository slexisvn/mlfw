import {
  PrimFunc, ForNode, BlockNode, SeqNode, BufferStoreNode, BufferLoadNode,
  VariableNode, IntImmNode, MathOpNode, AllocateNode, ForKind, LetStmtNode,
  IfThenElseNode
} from '../ir/tensor/nodes.js';
import { Buffer } from '../ir/tensor/buffer.js';
import { MemoryScope } from '../ir/tensor/tensor_types.js';
import { ScheduleTrace } from './trace.js';
import { ScheduleValidator } from './validator.js';
import { ScheduleState } from './schedule_state.js';

function collectBlocks(node, result = new Map()) {
  if (!node) return result;
  if (node instanceof BlockNode || node.type === 'BlockNode') {
    result.set(node.name, node);
  }
  if (node.body) collectBlocks(node.body, result);
  if (node.stmts) {
    for (const s of node.stmts) collectBlocks(s, result);
  }
  if (node.thenBody) collectBlocks(node.thenBody, result);
  if (node.elseBody) collectBlocks(node.elseBody, result);
  if (node.initBody) collectBlocks(node.initBody, result);
  return result;
}

function findLoopsForBlock(root, blockName) {
  const result = [];
  const search = (node, loopStack) => {
    if (!node) return false;
    if (node.type === 'ForNode') {
      loopStack.push(node);
      if (search(node.body, loopStack)) return true;
      loopStack.pop();
      return false;
    }
    if (node.type === 'BlockNode' && node.name === blockName) {
      result.push(...loopStack);
      return true;
    }
    if (node.type === 'SeqNode') {
      for (const s of node.stmts) {
        if (search(s, [...loopStack])) return true;
      }
    }
    if (node.type === 'IfThenElseNode') {
      if (search(node.thenBody, [...loopStack])) return true;
      if (search(node.elseBody, [...loopStack])) return true;
    }
    return false;
  };
  search(root, []);
  return result;
}

function findParent(root, target) {
  const search = (node) => {
    if (!node) return null;
    for (const key of Object.keys(node)) {
      const child = node[key];
      if (child === target) return { parent: node, key, index: -1 };
      if (Array.isArray(child)) {
        for (let i = 0; i < child.length; i++) {
          if (child[i] === target) return { parent: node, key, index: i };
          if (typeof child[i] === 'object' && child[i] !== null) {
            const r = search(child[i]);
            if (r) return r;
          }
        }
      } else if (typeof child === 'object' && child !== null && child.type) {
        const r = search(child);
        if (r) return r;
      }
    }
    return null;
  };
  return search(root);
}

function replaceInParent(root, oldNode, newNode) {
  if (root === oldNode) return newNode;
  const ref = findParent(root, oldNode);
  if (!ref) throw new Error('Cannot find node in tree for replacement');
  if (ref.index >= 0) {
    ref.parent[ref.key][ref.index] = newNode;
  } else {
    ref.parent[ref.key] = newNode;
  }
  return root;
}

function substituteVar(node, oldName, exprFactory) {
  if (!node || typeof node !== 'object') return node;
  if (node.type === 'VariableNode' && node.name === oldName) {
    return exprFactory();
  }
  for (const key of Object.keys(node)) {
    const child = node[key];
    if (Array.isArray(child)) {
      for (let i = 0; i < child.length; i++) {
        if (typeof child[i] === 'object' && child[i] !== null) {
          const replaced = substituteVar(child[i], oldName, exprFactory);
          if (replaced !== child[i]) child[i] = replaced;
        }
      }
    } else if (typeof child === 'object' && child !== null && child.type) {
      const replaced = substituteVar(child, oldName, exprFactory);
      if (replaced !== child) node[key] = replaced;
    }
  }
  return node;
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
    this._blockCache = null;
  }

  _invalidateCache() {
    this._blockCache = null;
    this.state.invalidate();
  }

  _blocks() {
    if (!this._blockCache) {
      this._blockCache = collectBlocks(this.func.body);
    }
    return this._blockCache;
  }

  getBlock(name) {
    const b = this._blocks().get(name);
    if (!b) throw new Error(`Block '${name}' not found`);
    return b;
  }

  getLoops(blockName) {
    return findLoopsForBlock(this.func.body, blockName);
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

    const innerLoop = new ForNode(
      innerVar,
      new IntImmNode(0),
      new IntImmNode(factor),
      loop.kind,
      loop.body,
      loop.threadTag
    );

    const needsGuard = extent % factor !== 0;
    if (needsGuard) {
      const flatIdx = new MathOpNode('+',
        new MathOpNode('*', outerVar, new IntImmNode(factor)),
        innerVar
      );
      const guard = new MathOpNode('<', flatIdx, new IntImmNode(extent));
      innerLoop.body = new IfThenElseNode(guard, innerLoop.body);
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

    replaceInParent(this.func, loop, outerLoop);
    this._invalidateCache();

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

    const topmostParent = findParent(this.func, topmostLoop);

    for (let i = 0; i < newOrder.length; i++) {
      if (i < newOrder.length - 1) {
        newOrder[i].body = newOrder[i + 1];
      } else {
        newOrder[i].body = innermostBody;
      }
    }

    if (topmostParent) {
      if (topmostParent.index >= 0) {
        topmostParent.parent[topmostParent.key][topmostParent.index] = newOrder[0];
      } else {
        topmostParent.parent[topmostParent.key] = newOrder[0];
      }
    } else {
      this.func.body = newOrder[0];
    }

    this._invalidateCache();

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

    replaceInParent(this.func, outer, fusedLoop);
    this._invalidateCache();

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
    loop.kind = ForKind.VECTORIZED;
    this._invalidateCache();
    if (!this._replaying) {
      this.trace.record('vectorize', [loop.loopVar.name]);
    }
  }

  unroll(loop) {
    if (loop.type !== 'ForNode') throw new Error('unroll expects ForNode');
    loop.kind = ForKind.UNROLLED;
    this._invalidateCache();
    if (!this._replaying) {
      this.trace.record('unroll', [loop.loopVar.name]);
    }
  }

  parallelize(loop) {
    if (loop.type !== 'ForNode') throw new Error('parallelize expects ForNode');
    loop.kind = ForKind.PARALLEL;
    this._invalidateCache();
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
    this._invalidateCache();
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
    const loopBinds = [];
    for (let i = 0; i < origBuf.shape.length; i++) {
      const v = freshVar(`cache_i${i}`);
      loopVars.push(v);
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
      const parentRef = findParent(this.func, outerLoop);
      const seq = new SeqNode([cacheBody, outerLoop]);
      if (parentRef) {
        if (parentRef.index >= 0) {
          parentRef.parent[parentRef.key][parentRef.index] = seq;
        } else {
          parentRef.parent[parentRef.key] = seq;
        }
      } else {
        this.func.body = seq;
      }
    }

    this._invalidateCache();
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
      const parentRef = findParent(this.func, outerLoop);
      const seq = new SeqNode([outerLoop, writebackBody]);
      if (parentRef) {
        if (parentRef.index >= 0) {
          parentRef.parent[parentRef.key][parentRef.index] = seq;
        } else {
          parentRef.parent[parentRef.key] = seq;
        }
      } else {
        this.func.body = seq;
      }
    }

    this._invalidateCache();
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
}

function substituteBufferInBlock(block, oldBuf, newBuf) {
  const visit = (node) => {
    if (!node || typeof node !== 'object') return;
    if ((node.type === 'BufferLoadNode' || node.type === 'BufferStoreNode') && node.buffer === oldBuf) {
      node.buffer = newBuf;
    }
    for (const key of Object.keys(node)) {
      const child = node[key];
      if (Array.isArray(child)) {
        for (const c of child) {
          if (typeof c === 'object' && c !== null) visit(c);
        }
      } else if (typeof child === 'object' && child !== null && child.type) {
        visit(child);
      }
    }
  };
  visit(block.body);
  if (block.initBody) visit(block.initBody);
}
