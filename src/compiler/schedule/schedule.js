import {
  ForNode, BlockNode, SeqNode, BufferStoreNode, BufferLoadNode, BlockRealizeNode,
  VariableNode, IntImmNode, MathOpNode, ForKind, IterVarKind, IfThenElseNode, AllocateNode
} from '../ir/tensor/nodes.js';
import { Buffer } from '../ir/tensor/buffer.js';
import { ScheduleTrace } from './trace.js';
import { ScheduleValidator } from './validator.js';
import { ScheduleState } from './schedule_state.js';
import { ScheduleMutator } from './mutator.js';
import { loopCarriedDependence, reorderLegality, collectVarsUsed, IterVarPolicy } from './legality.js';
import { cloneIRShared } from '../ir/clone_ir.js';
import { transform as irTransform, walk as irWalk } from '../ir/ir_visitor.js';
import { FuncAttr } from '../ir/func_attrs.js';
import { AccessKind, loadedBuffers } from '../analysis/buffer_access.js';
import { LinearForm, mixedRadixDecomposition, linearFormToNode } from '../analysis/iter_map.js';

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

function cloneExprTree(node) {
  return cloneIRShared(node, cloneExprTree, (n, copy, rec) => {
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
  });
}

function collectStores(node, out) {
  if (!node) return;
  if (node.type === 'BufferStoreNode') { out.push(node); return; }
  if (node.type === 'SeqNode') { for (const s of node.stmts) collectStores(s, out); return; }
  if (node.type === 'ForNode' || node.type === 'BlockNode') collectStores(node.body, out);
}

function usedVarNames(expr) {
  const names = new Set();
  irWalk(expr, (n) => { if (n.type === 'VariableNode') names.add(n.name); });
  return names;
}

function recoverIterVar(indexExpr, decomposition, factor) {
  let expr = decomposition.offset === 0
    ? indexExpr
    : new MathOpNode('-', indexExpr, new IntImmNode(decomposition.offset));
  if (factor.coeff !== 1) expr = new MathOpNode('//', expr, new IntImmNode(factor.coeff));
  if (factor.coeff * factor.extent !== decomposition.extent) {
    expr = new MathOpNode('%', expr, new IntImmNode(factor.extent));
  }
  return factor.min === 0 ? expr : new MathOpNode('+', expr, new IntImmNode(factor.min));
}

function invertWriteIndices(access) {
  const varRanges = new Map();
  for (const level of access.iterSpace) {
    if (level) varRanges.set(level.name, [level.min, level.extent]);
  }
  const decompositions = [];
  for (const form of access.forms) {
    const decomposition = mixedRadixDecomposition(form, varRanges);
    if (!decomposition) return null;
    decompositions.push(decomposition);
  }
  return decompositions;
}

function producerVarForms(access, blockInfo) {
  const forms = new Map();
  for (const level of access.iterSpace) {
    if (level) forms.set(level.name, LinearForm.variable(level.name));
  }
  if (blockInfo) {
    for (const binding of blockInfo.bindings) {
      if (binding.form) forms.set(binding.name, binding.form);
    }
  }
  return forms;
}

function substituteAffineVars(expr, varForms, loopVarExpr) {
  return irTransform(expr, (n) => {
    if (n.type !== 'VariableNode') return n;
    const form = varForms.get(n.name);
    if (!form) return n;
    return linearFormToNode(
      form,
      (name) => loopVarExpr.get(name)(),
      (value) => new IntImmNode(value),
      (op, a, b) => new MathOpNode(op, a, b)
    );
  }, { bindVars: false });
}

function inlineStoreValue(funcBody, store, decompositions, varForms) {
  const counter = { n: 0 };
  replaceBufferLoads(funcBody, store.buffer.name, (load) => {
    const loopVarExpr = new Map();
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

function dropBufferReads(funcBody, bufName) {
  const stack = [funcBody];
  while (stack.length > 0) {
    const node = stack.pop();
    if (!node || typeof node !== 'object') continue;
    if (node.type === 'BlockNode' && node.reads) {
      const kept = node.reads.filter((r) => !(r.buffer && r.buffer.name === bufName));
      if (kept.length !== node.reads.length) node.reads = kept;
    }
    if (node.body) stack.push(node.body);
    if (node.stmts) for (const s of node.stmts) stack.push(s);
    if (node.thenBody) stack.push(node.thenBody);
    if (node.elseBody) stack.push(node.elseBody);
    if (node.initBody) stack.push(node.initBody);
  }
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
    this.mutator = new ScheduleMutator(primFunc);
  }

  _replaceInTree(oldNode, newNode) {
    this.state.replaceNode(oldNode, newNode);
  }

  _removeFromTree(node) {
    this.state.removeNode(node);
  }

  getBlockSRef(name) {
    const sref = this.state.tree.getBlockSRef(name);
    if (!sref) throw new Error(`Block '${name}' not found`);
    return sref;
  }

  getBlock(name) {
    return this.getBlockSRef(name).node;
  }

  getLoops(blockName) {
    return this.state.tree.loopsOf(blockName).map(sref => sref.node);
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

    const topmostLoop = this._topmostOf(loopSet);
    const { links, innermostBody } = this._collectReorderChain(loopSet, topmostLoop);

    const after = links.slice();
    let slot = 0;
    for (let i = 0; i < links.length; i++) {
      if (loopSet.has(links[i])) after[i] = newOrder[slot++];
    }

    const isLoop = (n) => n.type === 'ForNode';
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
      this._setChainChild(chain[i], i + 1 < chain.length ? chain[i + 1] : innermostBody);
    }

    this._replaceInTree(topmostLoop, chain[0]);

    if (!this._replaying) {
      this.trace.record('reorder', [newOrder.map(l => l.loopVar.name)]);
    }
  }

  _topmostOf(loopSet) {
    let topmost = null;
    let found = 0;
    const stack = [{ node: this.func.body, depth: 0 }];
    let topmostDepth = Infinity;
    while (stack.length > 0) {
      const { node, depth } = stack.pop();
      if (!node) continue;
      if (node.type === 'ForNode') {
        if (loopSet.has(node)) {
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
        stack.push({ node: node.body, depth });
        if (node.initBody) stack.push({ node: node.initBody, depth });
      }
    }
    if (found !== loopSet.size) {
      throw new Error('reorder: not all requested loops were found in the function nest');
    }
    return topmost;
  }

  _arrangeChain(after) {
    const loopNames = new Set(after.filter((n) => n.type === 'ForNode').map((n) => n.loopVar.name));
    const guardOf = (node) => {
      if (node.type === 'IfThenElseNode') return node.condition;
      if (node.type === 'LetStmtNode') return node.value;
      return null;
    };
    const dependsOn = (node) => {
      const guard = guardOf(node);
      if (!guard) return new Set();
      const needs = new Set();
      for (const name of usedVarNames(guard)) if (loopNames.has(name)) needs.add(name);
      return needs;
    };

    const chain = [];
    const bound = new Set();
    const deferred = [];
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
        bound.add(node.loopVar.name);
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

  _setChainChild(node, child) {
    const key = node.type === 'IfThenElseNode' ? 'thenBody' : 'body';
    node[key] = child;
    node._setChild(key, child);
  }

  _collectReorderChain(loopSet, topmostLoop) {
    const remaining = new Set(loopSet);
    const links = [];
    let node = topmostLoop;
    let innermostBody = null;
    while (node) {
      if (node.type === 'ForNode') {
        links.push(node);
        remaining.delete(node);
        if (remaining.size === 0) { innermostBody = node.body; break; }
        node = node.body;
      } else if (node.type === 'IfThenElseNode') {
        if (node.elseBody) {
          throw new Error('reorder: a two-way conditional separates the reordered loops, so they do not form a single chain');
        }
        links.push(node);
        node = node.thenBody;
      } else if (node.type === 'AllocateNode' || node.type === 'LetStmtNode') {
        links.push(node);
        node = node.body;
      } else if (node.type === 'BlockNode') {
        throw new Error(`reorder: block '${node.name}' separates the reordered loops, which therefore belong to different block scopes`);
      } else if (node.type === 'SeqNode') {
        if (node.stmts.length !== 1) {
          throw new Error('reorder: multiple statements separate the reordered loops, so they do not form a single chain');
        }
        node = node.stmts[0];
      } else {
        break;
      }
    }
    if (remaining.size > 0) {
      throw new Error('reorder: loops do not form a single chain');
    }
    return { links, innermostBody };
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
    const carried = loopCarriedDependence(this.state, loop, IterVarPolicy.ACCUMULABLE);
    if (carried !== null) throw new Error(`Cannot vectorize: ${carried}`);
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
    const carried = loopCarriedDependence(this.state, loop, IterVarPolicy.SPATIAL);
    if (carried !== null) throw new Error(`Cannot parallelize: ${carried}`);
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
    const spatialLoopVars = new Set(spatialLoops.map(l => l.loopVar.name));
    const KO = K / factor;
    const partialBuf = new Buffer(`${acc.name}_rf`, [factor, ...acc.shape], acc.dtype, acc.scope);

    const kiVar = freshVar(`${reductionVarName}_rfi`);
    const koVar = freshVar(`${reductionVarName}_rfo`);
    const piVar = freshVar(`${reductionVarName}_rfp`);
    const kiIter = new BlockRealizeNode(freshVar(`${reductionVarName}_rfvi`), kiVar, IterVarKind.DATA_PAR);
    const piIter = new BlockRealizeNode(freshVar(`${reductionVarName}_rfvp`), piVar, IterVarKind.COMM_REDUCE);

    const splitK = () => new MathOpNode('+', new MathOpNode('*', koVar, new IntImmNode(factor)), kiVar);
    const cfIdx = (iterVar) => [iterVar, ...spatialIdx.map(cloneExprTree)];
    const partialUpdate = substituteVar(cloneExprTree(update), reductionVarName, splitK);
    const partialIterVars = block.iterVars.map((iv) => {
      const copy = cloneExprTree(iv);
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

  decomposeReduction(blockName) {
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

    const initStore = new BufferStoreNode(acc, store.indices.map(cloneExprTree), cloneExprTree(block.initBody.value));
    const initBlock = new BlockNode(`${blockName}_init`,
      this._iterVarsOver(block, spatialLoopVars), [], [{ buffer: acc }], initStore);
    let initNest = initBlock;
    for (let i = spatialLoops.length - 1; i >= 0; i--) {
      initNest = new ForNode(spatialLoops[i].loopVar, new IntImmNode(0), cloneExprTree(spatialLoops[i].extent), ForKind.SERIAL, initNest);
    }

    const updBlock = new BlockNode(`${blockName}_upd`,
      block.iterVars.map(cloneExprTree), block.reads.map(r => ({ buffer: r.buffer })), [{ buffer: acc }], cloneExprTree(store));
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

  _writeIndexLoopVars(primitive, block, store) {
    const info = this.state.blockAccessInfo(block);
    const access = (info ? info.accesses : []).find((a) => a.kind === AccessKind.WRITE && a.node === store);
    if (!access) throw new Error(`${primitive}: block '${block.name}' write is not reachable from the function body`);
    const names = new Set();
    for (const form of access.forms) {
      if (!form) throw new Error(`${primitive}: write index of '${access.buffer.name}' is not an affine map of its loop variables`);
      for (const name of form.terms.keys()) names.add(name);
    }
    return names;
  }

  _iterVarsOver(block, loopVarNames) {
    const info = this.state.blockAccessInfo(block);
    if (!info) return [];
    const formOf = new Map(info.bindings.map((b) => [b.name, b.form]));
    const kept = [];
    for (const iv of block.iterVars) {
      if (!iv || !iv.iterVar) continue;
      const form = formOf.get(iv.iterVar.name);
      if (form && [...form.terms.keys()].every((name) => loopVarNames.has(name))) kept.push(cloneExprTree(iv));
    }
    return kept;
  }

  _removeBlockNest(producerName, prod) {
    const loops = this.getLoops(producerName);
    const root = loops.length > 0 ? loops[0] : prod;
    const empty = new SeqNode([]);
    this.mutator.replaceNode(root, empty);
    this._replaceInTree(root, empty);
  }

  _inlinePlan(primitive, producerName) {
    const sref = this.getBlockSRef(producerName);
    const prod = sref.node;
    if (prod.initBody) throw new Error(`${primitive}: cannot inline a reduction block (has init)`);

    const stores = [];
    collectStores(prod.body, stores);
    if (stores.length === 0) throw new Error(`${primitive}: producer has no store to inline`);

    const blockInfo = this.state.blockAccessInfo(prod);
    const byNode = new Map();
    for (const access of (blockInfo ? blockInfo.accesses : [])) {
      if (access.kind === AccessKind.WRITE) byNode.set(access.node, access);
    }

    const produced = new Set(stores.map((s) => s.buffer));
    if (produced.size !== stores.length) throw new Error(`${primitive}: buffer written more than once in block`);

    const plans = [];
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
      const varForms = producerVarForms(access, blockInfo);
      for (const name of usedVarNames(store.value)) {
        if (!varForms.has(name)) {
          throw new Error(`${primitive}: producer value depends on '${name}', which is not an iteration variable of the block`);
        }
      }
      plans.push({ store, decompositions, varForms });
    }
    return { sref, prod, plans };
  }

  _applyInline(primitive, producerName, plan) {
    let total = 0;
    for (const { store, decompositions, varForms } of plan.plans) {
      total += inlineStoreValue(this.func.body, store, decompositions, varForms);
    }
    if (total === 0) throw new Error(`${primitive}: block '${producerName}' has no consumers to inline into`);
    for (const { store } of plan.plans) dropBufferReads(this.func.body, store.buffer.name);
    this._removeBlockNest(producerName, plan.prod);
    if (!this._replaying) this.trace.record(primitive, [producerName]);
  }

  computeInline(producerName) {
    const plan = this._inlinePlan('computeInline', producerName);
    if (plan.plans.length !== 1) {
      throw new Error('computeInline: block writes more than one buffer; use computeInlineBlock');
    }
    this._applyInline('computeInline', producerName, plan);
  }

  computeInlineBlock(producerName) {
    this._applyInline('computeInlineBlock', producerName, this._inlinePlan('computeInlineBlock', producerName));
  }

  _checkRelocationDependences(primitive, blockSRef, targetLoop, atStart) {
    const scope = this.state.scopeOf(blockSRef);
    const self = scope && scope.memberOf(blockSRef);
    if (!self) throw new Error(`${primitive}: block '${blockSRef.node.name}' is not part of any block scope`);
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
        `${primitive}: moving '${blockSRef.node.name}' across '${other.node.name}' would violate a ` +
        `${dep.kind} dependence on buffer '${dep.buffer.name}'`
      );
    }
  }

  _alignedLoopPairs(primitive, blockLoops, targetLoop) {
    const targetChain = [];
    for (let sref = this.state.tree.getSRef(targetLoop); sref; sref = sref.parent) {
      if (sref.isLoop) targetChain.unshift(sref.node);
    }
    if (blockLoops.length > targetChain.length) {
      throw new Error(`${primitive}: the moved block is nested deeper than the target loop`);
    }
    const aligned = targetChain.slice(targetChain.length - blockLoops.length);
    const pairs = [];
    for (let i = 0; i < blockLoops.length; i++) {
      const from = blockLoops[i];
      const to = aligned[i];
      const fromExtent = getConstExtent(from.extent);
      const toExtent = getConstExtent(to.extent);
      const fromMin = getConstExtent(from.min);
      const toMin = getConstExtent(to.min);
      if (fromExtent === null || toExtent === null || fromMin === null || toMin === null
          || fromExtent !== toExtent || fromMin !== toMin) {
        throw new Error(
          `${primitive}: iteration domain of '${from.loopVar.name}' does not match '${to.loopVar.name}'; ` +
          `relocation would change the block's iteration space`
        );
      }
      pairs.push([from, to]);
    }
    return pairs;
  }

  _relocateBlockToLoop(primitive, blockName, targetLoopRef, atStart) {
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

  computeAt(producerName, targetLoopRef) {
    const targetVar = this._relocateBlockToLoop('computeAt', producerName, targetLoopRef, true);
    if (!this._replaying) this.trace.record('computeAt', [producerName, targetVar]);
  }

  reverseComputeAt(consumerName, targetLoopRef) {
    const targetVar = this._relocateBlockToLoop('reverseComputeAt', consumerName, targetLoopRef, false);
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
    this.func.setAttr(FuncAttr.TENSOR_INTRIN, { name: intrinName, info });
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
