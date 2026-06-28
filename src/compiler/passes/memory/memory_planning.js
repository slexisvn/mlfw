import { BufferLiveness } from './buffer_liveness.js';
import { InplaceAnalysis } from './inplace_analysis.js';
import { BufferAssignment } from './buffer_assignment.js';
import { AllocateNode } from '../../ir/tensor/nodes.js';
import { MinHeap } from '../../../util/min_heap.js';

export class MemoryPlan {
  constructor(assignment, livenessResult, inplaceCandidates) {
    this.assignment = assignment;
    this.liveness = livenessResult;
    this.inplaceCandidates = inplaceCandidates;
    this.aliasMap = new Map();
  }

  peakMemory(scope = null) {
    return this.assignment.peakMemory(scope);
  }

  getReport() {
    const scopeBreakdown = new Map();
    for (const [scope, pool] of this.assignment.pools) {
      scopeBreakdown.set(scope, {
        peakUsage: pool.peakUsage,
        numBuffers: 0,
        numReused: 0
      });
    }

    for (const [buf, entry] of this.assignment.assignments) {
      const info = scopeBreakdown.get(entry.scope);
      if (info) {
        info.numBuffers++;
        if (entry.inplaceOf) info.numReused++;
      }
    }

    const totalTemporaries = this.liveness.getTemporaries().length;
    const totalInplace = this.inplaceCandidates.length;

    return {
      peakMemory: this.assignment.peakMemory(),
      scopeBreakdown,
      totalTemporaries,
      totalInplace,
      materializedReuse: this.aliasMap.size,
      assignments: this.assignment.assignments
    };
  }
}

export class MemoryPlanner {
  constructor(config = {}) {
    this.alignment = config.alignment || 64;
    this.enableInplace = config.enableInplace !== false;
    this.allocStrategy = config.allocStrategy || 'best-fit';
    this.poolAllocation = config.poolAllocation || false;
  }

  plan(primFunc) {
    const livenessResult = BufferLiveness.analyze(primFunc);
    const temporaries = livenessResult.getTemporaries();

    let inplaceCandidates = [];
    if (this.enableInplace) {
      inplaceCandidates = InplaceAnalysis.analyze(primFunc, livenessResult);
    }

    const assignment = new BufferAssignment();
    assignment.assign(temporaries, inplaceCandidates, this.alignment, this.allocStrategy);

    return new MemoryPlan(assignment, livenessResult, inplaceCandidates);
  }

  planAndRewrite(primFunc) {
    const plan = this.plan(primFunc);
    const rewritten = this._insertAllocations(primFunc, plan);
    return { func: rewritten, plan };
  }

  _insertAllocations(primFunc, plan) {
    const temporaries = plan.liveness.getTemporaries();
    if (temporaries.length === 0) return primFunc;

    let aliasMap = new Map();
    if (this.poolAllocation) {
      this._assignPoolOffsets(primFunc, plan, temporaries);
    } else {
      aliasMap = this._buildReuseAliases(temporaries, plan, primFunc);
      if (aliasMap.size > 0) {
        rewriteBufferAliases(primFunc.body, aliasMap);
      }
    }
    plan.aliasMap = aliasMap;

    const sorted = [...temporaries].sort((a, b) => b.firstUse - a.firstUse);

    let body = primFunc.body;
    const allocated = new Set();
    for (const interval of sorted) {
      const buf = interval.buffer;
      if (aliasMap.has(buf)) continue;
      const assignment = plan.assignment.getAssignment(buf);
      if (!assignment) continue;
      if (assignment.inplaceOf) continue;
      if (allocated.has(buf)) continue;
      allocated.add(buf);
      body = new AllocateNode(buf, assignment.isDynamic ? 'dynamic' : assignment.scope, body);
    }

    primFunc.body = body;
    primFunc._setChild('body', body);
    return primFunc;
  }

  _assignPoolOffsets(primFunc, plan, temporaries) {
    const unsafe = collectFreshZeroDependent(primFunc);
    for (const interval of temporaries) {
      const buf = interval.buffer;
      if (unsafe.has(buf)) continue;
      if (buf.scope !== 'global') continue;
      const assignment = plan.assignment.getAssignment(buf);
      if (!assignment || assignment.inplaceOf || assignment.isDynamic) continue;
      if (!(assignment.size > 0)) continue;
      buf.poolByteOffset = assignment.offset;
    }
  }

  _buildReuseAliases(temporaries, plan, primFunc) {
    const unsafe = collectFreshZeroDependent(primFunc);
    const inplaceSources = new Set(plan.assignment.inplaceMap.values());
    const effLastUse = plan.assignment.effLastUse;
    const lastUseOf = (interval) => effLastUse.get(interval.buffer) ?? interval.lastUse;

    const groups = new Map();
    for (const interval of temporaries) {
      const buf = interval.buffer;
      const assignment = plan.assignment.getAssignment(buf);
      if (!assignment || assignment.inplaceOf || assignment.isDynamic) continue;
      if (inplaceSources.has(buf)) continue;
      if (buf.numel() <= 0) continue;
      if (unsafe.has(buf)) continue;
      const key = `${buf.scope}|${buf.dtype}|${buf.shape.join(',')}|${buf.strides.join(',')}`;
      let bucket = groups.get(key);
      if (!bucket) { bucket = []; groups.set(key, bucket); }
      bucket.push(interval);
    }

    const aliasMap = new Map();
    for (const bucket of groups.values()) {
      if (bucket.length < 2) continue;
      bucket.sort((a, b) => a.firstUse - b.firstUse || lastUseOf(a) - lastUseOf(b));
      const slots = new MinHeap((x, y) => x.lastUse - y.lastUse);
      for (const interval of bucket) {
        const free = slots.peek();
        if (free && free.lastUse < interval.firstUse) {
          slots.pop();
          free.lastUse = lastUseOf(interval);
          slots.push(free);
          aliasMap.set(interval.buffer, free.rep);
        } else {
          slots.push({ rep: interval.buffer, lastUse: lastUseOf(interval) });
        }
      }
    }
    return aliasMap;
  }
}

function exprLoadsBuffer(node, target, seen) {
  if (!node || typeof node !== 'object' || seen.has(node)) return false;
  seen.add(node);
  if (node.type === 'BufferLoadNode') {
    if (!target || node.buffer === target) return true;
  }
  for (const key of Object.keys(node)) {
    if (key === '_parent' || key === '_parentKey' || key === '_parentIdx') continue;
    const value = node[key];
    if (!value || typeof value !== 'object') continue;
    if (Array.isArray(value)) {
      for (const item of value) if (exprLoadsBuffer(item, target, seen)) return true;
    } else if (exprLoadsBuffer(value, target, seen)) {
      return true;
    }
  }
  return false;
}

function collectFreshZeroDependent(primFunc) {
  const unsafe = new Set();
  const writeValues = new Map();

  const visited = new Set();
  const stack = [primFunc.body];
  while (stack.length > 0) {
    const node = stack.pop();
    if (!node || typeof node !== 'object' || visited.has(node)) continue;
    visited.add(node);

    if (node.type === 'BufferStoreNode' && node.buffer) {
      const buf = node.buffer;
      for (const idx of node.indices) {
        if (exprLoadsBuffer(idx, null, new Set())) { unsafe.add(buf); break; }
      }
      if (exprLoadsBuffer(node.value, buf, new Set())) unsafe.add(buf);
      let values = writeValues.get(buf);
      if (!values) { values = []; writeValues.set(buf, values); }
      values.push(node.value);
    }

    for (const key of Object.keys(node)) {
      if (key === '_parent' || key === '_parentKey' || key === '_parentIdx') continue;
      const value = node[key];
      if (Array.isArray(value)) {
        for (const item of value) if (item && typeof item === 'object') stack.push(item);
      } else if (value && typeof value === 'object') {
        stack.push(value);
      }
    }
  }

  for (const [buf, values] of writeValues) {
    const isConstZero = (v) => v && (v.type === 'IntImmNode' || v.type === 'FloatImmNode') && v.value === 0;
    if (values.every(isConstZero)) unsafe.add(buf);
    if (values.length === 1 && values[0] && (values[0].type === 'IntImmNode' || values[0].type === 'FloatImmNode')) {
      unsafe.add(buf);
    }
  }

  return unsafe;
}

function rewriteBufferAliases(root, aliasMap) {
  const visited = new Set();
  const stack = [root];
  while (stack.length > 0) {
    const node = stack.pop();
    if (!node || typeof node !== 'object' || visited.has(node)) continue;
    visited.add(node);

    for (const key of Object.keys(node)) {
      if (key === '_parent' || key === '_parentKey' || key === '_parentIdx') continue;
      const value = node[key];
      if (value && typeof value === 'object' && aliasMap.has(value)) {
        node[key] = aliasMap.get(value);
        continue;
      }
      if (Array.isArray(value)) {
        for (let i = 0; i < value.length; i++) {
          const item = value[i];
          if (item && typeof item === 'object' && aliasMap.has(item)) {
            value[i] = aliasMap.get(item);
          } else if (item && typeof item === 'object') {
            stack.push(item);
          }
        }
        continue;
      }
      if (value && typeof value === 'object') {
        stack.push(value);
      }
    }
  }
}
