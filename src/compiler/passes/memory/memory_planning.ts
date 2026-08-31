import { BufferLiveness } from './buffer_liveness.js';
import { InplaceAnalysis } from './inplace_analysis.js';
import { BufferAssignment } from './buffer_assignment.js';
import { AllocateNode } from '../../ir/tensor/nodes.js';
import { MinHeap } from '../../../util/min_heap.js';
import { buffersRequiringDefinedStorage } from '../../analysis/buffer_dataflow.js';
import { walk } from '../../ir/ir_visitor.js';
import type { IRNode } from '../../ir/ir_visitor.js';
import type { Buffer } from '../../ir/tensor/buffer.js';
import type { PrimFunc, TirNode } from '../../ir/tensor/nodes.js';
import type { BufferLifetime, MemoryPlanTrace } from '../../support/trace.js';
import type { BufferInterval, BufferLivenessResult } from './buffer_liveness.js';
import type { InplaceCandidate } from './inplace_analysis.js';
import type { BufferAssignmentEntry } from './buffer_assignment.js';

export type ScopeUsage = { peakUsage: number; numBuffers: number; numReused: number };
export type { BufferLifetime, MemoryPlanTrace } from '../../support/trace.js';
export type MemoryPlanReport = {
  peakMemory: number;
  scopeBreakdown: Map<string, ScopeUsage>;
  totalTemporaries: number;
  totalInplace: number;
  materializedReuse: number;
  assignments: Map<Buffer, BufferAssignmentEntry>;
};
export type MemoryPlannerConfig = {
  alignment?: number;
  enableInplace?: boolean;
  allocStrategy?: string;
  poolAllocation?: boolean;
};
type ReuseSlot = { rep: Buffer; lastUse: number };

export class MemoryPlan {
  assignment: BufferAssignment;
  liveness: BufferLivenessResult;
  inplaceCandidates: readonly InplaceCandidate[];
  aliasMap: Map<Buffer, Buffer>;
  shareable: ReadonlySet<Buffer>;
  preAllocated: ReadonlySet<string>;

  constructor(
    assignment: BufferAssignment,
    livenessResult: BufferLivenessResult,
    inplaceCandidates: readonly InplaceCandidate[],
    shareable: ReadonlySet<Buffer>,
    preAllocated: ReadonlySet<string>
  ) {
    this.assignment = assignment;
    this.liveness = livenessResult;
    this.inplaceCandidates = inplaceCandidates;
    this.aliasMap = new Map();
    this.shareable = shareable;
    this.preAllocated = preAllocated;
  }

  peakMemory(scope: string | null = null): number {
    return this.assignment.peakMemory(scope);
  }

  lifetimes(): MemoryPlanTrace {
    const buffers: BufferLifetime[] = [];
    let totalBytesIfNeverShared = 0;

    for (const interval of this.liveness.getTemporaries()) {
      const entry = this.assignment.assignments.get(interval.buffer);
      if (!entry) continue;
      const shared = this.assignment.inplaceMap.get(interval.buffer);
      totalBytesIfNeverShared += entry.size;
      buffers.push({
        name: interval.buffer.name,
        scope: entry.scope,
        bytes: entry.size,
        slot: entry.offset,
        firstUse: interval.firstUse,
        lastUse: interval.lastUse,
        sharesWith: shared ? shared.name : null,
      });
    }

    buffers.sort((a, b) => a.firstUse - b.firstUse || a.slot - b.slot);

    return {
      peakMemory: this.peakMemory(),
      totalBytesIfNeverShared,
      steps: this.liveness.stmtOrder.length,
      buffers,
    };
  }

  getReport(): MemoryPlanReport {
    const scopeBreakdown = new Map<string, ScopeUsage>();
    for (const [scope, pool] of this.assignment.pools) {
      scopeBreakdown.set(scope, {
        peakUsage: pool.peakUsage,
        numBuffers: 0,
        numReused: 0
      });
    }

    for (const entry of this.assignment.assignments.values()) {
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
  alignment: number;
  enableInplace: boolean;
  allocStrategy: string;
  poolAllocation: boolean;

  constructor(config: MemoryPlannerConfig = {}) {
    this.alignment = config.alignment || 64;
    this.enableInplace = config.enableInplace !== false;
    this.allocStrategy = config.allocStrategy || 'best-fit';
    this.poolAllocation = config.poolAllocation || false;
  }

  plan(primFunc: PrimFunc): MemoryPlan {
    const livenessResult = BufferLiveness.analyze(primFunc);
    const temporaries = livenessResult.getTemporaries();
    const preAllocated = collectPreAllocated(primFunc);
    const { shareable, donatable } = storageRoles(primFunc, temporaries, preAllocated);

    let inplaceCandidates: InplaceCandidate[] = [];
    if (this.enableInplace) {
      inplaceCandidates = InplaceAnalysis.analyze(primFunc, livenessResult)
        .filter((c) => donatable.has(c.srcBuffer) && shareable.has(c.dstBuffer));
    }

    const assignment = new BufferAssignment();
    assignment.assign(temporaries, inplaceCandidates, this.alignment, this.allocStrategy);

    return new MemoryPlan(assignment, livenessResult, inplaceCandidates, shareable, preAllocated);
  }

  planAndRewrite(primFunc: PrimFunc): { func: PrimFunc; plan: MemoryPlan } {
    const plan = this.plan(primFunc);
    const rewritten = this._insertAllocations(primFunc, plan);
    return { func: rewritten, plan };
  }

  _insertAllocations(primFunc: PrimFunc, plan: MemoryPlan): PrimFunc {
    const temporaries = plan.liveness.getTemporaries();
    if (temporaries.length === 0) return primFunc;

    let aliasMap = new Map<Buffer, Buffer>();
    if (this.poolAllocation) {
      this._assignPoolOffsets(plan, temporaries);
    } else {
      aliasMap = this._buildReuseAliases(plan, temporaries);
      if (aliasMap.size > 0) {
        rewriteBufferAliases(primFunc.body, aliasMap);
      }
    }
    plan.aliasMap = aliasMap;

    const sorted = [...temporaries].sort((a, b) => b.firstUse - a.firstUse);

    let body: TirNode = primFunc.body;
    const allocated = new Set<Buffer>();
    for (const interval of sorted) {
      const buf = interval.buffer;
      if (plan.preAllocated.has(buf.name)) continue;
      if (aliasMap.has(buf)) continue;
      const assignment = plan.assignment.getAssignment(buf);
      if (!assignment) continue;
      if (allocated.has(buf)) continue;
      allocated.add(buf);
      body = new AllocateNode(buf, assignment.isDynamic ? 'dynamic' : assignment.scope, body);
    }

    primFunc.body = body;
    primFunc._setChild('body', body);
    return primFunc;
  }

  _assignPoolOffsets(plan: MemoryPlan, temporaries: readonly BufferInterval[]): void {
    for (const interval of temporaries) {
      const buf = interval.buffer;
      if (!plan.shareable.has(buf)) continue;
      if (buf.scope !== 'global') continue;
      const assignment = plan.assignment.getAssignment(buf);
      if (!assignment || assignment.isDynamic) continue;
      if (!(assignment.size > 0)) continue;
      buf.poolByteOffset = assignment.offset;
    }
  }

  _buildReuseAliases(plan: MemoryPlan, temporaries: readonly BufferInterval[]): Map<Buffer, Buffer> {
    const effLastUse = plan.assignment.effLastUse;
    const lastUseOf = (interval: BufferInterval): number => effLastUse.get(interval.buffer) ?? interval.lastUse;

    const aliasMap = new Map<Buffer, Buffer>();
    for (const [buf, assignment] of plan.assignment.assignments) {
      if (assignment.inplaceOf) aliasMap.set(buf, assignment.inplaceOf);
    }

    const groups = new Map<string, BufferInterval[]>();
    for (const interval of temporaries) {
      const buf = interval.buffer;
      if (!plan.shareable.has(buf)) continue;
      const assignment = plan.assignment.getAssignment(buf);
      if (!assignment || assignment.inplaceOf || assignment.isDynamic) continue;
      const key = `${buf.scope}|${buf.dtype}|${buf.shape.join(',')}|${buf.strides.join(',')}`;
      let bucket = groups.get(key);
      if (!bucket) { bucket = []; groups.set(key, bucket); }
      bucket.push(interval);
    }

    for (const bucket of groups.values()) {
      if (bucket.length < 2) continue;
      bucket.sort((a, b) => a.firstUse - b.firstUse || lastUseOf(a) - lastUseOf(b));
      const slots = new MinHeap<ReuseSlot>((x: ReuseSlot, y: ReuseSlot) => x.lastUse - y.lastUse);
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
    return resolveAliasChains(aliasMap);
  }
}

function collectPreAllocated(primFunc: PrimFunc): Set<string> {
  const preAllocated = new Set<string>();
  walk(primFunc.body as unknown as IRNode, (node) => {
    if ((node as { type: string }).type === 'AllocateNode') preAllocated.add((node as unknown as AllocateNode).buffer.name);
  });
  return preAllocated;
}

function storageRoles(primFunc: PrimFunc, temporaries: readonly BufferInterval[], preAllocated: ReadonlySet<string>): { shareable: Set<Buffer>; donatable: Set<Buffer> } {
  const needsDefinedStorage = buffersRequiringDefinedStorage(primFunc);
  const shareable = new Set<Buffer>();
  const donatable = new Set<Buffer>();
  for (const interval of temporaries) {
    const buf = interval.buffer;
    if (buf.numel() <= 0) continue;
    if (preAllocated.has(buf.name)) continue;
    donatable.add(buf);
    if (!needsDefinedStorage.has(buf)) shareable.add(buf);
  }
  return { shareable, donatable };
}

function resolveAliasChains(aliasMap: ReadonlyMap<Buffer, Buffer>): Map<Buffer, Buffer> {
  const resolved = new Map<Buffer, Buffer>();
  for (const [from, target] of aliasMap) {
    let to = target;
    const seen = new Set<Buffer>([from, to]);
    for (let next = aliasMap.get(to); next && !seen.has(next); next = aliasMap.get(to)) {
      to = next;
      seen.add(to);
    }
    resolved.set(from, to);
  }
  return resolved;
}

function rewriteBufferAliases(root: TirNode, aliasMap: ReadonlyMap<Buffer, Buffer>): void {
  const visited = new Set<object>();
  const stack: unknown[] = [root];
  while (stack.length > 0) {
    const node = stack.pop() as Record<string, unknown>;
    if (!node || typeof node !== 'object' || visited.has(node)) continue;
    visited.add(node);

    for (const key of Object.keys(node)) {
      if (key === '_parent' || key === '_parentKey' || key === '_parentIdx') continue;
      const value = node[key];
      if (value && typeof value === 'object' && aliasMap.has(value as Buffer)) {
        node[key] = aliasMap.get(value as Buffer);
        continue;
      }
      if (Array.isArray(value)) {
        for (let i = 0; i < value.length; i++) {
          const item = value[i];
          if (item && typeof item === 'object' && aliasMap.has(item as Buffer)) {
            value[i] = aliasMap.get(item as Buffer);
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
