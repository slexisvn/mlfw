import { BufferLiveness } from './buffer_liveness.js';
import { InplaceAnalysis } from './inplace_analysis.js';
import { BufferAssignment } from './buffer_assignment.js';
import { AllocateNode } from '../../ir/tensor/nodes.js';
import { MinHeap } from '../../../util/min_heap.js';
import { buffersRequiringDefinedStorage } from '../../analysis/buffer_dataflow.js';
import type { Buffer } from '../../ir/tensor/buffer.js';
import type { PrimFunc, TirNode } from '../../ir/tensor/nodes.js';
import type { BufferInterval, BufferLivenessResult } from './buffer_liveness.js';
import type { InplaceCandidate } from './inplace_analysis.js';
import type { BufferAssignmentEntry } from './buffer_assignment.js';

export type ScopeUsage = { peakUsage: number; numBuffers: number; numReused: number };
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

  constructor(assignment: BufferAssignment, livenessResult: BufferLivenessResult, inplaceCandidates: readonly InplaceCandidate[]) {
    this.assignment = assignment;
    this.liveness = livenessResult;
    this.inplaceCandidates = inplaceCandidates;
    this.aliasMap = new Map();
  }

  peakMemory(scope: string | null = null): number {
    return this.assignment.peakMemory(scope);
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

    let inplaceCandidates: InplaceCandidate[] = [];
    if (this.enableInplace) {
      inplaceCandidates = InplaceAnalysis.analyze(primFunc, livenessResult);
    }

    const assignment = new BufferAssignment();
    assignment.assign(temporaries, inplaceCandidates, this.alignment, this.allocStrategy);

    return new MemoryPlan(assignment, livenessResult, inplaceCandidates);
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
      this._assignPoolOffsets(primFunc, plan, temporaries);
    } else {
      aliasMap = this._buildReuseAliases(temporaries, plan, primFunc);
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

  _assignPoolOffsets(primFunc: PrimFunc, plan: MemoryPlan, temporaries: readonly BufferInterval[]): void {
    const needsDefinedStorage = buffersRequiringDefinedStorage(primFunc);
    for (const interval of temporaries) {
      const buf = interval.buffer;
      if (needsDefinedStorage.has(buf)) continue;
      if (buf.scope !== 'global') continue;
      const assignment = plan.assignment.getAssignment(buf);
      if (!assignment || assignment.inplaceOf || assignment.isDynamic) continue;
      if (!(assignment.size > 0)) continue;
      buf.poolByteOffset = assignment.offset;
    }
  }

  _buildReuseAliases(temporaries: readonly BufferInterval[], plan: MemoryPlan, primFunc: PrimFunc): Map<Buffer, Buffer> {
    const needsDefinedStorage = buffersRequiringDefinedStorage(primFunc);
    const inplaceSources = new Set(plan.assignment.inplaceMap.values());
    const effLastUse = plan.assignment.effLastUse;
    const lastUseOf = (interval: BufferInterval): number => effLastUse.get(interval.buffer) ?? interval.lastUse;

    const groups = new Map<string, BufferInterval[]>();
    for (const interval of temporaries) {
      const buf = interval.buffer;
      const assignment = plan.assignment.getAssignment(buf);
      if (!assignment || assignment.inplaceOf || assignment.isDynamic) continue;
      if (inplaceSources.has(buf)) continue;
      if (buf.numel() <= 0) continue;
      if (needsDefinedStorage.has(buf)) continue;
      const key = `${buf.scope}|${buf.dtype}|${buf.shape.join(',')}|${buf.strides.join(',')}`;
      let bucket = groups.get(key);
      if (!bucket) { bucket = []; groups.set(key, bucket); }
      bucket.push(interval);
    }

    const aliasMap = new Map<Buffer, Buffer>();
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
    return aliasMap;
  }
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
