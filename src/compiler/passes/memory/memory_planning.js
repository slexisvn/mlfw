import { BufferLiveness } from './buffer_liveness.js';
import { InplaceAnalysis } from './inplace_analysis.js';
import { BufferAssignment } from './buffer_assignment.js';
import { AllocateNode } from '../../ir/tensor/nodes.js';

export class MemoryPlan {
  constructor(assignment, livenessResult, inplaceCandidates) {
    this.assignment = assignment;
    this.liveness = livenessResult;
    this.inplaceCandidates = inplaceCandidates;
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
      assignments: this.assignment.assignments
    };
  }
}

export class MemoryPlanner {
  constructor(config = {}) {
    this.alignment = config.alignment || 64;
    this.enableInplace = config.enableInplace !== false;
  }

  plan(primFunc) {
    const livenessResult = BufferLiveness.analyze(primFunc);
    const temporaries = livenessResult.getTemporaries();

    let inplaceCandidates = [];
    if (this.enableInplace) {
      inplaceCandidates = InplaceAnalysis.analyze(primFunc, livenessResult);
    }

    const assignment = new BufferAssignment();
    assignment.assign(temporaries, inplaceCandidates, this.alignment);

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

    const sorted = [...temporaries].sort((a, b) => b.firstUse - a.firstUse);

    let body = primFunc.body;
    for (const interval of sorted) {
      const buf = interval.buffer;
      const assignment = plan.assignment.getAssignment(buf);
      if (!assignment) continue;
      if (assignment.inplaceOf) continue;
      body = new AllocateNode(buf, assignment.isDynamic ? 'dynamic' : assignment.scope, body);
    }

    primFunc.body = body;
    primFunc._setChild('body', body);
    return primFunc;
  }
}
