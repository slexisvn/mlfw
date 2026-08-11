import { PrimFuncPass } from '../tir_pass.js';
import { SeqNode } from '../../ir/tensor/nodes.js';
import { walk } from '../../ir/ir_visitor.js';
import { collectBufferAccesses, AccessKind } from '../../analysis/buffer_access.js';
import { regionHull } from '../../schedule/dep_analysis.js';
import { linkAccessUnits } from '../../schedule/block_scope.js';

export const UNSEQUENCED_EFFECT_NODES = new Set(['CallExternNode', 'SyncThreadsNode', 'WhileNode']);

class ScheduleUnit {
  constructor(index, stmt) {
    this.index = index;
    this.stmt = stmt;
    this.reads = new Map();
    this.writes = new Map();
    this.temporaries = [];
    this.predecessors = new Set();
    this.successors = new Set();
  }
}

function buildUnits(stmts, accessInfo) {
  const accessOf = new Map();
  for (const access of accessInfo.order) accessOf.set(access.node, access);

  const units = [];
  for (let i = 0; i < stmts.length; i++) {
    const unit = new ScheduleUnit(i, stmts[i]);
    const readRegions = new Map();
    const writeRegions = new Map();
    let movable = true;
    walk(stmts[i], (node) => {
      if (UNSEQUENCED_EFFECT_NODES.has(node.type)) movable = false;
      const access = accessOf.get(node);
      if (!access) return;
      const target = access.kind === AccessKind.WRITE ? writeRegions : readRegions;
      let list = target.get(access.buffer);
      if (!list) { list = []; target.set(access.buffer, list); }
      list.push(access.regions);
    });
    if (!movable) return null;
    for (const [buffer, list] of readRegions) unit.reads.set(buffer, regionHull(list));
    for (const [buffer, list] of writeRegions) unit.writes.set(buffer, regionHull(list));
    units.push(unit);
  }
  return units;
}

function buildDependenceEdges(units) {
  const byBuffer = new Map();
  const entryFor = (buffer, unit) => {
    let entries = byBuffer.get(buffer);
    if (!entries) { entries = new Map(); byBuffer.set(buffer, entries); }
    let entry = entries.get(unit);
    if (!entry) { entry = { unit, position: unit.index, read: null, write: null }; entries.set(unit, entry); }
    return entry;
  };
  for (const unit of units) {
    for (const [buffer, hull] of unit.reads) entryFor(buffer, unit).read = hull;
    for (const [buffer, hull] of unit.writes) entryFor(buffer, unit).write = hull;
  }
  for (const [buffer, entries] of byBuffer) {
    linkAccessUnits([...entries.values()], buffer, (src, dst) => {
      src.unit.successors.add(dst.unit);
      dst.unit.predecessors.add(src.unit);
    });
  }
}

function assignTemporaries(units, paramBuffers) {
  const sizes = new Map();
  for (const unit of units) {
    const touched = new Set([...unit.reads.keys(), ...unit.writes.keys()]);
    for (const buffer of touched) {
      if (paramBuffers.has(buffer)) continue;
      if (!sizes.has(buffer)) {
        const bytes = buffer.sizeInBytes();
        if (!Number.isFinite(bytes) || bytes <= 0) continue;
        sizes.set(buffer, bytes);
      }
      unit.temporaries.push(buffer);
    }
  }
  return sizes;
}

function useCounts(units, sizes) {
  const counts = new Map();
  for (const buffer of sizes.keys()) counts.set(buffer, 0);
  for (const unit of units) {
    for (const buffer of unit.temporaries) counts.set(buffer, counts.get(buffer) + 1);
  }
  return counts;
}

function simulatePeak(order, sizes) {
  const remaining = useCounts(order, sizes);
  const live = new Set();
  let current = 0;
  let peak = 0;
  for (const unit of order) {
    for (const buffer of unit.temporaries) {
      if (live.has(buffer)) continue;
      live.add(buffer);
      current += sizes.get(buffer);
    }
    if (current > peak) peak = current;
    for (const buffer of unit.temporaries) {
      const left = remaining.get(buffer) - 1;
      remaining.set(buffer, left);
      if (left === 0 && live.delete(buffer)) current -= sizes.get(buffer);
    }
  }
  return peak;
}

function readyFrontier(units) {
  const pending = new Map();
  const ready = [];
  for (const unit of units) {
    pending.set(unit, unit.predecessors.size);
    if (unit.predecessors.size === 0) ready.push(unit);
  }
  return { pending, ready };
}

function listSchedule(units, sizes) {
  const remaining = useCounts(units, sizes);
  const live = new Set();
  const { pending, ready } = readyFrontier(units);
  const order = [];

  const benefit = (unit) => {
    let score = 0;
    for (const buffer of unit.temporaries) {
      const bytes = sizes.get(buffer);
      if (remaining.get(buffer) === 1) score += bytes;
      if (!live.has(buffer)) score -= bytes;
    }
    return score;
  };

  while (ready.length > 0) {
    let best = 0;
    let bestScore = benefit(ready[0]);
    for (let i = 1; i < ready.length; i++) {
      const score = benefit(ready[i]);
      if (score > bestScore || (score === bestScore && ready[i].index < ready[best].index)) {
        best = i;
        bestScore = score;
      }
    }
    const unit = ready.splice(best, 1)[0];
    order.push(unit);
    for (const buffer of unit.temporaries) {
      live.add(buffer);
      const left = remaining.get(buffer) - 1;
      remaining.set(buffer, left);
      if (left === 0) live.delete(buffer);
    }
    for (const next of unit.successors) {
      const left = pending.get(next) - 1;
      pending.set(next, left);
      if (left === 0) ready.push(next);
    }
  }
  return order.length === units.length ? order : null;
}

function subgraphBytes(units, sizes) {
  const total = new Map();
  for (let i = units.length - 1; i >= 0; i--) {
    const unit = units[i];
    let bytes = 0;
    for (const buffer of unit.temporaries) bytes += sizes.get(buffer);
    for (const next of unit.successors) bytes += total.get(next) || 0;
    total.set(unit, bytes);
  }
  return total;
}

function dfsSchedule(units, sizes) {
  const weight = subgraphBytes(units, sizes);
  const { pending, ready } = readyFrontier(units);
  const order = [];
  const stack = ready.sort((a, b) => (weight.get(b) - weight.get(a)) || (a.index - b.index));

  while (stack.length > 0) {
    const unit = stack.pop();
    order.push(unit);
    const released = [];
    for (const next of unit.successors) {
      const left = pending.get(next) - 1;
      pending.set(next, left);
      if (left === 0) released.push(next);
    }
    released.sort((a, b) => (weight.get(a) - weight.get(b)) || (b.index - a.index));
    for (const next of released) stack.push(next);
  }
  return order.length === units.length ? order : null;
}

export function scheduleForPeakMemory(units, paramBuffers) {
  const sizes = assignTemporaries(units, paramBuffers);
  if (sizes.size === 0) return null;

  const original = [...units];
  let bestOrder = original;
  let bestPeak = simulatePeak(original, sizes);

  for (const candidate of [listSchedule(units, sizes), dfsSchedule(units, sizes)]) {
    if (!candidate) continue;
    const peak = simulatePeak(candidate, sizes);
    if (peak < bestPeak) {
      bestPeak = peak;
      bestOrder = candidate;
    }
  }
  return bestOrder === original ? null : { order: bestOrder, peak: bestPeak, originalPeak: simulatePeak(original, sizes) };
}

export function buildScheduleUnits(stmts, paramBuffers) {
  const units = buildUnits(stmts, collectBufferAccesses(new SeqNode(stmts)));
  if (!units) return null;
  buildDependenceEdges(units);
  assignTemporaries(units, paramBuffers);
  return units;
}

export class MemorySchedulePass extends PrimFuncPass {
  constructor(config) {
    super('MemorySchedulePass', 'memoryScheduling');
    this.config = config;
  }

  run(primFunc, ctx) {
    const body = primFunc.body;
    if (!body || body.type !== 'SeqNode' || body.stmts.length < 2) return;

    const units = buildUnits(body.stmts, collectBufferAccesses(body));
    if (!units) return;
    buildDependenceEdges(units);

    const paramBuffers = new Set();
    for (const [, buffer] of primFunc.bufferMap) paramBuffers.add(buffer);

    const result = scheduleForPeakMemory(units, paramBuffers);
    if (!result) return;

    const replacement = new SeqNode(result.order.map((u) => u.stmt));
    primFunc.body = replacement;
    primFunc._setChild('body', replacement);
    ctx.trace.functionEvent(this.phase, primFunc.name, {
      peakBytes: result.peak,
      originalPeakBytes: result.originalPeak,
    });
  }
}
