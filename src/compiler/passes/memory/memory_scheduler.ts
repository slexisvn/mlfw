import { PrimFuncPass } from '../tir_pass.js';
import { SeqNode } from '../../ir/tensor/nodes.js';
import { walk } from '../../ir/ir_visitor.js';
import { collectBufferAccesses, AccessKind } from '../../analysis/buffer_access.js';
import { regionHull } from '../../schedule/dep_analysis.js';
import { linkAccessUnits } from '../../schedule/block_scope.js';
import type { Buffer } from '../../ir/tensor/buffer.js';
import type { PrimFunc, TirNode } from '../../ir/tensor/nodes.js';
import type { IRNode } from '../../ir/ir_visitor.js';
import type { TirPassCtx } from '../tir_pass.js';
import type { CompilerConfig } from '../../support/config_types.js';

type BufferRegionHull = ReturnType<typeof regionHull>;
type AccessInfo = ReturnType<typeof collectBufferAccesses>;
type DependenceEntry = { unit: ScheduleUnit; position: number; read: BufferRegionHull | null; write: BufferRegionHull | null };
export type PeakScheduleResult = { order: ScheduleUnit[]; peak: number; originalPeak: number };

export const UNSEQUENCED_EFFECT_NODES = new Set(['CallExternNode', 'SyncThreadsNode', 'WhileNode']);

class ScheduleUnit {
  index: number;
  stmt: TirNode;
  reads: Map<Buffer, BufferRegionHull>;
  writes: Map<Buffer, BufferRegionHull>;
  temporaries: Buffer[];
  predecessors: Set<ScheduleUnit>;
  successors: Set<ScheduleUnit>;

  constructor(index: number, stmt: TirNode) {
    this.index = index;
    this.stmt = stmt;
    this.reads = new Map();
    this.writes = new Map();
    this.temporaries = [];
    this.predecessors = new Set();
    this.successors = new Set();
  }
}

function buildUnits(stmts: readonly TirNode[], accessInfo: AccessInfo): ScheduleUnit[] | null {
  const accessOf = new Map<unknown, AccessInfo['order'][number]>();
  for (const access of accessInfo.order) accessOf.set(access.node, access);

  const units: ScheduleUnit[] = [];
  for (let i = 0; i < stmts.length; i++) {
    const unit = new ScheduleUnit(i, stmts[i]);
    const readRegions = new Map<Buffer, BufferRegionHull[]>();
    const writeRegions = new Map<Buffer, BufferRegionHull[]>();
    let movable = true;
    walk(stmts[i] as unknown as IRNode, (node: IRNode) => {
      if (UNSEQUENCED_EFFECT_NODES.has(node.type)) movable = false;
      const access = accessOf.get(node);
      if (!access) return;
      const target = access.kind === AccessKind.WRITE ? writeRegions : readRegions;
      let list = target.get(access.buffer);
      if (!list) { list = []; target.set(access.buffer, list); }
      list.push(access.regions as BufferRegionHull);
    });
    if (!movable) return null;
    for (const [buffer, list] of readRegions) unit.reads.set(buffer, regionHull(list));
    for (const [buffer, list] of writeRegions) unit.writes.set(buffer, regionHull(list));
    units.push(unit);
  }
  return units;
}

function buildDependenceEdges(units: readonly ScheduleUnit[]): void {
  const byBuffer = new Map<Buffer, Map<ScheduleUnit, DependenceEntry>>();
  const entryFor = (buffer: Buffer, unit: ScheduleUnit): DependenceEntry => {
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
    linkAccessUnits([...entries.values()] as never, buffer as never, ((src: DependenceEntry, dst: DependenceEntry) => {
      src.unit.successors.add(dst.unit);
      dst.unit.predecessors.add(src.unit);
    }) as never);
  }
}

function assignTemporaries(units: readonly ScheduleUnit[], paramBuffers: ReadonlySet<Buffer>): Map<Buffer, number> {
  const sizes = new Map<Buffer, number>();
  for (const unit of units) {
    const touched = new Set<Buffer>([...unit.reads.keys(), ...unit.writes.keys()]);
    for (const buffer of touched) {
      if (paramBuffers.has(buffer)) continue;
      if (!sizes.has(buffer)) {
        const bytes = buffer.sizeInBytes() as number;
        if (!Number.isFinite(bytes) || bytes <= 0) continue;
        sizes.set(buffer, bytes);
      }
      unit.temporaries.push(buffer);
    }
  }
  return sizes;
}

function useCounts(units: readonly ScheduleUnit[], sizes: ReadonlyMap<Buffer, number>): Map<Buffer, number> {
  const counts = new Map<Buffer, number>();
  for (const buffer of sizes.keys()) counts.set(buffer, 0);
  for (const unit of units) {
    for (const buffer of unit.temporaries) counts.set(buffer, (counts.get(buffer) as number) + 1);
  }
  return counts;
}

function simulatePeak(order: readonly ScheduleUnit[], sizes: ReadonlyMap<Buffer, number>): number {
  const remaining = useCounts(order, sizes);
  const live = new Set<Buffer>();
  let current = 0;
  let peak = 0;
  for (const unit of order) {
    for (const buffer of unit.temporaries) {
      if (live.has(buffer)) continue;
      live.add(buffer);
      current += sizes.get(buffer) as number;
    }
    if (current > peak) peak = current;
    for (const buffer of unit.temporaries) {
      const left = (remaining.get(buffer) as number) - 1;
      remaining.set(buffer, left);
      if (left === 0 && live.delete(buffer)) current -= sizes.get(buffer) as number;
    }
  }
  return peak;
}

function readyFrontier(units: readonly ScheduleUnit[]): { pending: Map<ScheduleUnit, number>; ready: ScheduleUnit[] } {
  const pending = new Map<ScheduleUnit, number>();
  const ready: ScheduleUnit[] = [];
  for (const unit of units) {
    pending.set(unit, unit.predecessors.size);
    if (unit.predecessors.size === 0) ready.push(unit);
  }
  return { pending, ready };
}

function listSchedule(units: readonly ScheduleUnit[], sizes: ReadonlyMap<Buffer, number>): ScheduleUnit[] | null {
  const remaining = useCounts(units, sizes);
  const live = new Set<Buffer>();
  const { pending, ready } = readyFrontier(units);
  const order: ScheduleUnit[] = [];

  const benefit = (unit: ScheduleUnit): number => {
    let score = 0;
    for (const buffer of unit.temporaries) {
      const bytes = sizes.get(buffer) as number;
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
      const left = (remaining.get(buffer) as number) - 1;
      remaining.set(buffer, left);
      if (left === 0) live.delete(buffer);
    }
    for (const next of unit.successors) {
      const left = (pending.get(next) as number) - 1;
      pending.set(next, left);
      if (left === 0) ready.push(next);
    }
  }
  return order.length === units.length ? order : null;
}

function subgraphBytes(units: readonly ScheduleUnit[], sizes: ReadonlyMap<Buffer, number>): Map<ScheduleUnit, number> {
  const total = new Map<ScheduleUnit, number>();
  for (let i = units.length - 1; i >= 0; i--) {
    const unit = units[i];
    let bytes = 0;
    for (const buffer of unit.temporaries) bytes += sizes.get(buffer) as number;
    for (const next of unit.successors) bytes += total.get(next) || 0;
    total.set(unit, bytes);
  }
  return total;
}

function dfsSchedule(units: readonly ScheduleUnit[], sizes: ReadonlyMap<Buffer, number>): ScheduleUnit[] | null {
  const weight = subgraphBytes(units, sizes);
  const { pending, ready } = readyFrontier(units);
  const order: ScheduleUnit[] = [];
  const stack = ready.sort((a, b) => ((weight.get(b) as number) - (weight.get(a) as number)) || (a.index - b.index));

  while (stack.length > 0) {
    const unit = stack.pop() as ScheduleUnit;
    order.push(unit);
    const released: ScheduleUnit[] = [];
    for (const next of unit.successors) {
      const left = (pending.get(next) as number) - 1;
      pending.set(next, left);
      if (left === 0) released.push(next);
    }
    released.sort((a, b) => ((weight.get(a) as number) - (weight.get(b) as number)) || (b.index - a.index));
    for (const next of released) stack.push(next);
  }
  return order.length === units.length ? order : null;
}

export function scheduleForPeakMemory(units: readonly ScheduleUnit[], paramBuffers: ReadonlySet<Buffer>): PeakScheduleResult | null {
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

export function buildScheduleUnits(stmts: TirNode[], paramBuffers: ReadonlySet<Buffer>): ScheduleUnit[] | null {
  const units = buildUnits(stmts, collectBufferAccesses(new SeqNode(stmts)));
  if (!units) return null;
  buildDependenceEdges(units);
  assignTemporaries(units, paramBuffers);
  return units;
}

export class MemorySchedulePass extends PrimFuncPass {
  config: CompilerConfig;

  constructor(config: CompilerConfig) {
    super('MemorySchedulePass', 'memoryScheduling');
    this.config = config;
  }

  _keepOrder(ctx: TirPassCtx, primFunc: PrimFunc, reason: string): void {
    if (ctx.trace.explainsEnabled) ctx.trace.explain('memory', primFunc.name, 'kept-order', reason, {});
  }

  override run(primFunc: PrimFunc, ctx: TirPassCtx): void {
    const body = primFunc.body as SeqNode;
    if (!body || body.type !== 'SeqNode' || body.stmts.length < 2) {
      return this._keepOrder(ctx, primFunc, 'the body is one statement, so there is no order left to choose');
    }

    const units = buildUnits(body.stmts, collectBufferAccesses(body));
    if (!units) {
      return this._keepOrder(ctx, primFunc, 'a statement has an effect that cannot be sequenced against the others, so nothing may move');
    }
    buildDependenceEdges(units);

    const paramBuffers = new Set<Buffer>();
    for (const [, buffer] of primFunc.bufferMap) paramBuffers.add(buffer);

    const result = scheduleForPeakMemory(units, paramBuffers);
    if (!result) {
      return this._keepOrder(ctx, primFunc, 'no order that respects the dependences held fewer bytes live than the one already there');
    }

    const replacement = new SeqNode(result.order.map((u) => u.stmt));
    primFunc.body = replacement;
    primFunc._setChild('body', replacement);
    ctx.trace.functionEvent(this.phase, primFunc.name, {
      peakBytes: result.peak,
      originalPeakBytes: result.originalPeak,
    });
    if (ctx.trace.explainsEnabled) {
      ctx.trace.explain('memory', primFunc.name, 'reordered',
        'moving each producer next to its consumer ends the buffer lifetime sooner, so fewer temporaries overlap',
        { peakBytes: result.peak, originalPeakBytes: result.originalPeak });
    }
  }
}
