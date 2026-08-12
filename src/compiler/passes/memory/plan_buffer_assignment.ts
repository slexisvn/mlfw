import { dtypeBytes } from '../../../util/dtype_map.js';
import { isElementwiseOp, isTerminatorOp } from '../../ir/graph/op_traits.js';
import { shapeProduct } from '../../ir/graph/types.js';
import type { GraphModule } from '../../ir/graph/module.js';
import type { Operation } from '../../ir/graph/operation.js';
import type { Value } from '../../ir/graph/value.js';
import type { Shape, TensorType } from '../../ir/graph/types.js';

export type PlanBuffers = { slotBuffer: number[]; bufferBytes: number[]; donated: number };
export type PlanDonation = { step: number; from: number; to: number };
export type PlanMemoryReport = { slotBytes: number; bufferBytes: number; buffers: number; donated: number };

type PlanStep = { name: string; inputSlots: readonly number[]; outputSlots: readonly number[] };
type PlanIntermediate = { slot: number; shape: readonly number[]; dtype: string };
type PlanFixup = { pos: number; kind: string; srcSlot?: number };
type PlanScanLoop = {
  loopStart: number;
  loopEnd: number;
  carry: readonly { a: number; b: number; initSlot: number; finalSlot: number }[];
  xs: readonly { xtSlot: number; xsSlot: number }[];
  ys: readonly { ytSlot: number; ysSlot: number }[];
};

export type AssignablePlan = {
  numSlots: number;
  argSlots: readonly number[];
  intermediates: readonly PlanIntermediate[];
  steps: readonly PlanStep[];
  returnFixups?: readonly PlanFixup[];
  scanLoops?: readonly PlanScanLoop[];
  buffers?: PlanBuffers;
};

function slotBytes(it: PlanIntermediate): number {
  const numel = shapeProduct(it.shape as Shape, -1);
  return numel < 0 ? 0 : Math.max(numel, 1) * dtypeBytes(it.dtype);
}

function pinnedSlots(plan: AssignablePlan, bytes: readonly number[]): Set<number> {
  const pinned = new Set<number>(plan.argSlots);
  for (const fx of plan.returnFixups || []) {
    if (typeof fx.srcSlot === 'number') pinned.add(fx.srcSlot);
  }
  for (const loop of plan.scanLoops || []) {
    for (const c of loop.carry) { pinned.add(c.a); pinned.add(c.b); pinned.add(c.initSlot); pinned.add(c.finalSlot); }
    for (const x of loop.xs) { pinned.add(x.xtSlot); pinned.add(x.xsSlot); }
    for (const y of loop.ys) { pinned.add(y.ytSlot); pinned.add(y.ysSlot); }
  }
  for (let s = 0; s < plan.numSlots; s++) {
    if (bytes[s] <= 0) pinned.add(s);
  }
  return pinned;
}

function liveRanges(plan: AssignablePlan, pinned: Set<number>): { def: number[]; lastUse: number[] } {
  const def = new Array<number>(plan.numSlots).fill(-1);
  const lastUse = new Array<number>(plan.numSlots).fill(-1);

  for (let k = 0; k < plan.steps.length; k++) {
    const step = plan.steps[k];
    for (const s of step.inputSlots) {
      if (def[s] < 0) pinned.add(s);
      if (k > lastUse[s]) lastUse[s] = k;
    }
    for (const s of step.outputSlots) {
      if (def[s] < 0) def[s] = k;
      if (k > lastUse[s]) lastUse[s] = k;
    }
  }

  for (const loop of plan.scanLoops || []) {
    const end = Math.max(loop.loopEnd - 1, loop.loopStart);
    const stretch = (s: number): void => {
      if (def[s] > loop.loopStart && def[s] < loop.loopEnd) def[s] = loop.loopStart;
      if (lastUse[s] >= loop.loopStart && lastUse[s] < end) lastUse[s] = end;
    };
    for (let k = loop.loopStart; k < loop.loopEnd && k < plan.steps.length; k++) {
      for (const s of plan.steps[k].inputSlots) stretch(s);
      for (const s of plan.steps[k].outputSlots) stretch(s);
    }
  }

  return { def, lastUse };
}

export function assignPlanBuffers(plan: AssignablePlan, donations: readonly PlanDonation[] = []): PlanBuffers | null {
  if (plan.numSlots === 0 || plan.steps.length === 0) return null;

  const bytes = new Array<number>(plan.numSlots).fill(0);
  for (const it of plan.intermediates) bytes[it.slot] = slotBytes(it);

  const pinned = pinnedSlots(plan, bytes);
  const { def, lastUse } = liveRanges(plan, pinned);

  const slotBuffer = new Array<number>(plan.numSlots).fill(-1);
  const bufferBytes: number[] = [];
  const bufferOwner: number[] = [];
  const newBuffer = (size: number, owner: number): number => {
    bufferBytes.push(size);
    bufferOwner.push(owner);
    return bufferBytes.length - 1;
  };

  for (let s = 0; s < plan.numSlots; s++) {
    if (pinned.has(s)) slotBuffer[s] = newBuffer(bytes[s], s);
  }

  const expiring: number[][] = plan.steps.map(() => []);
  for (let s = 0; s < plan.numSlots; s++) {
    if (!pinned.has(s) && lastUse[s] >= 0) expiring[lastUse[s]].push(s);
  }

  const donationsByStep = new Map<number, PlanDonation[]>();
  for (const d of donations) {
    const list = donationsByStep.get(d.step);
    if (list) list.push(d);
    else donationsByStep.set(d.step, [d]);
  }

  const freeList: number[] = [];
  const take = (size: number, owner: number): number => {
    let fit = -1;
    let fitBytes = Infinity;
    let largest = -1;
    let largestBytes = -1;
    for (let i = 0; i < freeList.length; i++) {
      const b = bufferBytes[freeList[i]];
      if (b >= size && b < fitBytes) { fitBytes = b; fit = i; }
      if (b > largestBytes) { largestBytes = b; largest = i; }
    }
    const pick = fit >= 0 ? fit : largest;
    if (pick < 0) return newBuffer(size, owner);
    const buf = freeList[pick];
    freeList[pick] = freeList[freeList.length - 1];
    freeList.pop();
    if (bufferBytes[buf] < size) bufferBytes[buf] = size;
    bufferOwner[buf] = owner;
    return buf;
  };

  let donated = 0;
  for (let k = 0; k < plan.steps.length; k++) {
    if (k > 0) {
      for (const s of expiring[k - 1]) {
        const buf = slotBuffer[s];
        if (buf >= 0 && bufferOwner[buf] === s) freeList.push(buf);
      }
    }
    for (const d of donationsByStep.get(k) || []) {
      if (pinned.has(d.from) || pinned.has(d.to)) continue;
      if (slotBuffer[d.to] >= 0 || lastUse[d.from] !== k || def[d.to] !== k) continue;
      const buf = slotBuffer[d.from];
      if (buf < 0 || bufferOwner[buf] !== d.from) continue;
      if (bufferBytes[buf] < bytes[d.to]) bufferBytes[buf] = bytes[d.to];
      bufferOwner[buf] = d.to;
      slotBuffer[d.to] = buf;
      donated++;
    }
    for (const s of plan.steps[k].outputSlots) {
      if (slotBuffer[s] >= 0) continue;
      slotBuffer[s] = take(bytes[s], s);
    }
  }

  for (let s = 0; s < plan.numSlots; s++) {
    if (slotBuffer[s] < 0) slotBuffer[s] = newBuffer(bytes[s], s);
  }

  return bufferBytes.length < plan.numSlots ? { slotBuffer, bufferBytes, donated } : null;
}

export function planMemoryReport(plan: AssignablePlan, buffers: PlanBuffers): PlanMemoryReport {
  let total = 0;
  for (const it of plan.intermediates) total += slotBytes(it);
  let assigned = 0;
  for (const b of buffers.bufferBytes) assigned += b;
  return { slotBytes: total, bufferBytes: assigned, buffers: buffers.bufferBytes.length, donated: buffers.donated };
}

function sameLayout(a: TensorType | null, b: TensorType | null): boolean {
  if (!a || !b || a.dtype !== b.dtype || a.rank !== b.rank) return false;
  for (let i = 0; i < a.rank; i++) {
    if (a.shape[i] !== b.shape[i]) return false;
  }
  return true;
}

function passThroughResults(user: Operation, operandIndex: number, valueType: TensorType): Value[] | null {
  const hasRegions = user.regions && user.regions.length > 0;
  if (isElementwiseOp(user.opName) && !hasRegions) {
    const results: Value[] = [];
    for (let i = 0; i < user.numResults; i++) {
      const result = user.getResult(i);
      if (!sameLayout(result.type as TensorType, valueType)) return null;
      results.push(result);
    }
    return results;
  }
  if (!hasRegions || user.regions.length !== 1) return null;

  const region = user.regions[0];
  const block = region.entryBlock;
  if (!block || region.blocks.length !== 1 || block.arguments.length !== user.numOperands) return null;
  const inner = block.arguments[operandIndex];
  if (!sameLayout(inner.type as TensorType, valueType)) return null;

  const positions = terminatorPositionsFrom(inner);
  if (positions === null) return null;
  const results: Value[] = [];
  for (const p of positions) {
    if (p >= user.numResults) return null;
    const result = user.getResult(p);
    if (!sameLayout(result.type as TensorType, valueType)) return null;
    results.push(result);
  }
  return results;
}

function terminatorPositionsFrom(source: Value): Set<number> | null {
  const seen = new Set<Value>([source]);
  const worklist: Value[] = [source];
  const positions = new Set<number>();
  while (worklist.length > 0) {
    const value = worklist.pop() as Value;
    const valueType = value.type as TensorType;
    for (const use of value.uses()) {
      if (isTerminatorOp(use.user.opName)) { positions.add(use.operandIndex); continue; }
      const results = passThroughResults(use.user, use.operandIndex, valueType);
      if (results === null) return null;
      for (const result of results) {
        if (seen.has(result)) continue;
        seen.add(result);
        worklist.push(result);
      }
    }
  }
  return positions;
}

export function computePlanDonations(module: GraphModule, plan: AssignablePlan): PlanDonation[] {
  const donations: PlanDonation[] = [];
  for (let k = 0; k < plan.steps.length; k++) {
    const step = plan.steps[k];
    if (step.outputSlots.length !== 1) continue;
    const func = module.getFunction(step.name);
    if (!func || func.args.length !== step.inputSlots.length) continue;
    const retOp = func.getReturnOp();
    if (!retOp || retOp.numOperands !== 1) continue;

    const outType = retOp.getOperand(0).type as TensorType;
    if (!outType || !outType.isFullyStatic) continue;

    for (let i = 0; i < func.args.length; i++) {
      const arg = func.args[i];
      if (!sameLayout(arg.type as TensorType, outType)) continue;
      const positions = terminatorPositionsFrom(arg);
      if (positions === null || !positions.has(0)) continue;
      donations.push({ step: k, from: step.inputSlots[i], to: step.outputSlots[0] });
      break;
    }
  }
  return donations;
}
