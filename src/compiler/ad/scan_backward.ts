import { GradAccumulator } from './grad_accumulator.js';
import { requireVJPRuleOrBarrier, registerRegionVJP, getRegionVJP } from './vjp_registry.js';
import { reduceGradToOperandShape } from './backward_builder.js';
import { regionFreeVars } from '../ir/graph/graph_algorithms.js';
import { scanGroups } from '../ir/graph/ops/control_flow.js';

export { regionFreeVars };
import { REGION_CONTROL_FLOW } from './control_flow_ops.js';
import { TensorType } from '../ir/graph/types.js';
import type { Shape } from '../ir/graph/types.js';
import type { Operation } from '../ir/graph/operation.js';
import type { Value } from '../ir/graph/value.js';
import type { Block } from '../ir/graph/block.js';
import type { IRBuilder } from '../ir/graph/builder.js';
import type { TensorValue } from './vjp_registry.js';

export type MaterializeFn = (v: Value) => Value;
export type NeedsGradSet = { has(id: number): boolean };
export type ScanCheckpoint = 'sqrt' | boolean | number | null;

export type RegionVJPCtx = {
  accumulator: GradAccumulator;
  builder: IRBuilder;
  materialize: MaterializeFn;
  needsGrad: NeedsGradSet;
  scanCheckpoint?: ScanCheckpoint;
};

type DiffBodyResult = {
  forwardYields: Value[];
  gradArgs?: (Value | null)[];
  gradFree?: Map<number, Value | null>;
};

const ALWAYS_NEEDS_GRAD: NeedsGradSet = { has: () => true };

registerRegionVJP('scan', ((op: Operation, ctx: RegionVJPCtx) => buildScanBackward(op, ctx.accumulator, ctx.builder, ctx.materialize, ctx.needsGrad, ctx.scanCheckpoint)) as never);
registerRegionVJP('if', ((op: Operation, ctx: RegionVJPCtx) => buildCondBackward(op, ctx.accumulator, ctx.builder, ctx.materialize, ctx.needsGrad)) as never);

function zeroLike(builder: IRBuilder, val: Value): Value {
  const z = builder.scalarConstant(0, (val.type as TensorType).dtype).getResult(0);
  return builder.broadcast(z, (val.type as TensorType).shape, [], [], val).getResult(0);
}

function sliceStep(builder: IRBuilder, v: Value, t: number): Value {
  const shape = (v.type as TensorType).shape;
  const starts = shape.map((_, i) => (i === 0 ? t : 0));
  const limits = shape.map((d, i) => (i === 0 ? t + 1 : d as number));
  const sliced = builder.slice(v, starts, limits).getResult(0);
  return builder.reshape(sliced, shape.slice(1)).getResult(0);
}

function stackSteps(builder: IRBuilder, steps: readonly Value[], fullShape: Shape): Value {
  const rest = fullShape.slice(1);
  const expanded = steps.map(s => builder.reshape(s, [1, ...rest]).getResult(0));
  if (expanded.length === 1) return builder.reshape(expanded[0], fullShape).getResult(0);
  return builder.concat(expanded, 0).getResult(0);
}

function diffBodyStep(builder: IRBuilder, bodyBlock: Block, argVals: readonly Value[], freeVarMap: ReadonlyMap<number, Value>, gradYields: readonly (Value | null)[] | null, forwardOnly: boolean, constCache: Map<number, Value> = new Map(), freeVars: readonly Value[] = []): DiffBodyResult {
  const map = new Map<number, Value>();
  const vmap = new Map<Value, Value>();
  for (let i = 0; i < bodyBlock.arguments.length; i++) {
    map.set(bodyBlock.arguments[i].id, argVals[i]);
    vmap.set(bodyBlock.arguments[i], argVals[i]);
  }
  for (const [id, v] of freeVarMap) map.set(id, v);
  for (const fv of freeVars) {
    if (freeVarMap.has(fv.id)) vmap.set(fv, freeVarMap.get(fv.id) as Value);
  }

  const bodyOps: Operation[] = [];
  let yieldOp: Operation | null = null;
  for (const op of bodyBlock.ops()) {
    if (op.opName === 'yield') yieldOp = op;
    else bodyOps.push(op);
  }

  const matOperand = (o: Value): Value => {
    if (map.has(o.id)) return map.get(o.id) as Value;
    const def = o.definingOp;
    if (def && def.opName === 'constant') {
      let clonedVal = constCache.get(o.id);
      if (clonedVal === undefined) {
        clonedVal = builder._buildOp('constant', [], [o.type], new Map(def.attributes), null).getResult(0);
        constCache.set(o.id, clonedVal);
      }
      map.set(o.id, clonedVal);
      vmap.set(o, clonedVal);
      return clonedVal;
    }
    return o;
  };

  for (const op of bodyOps) {
    if (REGION_CONTROL_FLOW.has(op.opName)) {
      for (const o of op.operands) { matOperand(o); vmap.set(o, map.get(o.id) ?? o); }
      const cloned = op.clone(vmap);
      builder.block.pushOp(cloned);
      for (let r = 0; r < op.numResults; r++) {
        map.set(op.getResult(r).id, cloned.getResult(r));
        vmap.set(op.getResult(r), cloned.getResult(r));
      }
      continue;
    }
    const operands = op.operands.map(matOperand);
    const cloned = builder._buildOp(op.opName, operands, op.results.map(r => r.type), new Map(op.attributes), null);
    for (let r = 0; r < op.numResults; r++) {
      map.set(op.getResult(r).id, cloned.getResult(r));
      vmap.set(op.getResult(r), cloned.getResult(r));
    }
  }

  const forwardYields = (yieldOp as Operation).operands.map(o => map.get(o.id) as Value);
  if (forwardOnly) return { forwardYields };

  const acc = new GradAccumulator(builder);
  const yields = (yieldOp as Operation).operands;
  const gy = gradYields as readonly (Value | null)[];
  for (let i = 0; i < yields.length; i++) {
    if (gy[i]) acc.accumulate(yields[i].id, gy[i]);
  }
  for (let i = bodyOps.length - 1; i >= 0; i--) {
    const op = bodyOps[i];
    if (op.opName === 'constant') continue;
    const gradOuts = op.results.map(r => acc.get(r.id));
    if (gradOuts.every(g => g === null)) continue;
    if (REGION_CONTROL_FLOW.has(op.opName)) {
      const regionFn = getRegionVJP(op.opName);
      if (regionFn) {
        (regionFn as unknown as (op: Operation, ctx: RegionVJPCtx) => void)(op, { accumulator: acc, builder, materialize: matOperand, needsGrad: ALWAYS_NEEDS_GRAD, scanCheckpoint: null });
      }
      continue;
    }
    const rule = requireVJPRuleOrBarrier(op.opName);
    if (!rule) continue;
    const ctx = {
      builder, op,
      operands: op.operands.map(matOperand) as TensorValue[],
      results: op.results.map(r => map.get(r.id) as Value) as TensorValue[],
      gradOutputs: gradOuts as (TensorValue | null)[],
      attrs: op.attributes,
      full: (value: number, type: TensorType) => builder.broadcast(builder.scalarConstant(value, type.dtype).getResult(0), type.shape, []).getResult(0) as TensorValue,
    };
    const gradIns = rule(ctx);
    if (!gradIns) continue;
    for (let o = 0; o < op.numOperands; o++) {
      if (!gradIns[o]) continue;
      acc.accumulate(op.getOperand(o).id, reduceGradToOperandShape(builder, gradIns[o] as Value, (op.getOperand(o).type as TensorType).shape));
    }
  }

  const gradArgs = bodyBlock.arguments.map(a => acc.get(a.id));
  const gradFree = new Map<number, Value | null>();
  for (const id of freeVarMap.keys()) gradFree.set(id, acc.get(id));
  return { forwardYields, gradArgs, gradFree };
}

export function buildCondBackward(ifOp: Operation, accumulator: GradAccumulator, builder: IRBuilder, materialize: MaterializeFn, needsGrad: NeedsGradSet): void {
  const thenBlock = ifOp.regions[0].blocks[0];
  const elseBlock = ifOp.regions[1].blocks[0];

  const pred = materialize(ifOp.getOperand(0));
  const gradResults: (Value | null)[] = [];
  for (let i = 0; i < ifOp.numResults; i++) gradResults.push(accumulator.get(ifOp.getResult(i).id));

  const thenFree = regionFreeVars(thenBlock);
  const elseFree = regionFreeVars(elseBlock);
  const thenMap = new Map(thenFree.map(v => [v.id, materialize(v)]));
  const elseMap = new Map(elseFree.map(v => [v.id, materialize(v)]));

  const inputOps = ifOp.operands.slice(1);
  const inputVals = inputOps.map(materialize);

  const thenStep = diffBodyStep(builder, thenBlock, inputVals, thenMap, gradResults, false, new Map(), thenFree);
  const elseStep = diffBodyStep(builder, elseBlock, inputVals, elseMap, gradResults, false, new Map(), elseFree);
  const gThenMap = thenStep.gradFree as Map<number, Value | null>;
  const gElseMap = elseStep.gradFree as Map<number, Value | null>;
  const gThenArgs = thenStep.gradArgs as (Value | null)[];
  const gElseArgs = elseStep.gradArgs as (Value | null)[];

  const selectBranch = (target: Value, gt: Value | null | undefined, ge: Value | null | undefined): void => {
    if (!gt && !ge) return;
    const zero = zeroLike(builder, target);
    const predBr = builder.broadcast(pred, (target.type as TensorType).shape, []).getResult(0);
    accumulator.accumulate(target.id, builder.select(predBr, gt ?? zero, ge ?? zero).getResult(0));
  };

  for (let i = 0; i < inputOps.length; i++) {
    if (!needsGrad.has(inputOps[i].id)) continue;
    selectBranch(inputOps[i], gThenArgs[i], gElseArgs[i]);
  }

  const allVars = new Map<number, Value>();
  for (const v of thenFree) allVars.set(v.id, v);
  for (const v of elseFree) allVars.set(v.id, v);

  for (const [id, val] of allVars) {
    if (!needsGrad.has(id)) continue;
    selectBranch(val, gThenMap.get(id), gElseMap.get(id));
  }
}

function resolveSegmentLength(scanCheckpoint: ScanCheckpoint | undefined, T: number): number | null {
  if (!scanCheckpoint || T <= 1) return null;
  if (scanCheckpoint === 'sqrt' || scanCheckpoint === true) return Math.max(1, Math.ceil(Math.sqrt(T)));
  if (typeof scanCheckpoint === 'number' && scanCheckpoint >= 1) {
    const k = Math.floor(scanCheckpoint);
    return k >= T ? null : k;
  }
  return null;
}

export function buildScanBackward(scanOp: Operation, accumulator: GradAccumulator, builder: IRBuilder, materialize: MaterializeFn, needsGrad: NeedsGradSet, scanCheckpoint: ScanCheckpoint | undefined = null): void {
  const bodyBlock = scanOp.regions[0].blocks[0];
  const numCarry = scanOp.getAttr<number>('num_carry') as number;
  const numXs = scanOp.getAttr<number>('num_xs') as number;
  const numYs = scanOp.numResults - numCarry;

  const { carries: carryOps, xs: xsOps, consts: constOps } = scanGroups(scanOp);
  const T = (xsOps[0].type as TensorType).shape[0] as number;

  const freeVars = regionFreeVars(bodyBlock);
  const xsB = xsOps.map(materialize);
  const initCarryB = carryOps.map(materialize);
  const constB = constOps.map(materialize);
  const freeVarMap = new Map<number, Value>(freeVars.map(v => [v.id, materialize(v)]));
  const bodyConstCache = new Map<number, Value>();

  const sliceX = (t: number): Value[] => xsB.map(v => sliceStep(builder, v, t));
  const stepForward = (xt: readonly Value[], c: readonly Value[]): Value[] =>
    diffBodyStep(builder, bodyBlock, [...c, ...xt, ...constB], freeVarMap, null, true, bodyConstCache, freeVars).forwardYields.slice(0, numCarry);

  const gYs: (Value | null)[] = [];
  for (let i = 0; i < numYs; i++) gYs.push(accumulator.get(scanOp.getResult(numCarry + i).id));
  let gCarry: Value[] = [];
  for (let i = 0; i < numCarry; i++) {
    const g = accumulator.get(scanOp.getResult(i).id);
    gCarry.push(g ?? zeroLike(builder, initCarryB[i]));
  }

  const gFree = new Map<number, Value>();
  const gConst: (Value | null)[] = constOps.map(() => null);
  const gXsSteps: Value[][] = xsB.map(() => new Array<Value>(T));

  const backwardStep = (t: number, xt: readonly Value[], carryIn: readonly Value[]): void => {
    const argVals = [...carryIn, ...xt, ...constB];
    const gY_t = gYs.map(g => (g === null ? null : sliceStep(builder, g, t)));
    const gradYields = [...gCarry, ...gY_t];
    const { gradArgs: rawGradArgs, gradFree: rawGradFree } = diffBodyStep(builder, bodyBlock, argVals, freeVarMap, gradYields, false, bodyConstCache, freeVars);
    const gradArgs = rawGradArgs as (Value | null)[];
    const gradFree = rawGradFree as Map<number, Value | null>;

    for (let i = 0; i < numXs; i++) gXsSteps[i][t] = gradArgs[numCarry + i] ?? zeroLike(builder, xt[i]);
    gCarry = [];
    for (let i = 0; i < numCarry; i++) gCarry.push(gradArgs[i] ?? zeroLike(builder, carryIn[i]));
    for (let i = 0; i < constOps.length; i++) {
      const g = gradArgs[numCarry + numXs + i];
      if (!g) continue;
      const prev = gConst[i];
      gConst[i] = prev ? builder.add(prev, g).getResult(0) : g;
    }
    for (const [id, g] of gradFree) {
      if (!g) continue;
      const prev = gFree.get(id);
      gFree.set(id, prev ? builder.add(prev, g).getResult(0) : g);
    }
  };

  const segLen = resolveSegmentLength(scanCheckpoint, T);

  if (!segLen) {
    let carry: Value[] = initCarryB;
    const carriesAtT: Value[][] = [carry];
    const xsSlices: Value[][] = [];
    for (let t = 0; t < T; t++) {
      const xt = sliceX(t);
      xsSlices.push(xt);
      carry = stepForward(xt, carry);
      carriesAtT.push(carry);
    }
    for (let t = T - 1; t >= 0; t--) backwardStep(t, xsSlices[t], carriesAtT[t]);
  } else {
    const numSeg = Math.ceil(T / segLen);
    const boundary = new Array<Value[]>(numSeg);
    let carry: Value[] = initCarryB;
    boundary[0] = carry;
    for (let t = 0; t < T; t++) {
      carry = stepForward(sliceX(t), carry);
      const s = (t + 1) / segLen;
      if (Number.isInteger(s) && s < numSeg) boundary[s] = carry;
    }
    for (let s = numSeg - 1; s >= 0; s--) {
      const start = s * segLen;
      const end = Math.min(start + segLen, T);
      const segXt: Value[][] = [];
      const carriesIn = [boundary[s]];
      let c = boundary[s];
      for (let t = start; t < end; t++) {
        const xt = sliceX(t);
        segXt.push(xt);
        c = stepForward(xt, c);
        carriesIn.push(c);
      }
      for (let t = end - 1; t >= start; t--) backwardStep(t, segXt[t - start], carriesIn[t - start]);
    }
  }

  for (let i = 0; i < numXs; i++) {
    if (!needsGrad.has(xsOps[i].id)) continue;
    accumulator.accumulate(xsOps[i].id, stackSteps(builder, gXsSteps[i], (xsB[i].type as TensorType).shape));
  }
  for (let i = 0; i < numCarry; i++) {
    if (needsGrad.has(carryOps[i].id)) accumulator.accumulate(carryOps[i].id, gCarry[i]);
  }
  for (let i = 0; i < constOps.length; i++) {
    const g = gConst[i];
    if (g && needsGrad.has(constOps[i].id)) accumulator.accumulate(constOps[i].id, g);
  }
  for (const [id, g] of gFree) {
    if (g && needsGrad.has(id)) accumulator.accumulate(id, g);
  }
}
