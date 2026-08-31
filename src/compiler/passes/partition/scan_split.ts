import { Operation } from '../../ir/graph/operation.js';
import { GraphFunction } from '../../ir/graph/function.js';
import { GraphModule } from '../../ir/graph/module.js';
import { materializePartition, isConstantOp, splitGraphForNative, topoSortOps } from './cublas_split.js';
import { isTerminatorOp } from '../../ir/graph/op_traits.js';
import { dtypeBytes } from '../../../util/dtype_map.js';
import { shapeProduct } from '../../ir/graph/types.js';
import type { Shape, TensorType } from '../../ir/graph/types.js';
import type { Block, Region } from '../../ir/graph/block.js';
import type { Value } from '../../ir/graph/value.js';
import type { CompileTarget } from '../../support/config_types.js';
import type { Partition } from '../../analysis/partition_core.js';

type PlanStep = { name: string; inputSlots: number[]; outputSlots: number[] };
type PlanIntermediate = { slot: number; shape: number[]; dtype: string };
type ReturnFixup = { pos: number; kind: string; srcSlot?: number };
type CarrySpec = { a: number; b: number; initSlot: number; finalSlot: number; bytes: number };
type XsSpec = { xtSlot: number; xsSlot: number; stepBytes: number };
type YsSpec = { ytSlot: number; ysSlot: number; stepBytes: number };
type ScanLoopSpec = { T: number; loopStart: number; loopEnd: number; carry: CarrySpec[]; xs: XsSpec[]; ys: YsSpec[] };
type SplitCtx = {
  getSlot(v: Value): number;
  newSlot(shape: Shape, dtype: string): number;
  steps: PlanStep[];
  scanLoops: ScanLoopSpec[];
  addedFuncs: GraphFunction[];
};
type ScanBodyBuild = { bodyFunc: GraphFunction; captured: Value[]; numCarry: number; numYs: number };
export type ScanSplitPlan = {
  numSlots: number;
  argSlots: number[];
  intermediates: PlanIntermediate[];
  steps: PlanStep[];
  returnFixups: ReturnFixup[];
  scanLoops: ScanLoopSpec[];
};

function numel(shape: Shape): number {
  return shapeProduct(shape, -1);
}

function isScanOversized(scanOp: Operation, region: Region, target: CompileTarget): boolean {
  const smem = target.sharedMemoryBytes || 16384;
  let maxBytes = 0;
  for (const op of (region.entryBlock as Block).ops()) {
    if (op.opName === 'yield') continue;
    for (let i = 0; i < op.numResults; i++) {
      const t = op.getResult(i).type as TensorType;
      if (!t || !t.shape) continue;
      const n = numel(t.shape);
      if (n < 0) return true;
      const b = n * dtypeBytes(t.dtype);
      if (b > maxBytes) maxBytes = b;
    }
  }
  const numCarry = scanOp.getAttr<number>('num_carry') as number;
  const numXs = scanOp.getAttr<number>('num_xs') as number;
  let carryBytes = 0;
  for (let i = 0; i < numCarry; i++) {
    const t = scanOp.getOperand(numXs + i).type as TensorType;
    const n = t && t.shape ? numel(t.shape) : -1;
    if (n > 0) carryBytes += n * dtypeBytes(t.dtype);
  }
  return (3 * maxBytes + 2 * carryBytes) > smem;
}

function buildScanBodyFunction(scanOp: Operation, name: string): ScanBodyBuild | null {
  const region = scanOp.regions[0];
  const bodyBlock = region.entryBlock as Block;
  const numCarry = scanOp.getAttr<number>('num_carry') as number;

  const bodyOps: Operation[] = [];
  let yieldOp: Operation | null = null;
  for (const op of bodyBlock.ops()) {
    if (op.opName === 'yield') { yieldOp = op; continue; }
    bodyOps.push(op);
  }
  if (!yieldOp) return null;

  const bodyOpSet = new Set<Operation>(bodyOps);
  const blockArgSet = new Set<Value>(bodyBlock.arguments);
  const captured: Value[] = [], capturedSet = new Set<Value>();
  const constDefs: Operation[] = [], constSet = new Set<Operation>();
  for (const op of bodyOps) {
    for (let i = 0; i < op.numOperands; i++) {
      const v = op.getOperand(i);
      const d = v.definingOp;
      if (blockArgSet.has(v)) continue;
      if (d && bodyOpSet.has(d)) continue;
      if (d && isConstantOp(d)) { if (!constSet.has(d)) { constSet.add(d); constDefs.push(d); } continue; }
      if (!capturedSet.has(v)) { capturedSet.add(v); captured.push(v); }
    }
  }

  const inArgs: Value[] = [...bodyBlock.arguments, ...captured];
  for (const v of inArgs) if (!v.type || !(v.type as TensorType).isFullyStatic) return null;
  const outValues: Value[] = [];
  for (let i = 0; i < yieldOp.numOperands; i++) outValues.push(yieldOp.getOperand(i));

  const bodyFunc = new GraphFunction(name, inArgs.map(v => v.type), outValues.map(v => v.type));
  const valueMap = new Map<Value, Value>();
  const entry = bodyFunc.entryBlock as Block;
  for (let i = 0; i < inArgs.length; i++) valueMap.set(inArgs[i], bodyFunc.args[i]);
  for (const c of constDefs) entry.pushOp(c.clone(valueMap));
  for (const op of bodyOps) entry.pushOp(op.clone(valueMap));
  const returnOperands = outValues.map(v => valueMap.get(v));
  if (returnOperands.some(v => v === undefined)) return null;
  entry.pushOp(new Operation('return', returnOperands as Value[], []));

  return { bodyFunc, captured, numCarry, numYs: outValues.length - numCarry };
}

function inlineNativeSplit(subFunc: GraphFunction, inputs: readonly Value[], outputs: readonly Value[], ctx: SplitCtx, minBoundaries: number): boolean {
  const { getSlot, newSlot, steps, addedFuncs } = ctx;
  const mod = new GraphModule(subFunc.name + '_mod');
  mod.addFunction(subFunc);
  const split = splitGraphForNative(mod, minBoundaries);
  if (split) {
    const pp = split.plan;
    const nin = inputs.length;
    const fixMap = new Map<number, number>();
    let fixOk = true;
    for (const fx of (pp.returnFixups || [])) { if (fx.kind !== 'copy') { fixOk = false; break; } fixMap.set(fx.pos, fx.srcSlot as number); }
    if (fixOk) {
      const retSlot = (j: number): number => { const pos = nin + j; return fixMap.has(pos) ? fixMap.get(pos) as number : pp.argSlots[pos]; };
      const remap = new Map<number, number>();
      for (let i = 0; i < nin; i++) remap.set(pp.argSlots[i], getSlot(inputs[i]));
      for (const it of pp.intermediates) remap.set(it.slot, newSlot(it.shape, it.dtype));
      for (let j = 0; j < outputs.length; j++) { const rs = retSlot(j); if (!remap.has(rs)) remap.set(rs, getSlot(outputs[j])); }
      const mapSlot = (s: number): number | null => { const m = remap.get(s); return m === undefined ? null : m; };
      const newSteps: PlanStep[] = [];
      let ok = true;
      for (const st of pp.steps) {
        const inS = st.inputSlots.map(mapSlot), outS = st.outputSlots.map(mapSlot);
        if (inS.includes(null) || outS.includes(null)) { ok = false; break; }
        newSteps.push({ name: st.name, inputSlots: inS as number[], outputSlots: outS as number[] });
      }
      if (ok) { for (const s of newSteps) steps.push(s); for (const f of mod.functions()) addedFuncs.push(f); return true; }
    }
  }
  steps.push({ name: subFunc.name, inputSlots: inputs.map(getSlot), outputSlots: outputs.map(getSlot) });
  addedFuncs.push(subFunc);
  return true;
}

function emitSegment(segOps: Operation[], name: string, ctx: SplitCtx): boolean {
  const mat = materializePartition({ ops: segOps, opSet: new Set(segOps) } as Partition, name, new Map());
  if (!mat) return false;
  return inlineNativeSplit(mat.subFunc, mat.inputs, mat.outputs, ctx, 1);
}

function emitScanLoop(scanOp: Operation, name: string, ctx: SplitCtx): boolean {
  const { getSlot, newSlot, steps, scanLoops, addedFuncs } = ctx;
  const region = scanOp.regions[0];
  if (!region || !region.entryBlock) return false;
  const built = buildScanBodyFunction(scanOp, name);
  if (!built) return false;
  const { bodyFunc, captured, numCarry, numYs } = built;
  const numXs = scanOp.getAttr<number>('num_xs') as number;

  const bodyModule = new GraphModule(name + '_mod');
  bodyModule.addFunction(bodyFunc);
  const bodySplit = splitGraphForNative(bodyModule, 2);
  const bodyPlan = bodySplit ? bodySplit.plan : null;

  const carryShapes: Shape[] = [], carryDtypes: string[] = [];
  for (let i = 0; i < numCarry; i++) { const t = scanOp.getOperand(numXs + i).type as TensorType; carryShapes.push(t.shape); carryDtypes.push(t.dtype); }
  const xtShapes: Shape[] = [], xtDtypes: string[] = [];
  for (let i = 0; i < numXs; i++) { const t = scanOp.getOperand(i).type as TensorType; xtShapes.push(t.shape.slice(1)); xtDtypes.push(t.dtype); }
  const ytShapes: Shape[] = [], ytDtypes: string[] = [];
  for (let i = 0; i < numYs; i++) { const t = scanOp.getResult(numCarry + i).type as TensorType; ytShapes.push(t.shape.slice(1)); ytDtypes.push(t.dtype); }

  const carryA = carryShapes.map((sh, i) => newSlot(sh, carryDtypes[i]));
  const carryB = carryShapes.map((sh, i) => newSlot(sh, carryDtypes[i]));
  const xtSlots = xtShapes.map((sh, i) => newSlot(sh, xtDtypes[i]));
  let ytSlots: number[] = [];

  const carryInitSlots: number[] = [], carryFinalSlots: number[] = [], invariantSlots: number[] = [];
  for (let i = 0; i < numCarry; i++) carryInitSlots.push(getSlot(scanOp.getOperand(numXs + i)));
  for (let i = 0; i < numCarry; i++) carryFinalSlots.push(getSlot(scanOp.getResult(i)));
  for (const v of captured) invariantSlots.push(getSlot(v));
  const xsSlots: number[] = [], ysSlots: number[] = [];
  for (let i = 0; i < numXs; i++) xsSlots.push(getSlot(scanOp.getOperand(i)));
  for (let i = 0; i < numYs; i++) ysSlots.push(getSlot(scanOp.getResult(numCarry + i)));

  const loopStart = steps.length;
  if (!bodyPlan) {
    ytSlots = ytShapes.map((sh, i) => newSlot(sh, ytDtypes[i]));
    steps.push({ name: bodyFunc.name, inputSlots: [...xtSlots, ...carryA, ...invariantSlots], outputSlots: [...carryB, ...ytSlots] });
    addedFuncs.push(bodyFunc);
  } else {
    const nargs = numXs + numCarry + captured.length;
    const fixMap = new Map<number, number>();
    for (const fx of (bodyPlan.returnFixups || [])) { if (fx.kind !== 'copy') return false; fixMap.set(fx.pos, fx.srcSlot as number); }
    const bodyReturnSlot = (retIdx: number): number => { const pos = nargs + retIdx; return fixMap.has(pos) ? fixMap.get(pos) as number : bodyPlan.argSlots[pos]; };
    const remap = new Map<number, number>();
    for (let i = 0; i < numXs; i++) remap.set(bodyPlan.argSlots[i], xtSlots[i]);
    for (let i = 0; i < numCarry; i++) remap.set(bodyPlan.argSlots[numXs + i], carryA[i]);
    for (let i = 0; i < captured.length; i++) remap.set(bodyPlan.argSlots[numXs + numCarry + i], invariantSlots[i]);
    for (const it of bodyPlan.intermediates) remap.set(it.slot, newSlot(it.shape, it.dtype));
    for (let i = 0; i < numCarry; i++) { const rs = bodyReturnSlot(i); if (!remap.has(rs)) remap.set(rs, carryB[i]); }
    for (let i = 0; i < numYs; i++) {
      const rs = bodyReturnSlot(numCarry + i);
      let ms = remap.get(rs);
      if (ms === undefined) { ms = newSlot(ytShapes[i], ytDtypes[i]); remap.set(rs, ms); }
      ytSlots.push(ms);
    }
    const mapSlot = (s: number): number | null => { const m = remap.get(s); if (m === undefined) return null; return m; };
    for (const st of bodyPlan.steps) {
      const inS = st.inputSlots.map(mapSlot), outS = st.outputSlots.map(mapSlot);
      if (inS.includes(null) || outS.includes(null)) return false;
      steps.push({ name: st.name, inputSlots: inS as number[], outputSlots: outS as number[] });
    }
    for (const f of bodyModule.functions()) addedFuncs.push(f);
  }
  const loopEnd = steps.length;

  const T = (scanOp.getOperand(0).type as TensorType).shape[0];
  if (typeof T !== 'number' || T < 0) return false;

  scanLoops.push({
    T, loopStart, loopEnd,
    carry: carryShapes.map((sh, i) => ({ a: carryA[i], b: carryB[i], initSlot: carryInitSlots[i], finalSlot: carryFinalSlots[i], bytes: numel(sh) * dtypeBytes(carryDtypes[i]) })),
    xs: xtSlots.map((s, i) => ({ xtSlot: s, xsSlot: xsSlots[i], stepBytes: numel(xtShapes[i]) * dtypeBytes(xtDtypes[i]) })),
    ys: ytSlots.map((s, i) => ({ ytSlot: s, ysSlot: ysSlots[i], stepBytes: numel(ytShapes[i]) * dtypeBytes(ytDtypes[i]) })),
  });
  return true;
}

export function splitGraphForScan(graphModule: GraphModule, target: CompileTarget | null, force = false): { plan: ScanSplitPlan } | null {
  if (!target || typeof target.isWebGPU !== 'function' || !target.isWebGPU()) return null;
  if (graphModule.functionCount !== 1) return null;
  const func = graphModule.functions().next().value as GraphFunction;
  const retOp = func.getReturnOp() as Operation;
  if (!retOp) return null;

  const scans: Operation[] = [];
  for (const op of func.ops()) if (op.opName === 'scan') scans.push(op);
  if (scans.length === 0) return null;

  const reach = new Set<Operation>();
  const visit = (op: Operation | null | undefined): void => { if (!op || reach.has(op)) return; reach.add(op); for (let i = 0; i < op.numOperands; i++) visit(op.getOperand(i).definingOp); };
  for (let i = 0; i < retOp.numOperands; i++) visit(retOp.getOperand(i).definingOp);
  for (const s of scans) if (!reach.has(s)) return null;

  if (!force) {
    let anyOversized = false;
    for (const s of scans) { const r = s.regions[0]; if (r && r.entryBlock && isScanOversized(s, r, target)) { anyOversized = true; break; } }
    if (!anyOversized) return null;
  }

  const reachList: Operation[] = [];
  for (const op of reach) {
    if (isConstantOp(op) || isTerminatorOp(op.opName)) continue;
    reachList.push(op);
  }
  const ordered = topoSortOps(reachList);

  const slotOf = new Map<Value, number>();
  let nextSlot = 0;
  const getSlot = (v: Value): number => { let s = slotOf.get(v); if (s === undefined) { s = nextSlot++; slotOf.set(v, s); } return s; };
  for (const arg of func.args) getSlot(arg);
  const intermediates: PlanIntermediate[] = [];
  const newSlot = (shape: Shape, dtype: string): number => { const s = nextSlot++; intermediates.push({ slot: s, shape: [...shape] as number[], dtype }); return s; };

  const steps: PlanStep[] = [];
  const scanLoops: ScanLoopSpec[] = [];
  const addedFuncs: GraphFunction[] = [];
  const ctx: SplitCtx = { getSlot, newSlot, steps, scanLoops, addedFuncs };
  const scanSet = new Set<Operation>(scans);

  let segOps: Operation[] = [];
  let segIdx = 0, scanIdx = 0;
  for (const op of ordered) {
    if (scanSet.has(op)) {
      if (segOps.length && !emitSegment(segOps, func.name + '_seg' + (segIdx++), ctx)) return null;
      segOps = [];
      if (!emitScanLoop(op, func.name + '_scan' + (scanIdx++), ctx)) return null;
    } else {
      segOps.push(op);
    }
  }
  if (segOps.length && !emitSegment(segOps, func.name + '_seg' + (segIdx++), ctx)) return null;

  const argSlots: number[] = func.args.map(getSlot);
  const returnFixups: ReturnFixup[] = [];
  const usedRet = new Set<number>();
  for (let i = 0; i < retOp.numOperands; i++) {
    const v = retOp.getOperand(i);
    const pos = argSlots.length;
    if (slotOf.has(v) && !(v.isBlockArgument && v.isBlockArgument())) {
      const s = getSlot(v);
      if (!usedRet.has(s)) { usedRet.add(s); argSlots.push(s); }
      else { argSlots.push(nextSlot++); returnFixups.push({ pos, kind: 'copy', srcSlot: s }); }
    } else if (v.isBlockArgument && v.isBlockArgument()) {
      argSlots.push(nextSlot++); returnFixups.push({ pos, kind: 'copy', srcSlot: getSlot(v) });
    } else return null;
  }

  const argSlotSet = new Set<number>(argSlots);
  const seen = new Set<number>();
  for (const [v, s] of slotOf) {
    if (argSlotSet.has(s) || seen.has(s)) continue;
    seen.add(s);
    const vt = v.type as TensorType;
    if (!vt || !vt.isFullyStatic) return null;
    if (!intermediates.some(it => it.slot === s)) intermediates.push({ slot: s, shape: [...vt.shape] as number[], dtype: vt.dtype });
  }

  graphModule.removeFunction(func.name);
  for (const f of addedFuncs) graphModule.addFunction(f);

  return { plan: { numSlots: nextSlot, argSlots, intermediates, steps, returnFixups, scanLoops } };
}
