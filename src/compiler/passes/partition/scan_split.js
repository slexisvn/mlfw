import { Operation } from '../../ir/graph/operation.js';
import { GraphFunction } from '../../ir/graph/function.js';
import { GraphModule } from '../../ir/graph/module.js';
import { materializePartition, isConstantOp, splitGraphForNative, topoSortOps } from './cublas_split.js';
import { isTerminatorOp } from '../../ir/graph/op_traits.js';
import { dtypeBytes } from '../../../util/dtype_map.js';
import { shapeProduct } from '../../ir/graph/types.js';

function numel(shape) {
  return shapeProduct(shape, -1);
}

function isScanOversized(scanOp, region, target) {
  const smem = target.sharedMemoryBytes || 16384;
  let maxBytes = 0;
  for (const op of region.entryBlock.ops()) {
    if (op.opName === 'yield') continue;
    for (let i = 0; i < op.numResults; i++) {
      const t = op.getResult(i).type;
      if (!t || !t.shape) continue;
      const n = numel(t.shape);
      if (n < 0) return true;
      const b = n * dtypeBytes(t.dtype);
      if (b > maxBytes) maxBytes = b;
    }
  }
  const numCarry = scanOp.getAttr('num_carry');
  const numXs = scanOp.getAttr('num_xs');
  let carryBytes = 0;
  for (let i = 0; i < numCarry; i++) {
    const t = scanOp.getOperand(numXs + i).type;
    const n = t && t.shape ? numel(t.shape) : -1;
    if (n > 0) carryBytes += n * dtypeBytes(t.dtype);
  }
  return (3 * maxBytes + 2 * carryBytes) > smem;
}

function buildScanBodyFunction(scanOp, name) {
  const region = scanOp.regions[0];
  const bodyBlock = region.entryBlock;
  const numCarry = scanOp.getAttr('num_carry');

  const bodyOps = [];
  let yieldOp = null;
  for (const op of bodyBlock.ops()) {
    if (op.opName === 'yield') { yieldOp = op; continue; }
    bodyOps.push(op);
  }
  if (!yieldOp) return null;

  const bodyOpSet = new Set(bodyOps);
  const blockArgSet = new Set(bodyBlock.arguments);
  const captured = [], capturedSet = new Set();
  const constDefs = [], constSet = new Set();
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

  const inArgs = [...bodyBlock.arguments, ...captured];
  for (const v of inArgs) if (!v.type || !v.type.isFullyStatic) return null;
  const outValues = [];
  for (let i = 0; i < yieldOp.numOperands; i++) outValues.push(yieldOp.getOperand(i));

  const bodyFunc = new GraphFunction(name, inArgs.map(v => v.type), outValues.map(v => v.type));
  const valueMap = new Map();
  for (let i = 0; i < inArgs.length; i++) valueMap.set(inArgs[i], bodyFunc.args[i]);
  for (const c of constDefs) bodyFunc.entryBlock.pushOp(c.clone(valueMap));
  for (const op of bodyOps) bodyFunc.entryBlock.pushOp(op.clone(valueMap));
  const returnOperands = outValues.map(v => valueMap.get(v));
  if (returnOperands.some(v => v === undefined)) return null;
  bodyFunc.entryBlock.pushOp(new Operation('return', returnOperands, []));

  return { bodyFunc, captured, numCarry, numYs: outValues.length - numCarry };
}

function inlineNativeSplit(subFunc, inputs, outputs, ctx, minBoundaries) {
  const { getSlot, newSlot, steps, addedFuncs } = ctx;
  const mod = new GraphModule(subFunc.name + '_mod');
  mod.addFunction(subFunc);
  const split = splitGraphForNative(mod, minBoundaries);
  if (split) {
    const pp = split.plan;
    const nin = inputs.length;
    const fixMap = new Map();
    let fixOk = true;
    for (const fx of (pp.returnFixups || [])) { if (fx.kind !== 'copy') { fixOk = false; break; } fixMap.set(fx.pos, fx.srcSlot); }
    if (fixOk) {
      const retSlot = (j) => { const pos = nin + j; return fixMap.has(pos) ? fixMap.get(pos) : pp.argSlots[pos]; };
      const remap = new Map();
      for (let i = 0; i < nin; i++) remap.set(pp.argSlots[i], getSlot(inputs[i]));
      for (const it of pp.intermediates) remap.set(it.slot, newSlot(it.shape, it.dtype));
      for (let j = 0; j < outputs.length; j++) { const rs = retSlot(j); if (!remap.has(rs)) remap.set(rs, getSlot(outputs[j])); }
      const mapSlot = (s) => { const m = remap.get(s); return m === undefined ? null : m; };
      const newSteps = [];
      let ok = true;
      for (const st of pp.steps) {
        const inS = st.inputSlots.map(mapSlot), outS = st.outputSlots.map(mapSlot);
        if (inS.includes(null) || outS.includes(null)) { ok = false; break; }
        newSteps.push({ name: st.name, inputSlots: inS, outputSlots: outS });
      }
      if (ok) { for (const s of newSteps) steps.push(s); for (const f of mod.functions()) addedFuncs.push(f); return true; }
    }
  }
  steps.push({ name: subFunc.name, inputSlots: inputs.map(getSlot), outputSlots: outputs.map(getSlot) });
  addedFuncs.push(subFunc);
  return true;
}

function emitSegment(segOps, name, ctx) {
  const mat = materializePartition({ ops: segOps, opSet: new Set(segOps) }, name, new Map());
  if (!mat) return false;
  return inlineNativeSplit(mat.subFunc, mat.inputs, mat.outputs, ctx, 1);
}

function emitScanLoop(scanOp, name, ctx) {
  const { getSlot, newSlot, steps, scanLoops, addedFuncs } = ctx;
  const region = scanOp.regions[0];
  if (!region || !region.entryBlock) return false;
  const built = buildScanBodyFunction(scanOp, name);
  if (!built) return false;
  const { bodyFunc, captured, numCarry, numYs } = built;
  const numXs = scanOp.getAttr('num_xs');

  const bodyModule = new GraphModule(name + '_mod');
  bodyModule.addFunction(bodyFunc);
  const bodySplit = splitGraphForNative(bodyModule, 2);
  const bodyPlan = bodySplit ? bodySplit.plan : null;

  const carryShapes = [], carryDtypes = [];
  for (let i = 0; i < numCarry; i++) { const t = scanOp.getOperand(numXs + i).type; carryShapes.push(t.shape); carryDtypes.push(t.dtype); }
  const xtShapes = [], xtDtypes = [];
  for (let i = 0; i < numXs; i++) { const t = scanOp.getOperand(i).type; xtShapes.push(t.shape.slice(1)); xtDtypes.push(t.dtype); }
  const ytShapes = [], ytDtypes = [];
  for (let i = 0; i < numYs; i++) { const t = scanOp.getResult(numCarry + i).type; ytShapes.push(t.shape.slice(1)); ytDtypes.push(t.dtype); }

  const carryA = carryShapes.map((sh, i) => newSlot(sh, carryDtypes[i]));
  const carryB = carryShapes.map((sh, i) => newSlot(sh, carryDtypes[i]));
  const xtSlots = xtShapes.map((sh, i) => newSlot(sh, xtDtypes[i]));
  let ytSlots = [];

  const carryInitSlots = [], carryFinalSlots = [], invariantSlots = [];
  for (let i = 0; i < numCarry; i++) carryInitSlots.push(getSlot(scanOp.getOperand(numXs + i)));
  for (let i = 0; i < numCarry; i++) carryFinalSlots.push(getSlot(scanOp.getResult(i)));
  for (const v of captured) invariantSlots.push(getSlot(v));
  const xsSlots = [], ysSlots = [];
  for (let i = 0; i < numXs; i++) xsSlots.push(getSlot(scanOp.getOperand(i)));
  for (let i = 0; i < numYs; i++) ysSlots.push(getSlot(scanOp.getResult(numCarry + i)));

  const loopStart = steps.length;
  if (!bodyPlan) {
    ytSlots = ytShapes.map((sh, i) => newSlot(sh, ytDtypes[i]));
    steps.push({ name: bodyFunc.name, inputSlots: [...xtSlots, ...carryA, ...invariantSlots], outputSlots: [...carryB, ...ytSlots] });
    addedFuncs.push(bodyFunc);
  } else {
    const nargs = numXs + numCarry + captured.length;
    const fixMap = new Map();
    for (const fx of (bodyPlan.returnFixups || [])) { if (fx.kind !== 'copy') return false; fixMap.set(fx.pos, fx.srcSlot); }
    const bodyReturnSlot = (retIdx) => { const pos = nargs + retIdx; return fixMap.has(pos) ? fixMap.get(pos) : bodyPlan.argSlots[pos]; };
    const remap = new Map();
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
    const mapSlot = (s) => { const m = remap.get(s); if (m === undefined) return null; return m; };
    for (const st of bodyPlan.steps) {
      const inS = st.inputSlots.map(mapSlot), outS = st.outputSlots.map(mapSlot);
      if (inS.includes(null) || outS.includes(null)) return false;
      steps.push({ name: st.name, inputSlots: inS, outputSlots: outS });
    }
    for (const f of bodyModule.functions()) addedFuncs.push(f);
  }
  const loopEnd = steps.length;

  const T = scanOp.getOperand(0).type.shape[0];
  if (typeof T !== 'number' || T < 0) return false;

  scanLoops.push({
    T, loopStart, loopEnd,
    carry: carryShapes.map((sh, i) => ({ a: carryA[i], b: carryB[i], initSlot: carryInitSlots[i], finalSlot: carryFinalSlots[i], bytes: numel(sh) * dtypeBytes(carryDtypes[i]) })),
    xs: xtSlots.map((s, i) => ({ xtSlot: s, xsSlot: xsSlots[i], stepBytes: numel(xtShapes[i]) * dtypeBytes(xtDtypes[i]) })),
    ys: ytSlots.map((s, i) => ({ ytSlot: s, ysSlot: ysSlots[i], stepBytes: numel(ytShapes[i]) * dtypeBytes(ytDtypes[i]) })),
  });
  return true;
}

export function splitGraphForScan(graphModule, target, force = false) {
  if (!target || typeof target.isWebGPU !== 'function' || !target.isWebGPU()) return null;
  if (graphModule.functionCount !== 1) return null;
  const func = graphModule.functions().next().value;
  const retOp = func.getReturnOp();
  if (!retOp) return null;

  const scans = [];
  for (const op of func.ops()) if (op.opName === 'scan') scans.push(op);
  if (scans.length === 0) return null;

  const reach = new Set();
  const visit = (op) => { if (!op || reach.has(op)) return; reach.add(op); for (let i = 0; i < op.numOperands; i++) visit(op.getOperand(i).definingOp); };
  for (let i = 0; i < retOp.numOperands; i++) visit(retOp.getOperand(i).definingOp);
  for (const s of scans) if (!reach.has(s)) return null;

  if (!force) {
    let anyOversized = false;
    for (const s of scans) { const r = s.regions[0]; if (r && r.entryBlock && isScanOversized(s, r, target)) { anyOversized = true; break; } }
    if (!anyOversized) return null;
  }

  const reachList = [];
  for (const op of reach) {
    if (isConstantOp(op) || isTerminatorOp(op.opName)) continue;
    reachList.push(op);
  }
  const ordered = topoSortOps(reachList);

  const slotOf = new Map();
  let nextSlot = 0;
  const getSlot = (v) => { let s = slotOf.get(v); if (s === undefined) { s = nextSlot++; slotOf.set(v, s); } return s; };
  for (const arg of func.args) getSlot(arg);
  const intermediates = [];
  const newSlot = (shape, dtype) => { const s = nextSlot++; intermediates.push({ slot: s, shape: [...shape], dtype }); return s; };

  const steps = [];
  const scanLoops = [];
  const addedFuncs = [];
  const ctx = { getSlot, newSlot, steps, scanLoops, addedFuncs };
  const scanSet = new Set(scans);

  let segOps = [];
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

  const argSlots = func.args.map(getSlot);
  const returnFixups = [];
  const usedRet = new Set();
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

  const argSlotSet = new Set(argSlots);
  const seen = new Set();
  for (const [v, s] of slotOf) {
    if (argSlotSet.has(s) || seen.has(s)) continue;
    seen.add(s);
    if (!v.type || !v.type.isFullyStatic) return null;
    if (!intermediates.some(it => it.slot === s)) intermediates.push({ slot: s, shape: [...v.type.shape], dtype: v.type.dtype });
  }

  graphModule.removeFunction(func.name);
  for (const f of addedFuncs) graphModule.addFunction(f);

  return { plan: { numSlots: nextSlot, argSlots, intermediates, steps, returnFixups, scanLoops } };
}
