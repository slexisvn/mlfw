import { GradAccumulator } from './grad_accumulator.js';
import { getVJPRule } from './vjp_registry.js';
import { reduceGradToOperandShape } from './backward_builder.js';

export function regionFreeVars(bodyBlock) {
  const local = new Set(bodyBlock.arguments.map(a => a.id));
  for (const op of bodyBlock.ops()) for (const r of op.results) local.add(r.id);
  const seen = new Set();
  const free = [];
  for (const op of bodyBlock.ops()) {
    for (const o of op.operands) {
      if (local.has(o.id) || seen.has(o.id)) continue;
      if (o.definingOp && o.definingOp.opName === 'constant') continue;
      seen.add(o.id);
      free.push(o);
    }
  }
  return free;
}

function zeroLike(builder, val) {
  const z = builder.scalarConstant(0, val.type.dtype).getResult(0);
  return builder.broadcast(z, val.type.shape, []).getResult(0);
}

function sliceStep(builder, v, t) {
  const shape = v.type.shape;
  const starts = shape.map((_, i) => (i === 0 ? t : 0));
  const limits = shape.map((d, i) => (i === 0 ? t + 1 : d));
  const sliced = builder.slice(v, starts, limits).getResult(0);
  return builder.reshape(sliced, shape.slice(1)).getResult(0);
}

function stackSteps(builder, steps, fullShape) {
  const rest = fullShape.slice(1);
  const expanded = steps.map(s => builder.reshape(s, [1, ...rest]).getResult(0));
  if (expanded.length === 1) return builder.reshape(expanded[0], fullShape).getResult(0);
  return builder.concat(expanded, 0).getResult(0);
}

function diffBodyStep(builder, bodyBlock, argVals, freeVarMap, gradYields, forwardOnly) {
  const map = new Map();
  for (let i = 0; i < bodyBlock.arguments.length; i++) map.set(bodyBlock.arguments[i].id, argVals[i]);
  for (const [id, v] of freeVarMap) map.set(id, v);

  const bodyOps = [];
  let yieldOp = null;
  for (const op of bodyBlock.ops()) {
    if (op.opName === 'yield') yieldOp = op;
    else bodyOps.push(op);
  }

  const matOperand = (o) => {
    if (map.has(o.id)) return map.get(o.id);
    const def = o.definingOp;
    if (def && def.opName === 'constant') {
      const cloned = builder._buildOp('constant', [], [o.type], new Map(def.attributes), null);
      map.set(o.id, cloned.getResult(0));
      return cloned.getResult(0);
    }
    return o;
  };

  for (const op of bodyOps) {
    const operands = op.operands.map(matOperand);
    const cloned = builder._buildOp(op.opName, operands, op.results.map(r => r.type), new Map(op.attributes), null);
    for (let r = 0; r < op.numResults; r++) map.set(op.getResult(r).id, cloned.getResult(r));
  }

  const forwardYields = yieldOp.operands.map(o => map.get(o.id));
  if (forwardOnly) return { forwardYields };

  const acc = new GradAccumulator(builder);
  for (let i = 0; i < yieldOp.operands.length; i++) {
    if (gradYields[i]) acc.accumulate(yieldOp.operands[i].id, gradYields[i]);
  }
  for (let i = bodyOps.length - 1; i >= 0; i--) {
    const op = bodyOps[i];
    if (op.opName === 'constant') continue;
    const gradOuts = op.results.map(r => acc.get(r.id));
    if (gradOuts.every(g => g === null)) continue;
    const rule = getVJPRule(op.opName);
    if (!rule) continue;
    const ctx = {
      builder, op,
      operands: op.operands.map(matOperand),
      results: op.results.map(r => map.get(r.id)),
      gradOutputs: gradOuts,
      attrs: op.attributes,
    };
    const gradIns = rule(ctx);
    if (!gradIns) continue;
    for (let o = 0; o < op.numOperands; o++) {
      if (!gradIns[o]) continue;
      acc.accumulate(op.getOperand(o).id, reduceGradToOperandShape(builder, gradIns[o], op.getOperand(o).type.shape));
    }
  }

  const gradArgs = bodyBlock.arguments.map(a => acc.get(a.id));
  const gradFree = new Map();
  for (const id of freeVarMap.keys()) gradFree.set(id, acc.get(id));
  return { forwardYields, gradArgs, gradFree };
}

const NESTED_CONTROL_FLOW = new Set(['scan', 'while', 'if']);

function assertNoNestedControlFlow(block, host) {
  for (const op of block.ops()) {
    if (NESTED_CONTROL_FLOW.has(op.opName)) {
      throw new Error(`${host} VJP does not support nested control flow ('${op.opName}') in the body`);
    }
  }
}

export function buildCondBackward(ifOp, accumulator, builder, materialize, needsGrad) {
  const thenBlock = ifOp.regions[0].blocks[0];
  const elseBlock = ifOp.regions[1].blocks[0];
  assertNoNestedControlFlow(thenBlock, 'if');
  assertNoNestedControlFlow(elseBlock, 'if');

  const pred = materialize(ifOp.getOperand(0));
  const gradResults = [];
  for (let i = 0; i < ifOp.numResults; i++) gradResults.push(accumulator.get(ifOp.getResult(i).id));

  const thenFree = regionFreeVars(thenBlock);
  const elseFree = regionFreeVars(elseBlock);
  const thenMap = new Map(thenFree.map(v => [v.id, materialize(v)]));
  const elseMap = new Map(elseFree.map(v => [v.id, materialize(v)]));

  const { gradFree: gThen } = diffBodyStep(builder, thenBlock, [], thenMap, gradResults, false);
  const { gradFree: gElse } = diffBodyStep(builder, elseBlock, [], elseMap, gradResults, false);

  const allVars = new Map();
  for (const v of thenFree) allVars.set(v.id, v);
  for (const v of elseFree) allVars.set(v.id, v);

  for (const [id, val] of allVars) {
    if (!needsGrad.has(id)) continue;
    const gt = gThen.get(id);
    const ge = gElse.get(id);
    if (!gt && !ge) continue;
    const zero = zeroLike(builder, val);
    const predBr = builder.broadcast(pred, val.type.shape, []).getResult(0);
    accumulator.accumulate(id, builder.select(predBr, gt ?? zero, ge ?? zero).getResult(0));
  }
}

export function buildScanBackward(scanOp, accumulator, builder, materialize, needsGrad) {
  const bodyBlock = scanOp.regions[0].blocks[0];
  assertNoNestedControlFlow(bodyBlock, 'scan');
  const numCarry = scanOp.getAttr('num_carry');
  const numXs = scanOp.getAttr('num_xs');
  const numYs = scanOp.numResults - numCarry;

  const xsOps = [];
  for (let i = 0; i < numXs; i++) xsOps.push(scanOp.getOperand(i));
  const carryOps = [];
  for (let i = 0; i < numCarry; i++) carryOps.push(scanOp.getOperand(numXs + i));
  const T = xsOps[0].type.shape[0];

  const freeVars = regionFreeVars(bodyBlock);
  const xsB = xsOps.map(materialize);
  const initCarryB = carryOps.map(materialize);
  const freeVarMap = new Map(freeVars.map(v => [v.id, materialize(v)]));

  let carry = initCarryB;
  const carriesAtT = [carry];
  const xsSlices = [];
  for (let t = 0; t < T; t++) {
    const xt = xsB.map(v => sliceStep(builder, v, t));
    xsSlices.push(xt);
    const { forwardYields } = diffBodyStep(builder, bodyBlock, [...xt, ...carry], freeVarMap, null, true);
    carry = forwardYields.slice(0, numCarry);
    carriesAtT.push(carry);
  }

  const gYs = [];
  for (let i = 0; i < numYs; i++) gYs.push(accumulator.get(scanOp.getResult(numCarry + i).id));
  let gCarry = [];
  for (let i = 0; i < numCarry; i++) {
    const g = accumulator.get(scanOp.getResult(i).id);
    gCarry.push(g ?? zeroLike(builder, initCarryB[i]));
  }

  const gFree = new Map();
  const gXsSteps = xsB.map(() => new Array(T));

  for (let t = T - 1; t >= 0; t--) {
    const argVals = [...xsSlices[t], ...carriesAtT[t]];
    const gY_t = gYs.map(g => (g === null ? null : sliceStep(builder, g, t)));
    const gradYields = [...gCarry, ...gY_t];
    const { gradArgs, gradFree } = diffBodyStep(builder, bodyBlock, argVals, freeVarMap, gradYields, false);

    for (let i = 0; i < numXs; i++) gXsSteps[i][t] = gradArgs[i] ?? zeroLike(builder, xsSlices[t][i]);
    gCarry = [];
    for (let i = 0; i < numCarry; i++) gCarry.push(gradArgs[numXs + i] ?? zeroLike(builder, carriesAtT[t][i]));
    for (const [id, g] of gradFree) {
      if (!g) continue;
      const prev = gFree.get(id);
      gFree.set(id, prev ? builder.add(prev, g).getResult(0) : g);
    }
  }

  for (let i = 0; i < numXs; i++) {
    if (!needsGrad.has(xsOps[i].id)) continue;
    accumulator.accumulate(xsOps[i].id, stackSteps(builder, gXsSteps[i], xsB[i].type.shape));
  }
  for (let i = 0; i < numCarry; i++) {
    if (needsGrad.has(carryOps[i].id)) accumulator.accumulate(carryOps[i].id, gCarry[i]);
  }
  for (const [id, g] of gFree) {
    if (g && needsGrad.has(id)) accumulator.accumulate(id, g);
  }
}
