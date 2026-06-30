import { GraphFunction } from '../ir/graph/function.js';
import { GraphModule } from '../ir/graph/module.js';
import { Operation } from '../ir/graph/operation.js';
import { TensorType, isFloatType } from '../ir/graph/types.js';
import { topoSortByOperands } from '../ir/graph/graph_algorithms.js';
import { CalibrationCollector } from './calibration.js';

const DEFAULT_QUANTIZABLE_OPS = new Set(['dot', 'conv', 'add', 'mul', 'sub']);

function activationTargets(func, quantizableOps) {
  const targets = [];
  const seen = new Set();
  for (const op of func.ops()) {
    if (!quantizableOps.has(op.opName)) continue;
    for (let i = 0; i < op.numOperands; i++) {
      const v = op.getOperand(i);
      if (seen.has(v)) continue;
      if (!(v.type instanceof TensorType) || !isFloatType(v.type.dtype)) continue;
      const def = v.definingOp;
      if (def && def.opName === 'constant') continue;
      seen.add(v);
      targets.push(v);
    }
  }
  return targets;
}

function buildCaptureFunction(func, targets) {
  const captureTypes = targets.map(v => v.type);
  const calib = new GraphFunction(func.name, func.inputTypes, [...func.outputTypes, ...captureTypes]);

  const valueMap = new Map();
  const srcArgs = func.entryBlock.arguments;
  const dstArgs = calib.entryBlock.arguments;
  for (let i = 0; i < srcArgs.length; i++) valueMap.set(srcArgs[i], dstArgs[i]);

  const ops = func.entryBlock.opsArray();
  const inBlock = new Set(ops);
  const ordered = topoSortByOperands(ops, (op) => inBlock.has(op), 'ignore');

  const clonedByOrig = new Map();
  for (const op of ordered) {
    if (op.opName === 'return') continue;
    clonedByOrig.set(op, op.clone(valueMap));
  }
  for (const op of ops) {
    if (op.opName === 'return') continue;
    calib.entryBlock.pushOp(clonedByOrig.get(op));
  }

  const origReturn = func.getReturnOp();
  const retOperands = [];
  if (origReturn) {
    for (const v of origReturn.operands) retOperands.push(valueMap.get(v) || v);
  }
  for (const v of targets) retOperands.push(valueMap.get(v));
  calib.entryBlock.pushOp(new Operation('return', retOperands, [], {}));

  return calib;
}

export function collectCalibration(func, target, batches, opts = {}) {
  if (target.isGPU && target.isGPU()) {
    throw new Error('collectCalibration: synchronous calibration is unavailable for async (GPU) targets; precompute calibration on a CPU/WASM target or supply quantization.calibration directly');
  }
  if (!batches || batches.length === 0) {
    throw new Error('collectCalibration: at least one calibration batch is required');
  }

  const quantizableOps = opts.quantizableOps || DEFAULT_QUANTIZABLE_OPS;
  const mode = opts.mode || 'minmax';
  const targets = activationTargets(func, quantizableOps);

  const collector = new CalibrationCollector(mode);
  collector.attach(func);
  const result = collector.getResult();
  if (targets.length === 0) return result;

  const captureFunc = buildCaptureFunction(func, targets);
  const module = new GraphModule('__calibrate__');
  module.addFunction(captureFunc);

  const numDeclaredOutputs = func.outputTypes.length;
  const compileFn = opts.compileFn || defaultCompileFn;
  const compiled = compileFn(module, target);

  for (const batch of batches) {
    const inputs = Array.isArray(batch) ? batch : [batch];
    const outBufs = captureFunc.outputTypes.map(t => new Float32Array(Math.max(1, t.numel())));
    compiled.run(func.name, ...inputs, ...outBufs);
    for (let i = 0; i < targets.length; i++) {
      collector.observe(targets[i], outBufs[numDeclaredOutputs + i]);
    }
  }

  return result;
}

function defaultCompileFn(module, target) {
  throw new Error('collectCalibration: opts.compileFn is required (pass the compileModule function to avoid a circular import)');
}
