import { FunctionPass, PassResult } from '../pass.js';
import { Operation } from '../../ir/graph/operation.js';
import { TensorType, ScalarType, isFloatType } from '../../ir/graph/types.js';
import { registry } from '../../ir/graph/ops.js';

export const PrecisionClass = Object.freeze({ ALWAYS: 'ALWAYS', ACCUMULATE: 'ACCUMULATE', FOLLOW: 'FOLLOW', NEVER: 'NEVER' });

const DEFAULT_PRECISION_CLASSES = {
  dot: PrecisionClass.ACCUMULATE,
  conv: PrecisionClass.ACCUMULATE,
  reshape: PrecisionClass.FOLLOW,
  transpose: PrecisionClass.FOLLOW,
  broadcast_in_dim: PrecisionClass.FOLLOW,
  slice: PrecisionClass.FOLLOW,
  concat: PrecisionClass.FOLLOW,
  reverse: PrecisionClass.FOLLOW,
  add: PrecisionClass.FOLLOW,
  sub: PrecisionClass.FOLLOW,
  mul: PrecisionClass.FOLLOW,
  maximum: PrecisionClass.FOLLOW,
  minimum: PrecisionClass.FOLLOW,
  relu: PrecisionClass.FOLLOW,
};
for (const [opName, cls] of Object.entries(DEFAULT_PRECISION_CLASSES)) {
  if (registry.has(opName)) registry.registerOpAttr(opName, 'precisionClass', cls);
}

export const DEFAULT_AUTOCAST_OPS = new Set(['dot', 'conv']);

const AUTOCAST_CLASSES = new Set([PrecisionClass.ALWAYS, PrecisionClass.ACCUMULATE]);

function accumulatesInHighPrecision(opName) {
  const def = registry.has(opName) ? registry.get(opName) : null;
  return !!def && def.getAttr('precisionClass') === PrecisionClass.ACCUMULATE;
}

function followOpSet(explicit) {
  if (explicit) return explicit;
  const follow = new Set();
  for (const def of registry.allOps()) {
    if (def.getAttr('precisionClass') === PrecisionClass.FOLLOW) follow.add(def.name);
  }
  return follow;
}

function autocastAllowSet(explicit) {
  if (explicit) return explicit;
  const allow = new Set(DEFAULT_AUTOCAST_OPS);
  for (const def of registry.allOps()) {
    if (AUTOCAST_CLASSES.has(def.getAttr('precisionClass'))) allow.add(def.name);
  }
  return allow;
}

function isCastableFloat(type, lowDtype) {
  return type instanceof TensorType && isFloatType(type.dtype) && type.dtype !== lowDtype;
}

function castOpToLowPrecision(op, lowDtype) {
  const block = op.parentBlock;
  if (!block) return false;

  const newOperands = [];
  let castedAny = false;
  for (let i = 0; i < op.numOperands; i++) {
    const v = op.getOperand(i);
    if (isCastableFloat(v.type, lowDtype)) {
      const lowType = new TensorType(v.type.shape, lowDtype);
      const down = new Operation('convert', [v], [lowType], { target_dtype: lowDtype });
      block.insertBefore(down, op);
      newOperands.push(down.getResult(0));
      castedAny = true;
    } else {
      newOperands.push(v);
    }
  }
  if (!castedAny) return false;

  const accumulates = accumulatesInHighPrecision(op.opName);
  const resultTypes = [];
  for (let r = 0; r < op.numResults; r++) {
    const rt = op.getResult(r).type;
    resultTypes.push(!accumulates && isCastableFloat(rt, lowDtype) ? new TensorType(rt.shape, lowDtype) : rt);
  }

  const attributes = new Map(op.attributes);
  if (accumulates && resultTypes[0] instanceof TensorType) attributes.set('out_dtype', resultTypes[0].dtype);
  const lowered = new Operation(op.opName, newOperands, resultTypes, attributes);
  block.insertBefore(lowered, op);

  for (let r = 0; r < op.numResults; r++) {
    const oldRes = op.getResult(r);
    const newRes = lowered.getResult(r);
    if (newRes.type instanceof TensorType && newRes.type.dtype !== oldRes.type.dtype) {
      const up = new Operation('convert', [newRes], [oldRes.type], { target_dtype: oldRes.type.dtype });
      block.insertBefore(up, op);
      oldRes.replaceAllUsesWith(up.getResult(0));
    } else {
      oldRes.replaceAllUsesWith(newRes);
    }
  }

  op.erase();
  return true;
}

function isUpConvert(value, lowDtype) {
  const d = value.definingOp;
  return d && d.opName === 'convert'
    && d.getOperand(0).type instanceof TensorType
    && d.getOperand(0).type.dtype === lowDtype;
}

function followHasLowInput(op, lowDtype) {
  for (let i = 0; i < op.numOperands; i++) {
    const t = op.getOperand(i).type;
    if (!(t instanceof TensorType) || !isFloatType(t.dtype)) continue;
    if (t.dtype === lowDtype || isUpConvert(op.getOperand(i), lowDtype)) return true;
  }
  return false;
}

function resultIsLow(op, lowDtype) {
  const r = op.getResult(0);
  return r && r.type instanceof TensorType && r.type.dtype === lowDtype;
}

function foldDoubleConverts(func) {
  let changed = false;
  for (const op of [...func.opsArray()]) {
    if (!op.parentBlock || op.opName !== 'convert') continue;
    const inner = op.getOperand(0).definingOp;
    if (!inner || inner.opName !== 'convert') continue;
    if (inner.getOperand(0).type.dtype !== op.getAttr('target_dtype')) continue;
    op.getResult(0).replaceAllUsesWith(inner.getOperand(0));
    op.erase();
    if (inner.getResult(0).uses && [...inner.getResult(0).uses()].length === 0) inner.erase();
    changed = true;
  }
  return changed;
}

export function applyAutocast(func, opts = {}) {
  const allow = autocastAllowSet(opts.allow);
  const lowDtype = opts.dtype || ScalarType.F16;
  let changed = false;
  for (const op of [...func.opsArray()]) {
    if (!op.parentBlock) continue;
    if (!allow.has(op.opName)) continue;
    changed = castOpToLowPrecision(op, lowDtype) || changed;
  }

  if (opts.propagateFollow && changed) {
    const follow = followOpSet(opts.follow);
    let progress = true;
    while (progress) {
      progress = false;
      for (const op of [...func.opsArray()]) {
        if (!op.parentBlock || !follow.has(op.opName)) continue;
        if (resultIsLow(op, lowDtype)) continue;
        if (!followHasLowInput(op, lowDtype)) continue;
        if (castOpToLowPrecision(op, lowDtype)) progress = true;
      }
      if (foldDoubleConverts(func)) progress = true;
    }
  }

  return changed;
}

export class MixedPrecisionPass extends FunctionPass {
  constructor(config = {}) {
    super('MixedPrecisionPass');
    this.allow = config.allow || DEFAULT_AUTOCAST_OPS;
    this.dtype = config.dtype || ScalarType.F16;
  }

  run(func) {
    const changed = applyAutocast(func, { allow: this.allow, dtype: this.dtype });
    return changed ? PassResult.CHANGED : PassResult.UNCHANGED;
  }
}
