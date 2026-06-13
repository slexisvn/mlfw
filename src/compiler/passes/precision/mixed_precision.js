import { FunctionPass, PassResult } from '../pass.js';
import { Operation } from '../../ir/graph/operation.js';
import { TensorType, ScalarType, isFloatType } from '../../ir/graph/types.js';

export const DEFAULT_AUTOCAST_OPS = new Set(['dot', 'conv']);

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

  const resultTypes = [];
  for (let r = 0; r < op.numResults; r++) {
    const rt = op.getResult(r).type;
    resultTypes.push(isCastableFloat(rt, lowDtype) ? new TensorType(rt.shape, lowDtype) : rt);
  }

  const lowered = new Operation(op.opName, newOperands, resultTypes, new Map(op.attributes));
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

export function applyAutocast(func, opts = {}) {
  const allow = opts.allow || DEFAULT_AUTOCAST_OPS;
  const lowDtype = opts.dtype || ScalarType.F16;
  let changed = false;
  for (const op of [...func.opsArray()]) {
    if (!op.parentBlock) continue;
    if (!allow.has(op.opName)) continue;
    changed = castOpToLowPrecision(op, lowDtype) || changed;
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
