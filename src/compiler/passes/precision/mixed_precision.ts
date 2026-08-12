import { FunctionPass, PassResult } from '../pass.js';
import { Operation } from '../../ir/graph/operation.js';
import { TensorType, ScalarType, isFloatType } from '../../ir/graph/types.js';
import { registry } from '../../ir/graph/ops.js';
import type { GraphFunction } from '../../ir/graph/function.js';
import type { Value } from '../../ir/graph/value.js';
import type { AttrValue, IRType, ScalarDType } from '../../ir/graph/types.js';
import type { PassResultValue, PassTarget } from '../pass.js';

export type PrecisionClassValue = (typeof PrecisionClass)[keyof typeof PrecisionClass];
export type AutocastOpts = {
  allow?: ReadonlySet<string>;
  dtype?: string;
  propagateFollow?: boolean;
  follow?: ReadonlySet<string>;
};
export type MixedPrecisionConfig = { allow?: ReadonlySet<string>; dtype?: string };

export const PrecisionClass = Object.freeze({ ALWAYS: 'ALWAYS', ACCUMULATE: 'ACCUMULATE', FOLLOW: 'FOLLOW', NEVER: 'NEVER' });

const DEFAULT_PRECISION_CLASSES: Record<string, PrecisionClassValue> = {
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

const AUTOCAST_CLASSES = new Set<PrecisionClassValue>([PrecisionClass.ALWAYS, PrecisionClass.ACCUMULATE]);

function accumulatesInHighPrecision(opName: string): boolean {
  const def = registry.has(opName) ? registry.get(opName) : null;
  return !!def && def.getAttr('precisionClass') === PrecisionClass.ACCUMULATE;
}

function followOpSet(explicit: ReadonlySet<string> | undefined): ReadonlySet<string> {
  if (explicit) return explicit;
  const follow = new Set<string>();
  for (const def of registry.allOps()) {
    if (def.getAttr('precisionClass') === PrecisionClass.FOLLOW) follow.add(def.name);
  }
  return follow;
}

function autocastAllowSet(explicit: ReadonlySet<string> | undefined): ReadonlySet<string> {
  if (explicit) return explicit;
  const allow = new Set<string>(DEFAULT_AUTOCAST_OPS);
  for (const def of registry.allOps()) {
    if (AUTOCAST_CLASSES.has(def.getAttr<PrecisionClassValue>('precisionClass') as PrecisionClassValue)) allow.add(def.name);
  }
  return allow;
}

function isCastableFloat(type: IRType, lowDtype: string): boolean {
  return type instanceof TensorType && isFloatType(type.dtype as ScalarDType) && type.dtype !== lowDtype;
}

function castOpToLowPrecision(op: Operation, lowDtype: string): boolean {
  const block = op.parentBlock;
  if (!block) return false;

  const newOperands: Value[] = [];
  let castedAny = false;
  for (let i = 0; i < op.numOperands; i++) {
    const v = op.getOperand(i);
    if (isCastableFloat(v.type, lowDtype)) {
      const lowType = new TensorType((v.type as TensorType).shape, lowDtype as ScalarDType);
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
  const resultTypes: IRType[] = [];
  for (let r = 0; r < op.numResults; r++) {
    const rt = op.getResult(r).type as TensorType;
    resultTypes.push(!accumulates && isCastableFloat(rt, lowDtype) ? new TensorType(rt.shape, lowDtype as ScalarDType) : rt);
  }

  const attributes = new Map<string, AttrValue>(op.attributes);
  if (accumulates && resultTypes[0] instanceof TensorType) attributes.set('out_dtype', (resultTypes[0] as TensorType).dtype);
  const lowered = new Operation(op.opName, newOperands, resultTypes, attributes);
  block.insertBefore(lowered, op);

  for (let r = 0; r < op.numResults; r++) {
    const oldRes = op.getResult(r);
    const newRes = lowered.getResult(r);
    if (newRes.type instanceof TensorType && newRes.type.dtype !== (oldRes.type as TensorType).dtype) {
      const up = new Operation('convert', [newRes], [oldRes.type], { target_dtype: (oldRes.type as TensorType).dtype });
      block.insertBefore(up, op);
      oldRes.replaceAllUsesWith(up.getResult(0));
    } else {
      oldRes.replaceAllUsesWith(newRes);
    }
  }

  op.erase();
  return true;
}

function isUpConvert(value: Value, lowDtype: string): boolean {
  const d = value.definingOp;
  return !!d && d.opName === 'convert'
    && d.getOperand(0).type instanceof TensorType
    && (d.getOperand(0).type as TensorType).dtype === lowDtype;
}

function followHasLowInput(op: Operation, lowDtype: string): boolean {
  for (let i = 0; i < op.numOperands; i++) {
    const t = op.getOperand(i).type;
    if (!(t instanceof TensorType) || !isFloatType(t.dtype as ScalarDType)) continue;
    if (t.dtype === lowDtype || isUpConvert(op.getOperand(i), lowDtype)) return true;
  }
  return false;
}

function resultIsLow(op: Operation, lowDtype: string): boolean {
  const r = op.getResult(0);
  return r && r.type instanceof TensorType && r.type.dtype === lowDtype;
}

function foldDoubleConverts(func: GraphFunction): boolean {
  let changed = false;
  for (const op of [...func.opsArray()]) {
    if (!op.parentBlock || op.opName !== 'convert') continue;
    const inner = op.getOperand(0).definingOp;
    if (!inner || inner.opName !== 'convert') continue;
    if ((inner.getOperand(0).type as TensorType).dtype !== op.getAttr<string>('target_dtype')) continue;
    op.getResult(0).replaceAllUsesWith(inner.getOperand(0));
    op.erase();
    if (inner.getResult(0).uses && [...inner.getResult(0).uses()].length === 0) inner.erase();
    changed = true;
  }
  return changed;
}

export function applyAutocast(func: GraphFunction, opts: AutocastOpts = {}): boolean {
  const allow = autocastAllowSet(opts.allow);
  const lowDtype: string = opts.dtype || ScalarType.F16;
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
  allow: ReadonlySet<string>;
  dtype: string;

  constructor(config: MixedPrecisionConfig = {}) {
    super('MixedPrecisionPass');
    this.allow = config.allow || DEFAULT_AUTOCAST_OPS;
    this.dtype = config.dtype || ScalarType.F16;
  }

  override run(func: PassTarget): PassResultValue {
    const changed = applyAutocast(func as GraphFunction, { allow: this.allow, dtype: this.dtype });
    return changed ? PassResult.CHANGED : PassResult.UNCHANGED;
  }
}
