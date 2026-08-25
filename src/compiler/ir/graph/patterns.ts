import { Pattern } from '../rewrite/pattern.js';
import { TensorType } from './types.js';
import { isDtypeInt } from '../../../util/dtype_map.js';
import { isOp, wildcard, matchPattern } from '../rewrite/dfpattern.js';
import { registry } from './ops.js';
import type { AttrValue } from './types.js';
import type { Value } from './value.js';
import type { Operation } from './operation.js';
import type { IRBuilder } from './builder.js';

const TRANSPOSE_TRANSPOSE_PAT = isOp('transpose', isOp('transpose', wildcard()));
const RESHAPE_RESHAPE_PAT = isOp('reshape', isOp('reshape', wildcard()));
const DOUBLE_NEG_PAT = isOp('neg', isOp('neg', wildcard()));
const EXP_LOG_PAT = isOp('exp', isOp('log', wildcard()));
const LOG_EXP_PAT = isOp('log', isOp('exp', wildcard()));

function isDense(value: unknown): value is ArrayLike<number | bigint> {
  return Array.isArray(value) || (ArrayBuffer.isView(value) && !(value instanceof DataView));
}

function constantValueOf(op: Operation | null): AttrValue | undefined {
  if (!op) return undefined;
  const def = registry.get(op.opName);
  if (!def) return undefined;
  if (def.isConstant) return op.getAttr<AttrValue>('value');
  if (def.isBroadcast && op.numOperands === 1) return constantValueOf(op.getOperand(0).definingOp);
  return undefined;
}

function isSplatOf(value: AttrValue, val: AttrValue): boolean {
  if (!isDense(value)) return value === val;
  if (value.length === 0) return false;
  const target = typeof value[0] === 'bigint' ? BigInt(val as number) : val;
  for (let i = 0; i < value.length; i++) {
    if (value[i] !== target) return false;
  }
  return true;
}

function isConstantVal(op: Operation | null, val: AttrValue): boolean {
  const value = constantValueOf(op);
  return value !== undefined && isSplatOf(value, val);
}

export class FoldTrivialReshape extends Pattern {
  constructor() { super('fold_trivial_reshape', 10); this.rootOpName = 'reshape'; }
  override match(op: Operation): boolean {
    const input = op.getOperand(0).type;
    const output = op.getResult(0).type;
    return input instanceof TensorType && output instanceof TensorType && input.shapeEquals(output);
  }
  override rewrite(op: Operation, builder: IRBuilder): boolean {
    op.replaceAllResultsWith([op.getOperand(0)]);
    op.erase();
    return true;
  }
}

export class ReshapeReshape extends Pattern {
  constructor() { super('reshape_reshape', 10); this.rootOpName = 'reshape'; }
  override match(op: Operation): boolean {
    return matchPattern(RESHAPE_RESHAPE_PAT, op) !== null;
  }
  override rewrite(op: Operation, builder: IRBuilder): boolean {
    const originalInput = op.getOperand(0).definingOp!.getOperand(0);
    const newShape = op.getAttr<readonly number[]>('new_shape')!;
    const newReshape = builder.reshape(originalInput, newShape);
    op.replaceAllResultsWith([newReshape.getResult(0)]);
    op.erase();
    return true;
  }
}

export class FoldTrivialTranspose extends Pattern {
  constructor() { super('fold_trivial_transpose', 10); this.rootOpName = 'transpose'; }
  override match(op: Operation): boolean {
    const perm = op.getAttr<readonly number[]>('permutation')!;
    if (!perm) return false;
    for (let i = 0; i < perm.length; i++) {
      if (perm[i] !== i) return false;
    }
    return true;
  }
  override rewrite(op: Operation, builder: IRBuilder): boolean {
    op.replaceAllResultsWith([op.getOperand(0)]);
    op.erase();
    return true;
  }
}

export class TransposeTranspose extends Pattern {
  constructor() { super('transpose_transpose', 10); this.rootOpName = 'transpose'; }
  override match(op: Operation): boolean {
    return matchPattern(TRANSPOSE_TRANSPOSE_PAT, op) !== null;
  }
  override rewrite(op: Operation, builder: IRBuilder): boolean {
    const originalInput = op.getOperand(0).definingOp!.getOperand(0);
    const perm1 = op.getOperand(0).definingOp!.getAttr<readonly number[]>('permutation')!;
    const perm2 = op.getAttr<readonly number[]>('permutation')!;
    const newPerm = new Array(perm2.length);
    for (let i = 0; i < perm2.length; i++) {
      newPerm[i] = perm1[perm2[i]];
    }
    const newTranspose = builder.transpose(originalInput, newPerm);
    op.replaceAllResultsWith([newTranspose.getResult(0)]);
    op.erase();
    return true;
  }
}

export class FoldTrivialPad extends Pattern {
  constructor() { super('fold_trivial_pad', 10); this.rootOpName = 'pad'; }
  override match(op: Operation): boolean {
    const low = op.getAttr<readonly number[]>('low')!;
    const high = op.getAttr<readonly number[]>('high')!;
    const interior = op.getAttr<readonly number[]>('interior')! || [];
    if (low && low.some(x => x !== 0)) return false;
    if (high && high.some(x => x !== 0)) return false;
    if (interior && interior.some(x => x !== 0)) return false;
    return true;
  }
  override rewrite(op: Operation, builder: IRBuilder): boolean {
    op.replaceAllResultsWith([op.getOperand(0)]);
    op.erase();
    return true;
  }
}

export class FoldTrivialSlice extends Pattern {
  constructor() { super('fold_trivial_slice', 10); this.rootOpName = 'slice'; }
  override match(op: Operation): boolean {
    const inputType = op.getOperand(0).type;
    const outputType = op.getResult(0).type;
    if (!(inputType instanceof TensorType) || !(outputType instanceof TensorType)) return false;
    const starts = op.getAttr<readonly number[]>('starts')!;
    const strides = op.getAttr<readonly number[]>('strides')! || starts.map(() => 1);
    if (starts.some(x => x !== 0)) return false;
    if (strides.some(x => x !== 1)) return false;
    return inputType.shapeEquals(outputType);
  }
  override rewrite(op: Operation, builder: IRBuilder): boolean {
    op.replaceAllResultsWith([op.getOperand(0)]);
    op.erase();
    return true;
  }
}

export class CommutativeConstantRight extends Pattern {
  constructor(opName: string | null = null) {
    super(`commutative_constant_right${opName ? '_' + opName : ''}`, 5);
    this.rootOpName = opName;
  }
  override match(op: Operation): boolean {
    if (op.numOperands !== 2) return false;
    const lhsDef = op.getOperand(0).definingOp;
    const rhsDef = op.getOperand(1).definingOp;
    return ((lhsDef && lhsDef.opName === 'constant') && !(rhsDef && rhsDef.opName === 'constant')) as boolean;
  }
  override rewrite(op: Operation, builder: IRBuilder): boolean {
    const lhs = op.getOperand(0);
    const rhs = op.getOperand(1);
    op.replaceOperand(0, rhs);
    op.replaceOperand(1, lhs);
    return true;
  }
}

export function commutativeConstantRightFor(opName: string): Pattern {
  return new CommutativeConstantRight(opName);
}

export class CanonicalizeCompare extends Pattern {
  constructor() { super('canonicalize_compare', 5); this.rootOpName = 'compare'; }
  override match(op: Operation): boolean {
    const lhsDef = op.getOperand(0).definingOp;
    const rhsDef = op.getOperand(1).definingOp;
    return ((lhsDef && lhsDef.opName === 'constant') && !(rhsDef && rhsDef.opName === 'constant')) as boolean;
  }
  override rewrite(op: Operation, builder: IRBuilder): boolean {
    const lhs = op.getOperand(0);
    const rhs = op.getOperand(1);
    op.replaceOperand(0, rhs);
    op.replaceOperand(1, lhs);
    const dir = op.getAttr<string>('direction')!;
    const invert: Record<string, string> = { 'eq': 'eq', 'ne': 'ne', 'lt': 'gt', 'le': 'ge', 'gt': 'lt', 'ge': 'le' };
    op.setAttr('direction', invert[dir]);
    return true;
  }
}

export class AddZero extends Pattern {
  fastMath: boolean;
  constructor(fastMath = false) { super('add_zero', 5); this.rootOpName = 'add'; this.fastMath = fastMath; }
  override match(op: Operation): boolean {
    if (!isDtypeInt((op.getResult(0).type as TensorType).dtype) && !this.fastMath) return false;
    return isConstantVal(op.getOperand(1).definingOp, 0) || isConstantVal(op.getOperand(0).definingOp, 0);
  }
  override rewrite(op: Operation, builder: IRBuilder): boolean {
    const keep = isConstantVal(op.getOperand(1).definingOp, 0) ? op.getOperand(0) : op.getOperand(1);
    if (keep.type.equals(op.getResult(0).type)) {
      op.replaceAllResultsWith([keep]);
      op.erase();
      return true;
    }
    return false;
  }
}

export class SubZero extends Pattern {
  constructor() { super('sub_zero', 5); this.rootOpName = 'sub'; }
  override match(op: Operation): boolean {
    return isConstantVal(op.getOperand(1).definingOp, 0);
  }
  override rewrite(op: Operation, builder: IRBuilder): boolean {
    const keep = op.getOperand(0);
    if (keep.type.equals(op.getResult(0).type)) {
      op.replaceAllResultsWith([keep]);
      op.erase();
      return true;
    }
    return false;
  }
}

export class SubSelf extends Pattern {
  fastMath: boolean;
  constructor(fastMath = false) { super('sub_self', 5); this.rootOpName = 'sub'; this.fastMath = fastMath; }
  override match(op: Operation): boolean {
    if (op.getOperand(0) !== op.getOperand(1)) return false;
    return isDtypeInt((op.getResult(0).type as TensorType).dtype) || this.fastMath;
  }
  override rewrite(op: Operation, builder: IRBuilder): boolean {
    const zero = builder.scalarConstant(0, (op.getResult(0).type as TensorType).dtype);
    let result = zero.getResult(0);
    const shape = (op.getResult(0).type as TensorType).shape;
    if (shape.length > 0) {
      result = builder.broadcast(result, shape, []).getResult(0);
    }
    op.replaceAllResultsWith([result]);
    op.erase();
    return true;
  }
}

export class MulOne extends Pattern {
  constructor() { super('mul_one', 5); this.rootOpName = 'mul'; }
  override match(op: Operation): boolean {
    return isConstantVal(op.getOperand(1).definingOp, 1) || isConstantVal(op.getOperand(0).definingOp, 1);
  }
  override rewrite(op: Operation, builder: IRBuilder): boolean {
    const keep = isConstantVal(op.getOperand(1).definingOp, 1) ? op.getOperand(0) : op.getOperand(1);
    if (keep.type.equals(op.getResult(0).type)) {
      op.replaceAllResultsWith([keep]);
      op.erase();
      return true;
    }
    return false;
  }
}

export class MulZero extends Pattern {
  fastMath: boolean;
  constructor(fastMath = false) { super('mul_zero', 5); this.rootOpName = 'mul'; this.fastMath = fastMath; }
  override match(op: Operation): boolean {
    if (!isDtypeInt((op.getResult(0).type as TensorType).dtype) && !this.fastMath) return false;
    return isConstantVal(op.getOperand(1).definingOp, 0) || isConstantVal(op.getOperand(0).definingOp, 0);
  }
  override rewrite(op: Operation, builder: IRBuilder): boolean {
    const zero = builder.scalarConstant(0, (op.getResult(0).type as TensorType).dtype);
    let result = zero.getResult(0);
    const shape = (op.getResult(0).type as TensorType).shape;
    if (shape.length > 0) {
      result = builder.broadcast(result, shape, []).getResult(0);
    }
    op.replaceAllResultsWith([result]);
    op.erase();
    return true;
  }
}

export class DivOne extends Pattern {
  constructor() { super('div_one', 5); this.rootOpName = 'div'; }
  override match(op: Operation): boolean {
    return isConstantVal(op.getOperand(1).definingOp, 1);
  }
  override rewrite(op: Operation, builder: IRBuilder): boolean {
    const keep = op.getOperand(0);
    if (keep.type.equals(op.getResult(0).type)) {
      op.replaceAllResultsWith([keep]);
      op.erase();
      return true;
    }
    return false;
  }
}

export class DoubleNeg extends Pattern {
  constructor() { super('double_neg', 5); this.rootOpName = 'neg'; }
  override match(op: Operation): boolean {
    return matchPattern(DOUBLE_NEG_PAT, op) !== null;
  }
  override rewrite(op: Operation, builder: IRBuilder): boolean {
    const original = op.getOperand(0).definingOp!.getOperand(0);
    op.replaceAllResultsWith([original]);
    op.erase();
    return true;
  }
}

export class ExpLog extends Pattern {
  fastMath: boolean;
  constructor(fastMath = false) { super('exp_log', 5); this.rootOpName = 'exp'; this.fastMath = fastMath; }
  override match(op: Operation): boolean {
    return this.fastMath && matchPattern(EXP_LOG_PAT, op) !== null;
  }
  override rewrite(op: Operation, builder: IRBuilder): boolean {
    op.replaceAllResultsWith([op.getOperand(0).definingOp!.getOperand(0)]);
    op.erase();
    return true;
  }
}

export class LogExp extends Pattern {
  fastMath: boolean;
  constructor(fastMath = false) { super('log_exp', 5); this.rootOpName = 'log'; this.fastMath = fastMath; }
  override match(op: Operation): boolean {
    return this.fastMath && matchPattern(LOG_EXP_PAT, op) !== null;
  }
  override rewrite(op: Operation, builder: IRBuilder): boolean {
    op.replaceAllResultsWith([op.getOperand(0).definingOp!.getOperand(0)]);
    op.erase();
    return true;
  }
}

export class DivSelf extends Pattern {
  fastMath: boolean;
  constructor(fastMath = false) { super('div_self', 5); this.rootOpName = 'div'; this.fastMath = fastMath; }
  override match(op: Operation): boolean {
    if (!this.fastMath) return false;
    return op.getOperand(0) === op.getOperand(1);
  }
  override rewrite(op: Operation, builder: IRBuilder): boolean {
    const one = builder.scalarConstant(1, (op.getResult(0).type as TensorType).dtype);
    let result = one.getResult(0);
    const shape = (op.getResult(0).type as TensorType).shape;
    if (shape.length > 0) {
      result = builder.broadcast(result, shape, []).getResult(0);
    }
    op.replaceAllResultsWith([result]);
    op.erase();
    return true;
  }
}

export class MulNegNeg extends Pattern {
  constructor() { super('mul_neg_neg', 4); this.rootOpName = 'mul'; }
  override match(op: Operation): boolean {
    const lDef = op.getOperand(0).definingOp;
    const rDef = op.getOperand(1).definingOp;
    return (lDef && lDef.opName === 'neg' && rDef && rDef.opName === 'neg') as boolean;
  }
  override rewrite(op: Operation, builder: IRBuilder): boolean {
    const a = op.getOperand(0).definingOp!.getOperand(0);
    const b = op.getOperand(1).definingOp!.getOperand(0);
    const newMul = builder.mul(a, b);
    op.replaceAllResultsWith([newMul.getResult(0)]);
    op.erase();
    return true;
  }
}

export class AddNegToSub extends Pattern {
  constructor() { super('add_neg_to_sub', 4); this.rootOpName = 'add'; }
  override match(op: Operation): boolean {
    const rDef = op.getOperand(1).definingOp;
    return (rDef && rDef.opName === 'neg') as boolean;
  }
  override rewrite(op: Operation, builder: IRBuilder): boolean {
    const a = op.getOperand(0);
    const b = op.getOperand(1).definingOp!.getOperand(0);
    const newSub = builder.sub(a, b);
    op.replaceAllResultsWith([newSub.getResult(0)]);
    op.erase();
    return true;
  }
}

export class SubNegToAdd extends Pattern {
  constructor() { super('sub_neg_to_add', 4); this.rootOpName = 'sub'; }
  override match(op: Operation): boolean {
    const rDef = op.getOperand(1).definingOp;
    return (rDef && rDef.opName === 'neg') as boolean;
  }
  override rewrite(op: Operation, builder: IRBuilder): boolean {
    const a = op.getOperand(0);
    const b = op.getOperand(1).definingOp!.getOperand(0);
    const newAdd = builder.add(a, b);
    op.replaceAllResultsWith([newAdd.getResult(0)]);
    op.erase();
    return true;
  }
}

export class DoubleConvert extends Pattern {
  constructor() { super('double_convert', 6); this.rootOpName = 'convert'; }
  override match(op: Operation): boolean {
    const inputOp = op.getOperand(0).definingOp;
    if (!inputOp || inputOp.opName !== 'convert') return false;
    const origDtype = (inputOp.getOperand(0).type as TensorType).dtype;
    const finalDtype = op.getAttr<string>('target_dtype')!;
    return origDtype === finalDtype;
  }
  override rewrite(op: Operation, builder: IRBuilder): boolean {
    op.replaceAllResultsWith([op.getOperand(0).definingOp!.getOperand(0)]);
    op.erase();
    return true;
  }
}

export class LayoutTransformIdentity extends Pattern {
  constructor() { super('layout_transform_identity', 10); this.rootOpName = 'layout_transform'; }
  override match(op: Operation): boolean {
    const src = op.getAttr<readonly number[]>('src_layout')!;
    const dst = op.getAttr<readonly number[]>('dst_layout')!;
    if (!src || !dst || src.length !== dst.length) return false;
    for (let i = 0; i < src.length; i++) {
      if (src[i] !== dst[i]) return false;
    }
    return true;
  }
  override rewrite(op: Operation, builder: IRBuilder): boolean {
    op.replaceAllResultsWith([op.getOperand(0)]);
    op.erase();
    return true;
  }
}

function foldableTransposeOperand(op: Operation, i: number): { source: Value; perm: readonly number[] } | null {
  const def = op.getOperand(i).definingOp;
  if (!def || def.opName !== 'transpose') return null;
  const perm = def.getAttr<readonly number[]>('permutation');
  if (!perm) return null;
  const rank = perm.length;
  const contracting = op.getAttr<readonly number[]>(i === 0 ? 'lhs_contracting' : 'rhs_contracting') || [];
  const batch = op.getAttr<readonly number[]>(i === 0 ? 'lhs_batch' : 'rhs_batch') || [];
  const bound = new Set<number>(contracting);
  for (const d of batch) bound.add(d);
  const free: number[] = [];
  for (let d = 0; d < rank; d++) {
    if (!bound.has(d)) free.push(d);
  }
  if (!isOrderPreserving(perm, free) || !isOrderPreserving(perm, batch)) return null;
  return { source: def.getOperand(0), perm };
}

function isOrderPreserving(perm: readonly number[], dims: readonly number[]): boolean {
  for (let j = 1; j < dims.length; j++) {
    if (perm[dims[j - 1]] >= perm[dims[j]]) return false;
  }
  return true;
}

export class FoldTransposeIntoDot extends Pattern {
  constructor() { super('fold_transpose_into_dot', 10); this.rootOpName = 'dot'; }
  override match(op: Operation): boolean {
    for (let i = 0; i < 2; i++) {
      if (foldableTransposeOperand(op, i)) return true;
    }
    return false;
  }
  override rewrite(op: Operation, builder: IRBuilder): boolean {
    const operands = [op.getOperand(0), op.getOperand(1)];
    const contracting = [[...op.getAttr<readonly number[]>('lhs_contracting')!], [...op.getAttr<readonly number[]>('rhs_contracting')!]];
    const batch = [[...(op.getAttr<readonly number[]>('lhs_batch') || [])], [...(op.getAttr<readonly number[]>('rhs_batch') || [])]];
    for (let i = 0; i < 2; i++) {
      const folded = foldableTransposeOperand(op, i);
      if (!folded) continue;
      operands[i] = folded.source;
      for (const dims of [contracting[i], batch[i]]) {
        for (let j = 0; j < dims.length; j++) dims[j] = folded.perm[dims[j]];
      }
    }
    const newDot = builder.dot(operands[0], operands[1], contracting[0], contracting[1], batch[0], batch[1]);
    const outDtype = op.getAttr<string>('out_dtype');
    if (outDtype) newDot.setAttr('out_dtype', outDtype);
    op.replaceAllResultsWith([newDot.getResult(0)]);
    op.erase();
    return true;
  }
}

export class LayoutTransformCompose extends Pattern {
  constructor() { super('layout_transform_compose', 10); this.rootOpName = 'layout_transform'; }
  override match(op: Operation): boolean {
    const inputOp = op.getOperand(0).definingOp;
    return (inputOp && inputOp.opName === 'layout_transform') as boolean;
  }
  override rewrite(op: Operation, builder: IRBuilder): boolean {
    const innerOp = op.getOperand(0).definingOp;
    const srcLayout = innerOp!.getAttr<readonly number[]>('src_layout')!;
    const midLayout = innerOp!.getAttr<readonly number[]>('dst_layout')!;
    const dstLayout = op.getAttr<readonly number[]>('dst_layout')!;
    const composed = new Array(srcLayout.length);
    for (let i = 0; i < dstLayout.length; i++) {
      composed[i] = srcLayout[midLayout.indexOf(dstLayout[i])];
    }
    const original = innerOp!.getOperand(0);
    const newOp = builder._inferAndBuild('layout_transform', [original],
      { src_layout: srcLayout, dst_layout: composed });
    op.replaceAllResultsWith([newOp.getResult(0)]);
    op.erase();
    return true;
  }
}

export class IdempotentSelf extends Pattern {
  constructor(opName: string) {
    super(`idempotent_self_${opName}`, 10);
    this.rootOpName = opName;
  }
  override match(op: Operation): boolean {
    return op.numOperands === 2 && op.getOperand(0) === op.getOperand(1);
  }
  override rewrite(op: Operation, builder: IRBuilder): boolean {
    op.replaceAllResultsWith([op.getOperand(0)]);
    op.erase();
    return true;
  }
}

export class AssociativeConstantReassoc extends Pattern {
  fastMath: boolean;
  constructor(opName: string, fastMath = false) {
    super(`associative_constant_reassoc_${opName}`, 4);
    this.rootOpName = opName;
    this.fastMath = fastMath;
  }
  _parts(op: Operation): { inner: Operation; x: Value; outerConst: Operation; innerConst: Operation } | null {
    if (op.numOperands !== 2) return null;
    const outerConst = op.getOperand(1).definingOp;
    if (!outerConst || outerConst.opName !== 'constant') return null;
    const innerValue = op.getOperand(0);
    if (innerValue.useCount !== 1) return null;
    const inner = innerValue.definingOp;
    if (!inner || inner.opName !== op.opName || inner.numOperands !== 2) return null;
    const innerConst = inner.getOperand(1).definingOp;
    if (!innerConst || innerConst.opName !== 'constant') return null;
    if (!innerConst.getResult(0).type.equals(outerConst.getResult(0).type)) return null;
    return { inner, x: inner.getOperand(0), outerConst, innerConst };
  }
  _folded(op: Operation, parts: { outerConst: Operation; innerConst: Operation }): AttrValue | undefined {
    const def = registry.get(op.opName);
    if (!def || !def.fold) return undefined;
    const a = parts.innerConst.getAttr<AttrValue>('value');
    const b = parts.outerConst.getAttr<AttrValue>('value');
    if (a === undefined || b === undefined) return undefined;
    return def.fold([a, b], op.attributes, [parts.innerConst, parts.outerConst]);
  }
  override match(op: Operation): boolean {
    if (!isDtypeInt((op.getResult(0).type as TensorType).dtype) && !this.fastMath) return false;
    const parts = this._parts(op);
    return parts !== null && this._folded(op, parts) !== undefined;
  }
  override rewrite(op: Operation, builder: IRBuilder): boolean {
    if (!isDtypeInt((op.getResult(0).type as TensorType).dtype) && !this.fastMath) return false;
    const parts = this._parts(op);
    if (!parts) return false;
    const value = this._folded(op, parts);
    if (value === undefined) return false;
    const folded = builder.constant(value, parts.outerConst.getResult(0).type as TensorType);
    const combined = builder.create(op.opName, [parts.x, folded.getResult(0)], op.attributes);
    op.replaceAllResultsWith([combined.getResult(0)]);
    op.erase();
    parts.inner.erase();
    return true;
  }
}
