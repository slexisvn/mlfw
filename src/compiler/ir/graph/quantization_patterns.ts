import { Pattern } from '../rewrite/pattern.js';
import { TensorType, ScalarType, scalarBytes } from './types.js';
import type { ScalarDType } from './types.js';
import type { Operation } from './operation.js';
import type { IRBuilder } from './builder.js';

function quantParamsMatch(opA: Operation, opB: Operation): boolean {
  const scaleA = opA.getAttr('scale');
  const scaleB = opB.getAttr('scale');
  const zpA = opA.getAttr('zero_point');
  const zpB = opB.getAttr('zero_point');
  const schemeA = opA.getAttr('scheme');
  const schemeB = opB.getAttr('scheme');
  if (schemeA !== schemeB) return false;
  if (typeof scaleA === 'number' && typeof scaleB === 'number') {
    if (scaleA !== scaleB) return false;
  } else {
    return false;
  }
  if (typeof zpA === 'number' && typeof zpB === 'number') {
    if (zpA !== zpB) return false;
  } else {
    return false;
  }
  return true;
}

export class QuantizeDequantizeIdentity extends Pattern {
  constructor() {
    super('quantize_dequantize_identity', 20);
    this.rootOpName = 'quantize';
  }

  override match(op: Operation): boolean {
    const inputOp = op.getOperand(0).definingOp;
    if (!inputOp || inputOp.opName !== 'dequantize') return false;
    return quantParamsMatch(op, inputOp!);
  }

  override rewrite(op: Operation, builder: IRBuilder): boolean {
    const original = op.getOperand(0).definingOp!.getOperand(0);
    if (!original.type || !op.getResult(0).type) return false;
    const srcType = original.type as TensorType;
    const dstType = op.getResult(0).type;
    if (srcType instanceof TensorType && dstType instanceof TensorType && srcType.dtype === dstType.dtype) {
      op.replaceAllResultsWith([original]);
      op.erase();
      return true;
    }
    return false;
  }
}

export class ConstantQuantize extends Pattern {
  constructor() {
    super('constant_quantize', 15);
    this.rootOpName = 'quantize';
  }

  override match(op: Operation): boolean {
    const inputOp = op.getOperand(0).definingOp;
    if (!inputOp || inputOp.opName !== 'constant') return false;
    const val = inputOp!.getAttr<number>('value')!;
    return typeof val === 'number';
  }

  override rewrite(op: Operation, builder: IRBuilder): boolean {
    const inputOp = op.getOperand(0).definingOp;
    const val = inputOp!.getAttr<number>('value')!;
    const scale = op.getAttr('scale');
    const zp = op.getAttr('zero_point');
    if (typeof scale !== 'number' || typeof zp !== 'number') return false;
    const tgtDtype = op.getAttr<ScalarDType>('target_dtype')!;
    const bits = scalarBytes(tgtDtype) * 8;
    const isUnsigned = tgtDtype === ScalarType.UI8;
    const cMin = isUnsigned ? 0 : -(1 << (bits - 1));
    const cMax = isUnsigned ? (1 << bits) - 1 : (1 << (bits - 1)) - 1;
    const quantized = Math.max(cMin, Math.min(cMax, Math.round(val / scale + zp)));
    const resultType = op.getResult(0).type as TensorType;
    const newConst = builder.constant(quantized, resultType);
    op.replaceAllResultsWith([newConst.getResult(0)]);
    op.erase();
    return true;
  }
}

export class DequantizeFoldIntoDot extends Pattern {
  constructor() {
    super('dequantize_fold_into_dot', 15);
    this.rootOpName = 'dot';
  }

  override match(op: Operation): boolean {
    if (op.numOperands !== 2) return false;
    const lhsDef = op.getOperand(0).definingOp;
    const rhsDef = op.getOperand(1).definingOp;
    return (lhsDef && lhsDef.opName === 'dequantize'
        && rhsDef && rhsDef.opName === 'dequantize') as boolean;
  }

  override rewrite(op: Operation, builder: IRBuilder): boolean {
    const lhsDequant = op.getOperand(0).definingOp;
    const rhsDequant = op.getOperand(1).definingOp;
    const lhsInput = lhsDequant!.getOperand(0);
    const rhsInput = rhsDequant!.getOperand(0);
    const lhsScale = lhsDequant!.getAttr('scale');
    const rhsScale = rhsDequant!.getAttr('scale');
    const lhsZP = lhsDequant!.getAttr('zero_point');
    const rhsZP = rhsDequant!.getAttr('zero_point');
    if (typeof lhsScale !== 'number' || typeof rhsScale !== 'number') return false;
    if (typeof lhsZP !== 'number' || typeof rhsZP !== 'number') return false;
    if (lhsZP !== 0 || rhsZP !== 0) return false;

    const outputScale = lhsScale * rhsScale;
    const lhsC = op.getAttr<readonly number[]>('lhs_contracting')!;
    const rhsC = op.getAttr<readonly number[]>('rhs_contracting')!;
    const lhsB = op.getAttr<readonly number[]>('lhs_batch')! || [];
    const rhsB = op.getAttr<readonly number[]>('rhs_batch')! || [];

    const resultType = new TensorType((op.getResult(0).type as TensorType).shape, ScalarType.I32);
    const qdot = builder._buildOp('quantized_dot', [lhsInput, rhsInput], [resultType], {
      lhs_contracting: lhsC,
      rhs_contracting: rhsC,
      lhs_batch: lhsB,
      rhs_batch: rhsB,
      lhs_scale: lhsScale,
      lhs_zero_point: lhsZP,
      rhs_scale: rhsScale,
      rhs_zero_point: rhsZP,
      output_scale: outputScale,
      output_zero_point: 0
    });

    const dequant = builder._inferAndBuild('dequantize', [qdot.getResult(0)], {
      scale: outputScale,
      zero_point: 0,
      scheme: lhsDequant!.getAttr('scheme') ?? null,
      target_dtype: (op.getResult(0).type as TensorType).dtype
    });

    op.replaceAllResultsWith([dequant.getResult(0)]);
    op.erase();
    return true;
  }
}
