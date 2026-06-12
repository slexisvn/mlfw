import { OpDef, OpTrait } from '../op_registry.js';
import { TensorType, ScalarType, DYNAMIC, isFloatType } from '../types.js';
import * as qpat from '../quantization_patterns.js';

const QUANTIZED_INT_DTYPES = new Set([ScalarType.I8, ScalarType.UI8]);
const DEQUANT_INPUT_DTYPES = new Set([ScalarType.I8, ScalarType.UI8, ScalarType.I32]);

export function register(registry) {
  registry.register(new OpDef({
    name: 'quantize',
    numOperands: 1,
    numResults: 1,
    traits: [OpTrait.ELEMENTWISE],
    attrs: [
      { name: 'scale', type: 'any', required: true },
      { name: 'zero_point', type: 'any', required: true },
      { name: 'scheme', type: 'string', required: true },
      { name: 'target_dtype', type: 'string', required: true },
      { name: 'axis', type: 'number', required: false }
    ],
    inferResultTypes(operandTypes, attrs) {
      if (operandTypes.length !== 1) return null;
      const inp = operandTypes[0];
      if (!(inp instanceof TensorType)) return null;
      const dt = attrs.get ? attrs.get('target_dtype') : attrs.target_dtype;
      if (!dt) return null;
      return [new TensorType(inp.shape, dt)];
    },
    verify(op) {
      const errs = [];
      if (op.numOperands !== 1) { errs.push('quantize expects 1 operand'); return errs; }
      const inp = op.getOperand(0).type;
      if (inp instanceof TensorType && !isFloatType(inp.dtype)) {
        errs.push(`quantize input must be float, got ${inp.dtype}`);
      }
      const tgt = op.getAttr('target_dtype');
      if (tgt && !QUANTIZED_INT_DTYPES.has(tgt)) {
        errs.push(`quantize target_dtype must be i8 or ui8, got ${tgt}`);
      }
      const scale = op.getAttr('scale');
      if (typeof scale === 'number' && scale <= 0) {
        errs.push('quantize scale must be positive');
      }
      return errs;
    },
    fold(constValues, attrs) {
      if (constValues[0] === undefined) return undefined;
      const scale = attrs.get ? attrs.get('scale') : attrs.scale;
      const zp = attrs.get ? attrs.get('zero_point') : attrs.zero_point;
      if (typeof scale !== 'number' || typeof zp !== 'number') return undefined;
      return Math.round(constValues[0] / scale + zp);
    },
    getCanonicalizationPatterns() {
      return [new qpat.QuantizeDequantizeIdentity()];
    }
  }));

  registry.register(new OpDef({
    name: 'dequantize',
    numOperands: 1,
    numResults: 1,
    traits: [OpTrait.ELEMENTWISE],
    attrs: [
      { name: 'scale', type: 'any', required: true },
      { name: 'zero_point', type: 'any', required: true },
      { name: 'scheme', type: 'string', required: true },
      { name: 'target_dtype', type: 'string', required: true },
      { name: 'axis', type: 'number', required: false }
    ],
    inferResultTypes(operandTypes, attrs) {
      if (operandTypes.length !== 1) return null;
      const inp = operandTypes[0];
      if (!(inp instanceof TensorType)) return null;
      const dt = attrs.get ? attrs.get('target_dtype') : attrs.target_dtype;
      if (!dt) return null;
      return [new TensorType(inp.shape, dt)];
    },
    verify(op) {
      const errs = [];
      if (op.numOperands !== 1) { errs.push('dequantize expects 1 operand'); return errs; }
      const inp = op.getOperand(0).type;
      if (inp instanceof TensorType && !DEQUANT_INPUT_DTYPES.has(inp.dtype)) {
        errs.push('dequantize input must be i8, ui8, or i32, got ' + inp.dtype);
      }
      const tgt = op.getAttr('target_dtype');
      if (tgt && !isFloatType(tgt)) {
        errs.push(`dequantize target_dtype must be float, got ${tgt}`);
      }
      return errs;
    },
    fold(constValues, attrs) {
      if (constValues[0] === undefined) return undefined;
      const scale = attrs.get ? attrs.get('scale') : attrs.scale;
      const zp = attrs.get ? attrs.get('zero_point') : attrs.zero_point;
      if (typeof scale !== 'number' || typeof zp !== 'number') return undefined;
      return (constValues[0] - zp) * scale;
    }
  }));

  registry.register(new OpDef({
    name: 'quantized_dot',
    numOperands: 2,
    numResults: 1,
    traits: [OpTrait.OPAQUE],
    attrs: [
      { name: 'lhs_contracting', type: 'array', required: true },
      { name: 'rhs_contracting', type: 'array', required: true },
      { name: 'lhs_batch', type: 'array', required: false },
      { name: 'rhs_batch', type: 'array', required: false },
      { name: 'lhs_scale', type: 'number', required: true },
      { name: 'lhs_zero_point', type: 'number', required: true },
      { name: 'rhs_scale', type: 'number', required: true },
      { name: 'rhs_zero_point', type: 'number', required: true },
      { name: 'output_scale', type: 'number', required: true },
      { name: 'output_zero_point', type: 'number', required: true }
    ],
    getFlops(op) {
      const lhs = op.getOperand(0).type;
      const rhs = op.getOperand(1).type;
      if (!(lhs instanceof TensorType) || !(rhs instanceof TensorType)) return 0;
      const lhsC = op.getAttr('lhs_contracting') || [];
      let contractDim = 1;
      for (const d of lhsC) {
        if (lhs.shape[d] !== DYNAMIC) contractDim *= lhs.shape[d];
      }
      const result = op.getResult(0).type;
      if (!(result instanceof TensorType)) return 0;
      const outputElements = result.numel();
      if (outputElements === DYNAMIC) return 0;
      return 2 * outputElements * contractDim;
    },
    inferResultTypes(operandTypes, attrs) {
      if (operandTypes.length !== 2) return null;
      const lhs = operandTypes[0], rhs = operandTypes[1];
      if (!(lhs instanceof TensorType) || !(rhs instanceof TensorType)) return null;
      const lhsC = new Set(attrs.get ? attrs.get('lhs_contracting') : attrs.lhs_contracting);
      const rhsC = new Set(attrs.get ? attrs.get('rhs_contracting') : attrs.rhs_contracting);
      const lhsB = new Set((attrs.get ? attrs.get('lhs_batch') : attrs.lhs_batch) || []);
      const rhsB = new Set((attrs.get ? attrs.get('rhs_batch') : attrs.rhs_batch) || []);
      const shape = [];
      for (let i = 0; i < lhs.rank; i++) {
        if (lhsB.has(i)) shape.push(lhs.shape[i]);
      }
      for (let i = 0; i < lhs.rank; i++) {
        if (!lhsC.has(i) && !lhsB.has(i)) shape.push(lhs.shape[i]);
      }
      for (let i = 0; i < rhs.rank; i++) {
        if (!rhsC.has(i) && !rhsB.has(i)) shape.push(rhs.shape[i]);
      }
      return [new TensorType(shape, ScalarType.I32)];
    },
    verify(op) {
      const errs = [];
      if (op.numOperands !== 2) { errs.push('quantized_dot expects 2 operands'); return errs; }
      if (!op.hasAttr('lhs_contracting')) errs.push('quantized_dot missing lhs_contracting');
      if (!op.hasAttr('rhs_contracting')) errs.push('quantized_dot missing rhs_contracting');
      if (!op.hasAttr('lhs_scale')) errs.push('quantized_dot missing lhs_scale');
      if (!op.hasAttr('rhs_scale')) errs.push('quantized_dot missing rhs_scale');
      const lhs = op.getOperand(0).type, rhs = op.getOperand(1).type;
      if (lhs instanceof TensorType && !QUANTIZED_INT_DTYPES.has(lhs.dtype)) {
        errs.push(`quantized_dot lhs must be i8/ui8, got ${lhs.dtype}`);
      }
      if (rhs instanceof TensorType && !QUANTIZED_INT_DTYPES.has(rhs.dtype)) {
        errs.push(`quantized_dot rhs must be i8/ui8, got ${rhs.dtype}`);
      }
      return errs;
    }
  }));

  registry.register(new OpDef({
    name: 'quantized_conv',
    numOperands: 2,
    numResults: 1,
    traits: [OpTrait.OPAQUE],
    attrs: [
      { name: 'strides', type: 'array', required: true },
      { name: 'padding', type: 'array', required: true },
      { name: 'dilation', type: 'array', required: false },
      { name: 'groups', type: 'number', required: false },
      { name: 'input_layout', type: 'string', required: true },
      { name: 'kernel_layout', type: 'string', required: true },
      { name: 'input_scale', type: 'number', required: true },
      { name: 'input_zero_point', type: 'number', required: true },
      { name: 'kernel_scale', type: 'number', required: true },
      { name: 'kernel_zero_point', type: 'number', required: true },
      { name: 'output_scale', type: 'number', required: true },
      { name: 'output_zero_point', type: 'number', required: true }
    ],
    inferResultTypes(operandTypes, attrs) {
      if (operandTypes.length !== 2) return null;
      const inp = operandTypes[0], kernel = operandTypes[1];
      if (!(inp instanceof TensorType) || !(kernel instanceof TensorType)) return null;
      const strides = attrs.get ? attrs.get('strides') : attrs.strides;
      const padding = attrs.get ? attrs.get('padding') : attrs.padding;
      const dilation = (attrs.get ? attrs.get('dilation') : attrs.dilation) || strides.map(() => 1);
      const spatialDims = strides.length;
      const batch = inp.shape[0];
      const outChannels = kernel.shape[0];
      const outSpatial = [];
      for (let i = 0; i < spatialDims; i++) {
        const inDim = inp.shape[i + 2];
        const kDim = kernel.shape[i + 2];
        const padTotal = padding[i][0] + padding[i][1];
        if (inDim === DYNAMIC || kDim === DYNAMIC) {
          outSpatial.push(DYNAMIC);
        } else {
          const effectiveK = (kDim - 1) * dilation[i] + 1;
          outSpatial.push(Math.floor((inDim + padTotal - effectiveK) / strides[i]) + 1);
        }
      }
      return [new TensorType([batch, outChannels, ...outSpatial], ScalarType.I32)];
    },
    verify(op) {
      const errs = [];
      if (op.numOperands !== 2) { errs.push('quantized_conv expects 2 operands'); return errs; }
      if (!op.hasAttr('strides')) errs.push('quantized_conv missing strides');
      if (!op.hasAttr('padding')) errs.push('quantized_conv missing padding');
      if (!op.hasAttr('input_layout')) errs.push('quantized_conv missing input_layout');
      if (!op.hasAttr('kernel_layout')) errs.push('quantized_conv missing kernel_layout');
      if (!op.hasAttr('input_scale')) errs.push('quantized_conv missing input_scale');
      if (!op.hasAttr('kernel_scale')) errs.push('quantized_conv missing kernel_scale');
      return errs;
    }
  }));
}
