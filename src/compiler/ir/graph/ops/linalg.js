import { OpDef, OpTrait } from '../op_registry.js';
import { TensorType, DYNAMIC } from '../types.js';
import * as pat from '../patterns.js';

export function register(registry) {
  registry.register(new OpDef({
    name: 'dot',
    numOperands: 2,
    numResults: 1,
    traits: [OpTrait.OPAQUE],
    attrs: [
      { name: 'lhs_contracting', type: 'array', required: true },
      { name: 'rhs_contracting', type: 'array', required: true },
      { name: 'lhs_batch', type: 'array', required: false },
      { name: 'rhs_batch', type: 'array', required: false }
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
    inferResultTypes: inferDotResultTypes,
    getCanonicalizationPatterns() { return [new pat.FoldTransposeIntoDot()]; },
    verify(op) {
      const errs = [];
      if (op.numOperands !== 2) { errs.push('dot expects 2 operands'); return errs; }
      if (!op.hasAttr('lhs_contracting')) errs.push('dot missing lhs_contracting');
      if (!op.hasAttr('rhs_contracting')) errs.push('dot missing rhs_contracting');
      const lhs = op.getOperand(0).type, rhs = op.getOperand(1).type;
      if (lhs instanceof TensorType && rhs instanceof TensorType) {
        if (lhs.dtype !== rhs.dtype) {
          errs.push(`dot dtype mismatch: ${lhs.dtype} vs ${rhs.dtype}`);
        }
        const lhsC = op.getAttr('lhs_contracting') || [];
        const rhsC = op.getAttr('rhs_contracting') || [];
        if (lhsC.length !== rhsC.length) {
          errs.push(`dot contracting dimensions count mismatch: lhs ${lhsC.length} vs rhs ${rhsC.length}`);
        } else {
          for (let i = 0; i < lhsC.length; i++) {
            const ld = lhsC[i], rd = rhsC[i];
            if (ld >= lhs.rank) errs.push(`dot lhs_contracting[${i}]=${ld} out of range (rank ${lhs.rank})`);
            if (rd >= rhs.rank) errs.push(`dot rhs_contracting[${i}]=${rd} out of range (rank ${rhs.rank})`);
            if (ld < lhs.rank && rd < rhs.rank && lhs.shape[ld] !== DYNAMIC && rhs.shape[rd] !== DYNAMIC && lhs.shape[ld] !== rhs.shape[rd]) {
              errs.push(`dot contracting dim size mismatch at [${i}]: lhs dim ${ld} size ${lhs.shape[ld]} vs rhs dim ${rd} size ${rhs.shape[rd]}`);
            }
          }
        }
      }
      return errs;
    }
  }));

  registry.register(new OpDef({
    name: 'conv',
    numOperands: 2,
    numResults: 1,
    traits: [OpTrait.OPAQUE],
    attrs: [
      { name: 'strides', type: 'array', required: true },
      { name: 'padding', type: 'array', required: true },
      { name: 'dilation', type: 'array', required: false },
      { name: 'groups', type: 'number', required: false },
      { name: 'input_layout', type: 'string', required: true },
      { name: 'kernel_layout', type: 'string', required: true }
    ],
    getFlops(op) {
      const output = op.getResult(0).type;
      const kernel = op.getOperand(1).type;
      if (!(output instanceof TensorType) || !(kernel instanceof TensorType)) return 0;
      const outElements = output.numel();
      const kElements = kernel.numel();
      if (outElements === DYNAMIC || kElements === DYNAMIC) return 0;
      return 2 * outElements * kElements / (kernel.shape[0] || 1);
    },
    inferResultTypes(operandTypes, attrs) {
      if (operandTypes.length !== 2) return null;
      const inp = operandTypes[0], kernel = operandTypes[1];
      if (!(inp instanceof TensorType) || !(kernel instanceof TensorType)) return null;
      if (inp.dtype !== kernel.dtype) return null;
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
      return [new TensorType([batch, outChannels, ...outSpatial], inp.dtype)];
    },
    verify(op) {
      const errs = [];
      if (op.numOperands !== 2) { errs.push('conv expects 2 operands'); return errs; }
      if (!op.hasAttr('strides')) errs.push('conv missing strides');
      if (!op.hasAttr('padding')) errs.push('conv missing padding');
      if (!op.hasAttr('input_layout')) errs.push('conv missing input_layout');
      if (!op.hasAttr('kernel_layout')) errs.push('conv missing kernel_layout');
      return errs;
    }
  }));
}

function inferDotResultTypes(operandTypes, attrs) {
  if (operandTypes.length !== 2) return null;
  const lhs = operandTypes[0], rhs = operandTypes[1];
  if (!(lhs instanceof TensorType) || !(rhs instanceof TensorType)) return null;
  if (lhs.dtype !== rhs.dtype) return null;
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
  return [new TensorType(shape, lhs.dtype)];
}

export { inferDotResultTypes };
