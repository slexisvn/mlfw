import { OpAttrKey, OpDef, OpTrait } from '../op_registry.js';
import type { OpAttrMap, OpAttrRecord, OpRegistry } from '../op_registry.js';
import type { IRType } from '../types.js';

type LinalgInferOpts = Readonly<{ allowMixedDtype?: boolean; outputDtype?: ScalarDType }>;
type LinalgInferExtra = readonly IRType[] | LinalgInferOpts | null;
import { TensorType, DYNAMIC, Layout } from '../types.js';
import { LayoutPreference } from '../layout_pref.js';
import type { LayoutTarget } from '../layout_pref.js';
import type { Operation } from '../operation.js';
import type { ScalarDType } from '../types.js';
import * as pat from '../patterns.js';
import * as qpat from '../quantization_patterns.js';

const NHWC_RANK = 4;

function convLayout(op: Operation, target: LayoutTarget): LayoutPreference | null {
  const input = op.getOperand(0).type as TensorType;
  const rank = input ? input.rank : NHWC_RANK;
  if (target.preferredConvLayout) {
    const preferred = target.preferredConvLayout as unknown as Layout;
    return new LayoutPreference([preferred, null], [preferred]);
  }
  if (rank !== NHWC_RANK || !(target.isGPU() || target.isCPU())) return null;
  const nhwc = new Layout([0, 2, 3, 1]);
  return new LayoutPreference([nhwc, null], [nhwc]);
}

function dotLayout(op: Operation, target: LayoutTarget): LayoutPreference | null {
  const lhsType = op.getOperand(0).type as TensorType;
  const rhsType = op.getOperand(1).type as TensorType;
  if (!lhsType || !rhsType) return null;
  const lhsLayout = Layout.rowMajor(lhsType.rank);
  if (target.isCPU() && rhsType.rank === 2) {
    return new LayoutPreference([lhsLayout, Layout.columnMajor(rhsType.rank)], [lhsLayout]);
  }
  return new LayoutPreference([lhsLayout, Layout.rowMajor(rhsType.rank)], [lhsLayout]);
}

export function register(registry: OpRegistry) {
  registry.register(new OpDef({
    name: 'dot',
    numOperands: 2,
    numResults: 1,
    opAttrs: { [OpAttrKey.GPU_CAPABLE]: true, [OpAttrKey.LAUNCH_BOUNDARY]: 'matmul', [OpAttrKey.LAYOUT_SENSITIVITY]: 4, [OpAttrKey.INFER_LAYOUT]: dotLayout },
    traits: [OpTrait.OPAQUE, OpTrait.OUT_EWISE_FUSABLE],
    attrs: [
      { name: 'lhs_contracting', type: 'array', required: true },
      { name: 'rhs_contracting', type: 'array', required: true },
      { name: 'lhs_batch', type: 'array', required: false },
      { name: 'rhs_batch', type: 'array', required: false },
      { name: 'out_dtype', type: 'string', required: false }
    ],
    getFlops(op) {
      const lhs = op.getOperand(0).type;
      const rhs = op.getOperand(1).type;
      if (!(lhs instanceof TensorType) || !(rhs instanceof TensorType)) return 0;
      const lhsC = op.getAttr<readonly number[]>('lhs_contracting')! || [];
      let contractDim = 1;
      for (const d of lhsC) {
        if (lhs.shape[d] !== DYNAMIC) contractDim *= lhs.shape[d] as number;
      }
      const result = op.getResult(0).type;
      if (!(result instanceof TensorType)) return 0;
      const outputElements = result.numel();
      if (outputElements === DYNAMIC) return 0;
      return 2 * outputElements * contractDim;
    },
    inferResultTypes: inferDotResultTypes,
    getCanonicalizationPatterns() { return [new pat.FoldTransposeIntoDot(), new qpat.DequantizeFoldIntoDot()]; },
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
        const lhsC = op.getAttr<readonly number[]>('lhs_contracting')! || [];
        const rhsC = op.getAttr<readonly number[]>('rhs_contracting')! || [];
        if (lhsC.length !== rhsC.length) {
          errs.push(`dot contracting dimensions count mismatch: lhs ${lhsC.length} vs rhs ${rhsC.length}`);
        } else {
          for (let i = 0; i < lhsC.length; i++) {
            const ld = lhsC[i], rd = rhsC[i];
            if (ld >= lhs.rank) errs.push(`dot lhs_contracting[${i}]=${ld} out of range (rank ${lhs.rank})`);
            if (rd >= rhs.rank) errs.push(`dot rhs_contracting[${i}]=${rd} out of range (rank ${rhs.rank})`);
            const ldDim = lhs.shape[ld], rdDim = rhs.shape[rd];
            if (ld < lhs.rank && rd < rhs.rank && typeof ldDim === 'number' && ldDim !== DYNAMIC && typeof rdDim === 'number' && rdDim !== DYNAMIC && ldDim !== rdDim) {
              errs.push(`dot contracting dim size mismatch at [${i}]: lhs dim ${ld} size ${ldDim} vs rhs dim ${rd} size ${rdDim}`);
            }
          }
        }
      }
      return errs;
    }
  }));

  registry.register(new OpDef({
    name: 'cublas_gemm',
    numOperands: 2,
    numResults: 1,
    opAttrs: { [OpAttrKey.LAUNCH_BOUNDARY]: 'matmul' },
    traits: [OpTrait.OPAQUE],
    attrs: [
      { name: 'lhs_contracting', type: 'array', required: true },
      { name: 'rhs_contracting', type: 'array', required: true },
      { name: 'lhs_batch', type: 'array', required: false },
      { name: 'rhs_batch', type: 'array', required: false }
    ],
    inferResultTypes: inferDotResultTypes,
    getFlops(op) {
      const lhs = op.getOperand(0).type;
      const result = op.getResult(0).type;
      if (!(lhs instanceof TensorType) || !(result instanceof TensorType)) return 0;
      let contractDim = 1;
      for (const d of (op.getAttr<readonly number[]>('lhs_contracting')! || [])) {
        if (lhs.shape[d] !== DYNAMIC) contractDim *= lhs.shape[d] as number;
      }
      const outputElements = result.numel();
      if (outputElements === DYNAMIC) return 0;
      return 2 * outputElements * contractDim;
    },
    verify(op) {
      const errs = [];
      if (op.numOperands !== 2) { errs.push('cublas_gemm expects 2 operands'); return errs; }
      if (!op.hasAttr('lhs_contracting')) errs.push('cublas_gemm missing lhs_contracting');
      if (!op.hasAttr('rhs_contracting')) errs.push('cublas_gemm missing rhs_contracting');
      return errs;
    }
  }));

  registry.register(new OpDef({
    name: 'conv',
    numOperands: 2,
    numResults: 1,
    opAttrs: { [OpAttrKey.GPU_CAPABLE]: true, [OpAttrKey.LAUNCH_BOUNDARY]: 'conv', [OpAttrKey.LAYOUT_SENSITIVITY]: 4, [OpAttrKey.INFER_LAYOUT]: convLayout },
    traits: [OpTrait.OPAQUE],
    attrs: [
      { name: 'strides', type: 'array', required: true },
      { name: 'padding', type: 'array', required: true },
      { name: 'dilation', type: 'array', required: false },
      { name: 'groups', type: 'number', required: false },
      { name: 'input_layout', type: 'string', required: true },
      { name: 'kernel_layout', type: 'string', required: true },
      { name: 'out_dtype', type: 'string', required: false }
    ],
    getFlops(op) {
      const output = op.getResult(0).type;
      const kernel = op.getOperand(1).type;
      if (!(output instanceof TensorType) || !(kernel instanceof TensorType)) return 0;
      const outElements = output.numel();
      const kElements = kernel.numel();
      if (outElements === DYNAMIC || kElements === DYNAMIC) return 0;
      return 2 * outElements * kElements / ((kernel.shape[0] as number) || 1);
    },
    inferResultTypes: inferConvResultTypes,
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

function inferDotResultTypes(operandTypes: readonly IRType[], attrs: OpAttrMap, opts: LinalgInferExtra): readonly IRType[] | null {
  if (operandTypes.length !== 2) return null;
  const lhs = operandTypes[0], rhs = operandTypes[1];
  if (!(lhs instanceof TensorType) || !(rhs instanceof TensorType)) return null;
  const o = (opts && !Array.isArray(opts) ? opts : {}) as LinalgInferOpts;
  if (!o.allowMixedDtype && lhs.dtype !== rhs.dtype) return null;
  const declared = (attrs.get ? attrs.get('out_dtype') : (attrs as unknown as OpAttrRecord).out_dtype) as ScalarDType;
  const lhsC = new Set((attrs.get ? attrs.get('lhs_contracting') : (attrs as unknown as OpAttrRecord).lhs_contracting) as readonly number[]);
  const rhsC = new Set((attrs.get ? attrs.get('rhs_contracting') : (attrs as unknown as OpAttrRecord).rhs_contracting) as readonly number[]);
  const lhsB = new Set(((attrs.get ? attrs.get('lhs_batch') : (attrs as unknown as OpAttrRecord).lhs_batch) as readonly number[]) || []);
  const rhsB = new Set(((attrs.get ? attrs.get('rhs_batch') : (attrs as unknown as OpAttrRecord).rhs_batch) as readonly number[]) || []);
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
  return [new TensorType(shape, declared || o.outputDtype || lhs.dtype)];
}

function inferConvResultTypes(operandTypes: readonly IRType[], attrs: OpAttrMap, opts: LinalgInferExtra): readonly IRType[] | null {
  if (operandTypes.length !== 2) return null;
  const inp = operandTypes[0], kernel = operandTypes[1];
  if (!(inp instanceof TensorType) || !(kernel instanceof TensorType)) return null;
  const o = (opts && !Array.isArray(opts) ? opts : {}) as LinalgInferOpts;
  if (!o.allowMixedDtype && inp.dtype !== kernel.dtype) return null;
  const declared = (attrs.get ? attrs.get('out_dtype') : (attrs as unknown as OpAttrRecord).out_dtype) as ScalarDType;
  const strides = (attrs.get ? attrs.get('strides') : (attrs as unknown as OpAttrRecord).strides) as readonly number[];
  const padding = (attrs.get ? attrs.get('padding') : (attrs as unknown as OpAttrRecord).padding) as readonly (readonly number[])[];
  const dilation = ((attrs.get ? attrs.get('dilation') : (attrs as unknown as OpAttrRecord).dilation) as readonly number[]) || strides.map(() => 1);
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
      const effectiveK = ((kDim as number) - 1) * dilation[i] + 1;
      outSpatial.push(Math.floor(((inDim as number) + padTotal - effectiveK) / strides[i]) + 1);
    }
  }
  return [new TensorType([batch, outChannels, ...outSpatial], declared || o.outputDtype || inp.dtype)];
}

export { inferDotResultTypes, inferConvResultTypes };
