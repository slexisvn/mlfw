import { registerVJPRule } from '../vjp_registry.js';
import type { TensorValue, VJPContext } from '../vjp_registry.js';
import type { IRBuilder } from '../../ir/graph/builder.js';

registerVJPRule('dot', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  const [lhs, rhs] = ctx.operands;
  const lhsContracting = ctx.op.getAttr<readonly number[]>('lhs_contracting')!;
  const rhsContracting = ctx.op.getAttr<readonly number[]>('rhs_contracting')!;
  const lhsBatch = ctx.op.getAttr<readonly number[]>('lhs_batch')! || [];
  const rhsBatch = ctx.op.getAttr<readonly number[]>('rhs_batch')! || [];

  const lhsRank = lhs.type.rank;
  const rhsRank = rhs.type.rank;

  const lhsFree = [];
  for (let i = 0; i < lhsRank; i++) {
    if (!lhsContracting.includes(i) && !lhsBatch.includes(i)) lhsFree.push(i);
  }
  const rhsFree = [];
  for (let i = 0; i < rhsRank; i++) {
    if (!rhsContracting.includes(i) && !rhsBatch.includes(i)) rhsFree.push(i);
  }

  const gradRhsFreeDims = [];
  for (let i = lhsBatch.length; i < lhsBatch.length + rhsFree.length; i++) {
    gradRhsFreeDims.push(i + lhsFree.length);
  }

  const gradLhsFreeDims = [];
  for (let i = lhsBatch.length; i < lhsBatch.length + lhsFree.length; i++) {
    gradLhsFreeDims.push(i);
  }

  const gradLhs = ctx.builder.dot(
    grad, rhs,
    gradRhsFreeDims, rhsFree,
    Array.from({ length: lhsBatch.length }, (_, i) => i), rhsBatch
  ).getResult(0);

  const gradRhs = ctx.builder.dot(
    lhs, grad,
    lhsFree, gradLhsFreeDims,
    lhsBatch, Array.from({ length: lhsBatch.length }, (_, i) => i)
  ).getResult(0);

  return [gradLhs, gradRhs];
});

registerVJPRule('matmul', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  const [lhs, rhs] = ctx.operands;
  const swapLastTwo = (rank: number) => {
    const perm = Array.from({ length: rank }, (_, i) => i);
    perm[rank - 2] = rank - 1;
    perm[rank - 1] = rank - 2;
    return perm;
  };
  const rhsT = ctx.builder.transpose(rhs, swapLastTwo(rhs.type.rank)).getResult(0);
  const gradLhs = ctx.builder.matmul(grad, rhsT).getResult(0);
  const lhsT = ctx.builder.transpose(lhs, swapLastTwo(lhs.type.rank)).getResult(0);
  const gradRhs = ctx.builder.matmul(lhsT, grad).getResult(0);
  return [gradLhs, gradRhs];
});

type ConvSpec = {
  strides: readonly number[];
  padding: readonly (readonly number[])[];
  dilation: readonly number[];
};

function _sliceChannels(b: IRBuilder, value: TensorValue, dim: number, from: number, to: number): TensorValue {
  const shape = value.type.shape;
  const starts = new Array(shape.length).fill(0);
  const limits = shape.map(d => d as number);
  starts[dim] = from;
  limits[dim] = to;
  return b.slice(value, starts, limits).getResult(0) as TensorValue;
}

function _sliceSpatial(b: IRBuilder, value: TensorValue, sizes: readonly number[]): TensorValue {
  const shape = value.type.shape;
  if (sizes.every((s, i) => s === shape[i + 2])) return value;
  const starts = new Array(shape.length).fill(0);
  const limits = shape.map(d => d as number);
  for (let i = 0; i < sizes.length; i++) limits[i + 2] = sizes[i];
  return b.slice(value, starts, limits).getResult(0) as TensorValue;
}

function _convGroupVJP(ctx: VJPContext, grad: TensorValue, input: TensorValue, kernel: TensorValue, spec: ConvSpec): [TensorValue, TensorValue] {
  const b = ctx.builder;
  const { strides, padding, dilation } = spec;
  const nSpatial = strides.length;
  const kSpatial = kernel.type.shape.slice(2).map(d => d as number);
  const inSpatial = input.type.shape.slice(2).map(d => d as number);
  const outSpatial = grad.type.shape.slice(2).map(d => d as number);

  const effective = kSpatial.map((k, i) => (k - 1) * dilation[i] + 1);
  const trailing = inSpatial.map((n, i) =>
    n + padding[i][0] + padding[i][1] - ((outSpatial[i] - 1) * strides[i] + effective[i]));

  let gradDilated = grad;
  if (strides.some(s => s > 1)) {
    const rank = grad.type.rank;
    const zero = b.scalarConstant(0, grad.type.dtype).getResult(0);
    const interior = new Array(rank).fill(0);
    for (let i = 0; i < nSpatial; i++) interior[i + 2] = strides[i] - 1;
    gradDilated = b.pad(grad, zero, new Array(rank).fill(0), new Array(rank).fill(0), interior).getResult(0) as TensorValue;
  }

  const flipDims = Array.from({ length: nSpatial }, (_, i) => i + 2);
  const swapOI = [1, 0, ...flipDims];
  const wFlip = b.reverse(b.transpose(kernel, swapOI).getResult(0), flipDims).getResult(0);
  const inputPad = kSpatial.map((_, i) => [
    effective[i] - 1 - padding[i][0],
    effective[i] - 1 - padding[i][1] + trailing[i],
  ]);
  const dInput = b.conv(gradDilated, wFlip, new Array(nSpatial).fill(1), inputPad as unknown as readonly number[], { dilation }).getResult(0) as TensorValue;

  const xSwap = b.transpose(input, swapOI).getResult(0);
  const gSwap = b.transpose(grad, swapOI).getResult(0);
  const dWconv = b.conv(xSwap, gSwap, dilation, padding as unknown as readonly number[], { dilation: strides }).getResult(0) as TensorValue;
  const dWeight = b.transpose(_sliceSpatial(b, dWconv, kSpatial), swapOI).getResult(0) as TensorValue;

  return [dInput, dWeight];
}

registerVJPRule('conv', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  const [input, kernel] = ctx.operands;
  const b = ctx.builder;
  const strides = ctx.op.getAttr<readonly number[]>('strides')!;
  const padding = ctx.op.getAttr<readonly (readonly number[])[]>('padding')!;
  const dilation = ctx.op.getAttr<readonly number[]>('dilation')! || strides.map(() => 1);
  const groups = ctx.op.getAttr<number>('groups')! || 1;
  const inLayout = ctx.op.getAttr<string>('input_layout')!;
  const kLayout = ctx.op.getAttr<string>('kernel_layout')!;

  if (inLayout !== 'NCHW' || kLayout !== 'OIHW') {
    throw new Error(`conv VJP supports only NCHW/OIHW layouts, got ${inLayout}/${kLayout}`);
  }

  const spec: ConvSpec = { strides, padding, dilation };
  if (groups === 1) {
    const [dInput, dWeight] = _convGroupVJP(ctx, grad, input, kernel, spec);
    return [dInput, dWeight];
  }

  const inPerGroup = (input.type.shape[1] as number) / groups;
  const outPerGroup = (grad.type.shape[1] as number) / groups;
  const dInputs: TensorValue[] = [];
  const dWeights: TensorValue[] = [];
  for (let g = 0; g < groups; g++) {
    const [dIn, dW] = _convGroupVJP(
      ctx,
      _sliceChannels(b, grad, 1, g * outPerGroup, (g + 1) * outPerGroup),
      _sliceChannels(b, input, 1, g * inPerGroup, (g + 1) * inPerGroup),
      _sliceChannels(b, kernel, 0, g * outPerGroup, (g + 1) * outPerGroup),
      spec,
    );
    dInputs.push(dIn);
    dWeights.push(dW);
  }

  return [b.concat(dInputs, 1).getResult(0), b.concat(dWeights, 0).getResult(0)];
});
