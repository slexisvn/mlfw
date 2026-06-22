import { registerVJPRule } from '../vjp_registry.js';

registerVJPRule('dot', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const [lhs, rhs] = ctx.operands;
  const lhsContracting = ctx.op.getAttr('lhs_contracting');
  const rhsContracting = ctx.op.getAttr('rhs_contracting');
  const lhsBatch = ctx.op.getAttr('lhs_batch') || [];
  const rhsBatch = ctx.op.getAttr('rhs_batch') || [];

  const lhsRank = lhs.type.rank;
  const rhsRank = rhs.type.rank;
  const gradRank = grad.type.rank;

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
  const grad = ctx.gradOutputs[0];
  const [lhs, rhs] = ctx.operands;
  const swapLastTwo = (rank) => {
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

registerVJPRule('conv', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const [input, kernel] = ctx.operands;
  const b = ctx.builder;
  const strides = ctx.op.getAttr('strides');
  const padding = ctx.op.getAttr('padding');
  const dilation = ctx.op.getAttr('dilation') || strides.map(() => 1);
  const groups = ctx.op.getAttr('groups') || 1;
  const inLayout = ctx.op.getAttr('input_layout');
  const kLayout = ctx.op.getAttr('kernel_layout');

  const supported = strides.length === 2 && strides.every(s => s === 1) &&
    dilation.every(d => d === 1) && groups === 1 && inLayout === 'NCHW' && kLayout === 'OIHW';
  if (!supported) {
    throw new Error('conv VJP supports only 2D stride-1 dilation-1 groups-1 NCHW/OIHW conv');
  }

  const kShape = kernel.type.shape;
  const KH = kShape[2], KW = kShape[3];
  const [padH, padW] = padding;

  const wFlip = b.reverse(b.transpose(kernel, [1, 0, 2, 3]).getResult(0), [2, 3]).getResult(0);
  const fullPad = [
    [KH - 1 - padH[0], KH - 1 - padH[1]],
    [KW - 1 - padW[0], KW - 1 - padW[1]]
  ];
  const dInput = b.conv(grad, wFlip, [1, 1], fullPad).getResult(0);

  const xSwap = b.transpose(input, [1, 0, 2, 3]).getResult(0);
  const gSwap = b.transpose(grad, [1, 0, 2, 3]).getResult(0);
  const dWconv = b.conv(xSwap, gSwap, [1, 1], [padH, padW]).getResult(0);
  const dWeight = b.transpose(dWconv, [1, 0, 2, 3]).getResult(0);

  return [dInput, dWeight];
});
