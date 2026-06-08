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
  const lhsRank = lhs.type.rank;
  const rhsRank = rhs.type.rank;

  let gradLhs, gradRhs;

  if (lhsRank === 2 && rhsRank === 2) {
    const rhsT = ctx.builder.transpose(rhs, [1, 0]).getResult(0);
    gradLhs = ctx.builder.matmul(grad, rhsT).getResult(0);
    const lhsT = ctx.builder.transpose(lhs, [1, 0]).getResult(0);
    gradRhs = ctx.builder.matmul(lhsT, grad).getResult(0);
  } else if (lhsRank === 3 && rhsRank === 3) {
    const rhsT = ctx.builder.transpose(rhs, [0, 2, 1]).getResult(0);
    gradLhs = ctx.builder.matmul(grad, rhsT).getResult(0);
    const lhsT = ctx.builder.transpose(lhs, [0, 2, 1]).getResult(0);
    gradRhs = ctx.builder.matmul(lhsT, grad).getResult(0);
  } else {
    const lhsPerm = Array.from({ length: lhsRank }, (_, i) => i);
    lhsPerm[lhsRank - 2] = lhsRank - 1;
    lhsPerm[lhsRank - 1] = lhsRank - 2;
    const rhsPerm = Array.from({ length: rhsRank }, (_, i) => i);
    rhsPerm[rhsRank - 2] = rhsRank - 1;
    rhsPerm[rhsRank - 1] = rhsRank - 2;
    const rhsT = ctx.builder.transpose(rhs, rhsPerm).getResult(0);
    gradLhs = ctx.builder.matmul(grad, rhsT).getResult(0);
    const lhsT = ctx.builder.transpose(lhs, lhsPerm).getResult(0);
    gradRhs = ctx.builder.matmul(lhsT, grad).getResult(0);
  }

  return [gradLhs, gradRhs];
});
