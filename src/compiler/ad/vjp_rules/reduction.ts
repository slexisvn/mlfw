import { registerVJPRule } from '../vjp_registry.js';

registerVJPRule('reduce', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  const [input] = ctx.operands;
  const reduceType = ctx.op.getAttr<string>('reduce_type')!;
  const dimensions = ctx.op.getAttr<readonly number[]>('dimensions')!;
  const inputShape = input.type.shape;
  const dtype = input.type.dtype;

  const dimSet = new Set(dimensions);
  const keepShape = inputShape.map((d, i) => (dimSet.has(i) ? 1 : d));
  const identityDims = inputShape.map((_, i) => i);
  const gradKept = ctx.builder.reshape(grad, keepShape).getResult(0);

  if (reduceType === 'sum') {
    const gradBroadcast = ctx.builder.broadcast(gradKept, inputShape, identityDims).getResult(0);
    return [gradBroadcast, null];
  }

  if (reduceType === 'mean') {
    let reduceSize = 1;
    for (let i = 0; i < inputShape.length; i++) {
      if (dimSet.has(i)) reduceSize *= inputShape[i] as number;
    }
    const gradBroadcast = ctx.builder.broadcast(gradKept, inputShape, identityDims).getResult(0);
    const divisor = ctx.builder.scalarConstant(reduceSize, dtype).getResult(0);
    const divisorBr = ctx.builder.broadcast(divisor, inputShape, []).getResult(0);
    const gradScaled = ctx.builder.div(gradBroadcast, divisorBr).getResult(0);
    return [gradScaled, null];
  }

  if (reduceType === 'max' || reduceType === 'min') {
    const resultKept = ctx.builder.reshape(ctx.results[0], keepShape).getResult(0);
    const reducedBroadcast = ctx.builder.broadcast(resultKept, inputShape, identityDims).getResult(0);
    const mask = ctx.builder.compare(input, reducedBroadcast, 'eq').getResult(0);
    const maskF = ctx.builder.convert(mask, dtype).getResult(0);
    const gradBroadcast = ctx.builder.broadcast(gradKept, inputShape, identityDims).getResult(0);
    const gradIn = ctx.builder.mul(gradBroadcast, maskF).getResult(0);
    return [gradIn, null];
  }

  if (reduceType === 'prod') {
    const b = ctx.builder;
    const zeroS = b.scalarConstant(0, dtype).getResult(0);
    const oneS = b.scalarConstant(1, dtype).getResult(0);
    const zeros = b.broadcast(zeroS, inputShape, []).getResult(0);
    const ones = b.broadcast(oneS, inputShape, []).getResult(0);

    const isZero = b.compare(input, zeros, 'eq').getResult(0);
    const withoutZeros = b.select(isZero, ones, input).getResult(0);
    const prodNonZero = b.reduce(withoutZeros, oneS, dimensions, 'prod').getResult(0);
    const prodNonZeroBr = b.broadcast(b.reshape(prodNonZero, keepShape).getResult(0), inputShape, identityDims).getResult(0);

    const zeroIndicator = b.select(isZero, ones, zeros).getResult(0);
    const zeroCount = b.reduce(zeroIndicator, zeroS, dimensions, 'sum').getResult(0);
    const zeroCountBr = b.broadcast(b.reshape(zeroCount, keepShape).getResult(0), inputShape, identityDims).getResult(0);

    const noZeros = b.compare(zeroCountBr, zeros, 'eq').getResult(0);
    const oneZero = b.compare(zeroCountBr, ones, 'eq').getResult(0);

    const quotient = b.div(prodNonZeroBr, withoutZeros).getResult(0);
    const nonZeroDeriv = b.select(noZeros, quotient, zeros).getResult(0);
    const zeroDeriv = b.select(oneZero, prodNonZeroBr, zeros).getResult(0);
    const deriv = b.select(isZero, zeroDeriv, nonZeroDeriv).getResult(0);

    const gradBroadcast = b.broadcast(gradKept, inputShape, identityDims).getResult(0);
    return [b.mul(gradBroadcast, deriv).getResult(0), null];
  }

  throw new Error(`reduce VJP: unsupported reduce_type '${reduceType}' on the gradient path (would silently drop the gradient)`);
});
