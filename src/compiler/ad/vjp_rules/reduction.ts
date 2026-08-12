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
    throw new Error("reduce VJP for reduce_type='prod' is not implemented; provide a gradient rule or avoid differentiating reduce_prod");
  }

  throw new Error(`reduce VJP: unsupported reduce_type '${reduceType}' on the gradient path (would silently drop the gradient)`);
});
