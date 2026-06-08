import { registerVJPRule } from '../vjp_registry.js';

registerVJPRule('reduce', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const [input] = ctx.operands;
  const reduceType = ctx.op.getAttr('reduce_type');
  const dimensions = ctx.op.getAttr('dimensions');
  const inputShape = input.type.shape;
  const dtype = input.type.dtype;

  if (reduceType === 'sum') {
    const broadcastDims = [];
    for (let i = 0; i < inputShape.length; i++) {
      if (!dimensions.includes(i)) broadcastDims.push(i);
    }
    const gradBroadcast = ctx.builder.broadcast(grad, inputShape, broadcastDims).getResult(0);
    return [gradBroadcast, null];
  }

  if (reduceType === 'mean') {
    const broadcastDims = [];
    let reduceSize = 1;
    for (let i = 0; i < inputShape.length; i++) {
      if (dimensions.includes(i)) {
        reduceSize *= inputShape[i];
      } else {
        broadcastDims.push(i);
      }
    }
    const gradBroadcast = ctx.builder.broadcast(grad, inputShape, broadcastDims).getResult(0);
    const divisor = ctx.builder.scalarConstant(reduceSize, dtype).getResult(0);
    const divisorBr = ctx.builder.broadcast(divisor, inputShape, []).getResult(0);
    const gradScaled = ctx.builder.div(gradBroadcast, divisorBr).getResult(0);
    return [gradScaled, null];
  }

  return [null, null];
});
