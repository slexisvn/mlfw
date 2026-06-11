import { registerVJPRule } from '../vjp_registry.js';

registerVJPRule('reshape', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const [input] = ctx.operands;
  const inputShape = input.type.shape;
  return [ctx.builder.reshape(grad, inputShape).getResult(0)];
});

registerVJPRule('transpose', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const permutation = ctx.op.getAttr('permutation');
  const inversePerm = new Array(permutation.length);
  for (let i = 0; i < permutation.length; i++) {
    inversePerm[permutation[i]] = i;
  }
  return [ctx.builder.transpose(grad, inversePerm).getResult(0)];
});

registerVJPRule('broadcast_in_dim', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const [input] = ctx.operands;
  const broadcastDimensions = ctx.op.getAttr('broadcast_dimensions');
  const inputShape = input.type.shape;
  const resultShape = ctx.results[0].type.shape;
  const dtype = input.type.dtype;

  const reduceDims = [];
  for (let i = 0; i < resultShape.length; i++) {
    if (!broadcastDimensions.includes(i)) {
      reduceDims.push(i);
    } else {
      const inputDimIdx = broadcastDimensions.indexOf(i);
      if (inputShape[inputDimIdx] === 1 && resultShape[i] !== 1) {
        reduceDims.push(i);
      }
    }
  }

  if (reduceDims.length === 0) {
    return [grad];
  }

  const initConst = ctx.builder.scalarConstant(0, dtype).getResult(0);
  const reduced = ctx.builder.reduce(grad, initConst, reduceDims, 'sum').getResult(0);

  const reducedShape = reduced.type.shape;
  if (reducedShape.length !== inputShape.length ||
      !reducedShape.every((d, i) => d === inputShape[i])) {
    return [ctx.builder.reshape(reduced, inputShape).getResult(0)];
  }

  return [reduced];
});

registerVJPRule('slice', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const [input] = ctx.operands;
  const starts = ctx.op.getAttr('starts');
  const inputShape = input.type.shape;
  const gradShape = grad.type.shape;
  const strides = ctx.op.getAttr('strides') || inputShape.map(() => 1);
  const dtype = input.type.dtype;

  const low = [...starts];
  const high = new Array(inputShape.length);
  const interior = new Array(inputShape.length);
  for (let i = 0; i < inputShape.length; i++) {
    interior[i] = strides[i] - 1;
    high[i] = inputShape[i] - starts[i] - (gradShape[i] - 1) * strides[i] - 1;
  }

  const paddingValue = ctx.builder.scalarConstant(0, dtype).getResult(0);
  return [ctx.builder.pad(grad, paddingValue, low, high, interior).getResult(0)];
});

registerVJPRule('concat', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const dimension = ctx.op.getAttr('dimension');
  const operands = ctx.operands;

  const grads = [];
  let offset = 0;
  for (let i = 0; i < operands.length; i++) {
    const shape = operands[i].type.shape;
    const starts = new Array(shape.length).fill(0);
    const limits = [...grad.type.shape];
    starts[dimension] = offset;
    limits[dimension] = offset + shape[dimension];
    grads.push(ctx.builder.slice(grad, starts, limits).getResult(0));
    offset += shape[dimension];
  }

  return grads;
});

registerVJPRule('gather', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const [operand, indices] = ctx.operands;
  const zero = ctx.builder.scalarConstant(0, operand.type.dtype).getResult(0);
  const zeros = ctx.builder.broadcast(zero, operand.type.shape, []).getResult(0);
  const gradOperand = ctx.builder.scatterAdd(zeros, indices, grad, {
    updateWindowDims: ctx.op.getAttr('offset_dims'),
    insertedWindowDims: ctx.op.getAttr('collapsed_slice_dims'),
    scatterDimsToOperandDims: ctx.op.getAttr('start_index_map'),
    indexVectorDim: ctx.op.getAttr('index_vector_dim'),
  }).getResult(0);
  return [gradOperand, null];
});

registerVJPRule('scatter', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const [operand, indices] = ctx.operands;
  const insertedWindowDims = ctx.op.getAttr('inserted_window_dims');
  const sliceSizes = operand.type.shape.map((s, i) => insertedWindowDims.includes(i) ? 1 : s);
  const gradUpdates = ctx.builder.gather(grad, indices, {
    offsetDims: ctx.op.getAttr('update_window_dims'),
    collapsedSliceDims: insertedWindowDims,
    startIndexMap: ctx.op.getAttr('scatter_dims_to_operand_dims'),
    indexVectorDim: ctx.op.getAttr('index_vector_dim'),
    sliceSizes,
  }).getResult(0);
  return [grad, null, gradUpdates];
});

registerVJPRule('pad', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const low = ctx.op.getAttr('low');
  const high = ctx.op.getAttr('high');
  const [input] = ctx.operands;
  const inputShape = input.type.shape;

  const starts = [...low];
  const limits = new Array(inputShape.length);
  for (let i = 0; i < inputShape.length; i++) {
    limits[i] = low[i] + inputShape[i];
  }

  return [ctx.builder.slice(grad, starts, limits).getResult(0), null];
});
