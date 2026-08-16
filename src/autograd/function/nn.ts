import { AutogradNode } from '../node.js';
import * as ops from '../../tensor/ops/ops.js';
import { zeros } from '../../tensor/factory/creation_ops.js';
import type { Tensor } from '../../tensor/core/tensor.js';
import type { GradInputList, GradOutputList } from '../types.js';

type Pair = readonly [number, number];

function _pair(value: unknown, fallback: number): Pair {
  if (Array.isArray(value)) {
    const arr = value as readonly number[];
    return [arr[0], arr.length > 1 ? arr[1] : arr[0]];
  }
  const v = typeof value === 'number' ? value : fallback;
  return [v, v];
}

function _padPairs(value: unknown): [Pair, Pair] {
  if (Array.isArray(value) && Array.isArray(value[0])) {
    const arr = value as readonly (readonly number[])[];
    return [[arr[0][0], arr[0][1]], [arr[1][0], arr[1][1]]];
  }
  const [h, w] = _pair(value, 0);
  return [[h, h], [w, w]];
}

function _sliceDim(t: Tensor, dim: number, start: number, length: number): Tensor {
  return ops.narrow(t, dim, start, length);
}

function _dilateSpatial(t: Tensor, strides: Pair): Tensor {
  if (strides[0] === 1 && strides[1] === 1) return t;
  const [n, c, h, w] = t.shape;
  const expanded = ops.reshape(t, [n, c, h, 1, w, 1]);
  const padded = ops.pad(expanded, [0, 0, 0, 0, 0, 0], [0, 0, 0, strides[0] - 1, 0, strides[1] - 1], 0);
  const flat = ops.reshape(padded, [n, c, h * strides[0], w * strides[1]]);
  return _sliceDim(_sliceDim(flat, 2, 0, (h - 1) * strides[0] + 1), 3, 0, (w - 1) * strides[1] + 1);
}

export class Conv2dBackward extends AutogradNode {
  constructor() { super(2); }

  apply(gradOutputs: GradOutputList): GradInputList {
    const grad = gradOutputs[0];
    const [input, weight] = this.savedTensors().map((t) => t.detach());
    const args = this.opArgs() || [];
    const strides = _pair(args[2], 1);
    const padding = _padPairs(args[3]);
    const dilation = _pair(args[4], 1);
    const groups = (args[5] as number) || 1;

    if (groups === 1) return this._groupGrad(grad, input, weight, strides, padding, dilation);

    const inPerGroup = input.shape[1] / groups;
    const outPerGroup = grad.shape[1] / groups;
    const dInputs: Tensor[] = [];
    const dWeights: Tensor[] = [];
    for (let g = 0; g < groups; g++) {
      const [dIn, dW] = this._groupGrad(
        _sliceDim(grad, 1, g * outPerGroup, outPerGroup),
        _sliceDim(input, 1, g * inPerGroup, inPerGroup),
        _sliceDim(weight, 0, g * outPerGroup, outPerGroup),
        strides, padding, dilation,
      );
      dInputs.push(dIn as Tensor);
      dWeights.push(dW as Tensor);
    }
    return [ops.cat(dInputs, 1), ops.cat(dWeights, 0)];
  }

  _groupGrad(grad: Tensor, input: Tensor, weight: Tensor, strides: Pair, padding: readonly [Pair, Pair], dilation: Pair): GradInputList {
    const kernel = [weight.shape[2], weight.shape[3]];
    const inSpatial = [input.shape[2], input.shape[3]];
    const outSpatial = [grad.shape[2], grad.shape[3]];
    const effective = kernel.map((k, i) => (k - 1) * dilation[i] + 1);
    const trailing = inSpatial.map((n, i) =>
      n + padding[i][0] + padding[i][1] - ((outSpatial[i] - 1) * strides[i] + effective[i]));

    const gradDilated = _dilateSpatial(grad, strides);

    const wFlip = ops.flip(ops.transpose(weight, 0, 1), [2, 3]);
    const inputPad: [Pair, Pair] = [
      [effective[0] - 1 - padding[0][0], effective[0] - 1 - padding[0][1] + trailing[0]],
      [effective[1] - 1 - padding[1][0], effective[1] - 1 - padding[1][1] + trailing[1]],
    ];
    const dInput = ops.conv2d(gradDilated, wFlip, [1, 1], inputPad, dilation, 1);

    const xSwap = ops.transpose(input, 0, 1);
    const gSwap = ops.transpose(grad, 0, 1);
    const dWfull = ops.conv2d(xSwap, gSwap, dilation, padding, strides, 1);
    const dWcropped = _sliceDim(_sliceDim(dWfull, 2, 0, kernel[0]), 3, 0, kernel[1]);
    return [dInput, ops.transpose(dWcropped, 0, 1)];
  }
}

export class Pool2dBackward extends AutogradNode {
  constructor() { super(1); }

  apply(gradOutputs: GradOutputList): GradInputList {
    const grad = gradOutputs[0];
    const [input] = this.savedTensors().map((t) => t.detach());
    const args = this.opArgs() || [];
    const poolType = (args[1] as string) || 'max';
    const kernel = _pair(args[2], 2);
    const strides = _pair(args[3], kernel[0]);
    const padding = _padPairs(args[4]);

    const noPad = padding[0][0] === 0 && padding[0][1] === 0 && padding[1][0] === 0 && padding[1][1] === 0;
    if (!noPad || strides[0] !== kernel[0] || strides[1] !== kernel[1]) {
      throw new Error('pool2d backward supports only non-overlapping (stride=kernel) pooling without padding');
    }

    const [n, c, oh, ow] = grad.shape;
    const upsample = (t: Tensor): Tensor => ops.reshape(
      ops.broadcast_in_dim(ops.reshape(t, [n, c, oh, 1, ow, 1]), [n, c, oh, kernel[0], ow, kernel[1]], [0, 1, 2, 3, 4, 5]),
      input.shape,
    );

    const upGrad = upsample(grad);
    if (poolType === 'avg') return [ops.div(upGrad, kernel[0] * kernel[1])];

    const output = this.savedTensors().length > 1 ? this.savedTensors()[1].detach() : null;
    const upOut = upsample(output || ops.pool2d(input, poolType, kernel, strides, padding));
    return [ops.where(ops.eq(input, upOut), upGrad, zeros(input.shape, { dtype: input.dtype, device: input.device }))];
  }
}

export class LayerNormBackward extends AutogradNode {
  constructor() { super(3); }

  apply(gradOutputs: GradOutputList): GradInputList {
    const grad = gradOutputs[0];
    const [input, weight] = this.savedTensors().map((t) => t.detach());
    const args = this.opArgs() || [];
    const rank = input.shape.length;
    const axisArg = (args[3] as number) ?? -1;
    const axis = axisArg < 0 ? rank + axisArg : axisArg;
    const eps = (args[4] as number) ?? 1e-5;
    const count = input.shape[axis];

    const mean = ops.mean(input, axis, true);
    const centered = ops.sub(input, mean);
    const variance = ops.mean(ops.mul(centered, centered), axis, true);
    const invStd = ops.rsqrt(ops.add(variance, eps));
    const normalized = ops.mul(centered, invStd);

    const gradWeight = _reduceToShape(ops.mul(grad, normalized), weight.shape);
    const gradBias = _reduceToShape(grad, weight.shape);

    const gN = ops.mul(grad, weight);
    const sumG = ops.sum(gN, axis, true);
    const sumGx = ops.sum(ops.mul(gN, normalized), axis, true);
    const inner = ops.sub(ops.sub(ops.mul(gN, count), sumG), ops.mul(normalized, sumGx));
    const gradInput = ops.div(ops.mul(invStd, inner), count);

    return [gradInput, gradWeight, gradBias];
  }
}

export class BatchNormBackward extends AutogradNode {
  constructor() { super(5); }

  apply(gradOutputs: GradOutputList): GradInputList {
    const grad = gradOutputs[0];
    const saved = this.savedTensors().map((t) => t.detach());
    const [input, weight, , mean, variance] = saved;
    const args = this.opArgs() || [];
    const rank = input.shape.length;
    const axisArg = (args[5] as number) ?? 1;
    const axis = axisArg < 0 ? rank + axisArg : axisArg;
    const eps = (args[6] as number) ?? 1e-5;

    const statShape = input.shape.map((d, i) => (i === axis ? d : 1));
    const meanB = ops.reshape(mean, statShape);
    const varB = ops.reshape(variance, statShape);
    const weightB = ops.reshape(weight, statShape);

    const invStd = ops.rsqrt(ops.add(varB, eps));
    const centered = ops.sub(input, meanB);
    const normalized = ops.mul(centered, invStd);

    const reduceDims: number[] = [];
    for (let i = 0; i < rank; i++) if (i !== axis) reduceDims.push(i);

    const gradWeight = ops.sum(ops.mul(grad, normalized), reduceDims, false);
    const gradBias = ops.sum(grad, reduceDims, false);
    const gradInput = ops.mul(ops.mul(grad, weightB), invStd);

    return [gradInput, gradWeight, gradBias, null, null];
  }
}

export class EmbeddingBackward extends AutogradNode {
  constructor() { super(2); }

  apply(gradOutputs: GradOutputList): GradInputList {
    const grad = gradOutputs[0];
    const [table, indices] = this.savedTensors().map((t) => t.detach());
    const flatIndices = ops.reshape(indices, [indices.numel]);
    const flatGrad = ops.reshape(grad, [indices.numel, table.shape[1]]);
    const base = zeros(table.shape, { dtype: grad.dtype, device: grad.device });
    const expanded = ops.broadcast_in_dim(ops.reshape(flatIndices, [indices.numel, 1]), [indices.numel, table.shape[1]], [0, 1]);
    return [ops.scatter_add(base, 0, expanded, flatGrad), null];
  }
}

function _reduceToShape(grad: Tensor, shape: readonly number[]): Tensor {
  const gradShape = grad.shape;
  if (gradShape.length === shape.length && gradShape.every((d, i) => d === shape[i])) return grad;
  const extra = gradShape.length - shape.length;
  const dims: number[] = [];
  for (let i = 0; i < extra; i++) dims.push(i);
  for (let i = 0; i < shape.length; i++) {
    if (shape[i] === 1 && gradShape[extra + i] !== 1) dims.push(extra + i);
  }
  const reduced = dims.length > 0 ? ops.sum(grad, dims, false) : grad;
  const reducedShape = reduced.shape;
  if (reducedShape.length === shape.length && reducedShape.every((d, i) => d === shape[i])) return reduced;
  return ops.reshape(reduced, shape);
}
