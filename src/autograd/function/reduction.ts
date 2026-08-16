import { AutogradNode } from '../node.js';
import * as ops from '../../tensor/ops/ops.js';
import { full, zeros } from '../../tensor/factory/creation_ops.js';
import { unsqueeze } from '../../tensor/ops/ops.js';
import type { Tensor } from '../../tensor/core/tensor.js';
import type { GradInputList, GradOutputList } from '../types.js';

function _normalizeDims(dim: unknown, rank: number): number[] {
  if (dim === undefined || dim === null) {
    const all = [];
    for (let i = 0; i < rank; i++) all.push(i);
    return all;
  }
  const list = Array.isArray(dim) ? dim : [dim];
  return list.map((d) => (d as number) < 0 ? (d as number) + rank : d as number).sort((a, b) => a - b);
}

function _unreduce(grad: Tensor, inputShape: readonly number[], dims: readonly number[], keepdim: unknown): Tensor {
  let g = grad;
  if (!keepdim) {
    for (const d of dims) {
      g = unsqueeze(g, d);
    }
  }
  const z = zeros(inputShape, { dtype: g.dtype, device: g.device });
  return ops.add(z, g);
}

export class SumBackward extends AutogradNode {
  constructor() { super(1); }

  apply(gradOutputs: GradOutputList): GradInputList {
    const g = gradOutputs[0];
    const meta = this.inputMetadata(0);
    const inputShape = meta!.shape;
    const args = this.opArgs();
    const dim = args ? args[1] : undefined;
    const keepdim = args ? args[2] : false;
    const dims = _normalizeDims(dim, inputShape.length);
    return [_unreduce(g, inputShape, dims, keepdim)];
  }
}

export class MeanBackward extends AutogradNode {
  constructor() { super(1); }

  apply(gradOutputs: GradOutputList): GradInputList {
    const g = gradOutputs[0];
    const meta = this.inputMetadata(0);
    const inputShape = meta!.shape;
    const args = this.opArgs();
    const dim = args ? args[1] : undefined;
    const keepdim = args ? args[2] : false;
    const dims = _normalizeDims(dim, inputShape.length);

    let count = 1;
    for (const d of dims) count *= inputShape[d];

    const expanded = _unreduce(g, inputShape, dims, keepdim);
    const n = full(inputShape, count, { dtype: g.dtype, device: g.device });
    return [ops.div(expanded, n)];
  }
}

function _selectMaskGrad(node: AutogradNode, gradOutputs: GradOutputList, input: Tensor, result: Tensor): GradInputList {
  const meta = node.inputMetadata(0);
  const inputShape = meta!.shape;
  const args = node.opArgs();
  const dim = args ? args[1] : undefined;
  const keepdim = args ? args[2] : false;
  const dims = _normalizeDims(dim, inputShape.length);

  const expandedResult = _unreduce(result, inputShape, dims, keepdim);
  const expandedGrad = _unreduce(gradOutputs[0], inputShape, dims, keepdim);
  const mask = ops.eq(input, expandedResult);
  const z = zeros(inputShape, { dtype: expandedGrad.dtype, device: expandedGrad.device });
  return [ops.where(mask, expandedGrad, z)];
}

export class MaxBackward extends AutogradNode {
  constructor() { super(1); }
  apply(gradOutputs: GradOutputList): GradInputList {
    const [input] = this.savedTensors();
    const x = input.detach();
    const args = this.opArgs();
    const result = ops.max(x, args ? args[1] as number : undefined, args ? args[2] as boolean : false);
    return _selectMaskGrad(this, gradOutputs, x, result);
  }
}

export class MinBackward extends AutogradNode {
  constructor() { super(1); }
  apply(gradOutputs: GradOutputList): GradInputList {
    const [input] = this.savedTensors();
    const x = input.detach();
    const args = this.opArgs();
    const result = ops.min(x, args ? args[1] as number : undefined, args ? args[2] as boolean : false);
    return _selectMaskGrad(this, gradOutputs, x, result);
  }
}

export class ProdBackward extends AutogradNode {
  constructor() { super(1); }
  apply(gradOutputs: GradOutputList): GradInputList {
    const [input] = this.savedTensors();
    const x = input.detach();
    const meta = this.inputMetadata(0);
    const inputShape = meta!.shape;
    const args = this.opArgs();
    const dim = args ? args[1] : undefined;
    const keepdim = args ? args[2] : false;
    const dims = _normalizeDims(dim, inputShape.length);

    const zerosT = zeros(inputShape, { dtype: x.dtype, device: x.device });
    const onesT = full(inputShape, 1, { dtype: x.dtype, device: x.device });
    const isZero = ops.eq(x, zerosT);
    const withoutZeros = ops.where(isZero, onesT, x);

    const prodNonZero = _unreduce(ops.prod(withoutZeros, dim as number, keepdim as boolean), inputShape, dims, keepdim);
    const zeroCount = _unreduce(ops.sum(ops.where(isZero, onesT, zerosT), dim as number, keepdim as boolean), inputShape, dims, keepdim);

    const noZeros = ops.eq(zeroCount, zerosT);
    const oneZero = ops.eq(zeroCount, onesT);
    const nonZeroDeriv = ops.where(noZeros, ops.div(prodNonZero, withoutZeros), zerosT);
    const zeroDeriv = ops.where(oneZero, prodNonZero, zerosT);
    const deriv = ops.where(isZero, zeroDeriv, nonZeroDeriv);

    return [ops.mul(_unreduce(gradOutputs[0], inputShape, dims, keepdim), deriv)];
  }
}
