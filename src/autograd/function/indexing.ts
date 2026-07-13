import { AutogradNode } from '../node.js';
import * as ops from '../../tensor/ops/ops.js';
import { zeros } from '../../tensor/factory/creation_ops.js';
import { narrow, contiguous, select } from '../../tensor/ops/ops.js';
import { normalizeAxis as _normDim } from '../../tensor/utils/shape_utils.js';
import type { Tensor } from '../../tensor/core/tensor.js';
import type { NumericTypedArray } from '../../tensor/types/dtype.js';
import type { GradInputList, GradOutputList } from '../types.js';

type GpuIndexSelectBackward = (grad: Tensor, index: Tensor, inputShape: readonly number[], dim: number) => Tensor | null;

function _addAt(data: NumericTypedArray, index: number, value: number | bigint): void {
  if (typeof value === 'bigint') {
    (data as BigInt64Array)[index] += value;
  } else {
    (data as Exclude<NumericTypedArray, BigInt64Array>)[index] += value;
  }
}

export class CatBackward extends AutogradNode {
  constructor() { super(0); }

  apply(gradOutputs: GradOutputList): GradInputList {
    const g = gradOutputs[0];
    const args = this.opArgs();
    const rank = g.shape.length;
    const dim = _normDim(args && args.length > 1 ? (args[1] ?? 0) as number : 0, rank);

    const grads = [];
    let offset = 0;
    let i = 0;
    while (this.inputMetadata(i)) {
      const meta = this.inputMetadata(i);
      const length = meta!.shape[dim];
      grads.push(contiguous(narrow(g, dim, offset, length)));
      offset += length;
      i++;
    }
    return grads;
  }
}

export class StackBackward extends AutogradNode {
  constructor() { super(0); }

  apply(gradOutputs: GradOutputList): GradInputList {
    const g = gradOutputs[0];
    const args = this.opArgs();
    const rank = g.shape.length;
    const dim = _normDim(args && args.length > 1 ? (args[1] ?? 0) as number : 0, rank);

    const grads = [];
    let i = 0;
    while (this.inputMetadata(i)) {
      grads.push(contiguous(select(g, dim, i)));
      i++;
    }
    return grads;
  }
}

export class ClampBackward extends AutogradNode {
  constructor() { super(3); }

  apply(gradOutputs: GradOutputList): GradInputList {
    const g = gradOutputs[0];
    const [self, lo, hi] = this.savedTensors();
    const z = zeros(g.shape, { dtype: g.dtype, device: g.device });
    const geLo = ops.ge(self.detach(), lo.detach());
    const gradAboveLo = ops.where(geLo, g, z);
    const leHi = ops.le(self.detach(), hi.detach());
    return [ops.where(leHi, gradAboveLo, z), null, null];
  }
}

export class PadBackward extends AutogradNode {
  constructor() { super(2); }

  apply(gradOutputs: GradOutputList): GradInputList {
    const g = gradOutputs[0];
    const args = this.opArgs();
    const low = args![2] as readonly number[];
    const meta = this.inputMetadata(0);
    const inputShape = meta!.shape;

    let out = g;
    for (let d = 0; d < inputShape.length; d++) {
      const start = low[d] || 0;
      out = narrow(out, d, start, inputShape[d]);
    }
    return [contiguous(out), null];
  }
}

export class IndexSelectBackward extends AutogradNode {
  static #gpuBackward: GpuIndexSelectBackward | null = null;
  static setGpuBackward(fn: GpuIndexSelectBackward | null) { IndexSelectBackward.#gpuBackward = fn; }
  constructor() { super(2); }

  apply(gradOutputs: GradOutputList): GradInputList {
    const g = gradOutputs[0];
    const [, index] = this.savedTensors();
    const meta = this.inputMetadata(0);
    const inputShape = meta!.shape;
    const rank = inputShape.length;
    const args = this.opArgs();
    const dim = _normDim(args && args.length > 2 ? (args[2] ?? 0) as number : 0, rank);

    if (IndexSelectBackward.#gpuBackward) {
      const r = IndexSelectBackward.#gpuBackward(g, index, inputShape, dim);
      if (r) return [r];
    }

    const result = zeros(inputShape, { dtype: g.dtype, device: g.device });
    const outData = result._impl.storage.data!;
    const resultStrides = result.strides;

    const gc = contiguous(g);
    const gData = gc._impl.storage.data!;
    const gOff = gc._impl.storageOffset;
    const gShape = gc.shape;
    const gStrides = gc.strides;

    const idxC = contiguous(index);
    const idxData = idxC._impl.storage.data!;
    const idxOff = idxC._impl.storageOffset;

    const ndim = gShape.length;
    const indices = new Int32Array(ndim);
    let gi = gOff;

    for (let i = 0; i < gc.numel; i++) {
      let oi = 0;
      for (let d = 0; d < ndim; d++) {
        const idx = d === dim ? Number(idxData[idxOff + indices[d]]) : indices[d];
        oi += idx * resultStrides[d];
      }
      _addAt(outData, oi, gData[gi]);

      for (let d = ndim - 1; d >= 0; d--) {
        indices[d]++;
        if (indices[d] < gShape[d]) { gi += gStrides[d]; break; }
        gi -= (gShape[d] - 1) * gStrides[d];
        indices[d] = 0;
      }
    }

    return [result, null];
  }
}

export class WhereBackward extends AutogradNode {
  constructor() { super(3); }

  apply(gradOutputs: GradOutputList): GradInputList {
    const g = gradOutputs[0];
    const [cond] = this.savedTensors();
    const z = zeros(g.shape, { dtype: g.dtype, device: g.device });
    const gradA = ops.where(cond.detach(), g, z);
    const gradB = ops.where(cond.detach(), z, g);
    return [null, gradA, gradB];
  }
}
