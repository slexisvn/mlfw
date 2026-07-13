import { Tensor } from '../tensor/core/tensor.js';
import { tensor, fromBuffer } from '../tensor/factory/from_ops.js';
import { typedArrayCtor } from '../tensor/types/dtype.js';
import type { MutableNumericArray, NumericSettable } from '../tensor/types/options.js';

type PlainRecord = Record<string, unknown>;

function stackTensors(tensors: readonly Tensor[], dim = 0): Tensor {
  const n = tensors.length;
  const elemShape = tensors[0].shape;
  const dtype = tensors[0].dtype;
  const outShape = [...elemShape];
  outShape.splice(dim, 0, n);

  const elemNumel = elemShape.reduce((a, b) => a * b, 1);
  const Ctor = typedArrayCtor(dtype);
  const outData = new Ctor(n * elemNumel);
  const writable = outData as MutableNumericArray & NumericSettable;

  for (let i = 0; i < n; i++) {
    const src = tensors[i];
    const srcData = src.data!;
    const srcOffset = src._impl ? src._impl.storageOffset : 0;
    if (src.isContiguous) {
      writable.set(srcData.subarray(srcOffset, srcOffset + elemNumel), i * elemNumel);
    } else {
      const shape = src.shape;
      const strides = src.strides;
      const storage = src._impl.storage.data!;
      const ndim = shape.length;
      const indices = new Int32Array(ndim);
      let si = srcOffset;
      for (let j = 0; j < elemNumel; j++) {
        writable[i * elemNumel + j] = storage[si];
        for (let d = ndim - 1; d >= 0; d--) {
          indices[d]++;
          if (indices[d] < shape[d]) { si += strides[d]; break; }
          si -= (shape[d] - 1) * strides[d];
          indices[d] = 0;
        }
      }
    }
  }

  return fromBuffer(outData, outShape, dtype);
}

export function defaultCollate(batch: readonly unknown[]): unknown {
  const elem = batch[0];

  if (elem instanceof Tensor) {
    return stackTensors(batch as readonly Tensor[], 0);
  }

  if (typeof elem === 'number') {
    return tensor(batch as readonly number[]);
  }

  if (Array.isArray(elem)) {
    const result = new Array(elem.length);
    for (let i = 0; i < elem.length; i++) {
      const column = new Array(batch.length);
      for (let j = 0; j < batch.length; j++) column[j] = (batch[j] as readonly unknown[])[i];
      result[i] = defaultCollate(column);
    }
    return result;
  }

  if (elem !== null && typeof elem === 'object' && elem.constructor === Object) {
    const keys = Object.keys(elem);
    const result: PlainRecord = {};
    for (let k = 0; k < keys.length; k++) {
      const key = keys[k];
      const column = new Array(batch.length);
      for (let j = 0; j < batch.length; j++) column[j] = (batch[j] as PlainRecord)[key];
      result[key] = defaultCollate(column);
    }
    return result;
  }

  throw new Error(`defaultCollate: unsupported element type ${typeof elem}`);
}
