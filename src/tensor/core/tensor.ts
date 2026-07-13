import { TensorImpl } from './tensor_impl.js';
import { AutogradMeta } from './autograd_meta.js';
import type { DType, NumericTypedArray } from '../types/dtype.js';
import type { Device } from '../types/device.js';
import type { DispatchKeySet } from '../../dispatcher/dispatch_key.js';

import { readFromStorage } from '../utils/half.js';

const _EXPAND_DTYPES = new Set(['f16', 'bf16', 'i64']);

export class Tensor {
  readonly _impl: TensorImpl;

  constructor(impl: TensorImpl) {
    this._impl = impl;
  }

  get impl(): TensorImpl {
    return this._impl;
  }

  get shape(): readonly number[] {
    return this._impl.sizes();
  }

  get strides(): readonly number[] {
    return this._impl.strides();
  }

  get dtype(): DType {
    return this._impl.dtype;
  }

  get device(): Device {
    return this._impl.device;
  }

  get ndim(): number {
    return this._impl.dim();
  }

  get rank(): number {
    return this._impl.dim();
  }

  get numel(): number {
    return this._impl.numel();
  }

  get length(): number {
    return this._impl.numel();
  }

  get isContiguous(): boolean {
    return this._impl.isContiguous();
  }

  get dispatchKeySet(): DispatchKeySet {
    return this._impl.keySet();
  }

  get storage() {
    return this._impl.storage;
  }

  get storageOffset(): number {
    return this._impl.storageOffset;
  }

  get data(): NumericTypedArray | null {
    const s = this._impl.storage;
    if (!s || s.isMeta) return null;
    const raw = s.data;
    const offset = this._impl.storageOffset;
    if (raw && offset === 0 && this._impl.isContiguous() && raw.length === this.numel) {
      return raw;
    }
    return raw;
  }

  get requiresGrad(): boolean {
    const meta = this._impl.autogradMeta;
    return meta ? meta.requiresGrad : false;
  }

  get gradFn(): unknown | null {
    const meta = this._impl.autogradMeta;
    return meta ? meta.gradFn : null;
  }

  get grad(): Tensor | null {
    const meta = this._impl.autogradMeta;
    return meta ? meta.grad : null;
  }

  set grad(t: Tensor | null) {
    this._ensureAutogradMeta().grad = t;
  }

  get isLeaf(): boolean {
    const meta = this._impl.autogradMeta;
    if (!meta) return true;
    return meta.isLeaf;
  }

  get version(): number {
    return this._impl.version;
  }

  requiresGrad_(flag = true): this {
    const meta = this._ensureAutogradMeta();
    meta.requiresGrad = flag;
    this._impl._updateKeySet();
    return this;
  }

  retainGrad(): this {
    this._ensureAutogradMeta().retainGrad = true;
    return this;
  }

  detach(): Tensor {
    const newImpl = new TensorImpl(
      this._impl.storage,
      this._impl.storageOffset,
      this._impl.sizes(),
      this._impl.strides(),
      this._impl.dtype,
      this._impl.device
    );
    return new Tensor(newImpl);
  }

  item(): number | bigint {
    if (this.numel !== 1) {
      throw new Error(`item() requires tensor with exactly 1 element, got ${this.numel}`);
    }
    const raw = this._impl.storage.data!;
    const v = raw[this._impl.storageOffset];
    return _EXPAND_DTYPES.has(this._impl.dtype) ? readFromStorage(this._impl.dtype, v) : v;
  }

  toArray(): number | bigint | NestedArray {
    const sizes = this.shape;
    const strides = this.strides;
    const data = this._impl.storage.data!;
    const offset = this._impl.storageOffset;
    const dtype = _EXPAND_DTYPES.has(this._impl.dtype) ? this._impl.dtype : null;

    if (sizes.length === 0) return dtype ? readFromStorage(dtype, data[offset]) : data[offset];
    return _toNestedArray(data, sizes, strides, offset, 0, dtype);
  }

  toString(): string {
    const shapeStr = this.shape.join(', ');
    return `Tensor(shape=[${shapeStr}], dtype=${this.dtype}, device=${this.device})`;
  }

  *[Symbol.iterator](): Generator<Tensor> {
    const dim0 = this.shape[0];
    if (dim0 === undefined) {
      throw new Error('Cannot iterate over a 0-d tensor');
    }
    for (let i = 0; i < dim0; i++) {
      yield this._select(0, i);
    }
  }

  _select(dim: number, index: number): Tensor {
    const sizes = this._impl.sizes();
    const strides = this._impl.strides();
    const newSizes: number[] = [];
    const newStrides: number[] = [];
    for (let d = 0; d < sizes.length; d++) {
      if (d === dim) continue;
      newSizes.push(sizes[d]);
      newStrides.push(strides[d]);
    }
    const newOffset = this._impl.storageOffset + index * strides[dim];
    const newImpl = new TensorImpl(
      this._impl.storage,
      newOffset,
      newSizes,
      newStrides,
      this._impl.dtype,
      this._impl.device
    );
    return new Tensor(newImpl);
  }

  _ensureAutogradMeta(): AutogradMeta {
    if (!this._impl.autogradMeta) {
      const meta = new AutogradMeta();
      meta.versionAtCreation = this._impl.version;
      this._impl.setAutogradMeta(meta);
    }
    return this._impl.autogradMeta!;
  }

  backward(gradOutput?: Tensor) {
    const { backward } = require_autograd_engine();
    backward(this, gradOutput);
  }
}

type NestedArray = Array<number | bigint | NestedArray>;

function _toNestedArray(data: NumericTypedArray, sizes: readonly number[], strides: readonly number[], offset: number, dim: number, dtype: string | null): NestedArray {
  const size = sizes[dim];
  if (dim === sizes.length - 1) {
    const arr = new Array<number | bigint>(size);
    if (dtype) {
      for (let i = 0; i < size; i++) arr[i] = readFromStorage(dtype, data[offset + i * strides[dim]]);
    } else {
      for (let i = 0; i < size; i++) arr[i] = data[offset + i * strides[dim]];
    }
    return arr;
  }
  const arr = new Array<number | bigint | NestedArray>(size);
  for (let i = 0; i < size; i++) {
    arr[i] = _toNestedArray(data, sizes, strides, offset + i * strides[dim], dim + 1, dtype);
  }
  return arr;
}

let _autogradEngine: { backward: (tensor: Tensor, gradOutput?: Tensor) => void } | null = null;

function require_autograd_engine() {
  if (!_autogradEngine) {
    throw new Error('Autograd engine not initialized. Import autograd/engine.js first.');
  }
  return _autogradEngine;
}

export function setAutogradEngine(engine: { backward: (tensor: Tensor, gradOutput?: Tensor) => void }) {
  _autogradEngine = engine;
}
