import { TensorImpl } from './tensor_impl.js';
import { AutogradMeta } from './autograd_meta.js';
import { flatIndex, computeNumel } from '../utils/shape_utils.js';

export class Tensor {
  constructor(impl) {
    this._impl = impl;
  }

  get impl() {
    return this._impl;
  }

  get shape() {
    return this._impl.sizes();
  }

  get strides() {
    return this._impl.strides();
  }

  get dtype() {
    return this._impl.dtype;
  }

  get device() {
    return this._impl.device;
  }

  get ndim() {
    return this._impl.dim();
  }

  get rank() {
    return this._impl.dim();
  }

  get numel() {
    return this._impl.numel();
  }

  get isContiguous() {
    return this._impl.isContiguous();
  }

  get dispatchKeySet() {
    return this._impl.keySet();
  }

  get storage() {
    return this._impl.storage;
  }

  get storageOffset() {
    return this._impl.storageOffset;
  }

  get data() {
    const s = this._impl.storage;
    if (!s || s.isMeta) return null;
    const raw = s.data;
    const offset = this._impl.storageOffset;
    if (offset === 0 && this._impl.isContiguous() && raw.length === this.numel) {
      return raw;
    }
    return raw;
  }

  get requiresGrad() {
    const meta = this._impl.autogradMeta;
    return meta ? meta.requiresGrad : false;
  }

  get gradFn() {
    const meta = this._impl.autogradMeta;
    return meta ? meta.gradFn : null;
  }

  get grad() {
    const meta = this._impl.autogradMeta;
    return meta ? meta.grad : null;
  }

  set grad(t) {
    this._ensureAutogradMeta().grad = t;
  }

  get isLeaf() {
    const meta = this._impl.autogradMeta;
    if (!meta) return true;
    return meta.isLeaf;
  }

  get version() {
    return this._impl.version;
  }

  requiresGrad_(flag = true) {
    const meta = this._ensureAutogradMeta();
    meta.requiresGrad = flag;
    this._impl._updateKeySet();
    return this;
  }

  retainGrad() {
    this._ensureAutogradMeta().retainGrad = true;
    return this;
  }

  detach() {
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

  item() {
    if (this.numel !== 1) {
      throw new Error(`item() requires tensor with exactly 1 element, got ${this.numel}`);
    }
    const raw = this._impl.storage.data;
    if (this._impl.dim() === 0) return raw[this._impl.storageOffset];
    return raw[this._impl.storageOffset];
  }

  toArray() {
    const sizes = this.shape;
    const strides = this.strides;
    const data = this._impl.storage.data;
    const offset = this._impl.storageOffset;

    if (sizes.length === 0) return data[offset];
    return _toNestedArray(data, sizes, strides, offset, 0);
  }

  toString() {
    const shapeStr = this.shape.join(', ');
    return `Tensor(shape=[${shapeStr}], dtype=${this.dtype}, device=${this.device})`;
  }

  *[Symbol.iterator]() {
    const dim0 = this.shape[0];
    if (dim0 === undefined) {
      throw new Error('Cannot iterate over a 0-d tensor');
    }
    for (let i = 0; i < dim0; i++) {
      yield this._select(0, i);
    }
  }

  _select(dim, index) {
    const sizes = this._impl.sizes();
    const strides = this._impl.strides();
    const newSizes = [];
    const newStrides = [];
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

  _ensureAutogradMeta() {
    if (!this._impl.autogradMeta) {
      const meta = new AutogradMeta();
      meta.versionAtCreation = this._impl.version;
      this._impl.setAutogradMeta(meta);
    }
    return this._impl.autogradMeta;
  }

  backward(gradOutput) {
    const { backward } = require_autograd_engine();
    backward(this, gradOutput);
  }
}

function _toNestedArray(data, sizes, strides, offset, dim) {
  const size = sizes[dim];
  if (dim === sizes.length - 1) {
    const arr = new Array(size);
    for (let i = 0; i < size; i++) arr[i] = data[offset + i * strides[dim]];
    return arr;
  }
  const arr = new Array(size);
  for (let i = 0; i < size; i++) {
    arr[i] = _toNestedArray(data, sizes, strides, offset + i * strides[dim], dim + 1);
  }
  return arr;
}

let _autogradEngine = null;

function require_autograd_engine() {
  if (!_autogradEngine) {
    throw new Error('Autograd engine not initialized. Import autograd/engine.js first.');
  }
  return _autogradEngine;
}

export function setAutogradEngine(engine) {
  _autogradEngine = engine;
}
