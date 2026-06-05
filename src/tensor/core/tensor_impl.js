import { DispatchKeySet, backendKeyForDevice, autogradKeyForBackend } from '../../dispatcher/dispatch_key.js';
import { computeStrides, computeNumel, isContiguous as checkContiguous } from '../utils/shape_utils.js';
import { MemoryFormat } from '../types/layout.js';
import { isChannelsLast } from '../types/layout.js';

export class TensorImpl {
  constructor(storage, storageOffset, sizes, strides, dtype, device) {
    this._storage = storage;
    this._storageOffset = storageOffset;
    this._sizes = Object.freeze([...sizes]);
    this._strides = strides ? Object.freeze([...strides]) : Object.freeze(computeStrides(sizes));
    this._dtype = dtype;
    this._device = device;
    this._numel = computeNumel(sizes);
    this._keySet = DispatchKeySet.fromKey(backendKeyForDevice(device.type));
    this._autogradMeta = null;
    this._version = 0;
    this._contiguousCache = null;
  }

  get storage() {
    return this._storage;
  }

  get storageOffset() {
    return this._storageOffset;
  }

  size(dim) {
    const d = dim < 0 ? this._sizes.length + dim : dim;
    return this._sizes[d];
  }

  stride(dim) {
    const d = dim < 0 ? this._strides.length + dim : dim;
    return this._strides[d];
  }

  sizes() {
    return this._sizes;
  }

  strides() {
    return this._strides;
  }

  dim() {
    return this._sizes.length;
  }

  numel() {
    return this._numel;
  }

  get dtype() {
    return this._dtype;
  }

  get device() {
    return this._device;
  }

  isContiguous(format) {
    if (!format || format === MemoryFormat.CONTIGUOUS) {
      if (this._contiguousCache === null) {
        this._contiguousCache = checkContiguous(this._sizes, this._strides);
      }
      return this._contiguousCache;
    }
    if (format === MemoryFormat.CHANNELS_LAST) {
      return isChannelsLast(this._sizes, this._strides);
    }
    return false;
  }

  setSizesAndStrides(sizes, strides) {
    this._sizes = Object.freeze([...sizes]);
    this._strides = strides ? Object.freeze([...strides]) : Object.freeze(computeStrides(sizes));
    this._numel = computeNumel(sizes);
    this._contiguousCache = null;
  }

  setStorageOffset(offset) {
    this._storageOffset = offset;
  }

  bumpVersion() {
    this._version++;
  }

  get version() {
    return this._version;
  }

  get autogradMeta() {
    return this._autogradMeta;
  }

  setAutogradMeta(meta) {
    this._autogradMeta = meta;
    this._updateKeySet();
  }

  keySet() {
    return this._keySet;
  }

  addKeyToSet(key) {
    this._keySet = this._keySet.add(key);
  }

  removeKeyFromSet(key) {
    this._keySet = this._keySet.remove(key);
  }

  _updateKeySet() {
    let ks = DispatchKeySet.fromKey(backendKeyForDevice(this._device.type));
    if (this._autogradMeta && this._autogradMeta.requiresGrad) {
      const backendKey = backendKeyForDevice(this._device.type);
      ks = ks.add(autogradKeyForBackend(backendKey));
    }
    this._keySet = ks;
  }

  get isMeta() {
    return this._storage && this._storage.isMeta;
  }

  shallowCopyFrom(other) {
    this._storage = other._storage;
    this._storageOffset = other._storageOffset;
    this._sizes = other._sizes;
    this._strides = other._strides;
    this._dtype = other._dtype;
    this._device = other._device;
    this._numel = other._numel;
    this._keySet = other._keySet;
    this._contiguousCache = null;
  }
}
