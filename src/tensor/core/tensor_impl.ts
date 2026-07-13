import { DispatchKeySet, backendKeyForDevice, autogradKeyForBackend } from '../../dispatcher/dispatch_key.js';
import type { DispatchKeyValue } from '../../dispatcher/dispatch_key.js';
import { computeStrides, computeNumel, isContiguous as checkContiguous } from '../utils/shape_utils.js';
import { MemoryFormat } from '../types/layout.js';
import type { MemoryFormatValue } from '../types/layout.js';
import { isChannelsLast } from '../types/layout.js';
import type { Storage } from './storage.js';
import type { AutogradMeta } from './autograd_meta.js';
import type { DType } from '../types/dtype.js';
import type { Device } from '../types/device.js';

export class TensorImpl {
  private _storage: Storage;
  private _storageOffset: number;
  private _sizes: readonly number[];
  private _strides: readonly number[];
  private _dtype: DType;
  private _device: Device;
  private _numel: number;
  private _keySet: DispatchKeySet;
  private _autogradMeta: AutogradMeta | null;
  private _version: number;
  private _contiguousCache: boolean | null;

  constructor(storage: Storage, storageOffset: number, sizes: readonly number[], strides: readonly number[] | null, dtype: DType, device: Device) {
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

  get storage(): Storage {
    return this._storage;
  }

  get storageOffset(): number {
    return this._storageOffset;
  }

  size(dim: number): number {
    const d = dim < 0 ? this._sizes.length + dim : dim;
    return this._sizes[d];
  }

  stride(dim: number): number {
    const d = dim < 0 ? this._strides.length + dim : dim;
    return this._strides[d];
  }

  sizes(): readonly number[] {
    return this._sizes;
  }

  strides(): readonly number[] {
    return this._strides;
  }

  dim(): number {
    return this._sizes.length;
  }

  numel(): number {
    return this._numel;
  }

  get dtype(): DType {
    return this._dtype;
  }

  get device(): Device {
    return this._device;
  }

  isContiguous(format?: MemoryFormatValue): boolean {
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

  setSizesAndStrides(sizes: readonly number[], strides?: readonly number[]) {
    this._sizes = Object.freeze([...sizes]);
    this._strides = strides ? Object.freeze([...strides]) : Object.freeze(computeStrides(sizes));
    this._numel = computeNumel(sizes);
    this._contiguousCache = null;
  }

  setStorageOffset(offset: number) {
    this._storageOffset = offset;
  }

  bumpVersion() {
    this._version++;
  }

  get version(): number {
    return this._version;
  }

  get autogradMeta(): AutogradMeta | null {
    return this._autogradMeta;
  }

  setAutogradMeta(meta: AutogradMeta | null) {
    this._autogradMeta = meta;
    this._updateKeySet();
  }

  keySet(): DispatchKeySet {
    return this._keySet;
  }

  addKeyToSet(key: DispatchKeyValue) {
    this._keySet = this._keySet.add(key);
  }

  removeKeyFromSet(key: DispatchKeyValue) {
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

  get isMeta(): boolean {
    return this._storage && this._storage.isMeta;
  }

  shallowCopyFrom(other: TensorImpl) {
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
