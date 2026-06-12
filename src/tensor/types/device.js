import { DispatchKey, backendKeyForDevice } from '../../dispatcher/dispatch_key.js';

export const DeviceType = Object.freeze({
  CPU: 'cpu',
  GPU: 'gpu',
  WASM: 'wasm',
  META: 'meta',
  LAZY: 'lazy',
});

const _TARGET_KIND_MAP = Object.freeze({
  cpu: DeviceType.CPU,
  gpu: DeviceType.GPU,
  wasm: DeviceType.WASM,
});

export function deviceTypeFromTargetKind(kind) {
  return _TARGET_KIND_MAP[kind] || null;
}

export class Device {
  constructor(type, index = 0) {
    this.type = type;
    this.index = index;
  }

  dispatchKey() {
    return backendKeyForDevice(this.type);
  }

  equals(other) {
    return this.type === other.type && this.index === other.index;
  }

  hash() {
    let h = 0x811c9dc5;
    for (let i = 0; i < this.type.length; i++) {
      h = ((h ^ this.type.charCodeAt(i)) * 0x01000193) & 0x7fffffff;
    }
    h = ((h ^ this.index) * 0x01000193) & 0x7fffffff;
    return h;
  }

  toString() {
    if (this.index === 0) return this.type;
    return `${this.type}:${this.index}`;
  }
}

export const CPU_DEVICE = new Device(DeviceType.CPU);
export const GPU_DEVICE = new Device(DeviceType.GPU);
export const WASM_DEVICE = new Device(DeviceType.WASM);
export const META_DEVICE = new Device(DeviceType.META);
export const LAZY_DEVICE = new Device(DeviceType.LAZY);

let _defaultDevice = CPU_DEVICE;
export function getDefaultDevice() {
  return _defaultDevice;
}
export function setDefaultDevice(device) {
  _defaultDevice = device;
}
