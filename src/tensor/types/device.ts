import { backendKeyForDevice } from '../../dispatcher/dispatch_key.js';

export enum DeviceType {
  CPU = 'cpu',
  GPU = 'gpu',
  WASM = 'wasm',
  WEBGPU = 'webgpu',
  META = 'meta',
  LAZY = 'lazy',
}

export type DeviceTypeName = `${DeviceType}`;

export class Device {
  readonly type: DeviceTypeName;
  readonly index: number;

  constructor(type: DeviceTypeName, index = 0) {
    this.type = type;
    this.index = index;
  }

  dispatchKey(): number {
    return backendKeyForDevice(this.type);
  }

  equals(other: Device): boolean {
    return this.type === other.type && this.index === other.index;
  }

  hash(): number {
    let h = 0x811c9dc5;
    for (let i = 0; i < this.type.length; i++) {
      h = ((h ^ this.type.charCodeAt(i)) * 0x01000193) & 0x7fffffff;
    }
    h = ((h ^ this.index) * 0x01000193) & 0x7fffffff;
    return h;
  }

  toString(): string {
    if (this.index === 0) return this.type;
    return `${this.type}:${this.index}`;
  }
}

export const CPU_DEVICE = new Device(DeviceType.CPU);
export const GPU_DEVICE = new Device(DeviceType.GPU);
export const WASM_DEVICE = new Device(DeviceType.WASM);
export const WEBGPU_DEVICE = new Device(DeviceType.WEBGPU);
export const META_DEVICE = new Device(DeviceType.META);

let _defaultDevice = CPU_DEVICE;
export function getDefaultDevice(): Device {
  return _defaultDevice;
}
export function setDefaultDevice(device: Device) {
  _defaultDevice = device;
}
