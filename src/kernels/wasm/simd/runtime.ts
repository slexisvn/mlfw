import { encodeWat } from '../../../backend/wasm/wat_encoder.js';

export const F64_BYTES = 8;
export const I32_BYTES = 4;
export const F64_LANES = 2;
export const VEC_BYTES = F64_LANES * F64_BYTES;
export const LANE_SHIFT = Math.log2(F64_LANES);

const PAGE_BYTES = 65536;
const ALIGN_BYTES = VEC_BYTES;
const RELEASE_ABOVE_BYTES = 64 * 1024 * 1024;

function alignUp(n: number, a: number): number {
  return Math.ceil(n / a) * a;
}

type SimdExportFn = (...args: number[]) => void;

class SimdModule {
  _module: WebAssembly.Module;
  _exportName: string;
  declare _instance: WebAssembly.Instance;
  declare _memory: WebAssembly.Memory;
  declare _fn: SimdExportFn;
  declare _top: number;

  constructor(module: WebAssembly.Module, exportName: string) {
    this._module = module;
    this._exportName = exportName;
    this._instantiate();
  }

  _instantiate(): void {
    this._instance = new WebAssembly.Instance(this._module, {});
    this._memory = this._instance.exports.memory as WebAssembly.Memory;
    this._fn = this._instance.exports[this._exportName] as SimdExportFn;
    this._top = 0;
  }

  reset(): void {
    this._top = 0;
  }

  reclaim(): void {
    if (this._memory.buffer.byteLength > RELEASE_ABOVE_BYTES) this._instantiate();
    else this._top = 0;
  }

  _ensure(bytes: number): void {
    const have = this._memory.buffer.byteLength;
    if (bytes <= have) return;
    this._memory.grow(Math.ceil((bytes - have) / PAGE_BYTES));
  }

  alloc(bytes: number): number {
    const ptr = alignUp(this._top, ALIGN_BYTES);
    this._top = ptr + alignUp(bytes, ALIGN_BYTES);
    this._ensure(this._top);
    return ptr;
  }

  allocF64(count: number): number {
    return this.alloc(count * F64_BYTES);
  }

  allocI32(count: number): number {
    return this.alloc(count * I32_BYTES);
  }

  f64(ptr: number, count: number): Float64Array {
    return new Float64Array(this._memory.buffer, ptr, count);
  }

  i32(ptr: number, count: number): Int32Array {
    return new Int32Array(this._memory.buffer, ptr, count);
  }

  writeF64(ptr: number, src: ArrayLike<number>): void {
    this.f64(ptr, src.length).set(src);
  }

  run(...args: number[]): void {
    this._fn(...args);
  }
}

const _cache = new Map<string, SimdModule>();

export type { SimdModule };

export function simdModule(key: string, buildWat: () => string, exportName: string): SimdModule {
  let mod = _cache.get(key);
  if (!mod) {
    mod = new SimdModule(new WebAssembly.Module(encodeWat(buildWat())), exportName);
    _cache.set(key, mod);
  }
  mod.reclaim();
  return mod;
}
