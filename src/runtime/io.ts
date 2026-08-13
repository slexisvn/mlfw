import type { WasmParallelInfo } from '../backend/wasm/codegen.js';
import type { ExternalKernelInfo } from '../compiler/pipeline/external_codegen.js';
import type { DType } from '../tensor/types/dtype.js';
import type { NumericSettable } from '../tensor/types/options.js';

export interface MemFs {
  readFile(path: string): string;
  readBinary(path: string): Uint8Array;
  writeFile(path: string, data: string): void;
  writeBinary(path: string, data: Uint8Array): void;
  appendFile(path: string, data: string): void;
  exists(path: string): boolean;
  mkdir(path: string): void;
  rename(from: string, to: string): void;
  readdir(path: string): string[];
  remove(path: string): void;
}

export interface WasmInstance {
  exports: WebAssembly.Exports;
  memory: WebAssembly.Memory;
  bufferOffsets: Map<string, number>;
  funcName: string;
  binary: Uint8Array;
  parallel: WasmParallelInfo | null;
  mathNames: string[];
}

export type WasmKernelFn = (...args: number[]) => void;

export type WasmBufferView = NumericSettable & ArrayLike<number | bigint>;

export type WasmBufferViewCtor = new (
  buffer: ArrayBufferLike,
  byteOffset: number,
  length: number,
) => WasmBufferView;

declare const CUDA_HANDLE: unique symbol;

export type CudaHandle = { readonly [CUDA_HANDLE]: true };

export type DevicePtr = bigint;

export type EagerGraph = { graph: CudaHandle | null; exec: CudaHandle | null };

export type CudaKernelMetadata = {
  cublas?: ExternalKernelInfo;
  gridDim?: number[];
  blockDim?: number[];
  sharedMemBytes?: number;
  outputIndices?: number[];
  scratch?: { size: number; dtype: DType }[];
  _outputSet?: Set<number>;
};

export type CudaKernel = {
  name: string;
  source: string;
  metadata: CudaKernelMetadata;
};
