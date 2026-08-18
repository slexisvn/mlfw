import { TargetAttr } from '../compiler/pipeline/target_attrs.js';
import { launchBoundaryClass } from '../compiler/ir/graph/op_traits.js';

export const TargetKind = Object.freeze({
  CPU: 'cpu',
  CUDA: 'cuda',
  WEBGPU: 'webgpu',
  WASM: 'wasm',
  ACCELERATOR: 'accelerator'
});

export type TargetKindValue = (typeof TargetKind)[keyof typeof TargetKind];

export type TargetFeaturesConfig = {
  kind: string;
  name: string;
  vectorWidth?: number;
  numCores?: number;
  maxThreadsPerBlock?: number;
  maxBlockDimX?: number;
  maxBlockDimY?: number;
  maxBlockDimZ?: number;
  maxGridDimX?: number;
  maxGridDimY?: number;
  maxGridDimZ?: number;
  sharedMemoryBytes?: number;
  memoryBudgetBytes?: number;
  registersPerThread?: number;
  warpSize?: number;
  memoryBandwidthGBs?: number;
  computeTFLOPs?: number;
  cacheLineSizeBytes?: number;
  l1CacheBytes?: number;
  l2CacheBytes?: number;
  supportsFloat16?: boolean;
  supportsTensorCore?: boolean;
  arch?: string | null;
  libraryClasses?: ReadonlySet<string>;
  enableEpilogueFusion?: boolean;
  preferredConvLayout?: string | null;
  layoutAwareOps?: ReadonlySet<string> | Iterable<string>;
  preferredBlockFactor?: number;
  supportsBlockedLayout?: boolean;
  supportsInt8?: boolean;
  supportsConstBuffers?: boolean;
  simd?: boolean;
  host?: TargetFeatures | null;
  attrs?: Readonly<Record<string, unknown>>;
};

export type TargetOverrides = Partial<TargetFeaturesConfig>;

const TENSOR_CORE_MIN_MAJOR = 7;

export function archMajor(arch: string | null | undefined): number | null {
  if (!arch) return null;
  const m = /^sm_(\d)(\d+)$/.exec(arch);
  return m ? Number(m[1]) : null;
}

export function archSupportsTensorCore(arch: string | null | undefined): boolean {
  const major = archMajor(arch);
  return major !== null && major >= TENSOR_CORE_MIN_MAJOR;
}

export class TargetFeatures {
  kind: string;
  name: string;
  vectorWidth: number;
  numCores: number;
  maxThreadsPerBlock: number;
  maxBlockDimX: number;
  maxBlockDimY: number;
  maxBlockDimZ: number;
  maxGridDimX: number;
  maxGridDimY: number;
  maxGridDimZ: number;
  sharedMemoryBytes: number;
  memoryBudgetBytes: number;
  registersPerThread: number;
  warpSize: number;
  memoryBandwidthGBs: number;
  computeTFLOPs: number;
  cacheLineSizeBytes: number;
  l1CacheBytes: number;
  l2CacheBytes: number;
  supportsFloat16: boolean;
  supportsTensorCore: boolean;
  arch: string | null;
  libraryClasses: ReadonlySet<string>;
  enableEpilogueFusion: boolean;
  preferredConvLayout: string | null;
  layoutAwareOps: ReadonlySet<string>;
  preferredBlockFactor: number;
  supportsBlockedLayout: boolean;
  supportsInt8: boolean;
  supportsConstBuffers: boolean;
  simd: boolean;
  host: TargetFeatures | null;
  attrs: Map<string, unknown>;

  constructor(config: TargetFeaturesConfig) {
    this.kind = config.kind;
    this.name = config.name;
    this.vectorWidth = config.vectorWidth || 1;
    this.numCores = config.numCores || 1;
    this.maxThreadsPerBlock = config.maxThreadsPerBlock || 1;
    this.maxBlockDimX = config.maxBlockDimX || 1;
    this.maxBlockDimY = config.maxBlockDimY || 1;
    this.maxBlockDimZ = config.maxBlockDimZ || 1;
    this.maxGridDimX = config.maxGridDimX || 1;
    this.maxGridDimY = config.maxGridDimY || 1;
    this.maxGridDimZ = config.maxGridDimZ || 1;
    this.sharedMemoryBytes = config.sharedMemoryBytes || 0;
    this.memoryBudgetBytes = config.memoryBudgetBytes || 0;
    this.registersPerThread = config.registersPerThread || 0;
    this.warpSize = config.warpSize || 1;
    this.memoryBandwidthGBs = config.memoryBandwidthGBs || 0;
    this.computeTFLOPs = config.computeTFLOPs || 0;
    this.cacheLineSizeBytes = config.cacheLineSizeBytes || 64;
    this.l1CacheBytes = config.l1CacheBytes || 0;
    this.l2CacheBytes = config.l2CacheBytes || 0;
    this.arch = config.arch ?? null;
    this.supportsFloat16 = config.supportsFloat16 ?? false;
    this.supportsTensorCore = config.supportsTensorCore ?? archSupportsTensorCore(this.arch);
    this.libraryClasses = config.libraryClasses || new Set();
    this.enableEpilogueFusion = config.enableEpilogueFusion ?? false;
    this.preferredConvLayout = config.preferredConvLayout || null;
    this.layoutAwareOps = config.layoutAwareOps instanceof Set ? config.layoutAwareOps : new Set(config.layoutAwareOps || []);
    this.preferredBlockFactor = config.preferredBlockFactor || 0;
    this.supportsBlockedLayout = config.supportsBlockedLayout ?? false;
    this.supportsInt8 = config.supportsInt8 ?? false;
    this.supportsConstBuffers = config.supportsConstBuffers ?? false;
    this.simd = config.simd ?? false;
    this.host = config.host || null;
    this.attrs = new Map(Object.entries(config.attrs || {}));
  }

  getAttr<T = unknown>(key: string, fallback: T | null = null): T | null { return this.attrs.has(key) ? this.attrs.get(key) as T : fallback; }
  hasAttr(key: string): boolean { return this.attrs.has(key); }
  withAttr(key: string, value: unknown): this { this.attrs.set(key, value); return this; }

  isGPU(): boolean {
    return this.kind === TargetKind.CUDA || this.kind === TargetKind.WEBGPU;
  }

  isWebGPU(): boolean {
    return this.kind === TargetKind.WEBGPU;
  }

  isCPU(): boolean {
    return this.kind === TargetKind.CPU;
  }

  isWasm(): boolean {
    return this.kind === TargetKind.WASM;
  }

  supportsThreadBinding(): boolean {
    return this.isGPU() || this.isWebGPU();
  }

  supportsVectorization(): boolean {
    return this.vectorWidth > 1;
  }

  maxParallelism(): number {
    if (this.isGPU()) {
      return this.maxThreadsPerBlock * this.maxGridDimX;
    }
    return this.numCores;
  }

  supportsSimd(): boolean {
    return this.simd && this.vectorWidth > 1;
  }

  hasLibraryOp(opName: string): boolean {
    const cls = launchBoundaryClass(opName);
    return cls !== null && this.libraryClasses.has(cls);
  }
}

export const CPUTarget = (overrides: TargetOverrides = {}): TargetFeatures => new TargetFeatures({
  kind: TargetKind.CPU,
  name: 'cpu_generic',
  vectorWidth: 8,
  numCores: 8,
  cacheLineSizeBytes: 64,
  l1CacheBytes: 32 * 1024,
  l2CacheBytes: 256 * 1024,
  memoryBandwidthGBs: 50,
  computeTFLOPs: 0.5,
  supportsBlockedLayout: true,
  preferredBlockFactor: 8,
  supportsInt8: true,
  supportsConstBuffers: true,
  ...overrides
});

export const CUDATarget = (overrides: TargetOverrides = {}): TargetFeatures => new TargetFeatures({
  kind: TargetKind.CUDA,
  name: 'cuda_generic',
  vectorWidth: 1,
  numCores: 80,
  maxThreadsPerBlock: 1024,
  maxBlockDimX: 1024,
  maxBlockDimY: 1024,
  maxBlockDimZ: 64,
  maxGridDimX: 2147483647,
  maxGridDimY: 65535,
  maxGridDimZ: 65535,
  sharedMemoryBytes: 48 * 1024,
  registersPerThread: 255,
  warpSize: 32,
  memoryBandwidthGBs: 900,
  computeTFLOPs: 15,
  supportsFloat16: true,
  libraryClasses: new Set(['matmul', 'conv']),
  enableEpilogueFusion: true,
  supportsInt8: true,
  supportsConstBuffers: true,
  attrs: {
    [TargetAttr.GRAPH_SPLIT]: { matmul: 2, conv: 2, attention: 1 },
    [TargetAttr.SCHEDULING]: { gpuTiling: true },
  },
  ...overrides
});

export const WasmTarget = (overrides: TargetOverrides = {}): TargetFeatures => new TargetFeatures({
  kind: TargetKind.WASM,
  name: 'wasm_generic',
  vectorWidth: 4,
  numCores: 1,
  cacheLineSizeBytes: 64,
  memoryBandwidthGBs: 10,
  computeTFLOPs: 0.1,
  supportsInt8: true,
  supportsConstBuffers: true,
  simd: true,
  ...overrides
});

export const WebGPUTarget = (overrides: TargetOverrides = {}): TargetFeatures => new TargetFeatures({
  kind: TargetKind.WEBGPU,
  name: 'webgpu_generic',
  vectorWidth: 1,
  numCores: 32,
  maxThreadsPerBlock: 256,
  maxBlockDimX: 256,
  maxBlockDimY: 256,
  maxBlockDimZ: 64,
  maxGridDimX: 65535,
  maxGridDimY: 65535,
  maxGridDimZ: 65535,
  sharedMemoryBytes: 16384,
  warpSize: 32,
  memoryBandwidthGBs: 400,
  computeTFLOPs: 8,
  supportsFloat16: true,
  attrs: { [TargetAttr.SCHEDULING]: { enabled: true } },
  ...overrides
});
