export const TargetKind = Object.freeze({
  CPU: 'cpu',
  CUDA: 'cuda',
  WEBGPU: 'webgpu',
  WASM: 'wasm',
  ACCELERATOR: 'accelerator'
});

export class TargetFeatures {
  constructor(config) {
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
    this.supportsFloat16 = config.supportsFloat16 ?? false;
    this.supportsTensorCore = config.supportsTensorCore ?? false;
    this.libraryOps = config.libraryOps || new Set();
    this.enableEpilogueFusion = config.enableEpilogueFusion ?? false;
    this.preferredConvLayout = config.preferredConvLayout || null;
    this.preferredBlockFactor = config.preferredBlockFactor || 0;
    this.supportsBlockedLayout = config.supportsBlockedLayout ?? false;
    this.supportsInt8 = config.supportsInt8 ?? false;
    this.simd = config.simd ?? false;
  }

  isGPU() {
    return this.kind === TargetKind.CUDA || this.kind === TargetKind.WEBGPU;
  }

  isWebGPU() {
    return this.kind === TargetKind.WEBGPU;
  }

  isCPU() {
    return this.kind === TargetKind.CPU;
  }

  isWasm() {
    return this.kind === TargetKind.WASM;
  }

  supportsThreadBinding() {
    return this.isGPU() || this.isWebGPU();
  }

  supportsVectorization() {
    return this.vectorWidth > 1;
  }

  maxParallelism() {
    if (this.isGPU()) {
      return this.maxThreadsPerBlock * this.maxGridDimX;
    }
    return this.numCores;
  }

  supportsSimd() {
    return this.simd && this.vectorWidth > 1;
  }

  hasLibraryOp(opName) {
    return this.libraryOps.has(opName);
  }
}

export const CPUTarget = (overrides = {}) => new TargetFeatures({
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
  ...overrides
});

export const CUDATarget = (overrides = {}) => new TargetFeatures({
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
  supportsTensorCore: false,
  libraryOps: new Set(['dot', 'conv']),
  enableEpilogueFusion: true,
  supportsInt8: true,
  ...overrides
});

export const WasmTarget = (overrides = {}) => new TargetFeatures({
  kind: TargetKind.WASM,
  name: 'wasm_generic',
  vectorWidth: 4,
  numCores: 1,
  cacheLineSizeBytes: 64,
  memoryBandwidthGBs: 10,
  computeTFLOPs: 0.1,
  supportsInt8: true,
  simd: true,
  ...overrides
});

export const WebGPUTarget = (overrides = {}) => new TargetFeatures({
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
  ...overrides
});
