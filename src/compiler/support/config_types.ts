import type { TargetFeatures } from './target.js';
import type { PassInstrument } from './pass_instrument.js';
import type { VerifyLevelValue } from './invariant_check.js';
import type { IRSnapshotFlags, TraceSink } from './trace.js';

export type PassPhase = 'pre' | 'post';

export type CompileTarget = TargetFeatures;
export type FusionAwareTarget = TargetFeatures & { maxFusionSize?: number };

export type PartitionTarget = {
  name: string;
  kind?: string;
  computeTFLOPs: number;
  hasLibraryClass(cls: string | null): boolean;
  isGPU(): boolean;
  isCPU(): boolean;
  isWasm(): boolean;
};

export type OptimizationConfig = {
  splitSerializedKernels?: boolean;
  tensorize?: boolean;
  loopPartition?: boolean;
  detectAccumulators?: boolean;
  fastMath?: boolean;
  layout?: boolean;
  rematerialization?: boolean;
  maxSimplifyIterations?: number;
  rematConfig?: Record<string, unknown>;
  [key: string]: unknown;
};

export type MemoryConfig = {
  scheduleForPeak?: boolean;
  alignment?: number;
  inplaceReuse?: boolean;
  allocStrategy?: string;
  poolAllocation?: boolean;
  planReuse?: boolean;
  planDonation?: boolean;
  [key: string]: unknown;
};

export type FusionConfig = {
  enabled?: boolean;
  strategy?: string;
  launchOverheadUs?: number;
  maxFusionSize?: number;
  [key: string]: unknown;
};

export type QuantizationConfig = {
  enabled?: boolean;
  [key: string]: unknown;
};

export type SchedulingConfig = {
  enabled?: boolean;
  autotune?: boolean;
  gpuTiling?: boolean;
  [key: string]: unknown;
};

export type PartitionConfig = {
  enabled: boolean;
  targets: readonly PartitionTarget[];
  defaultTarget: PartitionTarget | null;
  opTargetOverrides: Map<string, PartitionTarget>;
  memoryLimits: Map<string, number>;
  minPartitionSize: number;
  costWeights: Record<string, number>;
  [key: string]: unknown;
};

export type TraceConfig = {
  level: number;
  sink: TraceSink | null;
  irSnapshot: IRSnapshotFlags;
};

export type CompilerConfig = {
  target: CompileTarget;
  verify: VerifyLevelValue;
  errorMode: string;
  fusion: FusionConfig;
  scheduling: SchedulingConfig;
  matmulBackend: string;
  quantization: QuantizationConfig;
  optimization: OptimizationConfig;
  memory: MemoryConfig;
  partition: PartitionConfig;
  passContext: unknown;
  loweringRules: ReadonlyMap<string, unknown> | Record<string, unknown> | null;
  codegenEntries: ReadonlyMap<string, unknown> | Record<string, unknown> | null;
  instruments: readonly PassInstrument[];
  trace: TraceConfig;
};
