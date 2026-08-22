import type { TargetFeatures } from '../../backend/target.js';

export type PassPhase = 'pre' | 'post';

export type { PassManagerEntry as GraphPass } from '../passes/pass_manager.js';
export type { TirPassAny as TirPass } from '../passes/tir_pass_manager.js';

export type { TargetFeatures as CompileTarget } from '../../backend/target.js';
export type FusionAwareTarget = TargetFeatures & { maxFusionSize?: number };

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

export type { CompilerConfig } from './compiler.js';
