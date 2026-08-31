export type { PassManagerEntry as GraphPass } from '../passes/pass_manager.js';
export type { TirPassAny as TirPass } from '../passes/tir_pass_manager.js';

export type {
  PassPhase,
  CompileTarget,
  FusionAwareTarget,
  PartitionTarget,
  OptimizationConfig,
  MemoryConfig,
  FusionConfig,
  QuantizationConfig,
  SchedulingConfig,
  PartitionConfig,
  TraceConfig,
  CompilerConfig,
} from '../support/config_types.js';
