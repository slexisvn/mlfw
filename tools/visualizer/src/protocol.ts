import type { IRLevelValue } from 'mlfw/compiler/ir/verify.js';
import type { TargetName } from './catalog/targets.js';

export type { TargetName };

export type IRLevelName = IRLevelValue;

export type PassOutcome = 'changed' | 'unchanged' | 'failed' | 'unreported';

export type DagNode = {
  id: number;
  opName: string;
  operands: number[];
  results: number[];
  resultTypes: string[];
  attrs: [string, string][];
  regions: DagNode[][];
  regionArgs: number[][];
  loc: string | null;
  lines: number[];
};

export type DagValue = {
  id: number;
  name: string;
  type: string;
  producer: number | null;
};

export type Dag = {
  func: string;
  args: DagValue[];
  values: DagValue[];
  nodes: DagNode[];
  returns: number[];
};

export type NestKind =
  | 'func' | 'for' | 'block' | 'seq' | 'store' | 'alloc'
  | 'let' | 'if' | 'while' | 'accumulator' | 'bindings' | 'stmt'
  | 'source' | 'line';

export type NestNode = {
  id: string;
  kind: NestKind;
  label: string;
  detail: string;
  op: string | null;
  opId: number | null;
  line: number | null;
  children: NestNode[];
};

export type Snapshot = {
  text: string;
  ops: number;
  bytes: number;
  flops: number;
  dags: Dag[];
  nests: NestNode[];
};

export type TraceEventLite = Record<string, unknown> & { type: string };

export type BufferLifetime = {
  name: string;
  scope: string;
  bytes: number;
  slot: number;
  firstUse: number;
  lastUse: number;
  sharesWith: string | null;
};

export type TuningParams = Record<string, number | number[]>;

export type TuningCandidate = { sketch: string; score: number; params: TuningParams };

export type TuningRound = {
  func: string;
  blockName: string;
  round: number;
  measured: boolean;
  scores: TuningCandidate[];
  bestSketch: string | null;
  bestParams: TuningParams | null;
  bestScore: number | null;
  bestMedianMs: number | null;
};

export type MemoryPlan = {
  func: string;
  peakMemory: number;
  totalBytesIfNeverShared: number;
  steps: number;
  buffers: BufferLifetime[];
};

export type StepKind = 'input' | 'pass' | 'lowering' | 'primitive';

export type VerifyLevelName = 'off' | 'boundaries' | 'each-pass';

export type VerifyReport = { introduced: string[]; carried: string[] };

export type SkippedPass = { pass: string; level: IRLevelName };

export type CompileStep = {
  index: number;
  kind: StepKind;
  parent: string | null;
  unit: string | null;
  level: IRLevelName;
  phase: string;
  pass: string;
  outcome: PassOutcome;
  durationMs: number;
  before: Snapshot;
  after: Snapshot;
  events: TraceEventLite[];
  verify: VerifyReport | null;
  interpretable: boolean;
};


export type LaunchDiagnosis = { reason: string; buffers: string[] };

export type Kernel = {
  name: string;
  source: string;
  language: string;
  metadata: Record<string, unknown> | null;
  diagnosis: LaunchDiagnosis | null;
};

export type TensorStats = {
  min: number | null;
  max: number | null;
  mean: number | null;
  std: number | null;
  norm: number;
  zeros: number;
  nan: number;
  inf: number;
};

export type TensorPreview = {
  name: string;
  shape: number[];
  dtype: string;
  numel: number;
  preview: number[];
  stats: TensorStats;
};

export type LayerActivation = {
  name: string;
  kind: string;
  line: number | null;
  outputs: TensorPreview[];
};

export type RunResult = {
  ran: boolean;
  skipped: string | null;
  error: string | null;
  inputs: TensorPreview[];
  outputs: TensorPreview[];
  eagerOutputs: TensorPreview[];
  gradients: TensorPreview[];
  eagerGradients: TensorPreview[];
  parameters: TensorPreview[];
  layers: LayerActivation[];
  maxAbsDiff: number | null;
  maxAbsGradDiff: number | null;
  compiledMs: number | null;
  eagerMs: number | null;
  iterations: number;
};

export type CompileRequest = {
  kind: 'compile';
  id: number;
  source: string;
  options: CompileOptions;
};

export type BackwardMode = 'off' | 'separate' | 'joint';

export type CompileOptions = {
  target: TargetName;
  backward: BackwardMode;
  verify: VerifyLevelName;
  fusionStrategy: 'priority' | 'dominator' | 'greedy';
  fusion: boolean;
  scheduling: boolean;
  autotune: boolean;
  layout: boolean;
  disabledPasses: string[];
};

export type InitRequest = { kind: 'init'; id: number };

export type BisectRequest = {
  kind: 'bisect';
  id: number;
  source: string;
  options: CompileOptions;
  tolerance: number;
};

export type WorkerRequest = CompileRequest | InitRequest | BisectRequest | SemanticsRequest;

export type WorkerRequestDraft =
  | Omit<CompileRequest, 'id'>
  | Omit<InitRequest, 'id'>
  | Omit<BisectRequest, 'id'>
  | Omit<SemanticsRequest, 'id'>;

export type InitResponse = {
  kind: 'init';
  id: number;
  globals: string[];
};

export type CompileResponse = {
  kind: 'compile';
  id: number;
  ok: boolean;
  error: string | null;
  errorPhase: string | null;
  steps: CompileStep[];
  kernels: Kernel[];
  events: TraceEventLite[];
  sourceLines: number[];
  memoryPlans: MemoryPlan[];
  tuningRounds: TuningRound[];
  skipped: SkippedPass[];
  kernelReports: KernelReport[];
  totalMs: number;
  run: RunResult;
};

export type KernelIssue = { kind: string; detail: string };

export type KernelReport = {
  name: string;
  language: string;
  bytes: number;
  lines: number;
  longestLine: number;
  loops: number;
  tempBuffers: number;
  boundsChecks: number;
  modulos: number;
  arithmeticNoise: string[];
  extent1Loops: number;
  zeroInits: number;
  issues: KernelIssue[];
  blownUp: boolean;
};

export type CellDiff = { cell: string; before: number; after: number };

export type SemanticReport = {
  ran: boolean;
  reason: string | null;
  truncated: boolean;
  storesBefore: number;
  storesAfter: number;
  compared: number;
  changed: CellDiff[];
  dropped: string[];
  added: string[];
  changedCount: number;
  droppedCount: number;
  addedCount: number;
  vanishedBuffers: string[];
  newBuffers: string[];
  storageReused: boolean;
  reordered: boolean;
  verdict: string;
};

export type SemanticsRequest = { kind: 'semantics'; id: number; step: number };

export type SemanticsResponse = {
  kind: 'semantics';
  id: number;
  step: number;
  report: SemanticReport | null;
  unavailable: string | null;
  ms: number;
};

export type BisectMode = 'compile' | 'numeric';

export type BisectProbe = {
  index: number;
  disabled: string[];
  ok: boolean;
  ran: boolean;
  error: string | null;
  diff: number | null;
  good: boolean;
  ms: number;
};

export type BisectProgress = {
  kind: 'bisect-progress';
  id: number;
  probe: BisectProbe;
  note: string;
};

export type BisectResponse = {
  kind: 'bisect';
  id: number;
  mode: BisectMode | null;
  tolerance: number;
  baseline: BisectProbe | null;
  allOff: BisectProbe | null;
  candidates: string[];
  culprits: string[];
  probes: BisectProbe[];
  conclusion: string;
  error: string | null;
  totalMs: number;
};

export type WorkerResponse = InitResponse | CompileResponse | BisectResponse | SemanticsResponse;

export type WorkerProgress = BisectProgress;

export type WorkerMessage = WorkerResponse | WorkerProgress;

export const DEFAULT_OPTIONS: CompileOptions = {
  target: 'cpu',
  backward: 'off',
  verify: 'each-pass',
  fusionStrategy: 'priority',
  fusion: true,
  scheduling: true,
  autotune: false,
  layout: false,
  disabledPasses: [],
};
