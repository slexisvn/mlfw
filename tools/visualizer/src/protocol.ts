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
  children: NestNode[];
};

export type Snapshot = {
  text: string;
  ops: number;
  dags: Dag[];
  nests: NestNode[];
};

export type TraceEventLite = Record<string, unknown> & { type: string };

export type StepKind = 'input' | 'pass' | 'lowering';

export type CompileStep = {
  index: number;
  kind: StepKind;
  level: IRLevelName;
  phase: string;
  pass: string;
  outcome: PassOutcome;
  durationMs: number;
  before: Snapshot;
  after: Snapshot;
  events: TraceEventLite[];
};

export type SourceLink = [opId: number, line: number];

export type Kernel = {
  name: string;
  source: string;
  language: string;
};

export type TensorPreview = {
  shape: number[];
  dtype: string;
  numel: number;
  preview: number[];
};

export type RunResult = {
  ran: boolean;
  skipped: string | null;
  error: string | null;
  inputs: TensorPreview[];
  outputs: TensorPreview[];
  eagerOutputs: TensorPreview[];
  maxAbsDiff: number | null;
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

export type CompileOptions = {
  target: TargetName;
  fusionStrategy: 'priority' | 'dominator' | 'greedy';
  fusion: boolean;
  scheduling: boolean;
  layout: boolean;
  disabledPasses: string[];
};

export type InitRequest = { kind: 'init'; id: number };

export type WorkerRequest = CompileRequest | InitRequest;

export type WorkerRequestDraft = Omit<CompileRequest, 'id'> | Omit<InitRequest, 'id'>;

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
  sourceLinks: SourceLink[];
  totalMs: number;
  run: RunResult;
};

export type WorkerResponse = InitResponse | CompileResponse;

export const DEFAULT_OPTIONS: CompileOptions = {
  target: 'cpu',
  fusionStrategy: 'priority',
  fusion: true,
  scheduling: true,
  layout: false,
  disabledPasses: [],
};
