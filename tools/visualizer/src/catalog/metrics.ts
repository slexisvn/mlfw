import type { CompileOptions, CompileResponse, IRLevelName, Snapshot } from '../protocol.js';

export type MetricKind = 'count' | 'ms' | 'bytes' | 'diff' | 'ratio';

export type Metric = {
  id: string;
  label: string;
  meaning: string;
  kind: MetricKind;
  lowerIsBetter: boolean;
  of: (response: CompileResponse) => number | null;
};

const GRAPH_LEVELS: readonly IRLevelName[] = ['graph-module', 'graph-func'];

export function graphCostOf(response: CompileResponse): { bytes: number; flops: number } | null {
  const snapshot = lastGraphSnapshot(response);
  return snapshot && snapshot.bytes > 0 ? { bytes: snapshot.bytes, flops: snapshot.flops } : null;
}

function lastGraphSnapshot(response: CompileResponse): Snapshot | null {
  for (let i = response.steps.length - 1; i >= 0; i--) {
    if (GRAPH_LEVELS.includes(response.steps[i].level)) return response.steps[i].after;
  }
  return null;
}

function lastAtLevel(response: CompileResponse, levels: readonly IRLevelName[]): number | null {
  for (let i = response.steps.length - 1; i >= 0; i--) {
    if (levels.includes(response.steps[i].level)) return response.steps[i].after.ops;
  }
  return null;
}

function codegenLines(response: CompileResponse): number | null {
  for (let i = response.steps.length - 1; i >= 0; i--) {
    if (response.steps[i].phase === 'codegen') return response.steps[i].after.ops;
  }
  return null;
}

function changedPasses(response: CompileResponse): number {
  return response.steps.filter(step => step.kind === 'pass' && step.outcome === 'changed').length;
}

function peakMemory(response: CompileResponse): number | null {
  if (response.memoryPlans.length === 0) return null;
  return response.memoryPlans.reduce((sum, plan) => sum + plan.peakMemory, 0);
}

export const METRICS: readonly Metric[] = [
  {
    id: 'graph-ops',
    label: 'graph ops',
    meaning: 'nodes left in the graph IR when the graph passes were done',
    kind: 'count',
    lowerIsBetter: true,
    of: response => lastAtLevel(response, ['graph-module', 'graph-func']),
  },
  {
    id: 'tir-nodes',
    label: 'tensor IR nodes',
    meaning: 'size of the loop nests once scheduling and memory planning were done',
    kind: 'count',
    lowerIsBetter: true,
    of: response => lastAtLevel(response, ['tir']),
  },
  {
    id: 'kernel-lines',
    label: 'kernel lines',
    meaning: 'lines of source the backend emitted',
    kind: 'count',
    lowerIsBetter: true,
    of: codegenLines,
  },
  {
    id: 'kernels',
    label: 'kernels',
    meaning: 'how many separate kernels the program was cut into',
    kind: 'count',
    lowerIsBetter: true,
    of: response => response.kernels.length,
  },
  {
    id: 'peak-memory',
    label: 'peak memory',
    meaning: 'bytes of temporaries the memory plan reserves',
    kind: 'bytes',
    lowerIsBetter: true,
    of: peakMemory,
  },
  {
    id: 'bytes-moved',
    label: 'bytes moved',
    meaning: 'every tensor each surviving op reads and writes — what fusion is trying to shrink',
    kind: 'bytes',
    lowerIsBetter: true,
    of: response => lastGraphSnapshot(response)?.bytes ?? null,
  },
  {
    id: 'intensity',
    label: 'arithmetic intensity',
    meaning: 'flops per byte moved: below the machine ratio the program is waiting on memory, above it on arithmetic',
    kind: 'ratio',
    lowerIsBetter: false,
    of: response => {
      const snapshot = lastGraphSnapshot(response);
      if (!snapshot || snapshot.bytes === 0) return null;
      return snapshot.flops / snapshot.bytes;
    },
  },
  {
    id: 'changed-passes',
    label: 'passes that changed the IR',
    meaning: 'how many pass runs actually rewrote something',
    kind: 'count',
    lowerIsBetter: false,
    of: changedPasses,
  },
  {
    id: 'compile-ms',
    label: 'compile time',
    meaning: 'how long the whole pipeline took',
    kind: 'ms',
    lowerIsBetter: true,
    of: response => response.totalMs,
  },
  {
    id: 'run-ms',
    label: 'run time',
    meaning: 'one call to the compiled kernel',
    kind: 'ms',
    lowerIsBetter: true,
    of: response => response.run.compiledMs,
  },
  {
    id: 'max-diff',
    label: 'worst disagreement with eager',
    meaning: 'how far the compiled answer drifted from the same model run op by op',
    kind: 'diff',
    lowerIsBetter: true,
    of: response => (response.run.maxAbsGradDiff === null
      ? response.run.maxAbsDiff
      : Math.max(response.run.maxAbsDiff ?? 0, response.run.maxAbsGradDiff)),
  },
];

export type OptionDiff = { key: string; from: string; to: string };

function describeOption(value: unknown): string {
  if (Array.isArray(value)) return value.length === 0 ? 'none' : value.join(', ');
  if (typeof value === 'boolean') return value ? 'on' : 'off';
  return String(value);
}

export function optionDiffs(from: CompileOptions, to: CompileOptions): OptionDiff[] {
  const diffs: OptionDiff[] = [];
  for (const key of Object.keys(to) as (keyof CompileOptions)[]) {
    const before = describeOption(from[key]);
    const after = describeOption(to[key]);
    if (before !== after) diffs.push({ key, from: before, to: after });
  }
  return diffs;
}

export function formatMetric(kind: MetricKind, value: number | null): string {
  if (value === null) return '—';
  if (kind === 'ms') return `${value.toFixed(value < 10 ? 3 : 1)}ms`;
  if (kind === 'diff') return value === 0 ? 'exact' : value.toExponential(1);
  if (kind === 'ratio') return `${value.toFixed(2)} flop/byte`;
  if (kind === 'bytes') {
    if (value >= 1024 * 1024) return `${(value / (1024 * 1024)).toFixed(1)} MB`;
    if (value >= 1024) return `${(value / 1024).toFixed(1)} KB`;
    return `${value} B`;
  }
  return String(value);
}
