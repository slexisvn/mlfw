const CAMEL_BOUNDARY = /([a-z0-9])([A-Z])/g;
const SEPARATORS = /[_\-.]+/g;
const PASS_SUFFIX = /Pass$/;
const RUN_SUFFIX = /_\d+$/;

const ABBREVIATIONS: Record<string, string> = {
  dce: 'dead code elimination',
  cse: 'common subexpression elimination',
  tir: 'tensor IR',
  lir: 'low-level IR',
  ir: 'IR',
};

const OP_LABELS: Record<string, string> = {
  dot: 'matrix multiply',
  matmul: 'matrix multiply',
  maximum: 'element-wise maximum',
  minimum: 'element-wise minimum',
  broadcast_in_dim: 'broadcast to shape',
  transpose: 'swap axes',
  reshape: 'reinterpret shape',
  constant: 'constant value',
  add: 'add',
  sub: 'subtract',
  mul: 'multiply',
  div: 'divide',
  neg: 'negate',
  exp: 'exponential',
  log: 'natural log',
  tanh: 'tanh',
  sqrt: 'square root',
  rsqrt: 'reciprocal square root',
  reduce_sum: 'sum along axes',
  reduce_max: 'maximum along axes',
  reduce_mean: 'mean along axes',
  select: 'pick per element',
  compare: 'compare per element',
  convert: 'change dtype',
  slice: 'take a sub-tensor',
  concatenate: 'join tensors',
  pad: 'pad with a value',
  convolution: 'convolution',
  fusion: 'fused kernel',
  reduce_window: 'sliding-window reduce',
  iota: 'index ramp',
  return: 'function result',
};

const METRIC_UNITS: Record<string, string> = {
  peakMemory: 'bytes',
  totalBytes: 'bytes',
  bytes: 'bytes',
  durationMs: 'ms',
  elapsedMs: 'ms',
};

const HIDDEN_METRICS = new Set(['passName', 'level', 'timestamp', 'type']);

const ZERO_BYTE_NOTES: Record<string, string> = {
  peakMemory: 'none — every value stayed in registers',
};

const BYTE_STEPS = [
  { limit: 1024 ** 3, suffix: 'GB' },
  { limit: 1024 ** 2, suffix: 'MB' },
  { limit: 1024, suffix: 'KB' },
];

export function humanize(identifier: string): string {
  return identifier
    .replace(CAMEL_BOUNDARY, '$1 $2')
    .replace(SEPARATORS, ' ')
    .trim()
    .toLowerCase();
}

export function passLabel(name: string): string {
  return ABBREVIATIONS[name] ?? humanize(name.replace(PASS_SUFFIX, ''));
}

export function phaseLabel(phase: string): string {
  return humanize(phase);
}

export function levelLabel(level: string): string {
  return ABBREVIATIONS[level] ?? humanize(level);
}

export function opLabel(opName: string): string {
  const base = opName.replace(RUN_SUFFIX, '');
  return OP_LABELS[base] ?? humanize(base);
}

export function metricLabel(key: string): string {
  const unit = METRIC_UNITS[key];
  const label = humanize(key.replace(/Ms$/, '').replace(/Bytes?$/, ''));
  return unit && unit !== 'bytes' ? `${label} (${unit})` : label;
}

export function isHiddenMetric(key: string): boolean {
  return HIDDEN_METRICS.has(key);
}

function formatBytes(key: string, value: number): string {
  if (value === 0) return ZERO_BYTE_NOTES[key] ?? 'none';
  for (const step of BYTE_STEPS) {
    if (value >= step.limit) return `${(value / step.limit).toFixed(1)} ${step.suffix}`;
  }
  return `${value} B`;
}

export function metricValue(key: string, value: unknown): string {
  if (typeof value === 'number') {
    if (METRIC_UNITS[key] === 'bytes') return formatBytes(key, value);
    if (METRIC_UNITS[key] === 'ms') return `${value.toFixed(2)}ms`;
    return Number.isInteger(value) ? String(value) : value.toFixed(2);
  }
  if (typeof value === 'string') return value;
  return JSON.stringify(value);
}
