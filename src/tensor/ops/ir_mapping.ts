import { indexSelectGatherOpts, IRBuilder } from '../../compiler/ir/graph/builder.js';
import { registry } from '../../compiler/ir/graph/ops.js';
import { ScalarType } from '../../compiler/ir/graph/types.js';
import type { TensorType } from '../../compiler/ir/graph/types.js';
import type { DType } from '../types/dtype.js';
import { reduceInitValue } from '../../util/dtype_map.js';

export type GraphType = {
  rank: number;
  shape: readonly number[];
  dtype: DType;
};

export type GraphValue = {
  type: GraphType;
};

export type GraphOperation = {
  getResult(index: number): GraphValue;
};

export type Attrs = Record<string, unknown>;

type BuilderFn = (...args: unknown[]) => GraphOperation;

export type BuilderLike = Record<string, unknown> & {
  scalarConstant(value: unknown, dtype: DType): GraphOperation;
  reduce(value: GraphValue, init: GraphValue, dimensions: readonly number[], opName: string): GraphOperation;
  reshape(value: GraphValue, shape: readonly number[]): GraphOperation;
  _inferAndBuild(name: string, args: readonly GraphValue[], attrs: Attrs | null): GraphOperation;
};

type IrBuilder = (builder: BuilderLike, args: readonly GraphValue[], attrs: Attrs) => GraphOperation;

export const REDUCTION_OPS: Readonly<Record<string, string>> = Object.freeze({
  sum: 'sum',
  mean: 'mean',
  max: 'max',
  min: 'min',
  prod: 'prod',
});

export function buildMappedOp(builder: BuilderLike, opName: string, args: readonly GraphValue[], attrs?: Attrs | null): GraphOperation {
  if (REDUCTION_OPS[opName]) return buildReduce(builder, opName, args, attrs);

  const fn = IR_BUILDERS[opName];
  if (fn) return fn(builder, args, attrs || {});
  if (typeof builder[opName] === 'function') return callBuilder(builder, opName, args, attrs);
  return builder._inferAndBuild(opName, args, attrs || null);
}

export function canBuildMappedOp(opName: string): boolean {
  if (REDUCTION_OPS[opName]) return true;
  if (IR_BUILDERS[opName]) return true;
  if (typeof (IRBuilder.prototype as unknown as Record<string, unknown>)[opName] === 'function') return true;
  const opDef = registry.get(opName);
  return opDef !== null && opDef.inferResultTypes !== undefined;
}

function buildReduce(builder: BuilderLike, opName: string, args: readonly GraphValue[], attrs?: Attrs | null): GraphOperation {
  const rank = args[0].type.rank;
  const dims = attrs?.dim;
  const dimensions = dims !== undefined && dims !== null
    ? (Array.isArray(dims) ? dims : [dims]).map(d => (d as number) < 0 ? rank + (d as number) : d as number)
    : Array.from({ length: rank }, (_, i) => i);
  const initConst = builder.scalarConstant(reduceInitValue(opName, args[0].type.dtype), args[0].type.dtype);
  const reduced = builder.reduce(args[0], initConst.getResult(0), dimensions, opName);
  if (!attrs?.keepdim) return reduced;
  const dimSet = new Set(dimensions);
  const newShape = args[0].type.shape.map((d, i) => dimSet.has(i) ? 1 : d);
  return builder.reshape(reduced.getResult(0), newShape);
}

function callBuilder(builder: BuilderLike, opName: string, args: readonly GraphValue[], attrs?: Attrs | null): GraphOperation {
  if (args.length === 1) return callMethod(builder, opName, args[0]);
  if (args.length === 2) return callMethod(builder, opName, args[0], args[1]);
  if (args.length === 3) return callMethod(builder, opName, args[0], args[1], args[2]);
  return builder._inferAndBuild(opName, args, attrs || null);
}

function callMethod(builder: BuilderLike, name: string, ...args: unknown[]): GraphOperation {
  return (builder[name] as BuilderFn).call(builder, ...args);
}

function buildIdentity(builder: BuilderLike, value: GraphValue): GraphOperation {
  return builder._inferAndBuild('add', [value, builder.scalarConstant(0, value.type.dtype).getResult(0)], null);
}

const IR_BUILDERS: Readonly<Record<string, IrBuilder>> = Object.freeze({
  matmul: (b, a) => callMethod(b, 'matmul', a[0], a[1]),
  dot: (b, a) => callMethod(b, 'dot', a[0], a[1], [a[0].type.rank - 1], [0]),
  clone: (b, a) => buildIdentity(b, a[0]),
  contiguous: (b, a) => buildIdentity(b, a[0]),
  fill: (b, a, s) => callMethod(b, 'broadcast', b.scalarConstant(s.value, a[0].type.dtype).getResult(0), a[0].type.shape, []),
  relu: (b, a) => callMethod(b, 'relu', a[0]),
  sigmoid: (b, a) => callMethod(b, 'sigmoid', a[0]),
  gelu: (b, a) => callMethod(b, 'gelu', a[0]),
  silu: (b, a) => callMethod(b, 'silu', a[0]),
  softmax: (b, a, s) => callMethod(b, 'softmax', a[0], s?.dim ?? -1),
  log_softmax: (b, a, s) => callMethod(b, 'logSoftmax', a[0], s?.dim ?? -1),
  layer_norm: (b, a, s) => callMethod(b, 'layernorm', a[0], a[1], a[2], s?.axis ?? -1, s?.eps ?? 1e-5),
  batch_norm: (b, a, s) => callMethod(b, 'batchnorm', a[0], a[1], a[2], a[3], a[4], s?.axis ?? 1, s?.eps ?? 1e-5),
  conv2d: (b, a, s) => callMethod(b, 'conv', a[0], a[1], s?.strides ?? [1, 1], s?.padding ?? [[0, 0], [0, 0]], { dilation: s?.dilation ?? [1, 1], groups: s?.groups ?? 1 }),
  pool2d: (b, a, s) => callMethod(b, 'pool2d', a[0], s?.pool_type ?? 'max', s?.kernel_size ?? [2, 2], s?.strides ?? [2, 2], s?.padding ?? [[0, 0], [0, 0]]),
  embedding: (b, a) => callMethod(b, 'embedding', a[0], a[1]),
  argmax: (b, a, s) => callMethod(b, 'argmax', a[0], s?.dim ?? 0, s?.keepdim ?? false),
  argmin: (b, a, s) => callMethod(b, 'argmin', a[0], s?.dim ?? 0, s?.keepdim ?? false),
  maximum: (b, a) => callMethod(b, 'maximum', a[0], a[1]),
  minimum: (b, a) => callMethod(b, 'minimum', a[0], a[1]),
  eq: (b, a) => callMethod(b, 'compare', a[0], a[1], 'eq'),
  ne: (b, a) => callMethod(b, 'compare', a[0], a[1], 'ne'),
  lt: (b, a) => callMethod(b, 'compare', a[0], a[1], 'lt'),
  le: (b, a) => callMethod(b, 'compare', a[0], a[1], 'le'),
  gt: (b, a) => callMethod(b, 'compare', a[0], a[1], 'gt'),
  ge: (b, a) => callMethod(b, 'compare', a[0], a[1], 'ge'),
  clamp: (b, a) => callMethod(b, 'clamp', a[1], a[0], a[2]),
  pad: (b, a, s) => callMethod(b, 'pad', a[0], a[1], s.low, s.high),
  one_hot: (b, a, s) => callMethod(b, 'oneHot', a[0], s.depth, { dtype: ScalarType.F32 }),
  index_select: (b, a, s) => callMethod(b, 'gather', a[0], a[1], indexSelectGatherOpts(a[0].type as unknown as TensorType, s?.dim as number ?? 0, a[1].type.rank)),
  gather: (b, a, s) => callMethod(b, 'gatherDim', a[0], a[1], s?.dim ?? 0),
  scatter_add: (b, a, s) => callMethod(b, 'scatterAddDim', a[0], a[1], a[2], s?.dim ?? 0),
  cat: (b, a, s) => callMethod(b, 'concat', a, normalizeDim(s?.dim as number ?? 0, a[0].type.rank)),
  stack: (b, a, s) => {
    const dim = normalizeInsertDim(s?.dim as number ?? 0, a[0].type.rank);
    const expanded = a.map(arg => {
      const newShape = [...arg.type.shape];
      newShape.splice(dim, 0, 1);
      return b.reshape(arg, newShape).getResult(0);
    });
    return callMethod(b, 'concat', expanded, dim);
  },
  reshape: (b, a, s) => b.reshape(a[0], (s.shape ?? s.new_shape) as readonly number[]),
  transpose: (b, a, s) => {
    const rank = a[0].type.rank;
    const d0 = normalizeDim(s?.dim0 as number ?? 0, rank);
    const d1 = normalizeDim(s?.dim1 as number ?? 1, rank);
    const perm = Array.from({ length: rank }, (_, i) => i);
    perm[d0] = d1;
    perm[d1] = d0;
    return (b.transpose as BuilderFn)(a[0], perm);
  },
  permute: (b, a, s) => callMethod(b, 'transpose', a[0], s.dims),
  broadcast_in_dim: (b, a, s) => callMethod(b, 'broadcast', a[0], s.result_shape, s.broadcast_dimensions),
  expand: (b, a, s) => {
    const inRank = a[0].type.rank;
    const shape = s.shape as readonly number[];
    const offset = shape.length - inRank;
    const resultShape = shape.map((d, i) => d === -1 ? a[0].type.shape[i - offset] : d);
    const broadcastDimensions = Array.from({ length: inRank }, (_, i) => i + offset);
    return callMethod(b, 'broadcast', a[0], resultShape, broadcastDimensions);
  },
  slice: (b, a, s) => {
    const attrs = sliceAttrs(a[0].type.shape, s.dim as number, s.start as number, s.end as number, s.step as number);
    return callMethod(b, 'slice', a[0], attrs.starts, attrs.limits, attrs.strides);
  },
  unsqueeze: (b, a, s) => {
    const shape = [...a[0].type.shape];
    shape.splice(normalizeInsertDim(s.dim as number, shape.length), 0, 1);
    return b.reshape(a[0], shape);
  },
  squeeze: (b, a, s) => {
    const shape = [...a[0].type.shape];
    if (s.dim === undefined || s.dim === null) return b.reshape(a[0], shape.filter(d => d !== 1));
    const dim = normalizeDim(s.dim as number, shape.length);
    if (shape[dim] === 1) shape.splice(dim, 1);
    return b.reshape(a[0], shape);
  },
  narrow: (b, a, s) => {
    const dim = normalizeDim(s.dim as number, a[0].type.rank);
    const starts = new Array(a[0].type.rank).fill(0);
    const limits = [...a[0].type.shape];
    const strides = new Array(a[0].type.rank).fill(1);
    starts[dim] = s.start as number;
    limits[dim] = (s.start as number) + (s.length as number);
    return callMethod(b, 'slice', a[0], starts, limits, strides);
  },
  select: (b, a, s) => {
    const rank = a[0].type.rank;
    const dim = normalizeDim(s.dim as number, rank);
    const index = s.index as number;
    const idx = index < 0 ? a[0].type.shape[dim] + index : index;
    const starts = new Array(rank).fill(0);
    const limits = [...a[0].type.shape];
    starts[dim] = idx;
    limits[dim] = idx + 1;
    const sliced = callMethod(b, 'slice', a[0], starts, limits, new Array(rank).fill(1)).getResult(0);
    return b.reshape(sliced, a[0].type.shape.filter((_, i) => i !== dim));
  },
  split: (b, a, s) => callMethod(b, 'split', a[0], normalizeDim(s.dim as number ?? 0, a[0].type.rank), s.sizes),
  chunk: (b, a, s) => {
    const dim = normalizeDim(s.dim as number ?? 0, a[0].type.rank);
    const n = a[0].type.shape[dim];
    const size = Math.ceil(n / (s.chunks as number));
    const sizes: number[] = [];
    for (let off = 0; off < n; off += size) sizes.push(Math.min(size, n - off));
    return callMethod(b, 'split', a[0], dim, sizes);
  },
});

function normalizeDim(dim: number, rank: number): number {
  return dim < 0 ? rank + dim : dim;
}

function normalizeInsertDim(dim: number, rank: number): number {
  return dim < 0 ? rank + 1 + dim : dim;
}

function sliceAttrs(shape: readonly number[], dim: number, start: number, end: number, step: number) {
  const rank = shape.length;
  const d = normalizeDim(dim, rank);
  const dimSize = shape[d];
  let s = start ?? 0;
  let e = end ?? dimSize;
  const st = step ?? 1;
  if (s < 0) s += dimSize;
  if (e < 0) e += dimSize;
  s = Math.max(0, Math.min(s, dimSize));
  e = Math.max(0, Math.min(e, dimSize));
  const starts = new Array(rank).fill(0);
  const limits = [...shape];
  const strides = new Array(rank).fill(1);
  starts[d] = s;
  limits[d] = e;
  strides[d] = st;
  return { starts, limits, strides };
}
