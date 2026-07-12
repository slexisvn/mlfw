import { indexSelectGatherOpts } from '../../compiler/ir/graph/builder.js';
import { ScalarType } from '../../compiler/ir/graph/types.js';
import { reduceInitValue } from '../../util/dtype_map.js';

export const REDUCTION_OPS = Object.freeze({
  sum: 'sum',
  mean: 'mean',
  max: 'max',
  min: 'min',
  prod: 'prod',
});

export function buildMappedOp(builder, opName, args, attrs) {
  if (REDUCTION_OPS[opName]) return buildReduce(builder, opName, args, attrs);

  const fn = IR_BUILDERS[opName];
  if (fn) return fn(builder, args, attrs || {});
  if (typeof builder[opName] === 'function') return callBuilder(builder, opName, args, attrs);
  return builder._inferAndBuild(opName, args, attrs || null);
}

function buildReduce(builder, opName, args, attrs) {
  const rank = args[0].type.rank;
  const dims = attrs?.dim;
  const dimensions = dims !== undefined && dims !== null
    ? (Array.isArray(dims) ? dims : [dims]).map(d => d < 0 ? rank + d : d)
    : Array.from({ length: rank }, (_, i) => i);
  const initConst = builder.scalarConstant(reduceInitValue(opName, args[0].type.dtype), args[0].type.dtype);
  const reduced = builder.reduce(args[0], initConst.getResult(0), dimensions, opName);
  if (!attrs?.keepdim) return reduced;
  const dimSet = new Set(dimensions);
  const newShape = args[0].type.shape.map((d, i) => dimSet.has(i) ? 1 : d);
  return builder.reshape(reduced.getResult(0), newShape);
}

function callBuilder(builder, opName, args, attrs) {
  if (args.length === 1) return builder[opName](args[0]);
  if (args.length === 2) return builder[opName](args[0], args[1]);
  if (args.length === 3) return builder[opName](args[0], args[1], args[2]);
  return builder._inferAndBuild(opName, args, attrs || null);
}

const IR_BUILDERS = Object.freeze({
  matmul: (b, a) => b.matmul(a[0], a[1]),
  dot: (b, a) => b.dot(a[0], a[1], [a[0].type.rank - 1], [0]),
  clone: (b, a) => b._inferAndBuild('add', [a[0], b.scalarConstant(0, a[0].type.dtype).getResult(0)]),
  relu: (b, a) => b.relu(a[0]),
  sigmoid: (b, a) => b.sigmoid(a[0]),
  gelu: (b, a) => b.gelu(a[0]),
  silu: (b, a) => b.silu(a[0]),
  softmax: (b, a, s) => b.softmax(a[0], s?.dim ?? -1),
  log_softmax: (b, a, s) => b.logSoftmax(a[0], s?.dim ?? -1),
  layer_norm: (b, a, s) => b.layernorm(a[0], a[1], a[2], s?.axis ?? -1, s?.eps ?? 1e-5),
  batch_norm: (b, a, s) => b.batchnorm(a[0], a[1], a[2], a[3], a[4], s?.axis ?? 1, s?.eps ?? 1e-5),
  conv2d: (b, a, s) => b.conv(a[0], a[1], s?.strides ?? [1, 1], s?.padding ?? [[0, 0], [0, 0]], { dilation: s?.dilation ?? [1, 1], groups: s?.groups ?? 1 }),
  pool2d: (b, a, s) => b.pool2d(a[0], s?.pool_type ?? 'max', s?.kernel_size ?? [2, 2], s?.strides ?? [2, 2], s?.padding ?? [[0, 0], [0, 0]]),
  embedding: (b, a) => b.embedding(a[0], a[1]),
  argmax: (b, a, s) => b.argmax(a[0], s?.dim ?? 0, s?.keepdim ?? false),
  argmin: (b, a, s) => b.argmin(a[0], s?.dim ?? 0, s?.keepdim ?? false),
  maximum: (b, a) => b.maximum(a[0], a[1]),
  minimum: (b, a) => b.minimum(a[0], a[1]),
  eq: (b, a) => b.compare(a[0], a[1], 'eq'),
  ne: (b, a) => b.compare(a[0], a[1], 'ne'),
  lt: (b, a) => b.compare(a[0], a[1], 'lt'),
  le: (b, a) => b.compare(a[0], a[1], 'le'),
  gt: (b, a) => b.compare(a[0], a[1], 'gt'),
  ge: (b, a) => b.compare(a[0], a[1], 'ge'),
  clamp: (b, a) => b.clamp(a[1], a[0], a[2]),
  pad: (b, a, s) => b.pad(a[0], a[1], s.low, s.high),
  one_hot: (b, a, s) => b.oneHot(a[0], s.depth, { dtype: ScalarType.F32 }),
  index_select: (b, a, s) => b.gather(a[0], a[1], indexSelectGatherOpts(a[0].type, s?.dim ?? 0, a[1].type.rank)),
  gather: (b, a, s) => b.gatherDim(a[0], a[1], s?.dim ?? 0),
  scatter_add: (b, a, s) => b.scatterAddDim(a[0], a[1], a[2], s?.dim ?? 0),
  cat: (b, a, s) => b.concat(a, normalizeDim(s?.dim ?? 0, a[0].type.rank)),
  stack: (b, a, s) => {
    const dim = normalizeInsertDim(s?.dim ?? 0, a[0].type.rank);
    const expanded = a.map(arg => {
      const newShape = [...arg.type.shape];
      newShape.splice(dim, 0, 1);
      return b.reshape(arg, newShape).getResult(0);
    });
    return b.concat(expanded, dim);
  },
  reshape: (b, a, s) => b.reshape(a[0], resolveShape(s.shape ?? s.new_shape, a[0].type.shape)),
  transpose: (b, a, s) => {
    const rank = a[0].type.rank;
    const d0 = normalizeDim(s?.dim0 ?? 0, rank);
    const d1 = normalizeDim(s?.dim1 ?? 1, rank);
    const perm = Array.from({ length: rank }, (_, i) => i);
    perm[d0] = d1;
    perm[d1] = d0;
    return b.transpose(a[0], perm);
  },
  permute: (b, a, s) => b.transpose(a[0], s.dims),
  broadcast_in_dim: (b, a, s) => b.broadcast(a[0], s.result_shape, s.broadcast_dimensions),
  expand: (b, a, s) => {
    const inRank = a[0].type.rank;
    const shape = s.shape;
    const offset = shape.length - inRank;
    const resultShape = shape.map((d, i) => d === -1 ? a[0].type.shape[i - offset] : d);
    const broadcastDimensions = Array.from({ length: inRank }, (_, i) => i + offset);
    return b.broadcast(a[0], resultShape, broadcastDimensions);
  },
  slice: (b, a, s) => {
    const attrs = sliceAttrs(a[0].type.shape, s.dim, s.start, s.end, s.step);
    return b.slice(a[0], attrs.starts, attrs.limits, attrs.strides);
  },
  unsqueeze: (b, a, s) => {
    const shape = [...a[0].type.shape];
    shape.splice(normalizeInsertDim(s.dim, shape.length), 0, 1);
    return b.reshape(a[0], shape);
  },
  squeeze: (b, a, s) => {
    const shape = [...a[0].type.shape];
    if (s.dim === undefined || s.dim === null) return b.reshape(a[0], shape.filter(d => d !== 1));
    const dim = normalizeDim(s.dim, shape.length);
    if (shape[dim] === 1) shape.splice(dim, 1);
    return b.reshape(a[0], shape);
  },
  narrow: (b, a, s) => {
    const dim = normalizeDim(s.dim, a[0].type.rank);
    const starts = new Array(a[0].type.rank).fill(0);
    const limits = [...a[0].type.shape];
    const strides = new Array(a[0].type.rank).fill(1);
    starts[dim] = s.start;
    limits[dim] = s.start + s.length;
    return b.slice(a[0], starts, limits, strides);
  },
  select: (b, a, s) => {
    const rank = a[0].type.rank;
    const dim = normalizeDim(s.dim, rank);
    const idx = s.index < 0 ? a[0].type.shape[dim] + s.index : s.index;
    const starts = new Array(rank).fill(0);
    const limits = [...a[0].type.shape];
    starts[dim] = idx;
    limits[dim] = idx + 1;
    const sliced = b.slice(a[0], starts, limits, new Array(rank).fill(1)).getResult(0);
    return b.reshape(sliced, a[0].type.shape.filter((_, i) => i !== dim));
  },
  split: (b, a, s) => b.split(a[0], normalizeDim(s.dim ?? 0, a[0].type.rank), s.sizes),
  chunk: (b, a, s) => {
    const dim = normalizeDim(s.dim ?? 0, a[0].type.rank);
    const n = a[0].type.shape[dim];
    const size = Math.ceil(n / s.chunks);
    const sizes = [];
    for (let off = 0; off < n; off += size) sizes.push(Math.min(size, n - off));
    return b.split(a[0], dim, sizes);
  },
});

function normalizeDim(dim, rank) {
  return dim < 0 ? rank + dim : dim;
}

function resolveShape(shape, inputShape) {
  let known = 1;
  let infer = -1;
  let total = 1;
  for (const d of inputShape) total *= d;
  const out = [...shape];
  for (let i = 0; i < out.length; i++) {
    if (out[i] === -1) infer = i;
    else known *= out[i];
  }
  if (infer >= 0) out[infer] = known === 0 ? 0 : total / known;
  return out;
}

function normalizeInsertDim(dim, rank) {
  return dim < 0 ? rank + 1 + dim : dim;
}

function sliceAttrs(shape, dim, start, end, step) {
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
