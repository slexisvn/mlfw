import { TensorType, ScalarType, TupleType } from './types.js';
import { Operation } from './operation.js';
import { Block, Region } from './block.js';
import { GraphFunction } from './function.js';
import { GraphModule } from './module.js';
import { registry } from './ops.js';

function describeType(type) {
  if (!type) return '?';
  if (type instanceof TupleType) return `tuple(${type.types.map(describeType).join(', ')})`;
  if (type.shape !== undefined && type.dtype !== undefined) {
    return `[${type.shape.join(',')}]:${type.dtype}`;
  }
  return String(type);
}

export function indexSelectGatherOpts(operandType, dim, indicesRank) {
  const rank = operandType.rank;
  const d = dim < 0 ? rank + dim : dim;
  const sliceSizes = operandType.shape.map((s, i) => i === d ? 1 : s);
  const offsetDims = [];
  for (let i = 0; i < rank; i++) {
    if (i !== d) offsetDims.push(i < d ? i : i - 1 + indicesRank);
  }
  return {
    offsetDims,
    collapsedSliceDims: [d],
    startIndexMap: [d],
    indexVectorDim: indicesRank,
    sliceSizes,
  };
}

function bcastBatchDims(a, b) {
  const n = Math.max(a.length, b.length);
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const av = i < a.length ? a[a.length - 1 - i] : 1;
    const bv = i < b.length ? b[b.length - 1 - i] : 1;
    out[n - 1 - i] = av === 1 ? bv : av;
  }
  return out;
}

export class IRBuilder {
  constructor(func) {
    this.func = func;
    this.block = func.entryBlock;
    this._insertionPoint = null;
  }

  setInsertionPoint(op) {
    this._insertionPoint = op;
  }

  setInsertionPointToEnd() {
    this._insertionPoint = null;
  }

  _insert(op) {
    if (this._insertionPoint) {
      this.block.insertBefore(op, this._insertionPoint);
    } else {
      this.block.pushOp(op);
    }
    return op;
  }

  _buildOp(name, operands, resultTypes, attributes = null, regions = null) {
    const op = new Operation(name, operands, resultTypes, attributes, regions);
    return this._insert(op);
  }

  _inferAndBuild(name, operands, attributes = null, regions = null, explicitResultTypes = null) {
    const opDef = registry.get(name);
    let resultTypes = explicitResultTypes;
    if (!resultTypes && opDef && opDef.inferResultTypes) {
      const operandTypes = operands.map(o => o.type);
      const attrMap = attributes instanceof Map ? attributes : toMap(attributes);
      resultTypes = opDef.inferResultTypes(operandTypes, attrMap, explicitResultTypes);
    }
    if (!resultTypes) {
      const operandDesc = operands.map(o => describeType(o.type)).join(', ');
      const reason = opDef
        ? (opDef.inferResultTypes ? 'inferResultTypes returned no types' : 'op has no inferResultTypes and none were given')
        : 'op is not registered';
      throw new Error(`Cannot infer result types for op '${name}' (${reason}); operands: [${operandDesc}]`);
    }
    return this._buildOp(name, operands, resultTypes, attributes, regions);
  }

  constant(value, tensorType) {
    return this._buildOp('constant', [], [tensorType], {
      value,
      tensor_type: tensorType
    });
  }

  scalarConstant(value, dtype = ScalarType.F32) {
    const tt = new TensorType([], dtype);
    return this.constant(value, tt);
  }

  tensorConstant(value, shape, dtype = ScalarType.F32) {
    const tt = new TensorType(shape, dtype);
    return this.constant(value, tt);
  }

  iota(dim, tensorType) {
    return this._buildOp('iota', [], [tensorType], {
      iota_dimension: dim,
      tensor_type: tensorType
    });
  }

  add(lhs, rhs) { return this._inferAndBuild('add', [lhs, rhs]); }
  sub(lhs, rhs) { return this._inferAndBuild('sub', [lhs, rhs]); }
  mul(lhs, rhs) { return this._inferAndBuild('mul', [lhs, rhs]); }
  div(lhs, rhs) { return this._inferAndBuild('div', [lhs, rhs]); }
  rem(lhs, rhs) { return this._inferAndBuild('rem', [lhs, rhs]); }
  pow(lhs, rhs) { return this._inferAndBuild('pow', [lhs, rhs]); }
  maximum(lhs, rhs) { return this._inferAndBuild('maximum', [lhs, rhs]); }
  minimum(lhs, rhs) { return this._inferAndBuild('minimum', [lhs, rhs]); }

  neg(x) { return this._inferAndBuild('neg', [x]); }
  stopGradient(x) { return this._inferAndBuild('stop_gradient', [x]); }
  reverse(x, dimensions) { return this._inferAndBuild('reverse', [x], { dimensions }); }
  scaledDotProductAttention(q, k, v, scale, causal = false) {
    return this._inferAndBuild('scaled_dot_product_attention', [q, k, v], { scale, causal });
  }
  allReduce(x, opts = {}) {
    return this._inferAndBuild('all_reduce', [x], { reduce_op: opts.reduceOp || 'sum', mesh_axis: opts.meshAxis ?? 0 });
  }
  allGather(x, opts = {}) {
    return this._inferAndBuild('all_gather', [x], { mesh_axis: opts.meshAxis ?? 0, gather_dim: opts.gatherDim ?? 1 });
  }
  abs(x) { return this._inferAndBuild('abs', [x]); }
  exp(x) { return this._inferAndBuild('exp', [x]); }
  log(x) { return this._inferAndBuild('log', [x]); }
  sqrt(x) { return this._inferAndBuild('sqrt', [x]); }
  rsqrt(x) { return this._inferAndBuild('rsqrt', [x]); }
  tanh(x) { return this._inferAndBuild('tanh', [x]); }
  sin(x) { return this._inferAndBuild('sin', [x]); }
  cos(x) { return this._inferAndBuild('cos', [x]); }
  floor(x) { return this._inferAndBuild('floor', [x]); }
  ceil(x) { return this._inferAndBuild('ceil', [x]); }
  sign(x) { return this._inferAndBuild('sign', [x]); }
  erf(x) { return this._inferAndBuild('erf', [x]); }
  log2(x) { return this._inferAndBuild('log2', [x]); }
  log10(x) { return this._inferAndBuild('log10', [x]); }
  exp2(x) { return this._inferAndBuild('exp2', [x]); }
  square(x) { return this._inferAndBuild('square', [x]); }
  reciprocal(x) { return this._inferAndBuild('reciprocal', [x]); }

  logicalNot(x) { return this._inferAndBuild('logical_not', [x]); }
  logicalAnd(lhs, rhs) { return this._inferAndBuild('logical_and', [lhs, rhs]); }
  logicalOr(lhs, rhs) { return this._inferAndBuild('logical_or', [lhs, rhs]); }

  compare(lhs, rhs, direction) {
    return this._inferAndBuild('compare', [lhs, rhs], { direction });
  }

  select(pred, onTrue, onFalse) {
    return this._inferAndBuild('select', [pred, onTrue, onFalse]);
  }

  clamp(lo, x, hi) {
    return this._inferAndBuild('clamp', [lo, x, hi]);
  }

  broadcast(input, resultShape, broadcastDimensions) {
    return this._inferAndBuild('broadcast_in_dim', [input], {
      result_shape: resultShape,
      broadcast_dimensions: broadcastDimensions
    });
  }

  reshape(input, newShape) {
    return this._inferAndBuild('reshape', [input], { new_shape: newShape });
  }

  transpose(input, permutation) {
    return this._inferAndBuild('transpose', [input], { permutation });
  }

  slice(input, starts, limits, strides = null) {
    const attrs = { starts, limits };
    if (strides) attrs.strides = strides;
    return this._inferAndBuild('slice', [input], attrs);
  }

  concat(inputs, dimension) {
    return this._inferAndBuild('concat', inputs, { dimension });
  }

  pad(input, paddingValue, low, high, interior = null) {
    const attrs = { low, high };
    if (interior) attrs.interior = interior;
    return this._inferAndBuild('pad', [input, paddingValue], attrs);
  }

  reduce(input, initValue, dimensions, reduceType) {
    const scalarType = new TensorType([], input.type.dtype);
    const combinerRegion = new Region();
    const combinerBlock = new Block([scalarType, scalarType]);
    combinerRegion.addBlock(combinerBlock);
    const op = this._inferAndBuild('reduce', [input, initValue], {
      dimensions,
      reduce_type: reduceType
    }, [combinerRegion]);
    return op;
  }

  dot(lhs, rhs, lhsContracting, rhsContracting, lhsBatch = [], rhsBatch = []) {
    return this._inferAndBuild('dot', [lhs, rhs], {
      lhs_contracting: lhsContracting,
      rhs_contracting: rhsContracting,
      lhs_batch: lhsBatch,
      rhs_batch: rhsBatch
    });
  }

  matmul(lhs, rhs) {
    const lhsRank = lhs.type.rank;
    const rhsRank = rhs.type.rank;
    if (lhsRank === 1 && rhsRank === 1) {
      return this.dot(lhs, rhs, [0], [0]);
    }

    let L = lhs;
    let R = rhs;
    let squeezeLhs = false;
    let squeezeRhs = false;
    if (lhsRank === 1) {
      L = this.reshape(lhs, [1, lhs.type.shape[0]]).getResult(0);
      squeezeLhs = true;
    }
    if (rhsRank === 1) {
      R = this.reshape(rhs, [rhs.type.shape[0], 1]).getResult(0);
      squeezeRhs = true;
    }

    const lr = L.type.rank;
    const rr = R.type.rank;
    const rBatch = R.type.shape.slice(0, rr - 2);

    let result;
    if (rBatch.length === 0) {
      result = this.dot(L, R, [lr - 1], [0]);
    } else {
      const lBatch = L.type.shape.slice(0, lr - 2);
      const batch = bcastBatchDims(lBatch, rBatch);
      const nb = batch.length;
      const Lb = this._broadcastBatch(L, lBatch, batch);
      const Rb = this._broadcastBatch(R, rBatch, batch);
      const range = Array.from({ length: nb }, (_, i) => i);
      result = this.dot(Lb, Rb, [nb + 1], [nb], range, range);
    }

    if (!squeezeLhs && !squeezeRhs) return result;
    const outShape = result.getResult(0).type.shape;
    const drop = new Set();
    if (squeezeRhs) drop.add(outShape.length - 1);
    if (squeezeLhs) drop.add(outShape.length - 2);
    return this.reshape(result.getResult(0), outShape.filter((_, i) => !drop.has(i)));
  }

  _broadcastBatch(value, batch, targetBatch) {
    const matrixDims = value.type.shape.slice(value.type.rank - 2);
    const targetShape = [...targetBatch, ...matrixDims];
    if (batch.length === targetBatch.length && batch.every((d, i) => d === targetBatch[i])) {
      return value;
    }
    const offset = targetBatch.length - batch.length;
    const broadcastDims = [];
    for (let i = 0; i < batch.length; i++) broadcastDims.push(offset + i);
    broadcastDims.push(targetShape.length - 2, targetShape.length - 1);
    return this.broadcast(value, targetShape, broadcastDims).getResult(0);
  }

  conv(input, kernel, strides, padding, opts = {}) {
    return this._inferAndBuild('conv', [input, kernel], {
      strides,
      padding,
      dilation: opts.dilation || strides.map(() => 1),
      groups: opts.groups || 1,
      input_layout: opts.inputLayout || 'NCHW',
      kernel_layout: opts.kernelLayout || 'OIHW'
    });
  }

  convert(input, targetDtype) {
    return this._inferAndBuild('convert', [input], { target_dtype: targetDtype });
  }

  customCall(name, operands, resultTypes, backendConfig = null) {
    const attrs = { call_target_name: name };
    if (backendConfig) attrs.backend_config = backendConfig;
    return this._buildOp('custom_call', operands, resultTypes, attrs);
  }

  fusion(operands, resultTypes, fusionKind, bodyBuilder) {
    const bodyRegion = new Region();
    const bodyBlock = new Block(operands.map(o => o.type));
    bodyRegion.addBlock(bodyBlock);
    const op = this._buildOp('fusion', operands, resultTypes, { fusion_kind: fusionKind }, [bodyRegion]);
    if (bodyBuilder) {
      const innerBuilder = new IRBuilder(Object.create(this.func, {}));
      innerBuilder.block = bodyBlock;
      bodyBuilder(innerBuilder, bodyBlock.arguments);
    }
    return op;
  }

  ifOp(predicate, resultTypes, thenBuilder, elseBuilder) {
    const thenRegion = new Region();
    const thenBlock = new Block([]);
    thenRegion.addBlock(thenBlock);
    const elseRegion = new Region();
    const elseBlock = new Block([]);
    elseRegion.addBlock(elseBlock);
    const op = this._buildOp('if', [predicate], resultTypes, null, [thenRegion, elseRegion]);
    if (thenBuilder) {
      const tb = new IRBuilder(Object.create(this.func, {}));
      tb.block = thenBlock;
      thenBuilder(tb);
    }
    if (elseBuilder) {
      const eb = new IRBuilder(Object.create(this.func, {}));
      eb.block = elseBlock;
      elseBuilder(eb);
    }
    return op;
  }

  whileOp(initValues, condBuilder, bodyBuilder) {
    const types = initValues.map(v => v.type);
    const condRegion = new Region();
    const condBlock = new Block(types);
    condRegion.addBlock(condBlock);
    const bodyRegion = new Region();
    const bodyBlock = new Block(types);
    bodyRegion.addBlock(bodyBlock);
    const op = this._buildOp('while', initValues, types, null, [condRegion, bodyRegion]);
    if (condBuilder) {
      const cb = new IRBuilder(Object.create(this.func, {}));
      cb.block = condBlock;
      condBuilder(cb, condBlock.arguments);
    }
    if (bodyBuilder) {
      const bb = new IRBuilder(Object.create(this.func, {}));
      bb.block = bodyBlock;
      bodyBuilder(bb, bodyBlock.arguments);
    }
    return op;
  }

  scanOp(xsValues, initCarryValues, bodyFn) {
    const xtTypes = xsValues.map(v => v.type.withShape(v.type.shape.slice(1)));
    const carryTypes = initCarryValues.map(v => v.type);
    const bodyRegion = new Region();
    const bodyBlock = new Block([...xtTypes, ...carryTypes]);
    bodyRegion.addBlock(bodyBlock);
    const bb = new IRBuilder(Object.create(this.func, {}));
    bb.block = bodyBlock;
    const xtArgs = bodyBlock.arguments.slice(0, xtTypes.length);
    const carryArgs = bodyBlock.arguments.slice(xtTypes.length);
    const [newCarry, ys] = bodyFn(bb, xtArgs, carryArgs);
    bb.yieldOp([...newCarry, ...ys]);
    const length = xsValues[0].type.shape[0];
    const yTypes = ys.map(v => v.type.withShape([length, ...v.type.shape]));
    return this._buildOp('scan', [...xsValues, ...initCarryValues],
      [...carryTypes, ...yTypes],
      { num_carry: initCarryValues.length, num_xs: xsValues.length },
      [bodyRegion]);
  }

  returnOp(values) {
    return this._buildOp('return', values, []);
  }

  yieldOp(values) {
    return this._buildOp('yield', values, []);
  }

  relu(x) {
    const zero = this.scalarConstant(0, x.type.dtype);
    const broadcastZero = this.broadcast(
      zero.getResult(0),
      x.type.shape,
      []
    );
    return this.maximum(x, broadcastZero.getResult(0));
  }

  softmax(input, axis = -1) {
    const dim = axis < 0 ? input.type.rank + axis : axis;
    return this._inferAndBuild('softmax', [input], { axis: dim });
  }

  logSoftmax(input, axis = -1) {
    const dim = axis < 0 ? input.type.rank + axis : axis;
    return this._inferAndBuild('log_softmax', [input], { axis: dim });
  }

  sigmoid(x) {
    return this._inferAndBuild('sigmoid', [x]);
  }

  gelu(x) {
    return this._inferAndBuild('gelu', [x]);
  }

  silu(x) {
    return this._inferAndBuild('silu', [x]);
  }

  elu(x, alpha = 1.0) {
    return this._inferAndBuild('elu', [x], { alpha });
  }

  leakyRelu(x, negativeSlope = 0.01) {
    return this._inferAndBuild('leaky_relu', [x], { negative_slope: negativeSlope });
  }

  celu(x, alpha = 1.0) {
    return this._inferAndBuild('celu', [x], { alpha });
  }

  selu(x) { return this._inferAndBuild('selu', [x]); }
  mish(x) { return this._inferAndBuild('mish', [x]); }
  hardswish(x) { return this._inferAndBuild('hardswish', [x]); }
  hardsigmoid(x) { return this._inferAndBuild('hardsigmoid', [x]); }

  layernorm(input, gamma, beta, axis = -1, eps = 1e-5) {
    const dim = axis < 0 ? input.type.rank + axis : axis;
    return this._inferAndBuild('layer_norm', [input, gamma, beta], { axis: dim, epsilon: eps });
  }

  batchnorm(input, gamma, beta, mean, variance, axis = 1, eps = 1e-5) {
    return this._inferAndBuild('batch_norm', [input, gamma, beta, mean, variance], { axis, epsilon: eps });
  }

  where(condition, x, y) {
    return this._inferAndBuild('where', [condition, x, y]);
  }

  split(input, dimension, splitSizes) {
    return this._inferAndBuild('split', [input], { dimension, split_sizes: splitSizes });
  }

  oneHot(indices, depth, opts = {}) {
    return this._inferAndBuild('one_hot', [indices], {
      depth,
      axis: opts.axis ?? -1,
      on_value: opts.onValue ?? 1,
      off_value: opts.offValue ?? 0,
      dtype: opts.dtype || indices.type.dtype,
    });
  }

  embedding(weight, indices) {
    return this._inferAndBuild('embedding', [weight, indices]);
  }

  gather(operand, indices, opts) {
    return this._inferAndBuild('gather', [operand, indices], {
      offset_dims: opts.offsetDims,
      collapsed_slice_dims: opts.collapsedSliceDims,
      start_index_map: opts.startIndexMap,
      index_vector_dim: opts.indexVectorDim,
      slice_sizes: opts.sliceSizes,
    });
  }

  scatter(operand, indices, updates, opts) {
    const scalarType = new TensorType([], operand.type.dtype);
    const combinerRegion = new Region();
    combinerRegion.addBlock(new Block([scalarType, scalarType]));
    return this._inferAndBuild('scatter', [operand, indices, updates], {
      update_window_dims: opts.updateWindowDims,
      inserted_window_dims: opts.insertedWindowDims,
      scatter_dims_to_operand_dims: opts.scatterDimsToOperandDims,
      index_vector_dim: opts.indexVectorDim,
    }, [combinerRegion]);
  }

  scatterAdd(operand, indices, updates, opts) {
    return this.scatter(operand, indices, updates, opts);
  }

  _dimCoordIndices(index, dim, gridShape) {
    const r = gridShape.length;
    const idxI32 = index.type.dtype === 'i32' ? index : this.convert(index, 'i32').getResult(0);
    const coordShape = [...gridShape, 1];
    const coords = [];
    for (let j = 0; j < r; j++) {
      const cj = j === dim ? idxI32 : this.iota(j, new TensorType(gridShape, 'i32')).getResult(0);
      coords.push(this.reshape(cj, coordShape).getResult(0));
    }
    return this.concat(coords, r).getResult(0);
  }

  gatherDim(operand, index, dim) {
    const r = operand.type.rank;
    const d = dim < 0 ? r + dim : dim;
    const range = [];
    const ones = [];
    for (let j = 0; j < r; j++) { range.push(j); ones.push(1); }
    const fullIdx = this._dimCoordIndices(index, d, index.type.shape);
    return this.gather(operand, fullIdx, {
      offsetDims: [],
      collapsedSliceDims: range,
      startIndexMap: range,
      indexVectorDim: r,
      sliceSizes: ones,
    });
  }

  scatterAddDim(operand, index, updates, dim) {
    const r = operand.type.rank;
    const d = dim < 0 ? r + dim : dim;
    const range = [];
    for (let j = 0; j < r; j++) range.push(j);
    const fullIdx = this._dimCoordIndices(index, d, index.type.shape);
    return this.scatterAdd(operand, fullIdx, updates, {
      updateWindowDims: [],
      insertedWindowDims: range,
      scatterDimsToOperandDims: range,
      indexVectorDim: r,
    });
  }

  argmax(input, axis, keepDims = false) {
    const dim = axis < 0 ? input.type.rank + axis : axis;
    return this._inferAndBuild('argmax', [input], { axis: dim, keep_dims: keepDims });
  }

  argmin(input, axis, keepDims = false) {
    const dim = axis < 0 ? input.type.rank + axis : axis;
    return this._inferAndBuild('argmin', [input], { axis: dim, keep_dims: keepDims });
  }

  pool2d(input, poolType, kernelSize, strides, padding, opts = {}) {
    return this._inferAndBuild('pool2d', [input], {
      pool_type: poolType,
      kernel_size: kernelSize,
      strides,
      padding,
      ceil_mode: opts.ceilMode || false,
      count_include_pad: opts.countIncludePad || false,
      layout: opts.layout || 'NCHW',
    });
  }

  resize(input, outputSize, method, opts = {}) {
    return this._inferAndBuild('resize', [input], {
      output_size: outputSize,
      method,
      coordinate_mode: opts.coordinateMode || 'asymmetric',
      layout: opts.layout || 'NCHW',
    });
  }
}

export function broadcastDimsExcluding(rank, excludedDim) {
  const dims = [];
  for (let i = 0; i < rank; i++) {
    if (i !== excludedDim) dims.push(i);
  }
  return dims;
}

function toMap(obj) {
  if (!obj) return new Map();
  if (obj instanceof Map) return obj;
  const m = new Map();
  for (const k of Object.keys(obj)) m.set(k, obj[k]);
  return m;
}

export function buildFunction(name, inputTypes, outputTypes, bodyFn) {
  const func = new GraphFunction(name, inputTypes, outputTypes);
  const builder = new IRBuilder(func);
  bodyFn(builder, func.args);
  return func;
}

export function buildModule(name, funcBuilders) {
  const mod = new GraphModule(name);
  for (const [fname, inputTypes, outputTypes, bodyFn] of funcBuilders) {
    mod.addFunction(buildFunction(fname, inputTypes, outputTypes, bodyFn));
  }
  return mod;
}
