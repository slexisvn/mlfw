import { TensorType, ScalarType, TupleType, DYNAMIC, resultDtype } from './types.js';
import { Operation } from './operation.js';
import { Block, Region } from './block.js';
import { GraphFunction } from './function.js';
import { GraphModule } from './module.js';
import { registry } from './ops.js';
import { unifiedOperandIndices } from './op_traits.js';
import { currentLocation } from '../loc_source.js';
import type { Location } from '../location.js';
import type { AttrValue, Dim, IRType, ScalarDType, Shape } from './types.js';
import type { Value, BlockArgument } from './value.js';

export type AttrRecord = Record<string, AttrValue>;

export function propagateSymbolicShapes(op: Operation): void {
  const operandShapes = new Map<Value, Shape>();
  for (const operand of op.operands) {
    if (operand.symbolicShape) operandShapes.set(operand, operand.symbolicShape);
  }
  if (operandShapes.size === 0) return;

  const def = registry.get(op.opName);
  const propagated = def && def.propagateSymbolicShapes
    ? def.propagateSymbolicShapes(op, operandShapes as never) as unknown as (Shape | null)[] | null
    : null;

  for (let i = 0; i < op.numResults; i++) {
    const result = op.getResult(i);
    const resultType = result.type;
    if (!(resultType instanceof TensorType)) continue;
    const fromRule = propagated ? propagated[i] : null;
    result.symbolicShape = fromRule || alignSymbolicShape(resultType.shape, operandShapes);
  }
}

function alignSymbolicShape(resultShape: Shape, operandShapes: ReadonlyMap<Value, Shape>): Dim[] {
  const out: Dim[] = new Array(resultShape.length);
  for (let i = 0; i < resultShape.length; i++) {
    if (resultShape[i] !== DYNAMIC) { out[i] = resultShape[i]; continue; }
    out[i] = DYNAMIC;
    for (const symShape of operandShapes.values()) {
      const srcIdx = i - (resultShape.length - symShape.length);
      if (srcIdx < 0 || srcIdx >= symShape.length) continue;
      const sym = symShape[srcIdx];
      if (sym !== null && sym !== undefined && typeof sym !== 'number') {
        out[i] = sym;
        break;
      }
    }
  }
  return out;
}

export type AllReduceOpts = Readonly<{ reduceOp?: string; meshAxis?: number }>;
export type AllGatherOpts = Readonly<{ meshAxis?: number; gatherDim?: number }>;
export type ConvOpts = Readonly<{ dilation?: readonly number[]; groups?: number; inputLayout?: string; kernelLayout?: string }>;
export type OneHotOpts = Readonly<{ axis?: number; onValue?: number; offValue?: number; dtype?: ScalarDType }>;
export type Pool2dOpts = Readonly<{ ceilMode?: boolean; countIncludePad?: boolean; layout?: string }>;
export type ResizeOpts = Readonly<{ coordinateMode?: string; layout?: string }>;

export type GatherOpts = Readonly<{
  offsetDims: readonly number[];
  collapsedSliceDims: readonly number[];
  startIndexMap: readonly number[];
  indexVectorDim: number;
  sliceSizes: Shape;
}>;

export type ScatterOpts = Readonly<{
  updateWindowDims: readonly number[];
  insertedWindowDims: readonly number[];
  scatterDimsToOperandDims: readonly number[];
  indexVectorDim: number;
}>;

export type FusionBodyBuilder = (builder: IRBuilder, args: BlockArgument[]) => void;
export type BranchBuilder = (builder: IRBuilder) => void;
export type RegionBodyBuilder = (builder: IRBuilder, args: BlockArgument[]) => void;
export type ScanBodyFn = (builder: IRBuilder, xs: BlockArgument[], carry: BlockArgument[]) => [Value[], Value[]];
export type FunctionBodyFn = (builder: IRBuilder, args: BlockArgument[]) => void;
export type FunctionSpec = readonly [string, readonly IRType[], readonly IRType[], FunctionBodyFn];

function tensorTypeOf(value: Value): TensorType {
  return value.type as TensorType;
}

function describeType(type: IRType | null | undefined): string {
  if (!type) return '?';
  if (type instanceof TupleType) return `tuple(${(type as unknown as { types: readonly IRType[] }).types.map(describeType).join(', ')})`;
  const tensor = type as Partial<TensorType>;
  if (tensor.shape !== undefined && tensor.dtype !== undefined) {
    return `[${tensor.shape.join(',')}]:${tensor.dtype}`;
  }
  return String(type);
}

export function indexSelectGatherOpts(operandType: TensorType, dim: number, indicesRank: number): GatherOpts {
  const rank = operandType.rank;
  const d = dim < 0 ? rank + dim : dim;
  const sliceSizes = operandType.shape.map((s, i) => i === d ? 1 : s);
  const offsetDims: number[] = [];
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

function bcastBatchDims(a: Shape, b: Shape): Dim[] {
  const n = Math.max(a.length, b.length);
  const out = new Array<Dim>(n);
  for (let i = 0; i < n; i++) {
    const av = i < a.length ? a[a.length - 1 - i] : 1;
    const bv = i < b.length ? b[b.length - 1 - i] : 1;
    out[n - 1 - i] = av === 1 ? bv : av;
  }
  return out;
}

export class IRBuilder {
  func: GraphFunction;
  block: Block;
  location: Location | null;
  private _insertionPoint: Operation | null;

  constructor(func: GraphFunction) {
    this.func = func;
    this.block = func.entryBlock;
    this.location = null;
    this._insertionPoint = null;
  }

  withLocation<T>(location: Location | null, body: () => T): T {
    const previous = this.location;
    this.location = location;
    try {
      return body();
    } finally {
      this.location = previous;
    }
  }

  setInsertionPoint(op: Operation | null): void {
    this._insertionPoint = op;
  }

  setInsertionPointToEnd(): void {
    this._insertionPoint = null;
  }

  _insert(op: Operation): Operation {
    if (op.loc === null) op.loc = this.location === null ? currentLocation() : this.location;
    if (this._insertionPoint) {
      this.block.insertBefore(op, this._insertionPoint);
    } else {
      this.block.pushOp(op);
    }
    return op;
  }

  _buildOp(name: string, operands: readonly Value[], resultTypes: readonly IRType[], attributes: AttrRecord | ReadonlyMap<string, AttrValue> | null = null, regions: readonly Region[] | null = null): Operation {
    const op = new Operation(name, operands, resultTypes, attributes, regions);
    propagateSymbolicShapes(op);
    return this._insert(op);
  }

  _unifyOperandDtypes(name: string, operands: readonly Value[]): readonly Value[] {
    const indices = unifiedOperandIndices(name, operands.length);
    if (indices === null || indices.length < 2) return operands;

    let target: ScalarDType | null = null;
    for (const i of indices) {
      const type = operands[i]?.type;
      if (!(type instanceof TensorType)) return operands;
      target = target === null ? type.dtype : resultDtype(target, type.dtype) as ScalarDType;
    }

    let unified: Value[] | null = null;
    for (const i of indices) {
      if ((operands[i].type as TensorType).dtype === target) continue;
      if (unified === null) unified = [...operands];
      unified[i] = this.convert(operands[i], target!).getResult(0);
    }
    return unified || operands;
  }

  _inferAndBuild(name: string, rawOperands: readonly Value[], attributes: AttrRecord | ReadonlyMap<string, AttrValue> | null = null, regions: readonly Region[] | null = null, explicitResultTypes: readonly IRType[] | null = null): Operation {
    const opDef = registry.get(name);
    const operands = this._unifyOperandDtypes(name, rawOperands);
    let resultTypes: readonly IRType[] | null = explicitResultTypes;
    if (!resultTypes && opDef && opDef.inferResultTypes) {
      const operandTypes = operands.map(o => o.type);
      const attrMap = attributes instanceof Map ? attributes : toMap(attributes as AttrRecord | null);
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

  create(name: string, operands: readonly Value[], attributes: AttrRecord | ReadonlyMap<string, AttrValue> | null = null): Operation {
    return this._inferAndBuild(name, operands, attributes);
  }

  constant(value: AttrValue, tensorType: TensorType): Operation {
    return this._buildOp('constant', [], [tensorType], {
      value,
      tensor_type: tensorType
    });
  }

  scalarConstant(value: AttrValue, dtype: ScalarDType = ScalarType.F32): Operation {
    const tt = new TensorType([], dtype);
    return this.constant(value, tt);
  }

  tensorConstant(value: AttrValue, shape: Shape, dtype: ScalarDType = ScalarType.F32): Operation {
    const tt = new TensorType(shape, dtype);
    return this.constant(value, tt);
  }

  iota(dim: number, tensorType: TensorType): Operation {
    return this._buildOp('iota', [], [tensorType], {
      iota_dimension: dim,
      tensor_type: tensorType
    });
  }

  add(lhs: Value, rhs: Value): Operation { return this._inferAndBuild('add', [lhs, rhs]); }
  sub(lhs: Value, rhs: Value): Operation { return this._inferAndBuild('sub', [lhs, rhs]); }
  mul(lhs: Value, rhs: Value): Operation { return this._inferAndBuild('mul', [lhs, rhs]); }
  div(lhs: Value, rhs: Value): Operation { return this._inferAndBuild('div', [lhs, rhs]); }
  rem(lhs: Value, rhs: Value): Operation { return this._inferAndBuild('rem', [lhs, rhs]); }
  pow(lhs: Value, rhs: Value): Operation { return this._inferAndBuild('pow', [lhs, rhs]); }
  maximum(lhs: Value, rhs: Value): Operation { return this._inferAndBuild('maximum', [lhs, rhs]); }
  minimum(lhs: Value, rhs: Value): Operation { return this._inferAndBuild('minimum', [lhs, rhs]); }

  neg(x: Value): Operation { return this._inferAndBuild('neg', [x]); }
  stopGradient(x: Value): Operation { return this._inferAndBuild('stop_gradient', [x]); }
  reverse(x: Value, dimensions: readonly number[]): Operation { return this._inferAndBuild('reverse', [x], { dimensions }); }
  scaledDotProductAttention(q: Value, k: Value, v: Value, scale: number, causal = false): Operation {
    return this._inferAndBuild('scaled_dot_product_attention', [q, k, v], { scale, causal });
  }
  allReduce(x: Value, opts: AllReduceOpts = {}): Operation {
    return this._inferAndBuild('all_reduce', [x], { reduce_op: opts.reduceOp || 'sum', mesh_axis: opts.meshAxis ?? 0 });
  }
  allGather(x: Value, opts: AllGatherOpts = {}): Operation {
    return this._inferAndBuild('all_gather', [x], { mesh_axis: opts.meshAxis ?? 0, gather_dim: opts.gatherDim ?? 1 });
  }
  abs(x: Value): Operation { return this._inferAndBuild('abs', [x]); }
  exp(x: Value): Operation { return this._inferAndBuild('exp', [x]); }
  log(x: Value): Operation { return this._inferAndBuild('log', [x]); }
  sqrt(x: Value): Operation { return this._inferAndBuild('sqrt', [x]); }
  rsqrt(x: Value): Operation { return this._inferAndBuild('rsqrt', [x]); }
  tanh(x: Value): Operation { return this._inferAndBuild('tanh', [x]); }
  sin(x: Value): Operation { return this._inferAndBuild('sin', [x]); }
  cos(x: Value): Operation { return this._inferAndBuild('cos', [x]); }
  floor(x: Value): Operation { return this._inferAndBuild('floor', [x]); }
  ceil(x: Value): Operation { return this._inferAndBuild('ceil', [x]); }
  sign(x: Value): Operation { return this._inferAndBuild('sign', [x]); }
  erf(x: Value): Operation { return this._inferAndBuild('erf', [x]); }
  log2(x: Value): Operation { return this._inferAndBuild('log2', [x]); }
  log10(x: Value): Operation { return this._inferAndBuild('log10', [x]); }
  exp2(x: Value): Operation { return this._inferAndBuild('exp2', [x]); }
  square(x: Value): Operation { return this._inferAndBuild('square', [x]); }
  reciprocal(x: Value): Operation { return this._inferAndBuild('reciprocal', [x]); }

  logicalNot(x: Value): Operation { return this._inferAndBuild('logical_not', [x]); }
  logicalAnd(lhs: Value, rhs: Value): Operation { return this._inferAndBuild('logical_and', [lhs, rhs]); }
  logicalOr(lhs: Value, rhs: Value): Operation { return this._inferAndBuild('logical_or', [lhs, rhs]); }

  compare(lhs: Value, rhs: Value, direction: string): Operation {
    return this._inferAndBuild('compare', [lhs, rhs], { direction });
  }

  select(pred: Value, onTrue: Value, onFalse: Value): Operation {
    return this._inferAndBuild('select', [pred, onTrue, onFalse]);
  }

  clamp(lo: Value, x: Value, hi: Value): Operation {
    return this._inferAndBuild('clamp', [lo, x, hi]);
  }

  broadcast(input: Value, resultShape: Shape, broadcastDimensions: readonly number[]): Operation {
    return this._inferAndBuild('broadcast_in_dim', [input], {
      result_shape: resultShape,
      broadcast_dimensions: broadcastDimensions
    });
  }

  reshape(input: Value, newShape: Shape): Operation {
    return this._inferAndBuild('reshape', [input], { new_shape: newShape });
  }

  transpose(input: Value, permutation: readonly number[]): Operation {
    return this._inferAndBuild('transpose', [input], { permutation });
  }

  slice(input: Value, starts: readonly number[], limits: readonly number[], strides: readonly number[] | null = null): Operation {
    const attrs: AttrRecord = { starts, limits };
    if (strides) attrs.strides = strides;
    return this._inferAndBuild('slice', [input], attrs);
  }

  concat(inputs: readonly Value[], dimension: number): Operation {
    return this._inferAndBuild('concat', inputs, { dimension });
  }

  pad(input: Value, paddingValue: Value, low: readonly number[], high: readonly number[], interior: readonly number[] | null = null): Operation {
    const attrs: AttrRecord = { low, high };
    if (interior) attrs.interior = interior;
    return this._inferAndBuild('pad', [input, paddingValue], attrs);
  }

  reduce(input: Value, initValue: Value, dimensions: readonly number[], reduceType: string): Operation {
    const scalarType = new TensorType([], tensorTypeOf(input).dtype);
    const combinerRegion = new Region();
    const combinerBlock = new Block([scalarType, scalarType]);
    combinerRegion.addBlock(combinerBlock);
    const op = this._inferAndBuild('reduce', [input, initValue], {
      dimensions,
      reduce_type: reduceType
    }, [combinerRegion]);
    return op;
  }

  dot(lhs: Value, rhs: Value, lhsContracting: readonly number[], rhsContracting: readonly number[], lhsBatch: readonly number[] = [], rhsBatch: readonly number[] = []): Operation {
    return this._inferAndBuild('dot', [lhs, rhs], {
      lhs_contracting: lhsContracting,
      rhs_contracting: rhsContracting,
      lhs_batch: lhsBatch,
      rhs_batch: rhsBatch
    });
  }

  matmul(lhs: Value, rhs: Value): Operation {
    const lhsRank = tensorTypeOf(lhs).rank;
    const rhsRank = tensorTypeOf(rhs).rank;
    if (lhsRank === 1 && rhsRank === 1) {
      return this.dot(lhs, rhs, [0], [0]);
    }

    let L = lhs;
    let R = rhs;
    let squeezeLhs = false;
    let squeezeRhs = false;
    if (lhsRank === 1) {
      L = this.reshape(lhs, [1, tensorTypeOf(lhs).shape[0]]).getResult(0);
      squeezeLhs = true;
    }
    if (rhsRank === 1) {
      R = this.reshape(rhs, [tensorTypeOf(rhs).shape[0], 1]).getResult(0);
      squeezeRhs = true;
    }

    const lr = tensorTypeOf(L).rank;
    const rr = tensorTypeOf(R).rank;
    const rBatch = tensorTypeOf(R).shape.slice(0, rr - 2);

    let result: Operation;
    if (rBatch.length === 0) {
      result = this.dot(L, R, [lr - 1], [0]);
    } else {
      const lBatch = tensorTypeOf(L).shape.slice(0, lr - 2);
      const batch = bcastBatchDims(lBatch, rBatch);
      const nb = batch.length;
      const Lb = this._broadcastBatch(L, lBatch, batch);
      const Rb = this._broadcastBatch(R, rBatch, batch);
      const range = Array.from({ length: nb }, (_, i) => i);
      result = this.dot(Lb, Rb, [nb + 1], [nb], range, range);
    }

    if (!squeezeLhs && !squeezeRhs) return result;
    const outShape = tensorTypeOf(result.getResult(0)).shape;
    const drop = new Set<number>();
    if (squeezeRhs) drop.add(outShape.length - 1);
    if (squeezeLhs) drop.add(outShape.length - 2);
    return this.reshape(result.getResult(0), outShape.filter((_, i) => !drop.has(i)));
  }

  _broadcastBatch(value: Value, batch: Shape, targetBatch: Shape): Value {
    const matrixDims = tensorTypeOf(value).shape.slice(tensorTypeOf(value).rank - 2);
    const targetShape = [...targetBatch, ...matrixDims];
    if (batch.length === targetBatch.length && batch.every((d, i) => d === targetBatch[i])) {
      return value;
    }
    const offset = targetBatch.length - batch.length;
    const broadcastDims: number[] = [];
    for (let i = 0; i < batch.length; i++) broadcastDims.push(offset + i);
    broadcastDims.push(targetShape.length - 2, targetShape.length - 1);
    return this.broadcast(value, targetShape, broadcastDims).getResult(0);
  }

  conv(input: Value, kernel: Value, strides: readonly number[], padding: readonly number[], opts: ConvOpts = {}): Operation {
    return this._inferAndBuild('conv', [input, kernel], {
      strides,
      padding,
      dilation: opts.dilation || strides.map(() => 1),
      groups: opts.groups || 1,
      input_layout: opts.inputLayout || 'NCHW',
      kernel_layout: opts.kernelLayout || 'OIHW'
    });
  }

  convert(input: Value, targetDtype: ScalarDType): Operation {
    return this._inferAndBuild('convert', [input], { target_dtype: targetDtype });
  }

  customCall(name: string, operands: readonly Value[], resultTypes: readonly IRType[], backendConfig: AttrValue = null): Operation {
    const attrs: AttrRecord = { call_target_name: name };
    if (backendConfig) attrs.backend_config = backendConfig;
    return this._buildOp('custom_call', operands, resultTypes, attrs);
  }

  fusion(operands: readonly Value[], resultTypes: readonly IRType[], fusionKind: string, bodyBuilder: FusionBodyBuilder | null): Operation {
    const bodyRegion = new Region();
    const bodyBlock = new Block(operands.map(o => o.type));
    bodyRegion.addBlock(bodyBlock);
    const op = this._buildOp('fusion', operands, resultTypes, { fusion_kind: fusionKind }, [bodyRegion]);
    if (bodyBuilder) {
      const innerBuilder = new IRBuilder(this.func);
      innerBuilder.block = bodyBlock;
      bodyBuilder(innerBuilder, bodyBlock.arguments);
    }
    return op;
  }

  ifOp(predicate: Value, resultTypes: readonly IRType[], thenBuilder: BranchBuilder | null, elseBuilder: BranchBuilder | null): Operation {
    const thenRegion = new Region();
    const thenBlock = new Block([]);
    thenRegion.addBlock(thenBlock);
    const elseRegion = new Region();
    const elseBlock = new Block([]);
    elseRegion.addBlock(elseBlock);
    const op = this._buildOp('if', [predicate], resultTypes, null, [thenRegion, elseRegion]);
    if (thenBuilder) {
      const tb = new IRBuilder(this.func);
      tb.block = thenBlock;
      thenBuilder(tb);
    }
    if (elseBuilder) {
      const eb = new IRBuilder(this.func);
      eb.block = elseBlock;
      elseBuilder(eb);
    }
    return op;
  }

  whileOp(initValues: readonly Value[], condBuilder: RegionBodyBuilder | null, bodyBuilder: RegionBodyBuilder | null): Operation {
    const types = initValues.map(v => v.type);
    const condRegion = new Region();
    const condBlock = new Block(types);
    condRegion.addBlock(condBlock);
    const bodyRegion = new Region();
    const bodyBlock = new Block(types);
    bodyRegion.addBlock(bodyBlock);
    const op = this._buildOp('while', initValues, types, null, [condRegion, bodyRegion]);
    if (condBuilder) {
      const cb = new IRBuilder(this.func);
      cb.block = condBlock;
      condBuilder(cb, condBlock.arguments);
    }
    if (bodyBuilder) {
      const bb = new IRBuilder(this.func);
      bb.block = bodyBlock;
      bodyBuilder(bb, bodyBlock.arguments);
    }
    return op;
  }

  scanOp(xsValues: readonly Value[], initCarryValues: readonly Value[], bodyFn: ScanBodyFn): Operation {
    const xtTypes = xsValues.map(v => tensorTypeOf(v).withShape(tensorTypeOf(v).shape.slice(1)));
    const carryTypes = initCarryValues.map(v => v.type);
    const bodyRegion = new Region();
    const bodyBlock = new Block([...xtTypes, ...carryTypes]);
    bodyRegion.addBlock(bodyBlock);
    const bb = new IRBuilder(this.func);
    bb.block = bodyBlock;
    const xtArgs = bodyBlock.arguments.slice(0, xtTypes.length);
    const carryArgs = bodyBlock.arguments.slice(xtTypes.length);
    const [newCarry, ys] = bodyFn(bb, xtArgs, carryArgs);
    bb.yieldOp([...newCarry, ...ys]);
    if (xsValues.length === 0) throw new Error('scanOp requires at least one xs input');
    const length = tensorTypeOf(xsValues[0]).shape[0];
    if (typeof length !== 'number' || length < 0) {
      throw new Error(`scanOp requires a static, non-negative leading dim on xs, got ${length}`);
    }
    for (let i = 1; i < xsValues.length; i++) {
      if (tensorTypeOf(xsValues[i]).shape[0] !== length) {
        throw new Error('scanOp requires all xs inputs to share the same leading length');
      }
    }
    const yTypes = ys.map(v => tensorTypeOf(v).withShape([length, ...tensorTypeOf(v).shape]));
    return this._buildOp('scan', [...xsValues, ...initCarryValues],
      [...carryTypes, ...yTypes],
      { num_carry: initCarryValues.length, num_xs: xsValues.length },
      [bodyRegion]);
  }

  returnOp(values: readonly Value[]): Operation {
    return this._buildOp('return', values, []);
  }

  yieldOp(values: readonly Value[]): Operation {
    return this._buildOp('yield', values, []);
  }

  relu(x: Value): Operation {
    const zero = this.scalarConstant(0, tensorTypeOf(x).dtype);
    const broadcastZero = this.broadcast(
      zero.getResult(0),
      tensorTypeOf(x).shape,
      []
    );
    return this.maximum(x, broadcastZero.getResult(0));
  }

  softmax(input: Value, axis = -1): Operation {
    const dim = axis < 0 ? tensorTypeOf(input).rank + axis : axis;
    return this._inferAndBuild('softmax', [input], { axis: dim });
  }

  logSoftmax(input: Value, axis = -1): Operation {
    const dim = axis < 0 ? tensorTypeOf(input).rank + axis : axis;
    return this._inferAndBuild('log_softmax', [input], { axis: dim });
  }

  sigmoid(x: Value): Operation {
    return this._inferAndBuild('sigmoid', [x]);
  }

  gelu(x: Value): Operation {
    return this._inferAndBuild('gelu', [x]);
  }

  silu(x: Value): Operation {
    return this._inferAndBuild('silu', [x]);
  }

  elu(x: Value, alpha = 1.0): Operation {
    return this._inferAndBuild('elu', [x], { alpha });
  }

  leakyRelu(x: Value, negativeSlope = 0.01): Operation {
    return this._inferAndBuild('leaky_relu', [x], { negative_slope: negativeSlope });
  }

  celu(x: Value, alpha = 1.0): Operation {
    return this._inferAndBuild('celu', [x], { alpha });
  }

  selu(x: Value): Operation { return this._inferAndBuild('selu', [x]); }
  mish(x: Value): Operation { return this._inferAndBuild('mish', [x]); }
  hardswish(x: Value): Operation { return this._inferAndBuild('hardswish', [x]); }
  hardsigmoid(x: Value): Operation { return this._inferAndBuild('hardsigmoid', [x]); }

  layernorm(input: Value, gamma: Value, beta: Value, axis = -1, eps = 1e-5): Operation {
    const dim = axis < 0 ? tensorTypeOf(input).rank + axis : axis;
    return this._inferAndBuild('layer_norm', [input, gamma, beta], { axis: dim, epsilon: eps });
  }

  batchnorm(input: Value, gamma: Value, beta: Value, mean: Value, variance: Value, axis = 1, eps = 1e-5): Operation {
    return this._inferAndBuild('batch_norm', [input, gamma, beta, mean, variance], { axis, epsilon: eps });
  }

  where(condition: Value, x: Value, y: Value): Operation {
    return this._inferAndBuild('where', [condition, x, y]);
  }

  split(input: Value, dimension: number, splitSizes: readonly number[]): Operation {
    return this._inferAndBuild('split', [input], { dimension, split_sizes: splitSizes });
  }

  oneHot(indices: Value, depth: number, opts: OneHotOpts = {}): Operation {
    return this._inferAndBuild('one_hot', [indices], {
      depth,
      axis: opts.axis ?? -1,
      on_value: opts.onValue ?? 1,
      off_value: opts.offValue ?? 0,
      dtype: opts.dtype || tensorTypeOf(indices).dtype,
    });
  }

  embedding(weight: Value, indices: Value): Operation {
    return this._inferAndBuild('embedding', [weight, indices]);
  }

  gather(operand: Value, indices: Value, opts: GatherOpts): Operation {
    return this._inferAndBuild('gather', [operand, indices], {
      offset_dims: opts.offsetDims,
      collapsed_slice_dims: opts.collapsedSliceDims,
      start_index_map: opts.startIndexMap,
      index_vector_dim: opts.indexVectorDim,
      slice_sizes: opts.sliceSizes,
    });
  }

  scatter(operand: Value, indices: Value, updates: Value, opts: ScatterOpts): Operation {
    const scalarType = new TensorType([], tensorTypeOf(operand).dtype);
    const combinerRegion = new Region();
    combinerRegion.addBlock(new Block([scalarType, scalarType]));
    return this._inferAndBuild('scatter', [operand, indices, updates], {
      update_window_dims: opts.updateWindowDims,
      inserted_window_dims: opts.insertedWindowDims,
      scatter_dims_to_operand_dims: opts.scatterDimsToOperandDims,
      index_vector_dim: opts.indexVectorDim,
    }, [combinerRegion]);
  }

  scatterAdd(operand: Value, indices: Value, updates: Value, opts: ScatterOpts): Operation {
    return this.scatter(operand, indices, updates, opts);
  }

  _dimCoordIndices(index: Value, dim: number, gridShape: Shape): Value {
    const r = gridShape.length;
    const idxI32 = tensorTypeOf(index).dtype === 'i32' ? index : this.convert(index, 'i32').getResult(0);
    const coordShape = [...gridShape, 1];
    const coords: Value[] = [];
    for (let j = 0; j < r; j++) {
      const cj = j === dim ? idxI32 : this.iota(j, new TensorType(gridShape, 'i32')).getResult(0);
      coords.push(this.reshape(cj, coordShape).getResult(0));
    }
    return this.concat(coords, r).getResult(0);
  }

  gatherDim(operand: Value, index: Value, dim: number): Operation {
    const r = tensorTypeOf(operand).rank;
    const d = dim < 0 ? r + dim : dim;
    const range: number[] = [];
    const ones: number[] = [];
    for (let j = 0; j < r; j++) { range.push(j); ones.push(1); }
    const fullIdx = this._dimCoordIndices(index, d, tensorTypeOf(index).shape);
    return this.gather(operand, fullIdx, {
      offsetDims: [],
      collapsedSliceDims: range,
      startIndexMap: range,
      indexVectorDim: r,
      sliceSizes: ones,
    });
  }

  scatterAddDim(operand: Value, index: Value, updates: Value, dim: number): Operation {
    const r = tensorTypeOf(operand).rank;
    const d = dim < 0 ? r + dim : dim;
    const range: number[] = [];
    for (let j = 0; j < r; j++) range.push(j);
    const fullIdx = this._dimCoordIndices(index, d, tensorTypeOf(index).shape);
    return this.scatterAdd(operand, fullIdx, updates, {
      updateWindowDims: [],
      insertedWindowDims: range,
      scatterDimsToOperandDims: range,
      indexVectorDim: r,
    });
  }

  argmax(input: Value, axis: number, keepDims = false): Operation {
    const dim = axis < 0 ? tensorTypeOf(input).rank + axis : axis;
    return this._inferAndBuild('argmax', [input], { axis: dim, keep_dims: keepDims });
  }

  argmin(input: Value, axis: number, keepDims = false): Operation {
    const dim = axis < 0 ? tensorTypeOf(input).rank + axis : axis;
    return this._inferAndBuild('argmin', [input], { axis: dim, keep_dims: keepDims });
  }

  pool2d(input: Value, poolType: string, kernelSize: readonly number[], strides: readonly number[], padding: readonly number[], opts: Pool2dOpts = {}): Operation {
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

  resize(input: Value, outputSize: readonly number[], method: string, opts: ResizeOpts = {}): Operation {
    return this._inferAndBuild('resize', [input], {
      output_size: outputSize,
      method,
      coordinate_mode: opts.coordinateMode || 'asymmetric',
      layout: opts.layout || 'NCHW',
    });
  }
}

export function broadcastDimsExcluding(rank: number, excludedDim: number): number[] {
  const dims: number[] = [];
  for (let i = 0; i < rank; i++) {
    if (i !== excludedDim) dims.push(i);
  }
  return dims;
}

function toMap(obj: AttrRecord | null): Map<string, AttrValue> {
  if (!obj) return new Map();
  if (obj instanceof Map) return obj;
  const m = new Map<string, AttrValue>();
  for (const k of Object.keys(obj)) m.set(k, obj[k]);
  return m;
}

export function buildFunction(name: string, inputTypes: readonly IRType[], outputTypes: readonly IRType[], bodyFn: FunctionBodyFn): GraphFunction {
  const func = new GraphFunction(name, inputTypes, outputTypes);
  const builder = new IRBuilder(func);
  bodyFn(builder, func.args);
  return func;
}

export function buildModule(name: string, funcBuilders: readonly FunctionSpec[]): GraphModule {
  const mod = new GraphModule(name);
  for (const [fname, inputTypes, outputTypes, bodyFn] of funcBuilders) {
    mod.addFunction(buildFunction(fname, inputTypes, outputTypes, bodyFn));
  }
  return mod;
}
