import { TensorType, ScalarType, DYNAMIC, TupleType } from './types.js';
import { Operation } from './operation.js';
import { Block, Region } from './block.js';
import { GraphFunction } from './function.js';
import { GraphModule } from './module.js';
import { registry } from './ops.js';

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
      throw new Error(`Cannot infer result types for op '${name}'`);
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
    if (lhsRank === 2 && rhsRank === 2) {
      return this.dot(lhs, rhs, [1], [0]);
    }
    if (lhsRank === 3 && rhsRank === 3) {
      return this.dot(lhs, rhs, [2], [1], [0], [0]);
    }
    return this.dot(lhs, rhs, [lhsRank - 1], [rhsRank - 2]);
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
    const rank = input.type.rank;
    const dim = axis < 0 ? rank + axis : axis;
    const maxOp = this.reduce(input, this.scalarConstant(-Infinity, input.type.dtype).getResult(0), [dim], 'max');
    const broadcastMax = this.broadcast(maxOp.getResult(0), input.type.shape, broadcastDimsExcluding(rank, dim));
    const shifted = this.sub(input, broadcastMax.getResult(0));
    const exps = this.exp(shifted.getResult(0));
    const sumOp = this.reduce(exps.getResult(0), this.scalarConstant(0, input.type.dtype).getResult(0), [dim], 'sum');
    const broadcastSum = this.broadcast(sumOp.getResult(0), input.type.shape, broadcastDimsExcluding(rank, dim));
    return this.div(exps.getResult(0), broadcastSum.getResult(0));
  }

  layernorm(input, gamma, beta, axis = -1, eps = 1e-5) {
    const rank = input.type.rank;
    const dim = axis < 0 ? rank + axis : axis;
    const meanOp = this.reduce(input, this.scalarConstant(0, input.type.dtype).getResult(0), [dim], 'mean');
    const broadcastMean = this.broadcast(meanOp.getResult(0), input.type.shape, broadcastDimsExcluding(rank, dim));
    const centered = this.sub(input, broadcastMean.getResult(0));
    const sq = this.mul(centered.getResult(0), centered.getResult(0));
    const variance = this.reduce(sq.getResult(0), this.scalarConstant(0, input.type.dtype).getResult(0), [dim], 'mean');
    const epsConst = this.scalarConstant(eps, input.type.dtype);
    const broadcastEps = this.broadcast(epsConst.getResult(0), variance.getResult(0).type.shape, []);
    const varPlusEps = this.add(variance.getResult(0), broadcastEps.getResult(0));
    const rstd = this.rsqrt(varPlusEps.getResult(0));
    const broadcastRstd = this.broadcast(rstd.getResult(0), input.type.shape, broadcastDimsExcluding(rank, dim));
    const normalized = this.mul(centered.getResult(0), broadcastRstd.getResult(0));
    const broadcastGamma = this.broadcast(gamma, input.type.shape, [dim]);
    const scaled = this.mul(normalized.getResult(0), broadcastGamma.getResult(0));
    const broadcastBeta = this.broadcast(beta, input.type.shape, [dim]);
    return this.add(scaled.getResult(0), broadcastBeta.getResult(0));
  }
}

function broadcastDimsExcluding(rank, excludedDim) {
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
