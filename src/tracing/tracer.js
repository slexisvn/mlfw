import { IRBuilder, buildFunction } from '../compiler/ir/graph/builder.js';
import { GraphModule } from '../compiler/ir/graph/module.js';
import { GraphFunction } from '../compiler/ir/graph/function.js';
import { TensorType } from '../compiler/ir/graph/types.js';
import { SymbolicTensor } from './symbolic_tensor.js';
import { ShapeEnv } from './shape_env.js';

const _BUILDER_METHOD_MAP = {
  matmul: (b, args) => b.matmul(args[0], args[1]),
  softmax: (b, args, a) => b.softmax(args[0], a?.dim ?? -1),
  log_softmax: (b, args, a) => b.logSoftmax(args[0], a?.dim ?? -1),
  layer_norm: (b, args, a) => b.layernorm(args[0], args[1], args[2], a?.axis ?? -1, a?.eps ?? 1e-5),
  batch_norm: (b, args, a) => b.batchnorm(args[0], args[1], args[2], args[3], args[4], a?.axis ?? 1, a?.eps ?? 1e-5),
  embedding: (b, args) => b.embedding(args[0], args[1]),
  relu: (b, args) => b.relu(args[0]),
  sigmoid: (b, args) => b.sigmoid(args[0]),
  gelu: (b, args) => b.gelu(args[0]),
  silu: (b, args) => b.silu(args[0]),
  conv2d: (b, args, a) => b.conv(args[0], args[1], a?.strides ?? [1,1], a?.padding ?? [[0,0],[0,0]]),
  maximum: (b, args) => b.maximum(args[0], args[1]),
  minimum: (b, args) => b.minimum(args[0], args[1]),
  sum: (b, args, a) => {
    const dims = a?.dim;
    const dimensions = dims !== undefined && dims !== null
      ? (Array.isArray(dims) ? dims : [dims])
      : Array.from({ length: args[0].type.rank }, (_, i) => i);
    const initConst = b.scalarConstant(0, args[0].type.dtype);
    return b.reduce(args[0], initConst.getResult(0), dimensions, 'sum');
  },
  mean: (b, args, a) => {
    const dims = a?.dim;
    const dimensions = dims !== undefined && dims !== null
      ? (Array.isArray(dims) ? dims : [dims])
      : Array.from({ length: args[0].type.rank }, (_, i) => i);
    const initConst = b.scalarConstant(0, args[0].type.dtype);
    return b.reduce(args[0], initConst.getResult(0), dimensions, 'mean');
  },
  eq: (b, args) => b.compare(args[0], args[1], 'eq'),
  ne: (b, args) => b.compare(args[0], args[1], 'ne'),
  lt: (b, args) => b.compare(args[0], args[1], 'lt'),
  le: (b, args) => b.compare(args[0], args[1], 'le'),
  gt: (b, args) => b.compare(args[0], args[1], 'gt'),
  ge: (b, args) => b.compare(args[0], args[1], 'ge'),
  transpose: (b, args, a) => {
    const rank = args[0].type.rank;
    const d0 = a?.dim0 ?? 0;
    const d1 = a?.dim1 ?? 1;
    const perm = Array.from({ length: rank }, (_, i) => i);
    perm[d0] = d1;
    perm[d1] = d0;
    return b.transpose(args[0], perm);
  },
};

let _activeTracer = null;

export function getActiveTracer() {
  return _activeTracer;
}

export class Tracer {
  constructor(name) {
    this._name = name || 'traced';
    this._shapeEnv = new ShapeEnv();
    this._inputTypes = [];
    this._outputTypes = [];
    this._inputs = [];
    this._func = null;
    this._builder = null;
    this._module = null;
    this._capturedParams = new Map();
    this._capturedParamOrder = [];
  }

  get shapeEnv() {
    return this._shapeEnv;
  }

  createInput(shape, dtype) {
    const tensorType = new TensorType(shape, dtype);
    this._inputTypes.push(tensorType);
    return { shape: [...shape], dtype, tensorType };
  }

  _initGraph() {
    const placeholderOutputTypes = [];
    this._func = new GraphFunction(this._name, this._inputTypes, placeholderOutputTypes);
    this._builder = new IRBuilder(this._func);
    this._module = new GraphModule(this._name);

    const symbolicInputs = [];
    const args = this._func.args;
    for (let i = 0; i < args.length; i++) {
      const irValue = args[i];
      const tt = this._inputTypes[i];
      const st = new SymbolicTensor(irValue, tt.shape, tt.dtype, this);
      symbolicInputs.push(st);
    }
    this._inputs = symbolicInputs;
    return symbolicInputs;
  }

  recordOp(opName, tensorArgs, attrs) {
    const irOperands = [];
    for (const arg of tensorArgs) {
      if (arg instanceof SymbolicTensor) {
        irOperands.push(arg.irValue);
      }
    }

    let op;
    const alias = _BUILDER_METHOD_MAP[opName];
    if (alias) {
      op = alias(this._builder, irOperands, attrs);
    } else if (typeof this._builder[opName] === 'function') {
      op = this._builder[opName](...irOperands);
    } else {
      op = this._builder._inferAndBuild(opName, irOperands, attrs || null);
    }

    const resultValue = op.getResult(0);
    const resultType = resultValue.type;

    return new SymbolicTensor(
      resultValue,
      resultType.shape,
      resultType.dtype,
      this
    );
  }

  captureConstant(tensor) {
    let cached = this._capturedParams.get(tensor);
    if (cached) return cached;

    const tt = new TensorType(tensor.shape, tensor.dtype);
    this._func.inputTypes = Object.freeze([...this._func.inputTypes, tt]);

    const block = this._func.entryBlock;
    const irValue = block.addArgument(tt);

    const sym = new SymbolicTensor(irValue, tensor.shape, tensor.dtype, this);
    this._capturedParams.set(tensor, sym);
    this._capturedParamOrder.push(tensor);
    return sym;
  }

  get capturedParams() {
    return this._capturedParamOrder;
  }

  markOutput(symbolicTensor) {
    if (symbolicTensor instanceof SymbolicTensor) {
      this._builder.returnOp([symbolicTensor.irValue]);
    }
    this._outputTypes = [new TensorType(symbolicTensor.shape, symbolicTensor.dtype)];
  }

  markOutputs(symbolicTensors) {
    const irValues = symbolicTensors.map(st => st.irValue);
    this._builder.returnOp(irValues);
    this._outputTypes = symbolicTensors.map(
      st => new TensorType(st.shape, st.dtype)
    );
  }

  getGraphModule() {
    this._func.outputTypes = Object.freeze(this._outputTypes);
    this._module.addFunction(this._func);
    return this._module;
  }

  activate() {
    _activeTracer = this;
  }

  deactivate() {
    if (_activeTracer === this) _activeTracer = null;
  }
}
