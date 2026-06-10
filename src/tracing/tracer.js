import { IRBuilder } from '../compiler/ir/graph/builder.js';
import { GraphModule } from '../compiler/ir/graph/module.js';
import { GraphFunction } from '../compiler/ir/graph/function.js';
import { TensorType, DYNAMIC } from '../compiler/ir/graph/types.js';
import { registry } from '../compiler/ir/graph/ops.js';
import { SymbolicTensor } from './symbolic_tensor.js';
import { ShapeEnv } from './shape_env.js';
import { reduceInitValue } from '../backend/dtype_map.js';

function _traceReduce(b, args, a, reduceType) {
  const rank = args[0].type.rank;
  const dims = a?.dim;
  const dimensions = dims !== undefined && dims !== null
    ? (Array.isArray(dims) ? dims : [dims]).map(d => d < 0 ? rank + d : d)
    : Array.from({ length: rank }, (_, i) => i);
  const initConst = b.scalarConstant(reduceInitValue(reduceType, args[0].type.dtype), args[0].type.dtype);
  const reduced = b.reduce(args[0], initConst.getResult(0), dimensions, reduceType);
  if (!a?.keepdim) return reduced;
  const dimSet = new Set(dimensions);
  const newShape = args[0].type.shape.map((d, i) => dimSet.has(i) ? 1 : d);
  return b.reshape(reduced.getResult(0), newShape);
}

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
  pool2d: (b, args, a) => b.pool2d(args[0], a?.pool_type ?? 'max', a?.kernel_size ?? [2,2], a?.strides ?? [2,2], a?.padding ?? [[0,0],[0,0]]),
  maximum: (b, args) => b.maximum(args[0], args[1]),
  minimum: (b, args) => b.minimum(args[0], args[1]),
  sum: (b, args, a) => _traceReduce(b, args, a, 'sum'),
  mean: (b, args, a) => _traceReduce(b, args, a, 'mean'),
  max: (b, args, a) => _traceReduce(b, args, a, 'max'),
  min: (b, args, a) => _traceReduce(b, args, a, 'min'),
  prod: (b, args, a) => _traceReduce(b, args, a, 'prod'),
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
  permute: (b, args, a) => b.transpose(args[0], a.dims),
  reshape: (b, args, a) => b.reshape(args[0], a.new_shape),
  slice: (b, args, a) => b.slice(args[0], a.starts, a.limits, a.strides),
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
    this._inputSymShapes = [];
    this._outputTypes = [];
    this._outputSymShapes = [];
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

  createInput(shape, dtype, dynamicDims) {
    const inputIdx = this._inputTypes.length;
    const { irShape, symShape } = this._shapeEnv.produceShapeSpec(inputIdx, shape, dynamicDims);

    if (dynamicDims) {
      for (let i = 0; i < symShape.length; i++) {
        if (typeof symShape[i] === 'string') {
          this._shapeEnv.guardRelation(symShape[i], 'gt', 0);
        }
      }
    }

    const tensorType = new TensorType(irShape, dtype);
    this._inputTypes.push(tensorType);
    this._inputSymShapes.push(symShape);
    return { shape: irShape, dtype, tensorType };
  }

  _initGraph() {
    this._func = new GraphFunction(this._name, this._inputTypes, []);
    this._builder = new IRBuilder(this._func);
    this._module = new GraphModule(this._name);

    const symbolicInputs = [];
    const args = this._func.args;
    for (let i = 0; i < args.length; i++) {
      const irValue = args[i];
      const tt = this._inputTypes[i];
      const st = new SymbolicTensor(irValue, tt.shape, tt.dtype, this, this._inputSymShapes[i]);
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
    const resultSymShape = this._propagateSymbolicShape(opName, op, tensorArgs, resultType);

    return new SymbolicTensor(
      resultValue,
      resultType.shape,
      resultType.dtype,
      this,
      resultSymShape
    );
  }

  _propagateSymbolicShape(opName, op, tensorArgs, resultType) {
    const symTensorArgs = tensorArgs.filter(a => a instanceof SymbolicTensor);

    const def = registry.get(op.opName || opName);
    if (def && def.propagateSymbolicShapes) {
      const shapeMap = new Map();
      for (const arg of symTensorArgs) {
        shapeMap.set(arg.irValue, arg.symbolicShape);
      }
      const propagated = def.propagateSymbolicShapes(op, shapeMap);
      if (propagated && propagated[0]) return propagated[0];
    }

    const resultShape = resultType.shape;
    const outSym = new Array(resultShape.length);

    for (let i = 0; i < resultShape.length; i++) {
      if (resultShape[i] !== DYNAMIC) {
        outSym[i] = resultShape[i];
        continue;
      }

      let resolved = null;
      for (const arg of symTensorArgs) {
        const argSymShape = arg.symbolicShape;
        if (!argSymShape) continue;
        const offset = resultShape.length - argSymShape.length;
        const srcIdx = i - offset;
        if (srcIdx >= 0 && srcIdx < argSymShape.length && typeof argSymShape[srcIdx] === 'string') {
          resolved = argSymShape[srcIdx];
          break;
        }
      }

      outSym[i] = resolved !== null ? resolved : DYNAMIC;
    }

    return outSym;
  }

  captureConstant(tensor) {
    let cached = this._capturedParams.get(tensor);
    if (cached) return cached;

    if (tensor.shape.length === 0 && tensor.data) {
      const value = tensor.data[0];
      const op = this._builder.scalarConstant(value, tensor.dtype);
      const irValue = op.getResult(0);
      const sym = new SymbolicTensor(irValue, [], tensor.dtype, this, []);
      this._capturedParams.set(tensor, sym);
      return sym;
    }

    const tt = new TensorType(tensor.shape, tensor.dtype);
    this._func.inputTypes = Object.freeze([...this._func.inputTypes, tt]);

    const block = this._func.entryBlock;
    const irValue = block.addArgument(tt);

    const sym = new SymbolicTensor(irValue, tensor.shape, tensor.dtype, this, [...tensor.shape]);
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
      this._outputSymShapes = [symbolicTensor.symbolicShape];
    }
    this._outputTypes = [new TensorType(symbolicTensor.shape, symbolicTensor.dtype)];
  }

  markOutputs(symbolicTensors) {
    const irValues = symbolicTensors.map(st => st.irValue);
    this._builder.returnOp(irValues);
    this._outputTypes = symbolicTensors.map(
      st => new TensorType(st.shape, st.dtype)
    );
    this._outputSymShapes = symbolicTensors.map(
      st => st instanceof SymbolicTensor ? st.symbolicShape : [...st.shape]
    );
  }

  get outputSymShapes() {
    return this._outputSymShapes;
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
