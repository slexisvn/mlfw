import { IRBuilder } from '../compiler/ir/graph/builder.js';
import { GraphModule } from '../compiler/ir/graph/module.js';
import { GraphFunction } from '../compiler/ir/graph/function.js';
import { TensorType, DYNAMIC } from '../compiler/ir/graph/types.js';
import { registry } from '../compiler/ir/graph/ops.js';
import { SymbolicTensor } from './symbolic_tensor.js';
import { ShapeEnv } from './shape_env.js';
import { buildMappedOp } from '../tensor/ops/ir_mapping.js';

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
    this._func.inputTypes = [...this._func.inputTypes];
    this._builder = new IRBuilder(this._func);
    this._module = new GraphModule(this._name);

    const symbolicInputs = [];
    const args = this._func.args;
    for (let i = 0; i < args.length; i++) {
      const irValue = args[i];
      const tt = this._inputTypes[i];
      irValue.symbolicShape = this._inputSymShapes[i];
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

    const op = buildMappedOp(this._builder, opName, irOperands, attrs);

    const results = [];
    for (let i = 0; i < op.numResults; i++) {
      const resultValue = op.getResult(i);
      const resultType = resultValue.type;
      const resultSymShape = this._propagateSymbolicShape(opName, op, tensorArgs, resultType, i);
      resultValue.symbolicShape = resultSymShape;
      results.push(new SymbolicTensor(resultValue, resultType.shape, resultType.dtype, this, resultSymShape));
    }

    return results.length === 1 ? results[0] : results;
  }

  _propagateSymbolicShape(opName, op, tensorArgs, resultType, resultIndex = 0) {
    const symTensorArgs = tensorArgs.filter(a => a instanceof SymbolicTensor);

    const def = registry.get(op.opName || opName);
    if (def && def.propagateSymbolicShapes) {
      const shapeMap = new Map();
      for (const arg of symTensorArgs) {
        shapeMap.set(arg.irValue, arg.symbolicShape);
      }
      const propagated = def.propagateSymbolicShapes(op, shapeMap);
      if (propagated && propagated[resultIndex]) return propagated[resultIndex];
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

  scan(xsValues, carryValues, stepFn) {
    const toIr = (s) => s instanceof SymbolicTensor ? s.irValue : this.captureConstant(s).irValue;
    const xsIr = xsValues.map(toIr);
    const carryIr = carryValues.map(toIr);
    const op = this._builder.scanOp(xsIr, carryIr, (bb, xtArgs, carryArgs) => {
      const saved = this._builder;
      this._builder = bb;
      try {
        const wrap = (v) => new SymbolicTensor(v, v.type.shape, v.type.dtype, this, [...v.type.shape]);
        const [newCarry, ys] = stepFn(carryArgs.map(wrap), xtArgs.map(wrap));
        return [newCarry.map(s => s.irValue), ys.map(s => s.irValue)];
      } finally {
        this._builder = saved;
      }
    });
    const numCarry = carryValues.length;
    const carryOut = [];
    const ysOut = [];
    for (let i = 0; i < op.numResults; i++) {
      const rv = op.getResult(i);
      const sym = new SymbolicTensor(rv, rv.type.shape, rv.type.dtype, this, [...rv.type.shape]);
      if (i < numCarry) carryOut.push(sym); else ysOut.push(sym);
    }
    return [carryOut, ysOut];
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
    this._func.inputTypes.push(tt);

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
    if (!Object.isFrozen(this._func.inputTypes)) {
      this._func.inputTypes = Object.freeze(this._func.inputTypes);
    }
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
