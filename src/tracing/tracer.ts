import { IRBuilder } from '../compiler/ir/graph/builder.js';
import { GraphModule } from '../compiler/ir/graph/module.js';
import { GraphFunction } from '../compiler/ir/graph/function.js';
import { TensorType, DYNAMIC } from '../compiler/ir/graph/types.js';
import { registry } from '../compiler/ir/graph/ops.js';
import { SymbolicTensor } from './symbolic_tensor.js';
import { ShapeEnv } from './shape_env.js';
import { buildMappedOp } from '../tensor/ops/ir_mapping.js';
import type { DType } from '../tensor/types/dtype.js';
import type { Tensor } from '../tensor/core/tensor.js';
import type { Shape } from '../tensor/utils/shape_utils.js';
import type { AttrMap, GraphFunctionLike, GraphModuleLike, IRBuilderLike, IROperationLike, IRValueLike, MutableSymbolicShape, SymbolicShape, TensorOutput } from './types.js';

let _activeTracer: Tracer | null = null;

export function getActiveTracer(): Tracer | null {
  return _activeTracer;
}

export class Tracer {
  private _name: string;
  private _shapeEnv: ShapeEnv;
  private _inputTypes: TensorType[];
  private _inputSymShapes: SymbolicShape[];
  private _outputTypes: TensorType[];
  private _outputSymShapes: SymbolicShape[];
  private _inputs: SymbolicTensor[];
  private _func: GraphFunctionLike | null;
  private _builder: IRBuilderLike | null;
  private _module: GraphModuleLike | null;
  private _capturedParams: Map<Tensor, SymbolicTensor>;
  private _capturedParamOrder: Tensor[];

  constructor(name?: string) {
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

  get shapeEnv(): ShapeEnv {
    return this._shapeEnv;
  }

  createInput(shape: Shape, dtype: DType, dynamicDims?: Set<number> | null): { shape: number[]; dtype: DType; tensorType: TensorType } {
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

  _initGraph(): SymbolicTensor[] {
    this._func = new GraphFunction(this._name, this._inputTypes, []) as unknown as GraphFunctionLike;
    this._func.inputTypes = [...this._func.inputTypes];
    this._builder = new IRBuilder(this._func as unknown as GraphFunction) as unknown as IRBuilderLike;
    this._module = new GraphModule(this._name) as unknown as GraphModuleLike;

    const symbolicInputs: SymbolicTensor[] = [];
    const args = this._func.args;
    for (let i = 0; i < args.length; i++) {
      const irValue = args[i];
      const tt = this._inputTypes[i];
      irValue.symbolicShape = this._inputSymShapes[i];
      const st = new SymbolicTensor(irValue, tt.shape as readonly number[], tt.dtype, this, this._inputSymShapes[i]);
      symbolicInputs.push(st);
    }
    this._inputs = symbolicInputs;
    return symbolicInputs;
  }

  recordOp(opName: string, tensorArgs: readonly TensorOutput[], attrs: AttrMap): SymbolicTensor | SymbolicTensor[] {
    const irOperands: IRValueLike[] = [];
    for (const arg of tensorArgs) {
      if (arg instanceof SymbolicTensor) {
        irOperands.push(arg.irValue);
      }
    }

    const builder = this._requireBuilder();
    const op = buildMappedOp(
      builder as unknown as Parameters<typeof buildMappedOp>[0],
      opName,
      irOperands as unknown as Parameters<typeof buildMappedOp>[2],
      attrs
    ) as IROperationLike;

    const results: SymbolicTensor[] = [];
    for (let i = 0; i < op.numResults; i++) {
      const resultValue = op.getResult(i);
      const resultType = resultValue.type;
      const resultSymShape = this._propagateSymbolicShape(opName, op, tensorArgs, resultType, i);
      resultValue.symbolicShape = resultSymShape;
      results.push(new SymbolicTensor(resultValue, resultType.shape as readonly number[], resultType.dtype, this, resultSymShape));
    }

    return results.length === 1 ? results[0] : results;
  }

  _propagateSymbolicShape(opName: string, op: IROperationLike, tensorArgs: readonly TensorOutput[], resultType: TensorType, resultIndex = 0): SymbolicShape {
    const symTensorArgs = tensorArgs.filter((a): a is SymbolicTensor => a instanceof SymbolicTensor);

    const def = registry.get(op.opName || opName);
    if (def && def.propagateSymbolicShapes) {
      const shapeMap = new Map<IRValueLike, SymbolicShape>();
      for (const arg of symTensorArgs) {
        shapeMap.set(arg.irValue, arg.symbolicShape);
      }
      const propagated = (def.propagateSymbolicShapes as unknown as (op: IROperationLike, shapes: Map<IRValueLike, SymbolicShape>) => SymbolicShape[] | null | undefined)(op, shapeMap);
      if (propagated && propagated[resultIndex]) return propagated[resultIndex];
    }

    const resultShape = resultType.shape;
    const outSym = new Array<MutableSymbolicShape[number]>(resultShape.length);

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

  scan(
    xsValues: readonly TensorOutput[],
    carryValues: readonly TensorOutput[],
    stepFn: (carry: SymbolicTensor[], xs: SymbolicTensor[]) => [SymbolicTensor[], SymbolicTensor[]]
  ): [SymbolicTensor[], SymbolicTensor[]] {
    const toIr = (s: TensorOutput) => s instanceof SymbolicTensor ? s.irValue : this.captureConstant(s).irValue;
    const xsIr = xsValues.map(toIr);
    const carryIr = carryValues.map(toIr);
    const builder = this._requireBuilder();
    const op = builder.scanOp(xsIr, carryIr, (bb: IRBuilderLike, xtArgs: IRValueLike[], carryArgs: IRValueLike[]) => {
      const saved = this._requireBuilder();
      this._builder = bb;
      try {
        const wrap = (v: IRValueLike) => new SymbolicTensor(v, v.type.shape as readonly number[], v.type.dtype, this, [...v.type.shape]);
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
      const sym = new SymbolicTensor(rv, rv.type.shape as readonly number[], rv.type.dtype, this, [...rv.type.shape]);
      if (i < numCarry) carryOut.push(sym); else ysOut.push(sym);
    }
    return [carryOut, ysOut];
  }

  captureConstant(tensor: Tensor): SymbolicTensor {
    let cached = this._capturedParams.get(tensor);
    if (cached) return cached;

    if (tensor.shape.length === 0 && tensor.data) {
      const value = tensor.data[0];
      const op = this._requireBuilder().scalarConstant(value, tensor.dtype);
      const irValue = op.getResult(0);
      const sym = new SymbolicTensor(irValue, [], tensor.dtype, this, []);
      this._capturedParams.set(tensor, sym);
      return sym;
    }

    const tt = new TensorType(tensor.shape, tensor.dtype);
    const func = this._requireFunc();
    (func.inputTypes as TensorType[]).push(tt);

    const block = func.entryBlock;
    const irValue = block.addArgument(tt);

    const sym = new SymbolicTensor(irValue, tensor.shape, tensor.dtype, this, [...tensor.shape]);
    this._capturedParams.set(tensor, sym);
    this._capturedParamOrder.push(tensor);
    return sym;
  }

  get capturedParams(): Tensor[] {
    return this._capturedParamOrder;
  }

  markOutput(symbolicTensor: TensorOutput): void {
    if (symbolicTensor instanceof SymbolicTensor) {
      this._requireBuilder().returnOp([symbolicTensor.irValue]);
      this._outputSymShapes = [symbolicTensor.symbolicShape];
    }
    this._outputTypes = [new TensorType(symbolicTensor.shape, symbolicTensor.dtype)];
  }

  markOutputs(symbolicTensors: readonly SymbolicTensor[]): void {
    const irValues = symbolicTensors.map(st => st.irValue);
    this._requireBuilder().returnOp(irValues);
    this._outputTypes = symbolicTensors.map(
      st => new TensorType(st.shape, st.dtype)
    );
    this._outputSymShapes = symbolicTensors.map(
      st => st.symbolicShape
    );
  }

  get outputSymShapes(): SymbolicShape[] {
    return this._outputSymShapes;
  }

  getGraphModule(): GraphModuleLike {
    const func = this._requireFunc();
    const module = this._requireModule();
    func.outputTypes = Object.freeze(this._outputTypes);
    if (!Object.isFrozen(func.inputTypes)) {
      func.inputTypes = Object.freeze(func.inputTypes);
    }
    module.addFunction(func);
    return module;
  }

  activate(): void {
    _activeTracer = this;
  }

  deactivate(): void {
    if (_activeTracer === this) _activeTracer = null;
  }

  private _requireBuilder(): IRBuilderLike {
    if (!this._builder) throw new Error('Tracer graph has not been initialized');
    return this._builder;
  }

  private _requireFunc(): GraphFunctionLike {
    if (!this._func) throw new Error('Tracer graph has not been initialized');
    return this._func;
  }

  private _requireModule(): GraphModuleLike {
    if (!this._module) throw new Error('Tracer graph has not been initialized');
    return this._module;
  }
}
