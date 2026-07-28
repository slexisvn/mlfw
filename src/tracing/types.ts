import type { Tensor } from '../tensor/core/tensor.js';
import type { DType, NumericTypedArray } from '../tensor/types/dtype.js';
import type { Device } from '../tensor/types/device.js';
import type { Dim, TensorType } from '../compiler/ir/graph/types.js';
import type { ShapeEnv } from './shape_env.js';
import type { SymbolicTensor } from './symbolic_tensor.js';

export type SymbolicDim = Dim | string;
export type SymbolicShape = readonly SymbolicDim[];
export type MutableSymbolicShape = SymbolicDim[];
export type TensorInput = Tensor;
export type TensorOutput = Tensor | SymbolicTensor;
export type MaybePromise<T> = T | Promise<T>;
export type AttrValue = unknown;
export type AttrMap = Record<string, AttrValue>;
export type DynamicShapeSpec = true | Set<number> | null | undefined;
export type DynamicShapes = readonly DynamicShapeSpec[] | null | undefined;

export type IRValueLike = {
  id?: number;
  type: TensorType;
  symbolicShape?: SymbolicShape;
  addUse?: (link: unknown) => void;
  removeUse?: (link: unknown) => void;
  replaceAllUsesWith?: (value: IRValueLike) => void;
};

export type IROperationLike = {
  opName: string;
  operands: IRValueLike[];
  numOperands: number;
  numResults: number;
  getResult(index: number): IRValueLike;
  erase(): void;
};

export type IRBlockLike = {
  arguments: IRValueLike[];
  firstOp: IROperationLike | null;
  lastOp: IROperationLike | null;
  addArgument(type: TensorType): IRValueLike;
  getArgument(index: number): IRValueLike | undefined;
  insertBefore(op: IROperationLike, before: IROperationLike): void;
  pushOp(op: IROperationLike): void;
  removeArguments(indices: Set<number>): void;
};

export type GraphFunctionLike = {
  name: string;
  inputTypes: readonly TensorType[];
  outputTypes: readonly TensorType[];
  entryBlock: IRBlockLike;
  args: IRValueLike[];
  getReturnOp(): IROperationLike | null;
};

export type GraphModuleLike = {
  addFunction(func: GraphFunctionLike): void;
  functions(): IterableIterator<GraphFunctionLike>;
};

export type IRBuilderLike = {
  returnOp(values: readonly IRValueLike[]): IROperationLike;
  scalarConstant(value: number | bigint, dtype: DType): IROperationLike;
  scanOp(
    xs: readonly IRValueLike[],
    carry: readonly IRValueLike[],
    body: (builder: IRBuilderLike, xs: IRValueLike[], carry: IRValueLike[]) => [IRValueLike[], IRValueLike[]]
  ): IROperationLike;
};

export type TracedCore = {
  graph: GraphModuleLike;
  capturedParams: TensorInput[];
  numUserInputs: number;
  outputTypes: readonly TensorType[];
  shapeEnv: ShapeEnv;
  outputSymShapes: readonly SymbolicShape[];
};

export type CompiledResult = {
  module: {
    executionPlan?: unknown;
    runPlanAsync(plan: unknown, args: readonly RuntimeArg[], opts?: Record<string, unknown>): Promise<unknown>;
  };
  listKernels(): string[];
  isAsync(name: string): boolean;
  run(name: string, ...args: RuntimeArg[]): void;
  runAsync(name: string, ...args: RuntimeArg[]): Promise<unknown>;
  getSource(name: string): string;
};

export type CompiledEntry = {
  result: CompiledResult;
  graph: GraphModuleLike;
  capturedParams: TensorInput[];
  numUserInputs: number;
  outputTypes: readonly TensorType[];
  shapeEnv: ShapeEnv;
  outputSymShapes: readonly SymbolicShape[];
};

export type RuntimeArg = NumericTypedArray | RuntimeTensorLike;
export type RuntimeTensorLike = {
  resident?: { key: unknown; version: number };
};

export type CompilableModel = {
  constructor: { name?: string };
  forward(...inputs: TensorOutput[]): MaybePromise<TensorOutput | TensorOutput[]>;
};

export type CompileOptions = Record<string, unknown> & {
  name?: string;
  target?: unknown;
  backward?: unknown;
  dynamicShapes?: DynamicShapes;
  dynamic_shapes?: DynamicShapes;
  shapeBuckets?: readonly (readonly (readonly number[])[])[];
  foldWeights?: boolean;
  quantization?: { foldWeights?: boolean };
  mode?: string;
  rematPolicy?: unknown;
  remat?: Record<string, unknown>;
};

export type CompiledForward = {
  (...inputs: TensorInput[]): MaybePromise<TensorOutput | TensorOutput[]>;
  original: CompilableModel;
  graph(inputs?: TensorInput[]): MaybePromise<GraphModuleLike>;
  source(): string | null;
  kernels(): string[];
  result(): CompiledResult | null;
  _ready: Promise<void> | null;
};
