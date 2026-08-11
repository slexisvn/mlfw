import { registerNativeOps } from './tensor/native/registration.js';
import { registerCpuLinalg } from './kernels/cpu/linalg/register.js';
import { registerCpuMl } from './kernels/cpu/ml/register.js';
import { registerCpuNumeric } from './kernels/cpu/numeric/register.js';
import { registerWasmLinalg } from './kernels/wasm/linalg/register.js';
import { registerWasmMl } from './kernels/wasm/ml/register.js';
import { registerAutogradKernels } from './autograd/dispatch.js';
import { installOps } from './tensor/ops/install.js';
import { Tensor } from './tensor/core/tensor.js';

registerNativeOps();
registerCpuLinalg();
registerCpuMl();
registerCpuNumeric();
registerWasmLinalg();
registerWasmMl();
registerAutogradKernels();
installOps(Tensor as unknown as Parameters<typeof installOps>[0]);

export { Tensor } from './tensor/core/tensor.js';

export {
  CPU_DEVICE, GPU_DEVICE, WASM_DEVICE, WEBGPU_DEVICE,
  getDefaultDevice, setDefaultDevice,
} from './tensor/types/device.js';

export { GradMode, noGrad, enableGrad } from './autograd/grad_mode.js';
export { SymbolicTensor } from './tracing/symbolic_tensor.js';
export { TraceLevel } from './compiler/pipeline/trace.js';
export { printModule } from './compiler/ir/graph/printer.js';
export { preloadWebGPU, preloadCudaRuntime, releaseCudaMemory } from './runtime/backend_registry.js';
export { flushWebGPUEager } from './runtime/webgpu.js';
export { dispatcher } from './dispatcher/dispatcher.js';
export { manualSeed as manual_seed, seed, unseed } from './util/random.js';

export {
  zeros, ones, empty, full, randn, arange, eye, linspace, randperm,
} from './tensor/factory/creation_ops.js';

export {
  zerosLike, onesLike, emptyLike, fullLike, randnLike,
} from './tensor/factory/like_ops.js';

export {
  tensor, fromBuffer, scalar,
} from './tensor/factory/from_ops.js';

export {
  add, sub, mul, div, neg, pow, maximum, minimum,
  exp, log, sqrt, rsqrt, abs, sin, cos, tanh, sigmoid,
  erf, erfc, lgamma, gamma,
  relu, gelu, silu, sign, floor, ceil,
  eq, ne, lt, le, gt, ge, where, clamp, pad, one_hot, index_select, gather, scatter_add, scatter,
  sum, mean, max, min, argmax, argmin, prod,
  matmul, dot, cat, stack, clone,
  reshape, transpose, permute, broadcast_in_dim, expand, slice, unsqueeze, squeeze, narrow, select, contiguous,
  repeat, tile, split, chunk, roll, flip, cumsum, sort, topk, argsort,
  softmax, log_softmax,
} from './tensor/ops/ops.js';

export {
  Module, Parameter, F,
  Linear, Conv1d, Conv2d,
  ReLU, GELU, SiLU, Sigmoid, Tanh, LeakyReLU, ELU, Softmax, LogSoftmax,
  LayerNorm, BatchNorm1d, BatchNorm2d,
  MaxPool2d, AvgPool2d, AdaptiveAvgPool2d,
  Dropout,
  CrossEntropyLoss, MSELoss, NLLLoss, BCELoss,
  Embedding,
  GRU, GRUCell,
  LSTM, LSTMCell,
  Sequential, ModuleList, ModuleDict,
  Flatten,
  MultiheadAttention,
  TransformerEncoderLayer, TransformerDecoderLayer,
  TransformerEncoder, TransformerDecoder,
  Transformer,
  PositionalEncoding,
} from './nn/index.js';

export * as nn from './nn/index.js';
export * as init from './nn/init.js';

export {
  Dataset, TensorDataset, MapDataset,
  Sampler, SequentialSampler, RandomSampler, BatchSampler,
  DataLoader, defaultCollate,
} from './data/index.js';
export * as data from './data/index.js';

export { Tokenizer, Vocab } from './tokenizer/index.js';
export * as tokenizer from './tokenizer/index.js';

export {
  Optimizer, SGD, Adam, AdamW,
  LRScheduler, StepLR, CosineAnnealingLR, ReduceLROnPlateau,
  clipGradNorm_, clipGradValue_,
} from './optim/index.js';
export * as optim from './optim/index.js';

export { trace, compile } from './tracing/compile.js';
export { scan } from './tracing/scan.js';
export { compileWithBackward } from './tracing/compile_backward.js';
export { CPUTarget, CUDATarget, WasmTarget, WebGPUTarget } from './backend/target.js';

export {
  LightningModule, Trainer, Callback,
  ModelCheckpoint, loadCheckpoint, applyCheckpoint, serializeCheckpoint,
  EarlyStopping, ProgressCallback,
  LearningRateMonitor, Timer, GradientAccumulationScheduler,
  Logger, ConsoleLogger, CSVLogger,
  Metric, MeanMetric, SumMetric, MetricCollection,
  Accuracy, Precision, Recall, F1Score, ConfusionMatrix,
} from './lightning/index.js';
export * as lightning from './lightning/index.js';

export { fs as memfs } from '#io/fs';

export * as ops from './tensor/ops/ops.js';
export * as linalg from './tensor/ops/linalg.js';
export * as ml from './ml/index.js';
export * as numeric from './numeric/index.js';
