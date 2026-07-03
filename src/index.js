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
installOps(Tensor);

export { Tensor } from './tensor/core/tensor.js';

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
  eq, ne, lt, le, gt, ge, where, clamp, pad, one_hot, index_select, gather, scatter_add,
  sum, mean, max, min, argmax, argmin, prod,
  matmul, dot, cat, stack, clone,
  softmax, log_softmax,
} from './tensor/ops/ops.js';

export { roll, flip, cumsum, sort, topk, argsort, scatter } from './tensor/ops/composite.js';

export { noGrad, enableGrad } from './autograd/grad_mode.js';

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
  ModelCheckpoint, loadCheckpoint, applyCheckpoint,
  EarlyStopping, ProgressCallback,
  LearningRateMonitor, Timer, GradientAccumulationScheduler,
  Logger, ConsoleLogger, CSVLogger,
  Metric, MeanMetric, SumMetric, MetricCollection,
  Accuracy, Precision, Recall, F1Score, ConfusionMatrix,
} from './lightning/index.js';
export * as lightning from './lightning/index.js';

export { TeraRuntime } from './cli/runtime.js';
export { formatValue } from './cli/format.js';
export { fs as memfs } from '#io/fs';

export * as linalg from './tensor/ops/linalg.js';
export * as ml from './ml/index.js';
export * as numeric from './numeric/index.js';
