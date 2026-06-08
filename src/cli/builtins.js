import * as fw from '../index.js';
import * as ops from '../tensor/ops/ops.js';
import { Tensor } from '../tensor/core/tensor.js';
import { SymbolicTensor } from '../tracing/symbolic_tensor.js';
import { CompiledProgramView, formatTrace, formatValue } from './format.js';
import { printModule } from '../compiler/ir/graph/printer.js';

const FACTORIES = [
  'tensor', 'zeros', 'ones', 'empty', 'full', 'randn', 'arange', 'eye', 'linspace',
  'zerosLike', 'onesLike', 'emptyLike', 'fullLike', 'randnLike',
];

const FUNCTIONS = [
  'add', 'sub', 'mul', 'div', 'neg', 'pow', 'remainder', 'maximum', 'minimum',
  'exp', 'log', 'sqrt', 'rsqrt', 'abs', 'sin', 'cos', 'tanh', 'sigmoid', 'relu',
  'gelu', 'silu', 'sign', 'floor', 'ceil', 'eq', 'ne', 'lt', 'le', 'gt', 'ge',
  'where', 'matmul', 'dot', 'cat', 'stack', 'clone', 'softmax', 'log_softmax',
];

const REDUCTIONS = ['sum', 'mean', 'max', 'min', 'argmax', 'argmin', 'prod'];
const BINARY_TENSOR_FUNCTIONS = new Set([
  'add', 'sub', 'mul', 'div', 'pow', 'remainder', 'maximum', 'minimum',
  'eq', 'ne', 'lt', 'le', 'gt', 'ge', 'matmul', 'dot',
]);
const MODULES = [
  'Linear', 'ReLU', 'GELU', 'SiLU', 'Sigmoid', 'Tanh', 'LeakyReLU', 'ELU',
  'Softmax', 'LogSoftmax', 'Flatten', 'Dropout', 'LayerNorm', 'BatchNorm1d',
  'BatchNorm2d', 'Conv1d', 'Conv2d', 'MaxPool2d', 'AvgPool2d',
  'AdaptiveAvgPool2d', 'Embedding', 'CrossEntropyLoss', 'MSELoss', 'NLLLoss',
  'BCELoss',
];

export function installBuiltins(runtime, define) {
  for (const name of FACTORIES) define(name, (...args) => callWithOptions(fw[name], args));
  for (const name of FUNCTIONS) {
    define(name, (...args) => {
      if (BINARY_TENSOR_FUNCTIONS.has(name)) promoteScalarArgs(args);
      return callWithOptions(fw[name] ?? ops[name], args);
    });
  }
  for (const name of REDUCTIONS) {
    define(name, (input, ...args) => {
      const named = takeNamed(args);
      return fw[name](input, named.axis ?? args[0], named.keep ?? false);
    });
  }
  for (const name of MODULES) define(name, (...args) => constructWithNamed(fw[name], args));

  define('Sequential', (...args) => new fw.Sequential(...args));
  define('reshape', (value, shape) => value.reshape(shape));
  define('transpose', (value, dim0, dim1) => value.transpose(dim0, dim1));
  define('permute', (value, dims) => value.permute(dims));
  define('expand', (value, shape) => value.expand(shape));
  define('slice', (value, dim, start, end, step = 1) => value.slice(dim, start, end, step));
  define('unsqueeze', (value, dim) => value.unsqueeze(dim));
  define('squeeze', (value, dim) => value.squeeze(dim));
  define('narrow', (value, dim, start, length) => value.narrow(dim, start, length));
  define('select', (value, dim, index) => value.select(dim, index));
  define('contiguous', value => value.contiguous());
  define('detach', value => value.detach());
  define('requires_grad', (value, flag = true) => value.requiresGrad_(flag));
  define('grad', value => value.grad);
  define('backward', (value, gradient = undefined) => {
    value.backward(gradient);
    return value;
  });

  define('range', (...args) => {
    let start = 0, stop, step = 1;
    if (args.length === 1) stop = args[0];
    else if (args.length === 2) { start = args[0]; stop = args[1]; }
    else { start = args[0]; stop = args[1]; step = args[2]; }
    const result = [];
    if (step > 0) for (let i = start; i < stop; i += step) result.push(i);
    else if (step < 0) for (let i = start; i > stop; i += step) result.push(i);
    else throw new Error('range() step cannot be zero');
    return result;
  });

  define('len', value => {
    if (Array.isArray(value)) return value.length;
    if (typeof value === 'string') return value.length;
    if (value instanceof Tensor || value instanceof SymbolicTensor) return value.shape[0];
    throw new Error('len() expects an array, string, or tensor');
  });

  define('shape', value => value.shape);
  define('dtype', value => value.dtype);
  define('print', value => { runtime.output(formatValue(value)); return value; });
  define('trace', value => {
    const view = value?._isCompiled ? value._compiledView : value instanceof CompiledProgramView ? value : null;
    if (!view?.events) throw new Error('trace() expects a compiled program');
    const text = formatTrace(view.events);
    runtime.output(text);
    return text;
  });
  define('graph', value => {
    const graph = value?._isCompiled ? value._compiledView?.graph :
                  value instanceof CompiledProgramView ? value.graph : value;
    const text = printModule(graph);
    runtime.output(text);
    return text;
  });
  define('compile', (...args) => runtime.compile(...args));

  define('cpu', 'cpu');
  define('gpu', 'gpu');
  define('wasm', 'wasm');
  define('webgpu', 'webgpu');
  for (const dtype of ['f16', 'f32', 'f64', 'i32', 'i64', 'bool']) define(dtype, dtype);
}

export function takeNamed(args) {
  const last = args[args.length - 1];
  return last && last.__named ? args.pop() : {};
}

function callWithOptions(fn, args) {
  const named = takeNamed(args);
  if (Object.keys(named).length === 0) return fn(...args);
  delete named.__named;
  if ('grad' in named) {
    named.requiresGrad = named.grad;
    delete named.grad;
  }
  if ('axis' in named) {
    args.push(named.axis);
    delete named.axis;
  }
  return fn(...args, named);
}

function constructWithNamed(Type, args) {
  const named = takeNamed(args);
  delete named.__named;
  if (Type === fw.Softmax || Type === fw.LogSoftmax) return new Type(named.axis ?? args[0] ?? -1);
  if (Type === fw.Conv1d || Type === fw.Conv2d) return new Type(...args, named);
  return new Type(...args, ...Object.values(named));
}

function promoteScalarArgs(args) {
  const reference = args.find(value => value instanceof Tensor || value instanceof SymbolicTensor);
  if (!reference) return;
  const options = { dtype: reference.dtype, device: reference.device };
  for (let i = 0; i < Math.min(args.length, 2); i++) {
    if (typeof args[i] === 'number' || typeof args[i] === 'boolean') args[i] = fw.tensor(args[i], options);
  }
}

const FACTORY_SIGNATURES = {
  tensor: [{ name: 'data' }, { name: 'opts', isOptional: true }],
  zeros: [{ name: 'shape' }, { name: 'opts', isOptional: true }],
  ones: [{ name: 'shape' }, { name: 'opts', isOptional: true }],
  empty: [{ name: 'shape' }, { name: 'opts', isOptional: true }],
  full: [{ name: 'shape' }, { name: 'value' }, { name: 'opts', isOptional: true }],
  randn: [{ name: 'shape' }, { name: 'opts', isOptional: true }],
  arange: [{ name: 'start' }, { name: 'end', isOptional: true }, { name: 'step', isOptional: true }, { name: 'opts', isOptional: true }],
  eye: [{ name: 'n' }, { name: 'm', isOptional: true }, { name: 'opts', isOptional: true }],
  linspace: [{ name: 'start' }, { name: 'end' }, { name: 'steps' }, { name: 'opts', isOptional: true }],
  zerosLike: [{ name: 'tensor' }],
  onesLike: [{ name: 'tensor' }],
  emptyLike: [{ name: 'tensor' }],
  fullLike: [{ name: 'tensor' }, { name: 'value' }],
  randnLike: [{ name: 'tensor' }],
};

const MODULE_SIGNATURES = {
  Linear: [{ name: 'inFeatures' }, { name: 'outFeatures' }, { name: 'bias', defaultValue: 'true', isOptional: true }],
  Conv1d: [{ name: 'inChannels' }, { name: 'outChannels' }, { name: 'kernelSize' }, { name: 'stride', defaultValue: '1', isOptional: true }, { name: 'padding', defaultValue: '0', isOptional: true }],
  Conv2d: [{ name: 'inChannels' }, { name: 'outChannels' }, { name: 'kernelSize' }, { name: 'stride', defaultValue: '1', isOptional: true }, { name: 'padding', defaultValue: '0', isOptional: true }],
  LayerNorm: [{ name: 'normalizedShape' }, { name: 'eps', defaultValue: '1e-5', isOptional: true }],
  BatchNorm1d: [{ name: 'numFeatures' }, { name: 'eps', defaultValue: '1e-5', isOptional: true }, { name: 'momentum', defaultValue: '0.1', isOptional: true }],
  BatchNorm2d: [{ name: 'numFeatures' }, { name: 'eps', defaultValue: '1e-5', isOptional: true }, { name: 'momentum', defaultValue: '0.1', isOptional: true }],
  Dropout: [{ name: 'p', defaultValue: '0.5', isOptional: true }],
  Embedding: [{ name: 'numEmbeddings' }, { name: 'embeddingDim' }, { name: 'paddingIdx', isOptional: true }],
  MaxPool2d: [{ name: 'kernelSize' }, { name: 'stride', isOptional: true }, { name: 'padding', defaultValue: '0', isOptional: true }],
  AvgPool2d: [{ name: 'kernelSize' }, { name: 'stride', isOptional: true }, { name: 'padding', defaultValue: '0', isOptional: true }],
  AdaptiveAvgPool2d: [{ name: 'outputSize' }],
  LeakyReLU: [{ name: 'negativeSlope', defaultValue: '0.01', isOptional: true }],
  ELU: [{ name: 'alpha', defaultValue: '1.0', isOptional: true }],
  Softmax: [{ name: 'dim', defaultValue: '-1', isOptional: true }],
  LogSoftmax: [{ name: 'dim', defaultValue: '-1', isOptional: true }],
  Flatten: [{ name: 'startDim', defaultValue: '1', isOptional: true }, { name: 'endDim', defaultValue: '-1', isOptional: true }],
};

const BUILTIN_SIGNATURES = {
  reshape: [{ name: 'tensor' }, { name: 'shape' }],
  transpose: [{ name: 'tensor' }, { name: 'dim0' }, { name: 'dim1' }],
  permute: [{ name: 'tensor' }, { name: 'dims' }],
  expand: [{ name: 'tensor' }, { name: 'shape' }],
  slice: [{ name: 'tensor' }, { name: 'dim' }, { name: 'start' }, { name: 'end' }, { name: 'step', defaultValue: '1', isOptional: true }],
  unsqueeze: [{ name: 'tensor' }, { name: 'dim' }],
  squeeze: [{ name: 'tensor' }, { name: 'dim' }],
  narrow: [{ name: 'tensor' }, { name: 'dim' }, { name: 'start' }, { name: 'length' }],
  select: [{ name: 'tensor' }, { name: 'dim' }, { name: 'index' }],
  contiguous: [{ name: 'tensor' }],
  detach: [{ name: 'tensor' }],
  requires_grad: [{ name: 'tensor' }, { name: 'flag', defaultValue: 'true', isOptional: true }],
  grad: [{ name: 'tensor' }],
  backward: [{ name: 'tensor' }, { name: 'gradient', isOptional: true }],
  range: [{ name: 'start' }, { name: 'stop', isOptional: true }, { name: 'step', isOptional: true }],
  len: [{ name: 'value' }],
  shape: [{ name: 'tensor' }],
  dtype: [{ name: 'tensor' }],
  print: [{ name: 'value' }],
  trace: [{ name: 'compiled' }],
  graph: [{ name: 'compiled' }],
  compile: [
    { name: 'model' }, { name: 'input', isOptional: true }, { name: 'target', defaultValue: 'cpu', isOptional: true },
    { name: 'fusion', isOptional: true }, { name: 'scheduling', isOptional: true }, { name: 'autotune', isOptional: true },
    { name: 'quantization', isOptional: true }, { name: 'layout', isOptional: true }, { name: 'rematerialization', isOptional: true },
    { name: 'inplaceReuse', isOptional: true }, { name: 'partition', isOptional: true },
    { name: 'debug', isOptional: true }, { name: 'snippet', isOptional: true }, { name: 'verify', defaultValue: 'true', isOptional: true },
    { name: 'epilogue', isOptional: true }, { name: 'fusionStrategy', defaultValue: 'xla', isOptional: true },
    { name: 'numTrials', defaultValue: '64', isOptional: true }, { name: 'timeBudgetMs', defaultValue: '30000', isOptional: true },
  ],
  Sequential: [{ name: '...modules' }],
  sum: [{ name: 'input' }, { name: 'axis', isOptional: true }, { name: 'keep', isOptional: true }],
  mean: [{ name: 'input' }, { name: 'axis', isOptional: true }, { name: 'keep', isOptional: true }],
  max: [{ name: 'input' }, { name: 'axis', isOptional: true }, { name: 'keep', isOptional: true }],
  min: [{ name: 'input' }, { name: 'axis', isOptional: true }, { name: 'keep', isOptional: true }],
  argmax: [{ name: 'input' }, { name: 'axis', isOptional: true }, { name: 'keep', isOptional: true }],
  argmin: [{ name: 'input' }, { name: 'axis', isOptional: true }, { name: 'keep', isOptional: true }],
  prod: [{ name: 'input' }, { name: 'axis', isOptional: true }, { name: 'keep', isOptional: true }],
};

export function installSignatures(registry) {
  for (const [name, params] of Object.entries(FACTORY_SIGNATURES)) registry.register(name, params);
  for (const [name, params] of Object.entries(MODULE_SIGNATURES)) registry.register(name, params);
  for (const [name, params] of Object.entries(BUILTIN_SIGNATURES)) registry.register(name, params);
}
