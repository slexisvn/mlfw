import * as fw from '../index.js';
import * as ops from '../tensor/ops/ops.js';
import { Tensor } from '../tensor/core/tensor.js';
import { SymbolicTensor } from '../tracing/symbolic_tensor.js';
import { CompiledProgramView, formatTrace, formatValue } from './format.js';
import { printModule } from '../compiler/ir/printer/printer.js';

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

  define('shape', value => value.shape);
  define('dtype', value => value.dtype);
  define('print', value => { runtime.output(formatValue(value)); return value; });
  define('trace', value => {
    if (!(value instanceof CompiledProgramView)) throw new Error('trace() expects a compiled program');
    const text = formatTrace(value.events);
    runtime.output(text);
    return text;
  });
  define('graph', value => {
    const graph = value instanceof CompiledProgramView ? value.graph : value;
    const text = printModule(graph);
    runtime.output(text);
    return text;
  });
  define('compile', (...args) => runtime.compile(...args));

  define('cpu', 'cpu');
  define('gpu', 'gpu');
  define('wasm', 'wasm');
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
