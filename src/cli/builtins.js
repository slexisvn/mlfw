import './qe_config.js';
import * as fw from '../index.js';
import * as ops from '../tensor/ops/ops.js';
import { Tensor } from '../tensor/core/tensor.js';
import { CPU_DEVICE, GPU_DEVICE, WASM_DEVICE } from '../tensor/types/device.js';
import { CompiledProgramView, formatTrace, formatValue, formatValueCompact } from './format.js';
import { printModule } from '../compiler/ir/graph/printer.js';
import { DataLoader, TensorDataset } from '../data/index.js';
import { Tokenizer } from '../tokenizer/index.js';
import { loadCsvRows } from './csv.js';
import {
  createEngine, DataFrame, Col, InMemoryRelation,
  col as qcol, lit as qlit, expr as qexpr,
  sum as qsum, avg as qavg, min as qmin, max as qmax, count as qcount, countStar as qcountStar,
} from '#plugins/query-engine';
import { SGD, Adam, AdamW, StepLR, CosineAnnealingLR, ReduceLROnPlateau } from '../optim/index.js';
import {
  Trainer, EarlyStopping, ModelCheckpoint, ProgressCallback,
  LearningRateMonitor, Timer, GradientAccumulationScheduler,
  ConsoleLogger, CSVLogger,
  Accuracy, Precision, Recall, F1Score, ConfusionMatrix, MetricCollection,
  serializeCheckpoint, loadCheckpoint, applyCheckpoint,
} from '../lightning/index.js';
import { fs } from '#io/fs';

const FACTORIES = [
  'tensor', 'zeros', 'ones', 'empty', 'full', 'randn', 'arange', 'eye', 'linspace', 'randperm',
  'zerosLike', 'onesLike', 'emptyLike', 'fullLike', 'randnLike',
];

const FREE_TENSOR_FUNCTIONS = ['where', 'cat', 'stack'];
const COLUMN_AGGREGATES = ['sum', 'min', 'max'];
const MODULES = [
  'Linear', 'ReLU', 'GELU', 'SiLU', 'Sigmoid', 'Tanh', 'LeakyReLU', 'ELU',
  'Softmax', 'LogSoftmax', 'Flatten', 'Dropout', 'LayerNorm', 'BatchNorm1d',
  'BatchNorm2d', 'Conv1d', 'Conv2d', 'MaxPool2d', 'AvgPool2d',
  'AdaptiveAvgPool2d', 'Embedding', 'GRU', 'GRUCell', 'LSTM', 'LSTMCell', 'CrossEntropyLoss', 'MSELoss', 'NLLLoss',
  'BCELoss',
];

// One shared query engine backs every DataFrame so they can be joined together.
let _engine = null;
function engine() {
  return _engine ?? (_engine = createEngine());
}

let _uploadTableId = 0;
// Build the relation once from typed column arrays (no row objects) and register
// it as a scannable table; returns the table name.
function registerColumnsAsTable(columns) {
  const eng = engine();
  const relation = InMemoryRelation.fromColumns(columns);
  const name = `__upload${_uploadTableId++}`;
  eng.catalog.registerTable(name, relation.getSchema());
  eng.catalog.registerTableStorage(name, relation);
  return name;
}

function dropTable(name) {
  const eng = engine();
  const key = name.toUpperCase();
  eng.catalog.tables?.delete(key);
  eng.catalog.tableStorage?.delete(key);
}

export function createDataFrameFromColumns(columns) {
  return engine().sql(`SELECT * FROM ${registerColumnsAsTable(columns)}`);
}

// Uploaded CSV files (browser): file name -> registered table name. The relation
// is built once on upload; load_csv() re-scans it without rebuilding or keeping a
// second copy of the column data.
const _uploadedCsv = new Map();
export function setUploadedCsv(name, columns) { _uploadedCsv.set(name, registerColumnsAsTable(columns)); }

// Streaming ingest: append row batches into a RelationBuilder so the whole file
// is never held in memory at once. finish() registers the relation under `name`.
export function beginUploadedCsv(name) {
  const builder = InMemoryRelation.builder();
  return {
    appendRows(rows) { if (rows && rows.length) builder.appendRows(rows); },
    finish() {
      const relation = builder.finish();
      const eng = engine();
      const table = `__upload${_uploadTableId++}`;
      eng.catalog.registerTable(table, relation.getSchema());
      eng.catalog.registerTableStorage(table, relation);
      const old = _uploadedCsv.get(name);
      if (old) dropTable(old);
      _uploadedCsv.set(name, table);
    },
  };
}
export function removeUploadedCsv(name) {
  const table = _uploadedCsv.get(name);
  if (table) dropTable(table);
  _uploadedCsv.delete(name);
}

const AGG_FNS = { sum: qsum, min: qmin, max: qmax };

function isColumnArg(value) {
  return typeof value === 'string' || value instanceof Col;
}

// DataFrame is the query-engine class shipped in the plugin bundle. Augment its
// prototype with tera-facing conveniences: tensor conversion, label encoding,
// and a readable print form.
DataFrame.prototype.toString = function () {
  return `DataFrame(${this.columns().join(', ')})`;
};

// Pandas-style first-N-rows view; returns a (lazy) DataFrame.
DataFrame.prototype.head = function (n = 5) {
  return this.limit(n);
};

async function dfToTensor(df, cols) {
  const frame = cols.length > 0 ? df.select(...cols) : df;
  const names = frame.columns();
  const rows = await frame.collect();
  const k = names.length;
  const n = rows.length;
  const flat = new Float32Array(n * k);
  let idx = 0;
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < k; c++) {
      const v = rows[r][names[c]];
      if (typeof v !== 'number') {
        throw new Error(`Column '${names[c]}' contains non-numeric value '${v}' at row ${r}. Use encode() for categorical columns.`);
      }
      flat[idx++] = v;
    }
  }
  return fw.tensor(flat, { shape: [n, k] });
}

async function dfEncode(df, column, knownClasses) {
  const name = column ?? df.columns()[0];
  const rows = await df.collect();
  const classMap = new Map();
  const classes = knownClasses ? [...knownClasses] : [];
  for (let i = 0; i < classes.length; i++) classMap.set(String(classes[i]), i);
  const encoded = new Float32Array(rows.length);
  for (let i = 0; i < rows.length; i++) {
    const key = String(rows[i][name]);
    let idx = classMap.get(key);
    if (idx === undefined) {
      if (knownClasses) throw new Error(`Unknown class '${rows[i][name]}' not present in fitted classes`);
      idx = classes.length;
      classMap.set(key, idx);
      classes.push(rows[i][name]);
    }
    encoded[i] = idx;
  }
  return [fw.tensor(encoded, { shape: [encoded.length] }), classes];
}

DataFrame.prototype.toTensor = function (...cols) { return dfToTensor(this, cols); };
DataFrame.prototype.to_tensor = DataFrame.prototype.toTensor;
DataFrame.prototype.to_array = function () { return this.toArray(); };
DataFrame.prototype.encode = function (column, ...rest) {
  const named = takeNamed(rest);
  const knownClasses = named.classes ?? rest[0] ?? null;
  return dfEncode(this, column, knownClasses);
};

export function installBuiltins(runtime, define) {
  for (const name of FACTORIES) define(name, (...args) => callWithOptions(fw[name], args));
  for (const name of FREE_TENSOR_FUNCTIONS) define(name, (...args) => callWithOptions(fw[name] ?? ops[name], args));
  for (const name of COLUMN_AGGREGATES) {
    define(name, input => {
      if (!isColumnArg(input)) throw new Error(`${name}() expects a DataFrame column; call tensor.${name}() for tensors`);
      return AGG_FNS[name](input);
    });
  }
  for (const name of MODULES) define(name, (...args) => constructWithNamed(fw[name], args));

  define('Sequential', (...args) => new fw.Sequential(...args));

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

  define('print', (...args) => {
    const named = args.length > 0 && args[args.length - 1]?.__named ? args.pop() : null;
    const sep = named?.sep ?? ' ';
    const compact = args.length > 1;
    const text = args.map(v => compact ? formatValueCompact(v) : formatValue(v)).join(sep);
    runtime.output(text);
  });
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

  define('Tokenizer', (...args) => constructWithSnakeCase(Tokenizer, args));

  define('TensorDataset', (...args) => new TensorDataset(...args));
  define('DataLoader', (...args) => {
    const named = takeNamed(args);
    delete named.__named;
    return new DataLoader(args[0], snakeNamedToCamel(named));
  });

  define('SGD', (...args) => constructOptimizer(SGD, args));
  define('Adam', (...args) => constructOptimizer(Adam, args));
  define('AdamW', (...args) => constructOptimizer(AdamW, args));

  define('StepLR', (...args) => constructScheduler(StepLR, args, ['optimizer', 'stepSize', 'gamma']));
  define('CosineAnnealingLR', (...args) => constructScheduler(CosineAnnealingLR, args, ['optimizer', 'tMax', 'etaMin']));
  define('ReduceLROnPlateau', (...args) => {
    const named = takeNamed(args);
    delete named.__named;
    return new ReduceLROnPlateau(args[0], snakeNamedToCamel(named));
  });

  define('Trainer', (...args) => {
    const named = takeNamed(args);
    delete named.__named;
    return new Trainer(snakeNamedToCamel(named));
  });

  define('EarlyStopping', (...args) => constructWithSnakeCase(EarlyStopping, args));
  define('ModelCheckpoint', (...args) => constructWithSnakeCase(ModelCheckpoint, args));
  define('ProgressCallback', (...args) => constructWithSnakeCase(ProgressCallback, args));
  define('LearningRateMonitor', (...args) => constructWithSnakeCase(LearningRateMonitor, args));
  define('Timer', (...args) => constructWithSnakeCase(Timer, args));
  define('GradientAccumulationScheduler', (...args) => constructWithSnakeCase(GradientAccumulationScheduler, args));

  define('ConsoleLogger', (...args) => constructWithSnakeCase(ConsoleLogger, args));
  define('CSVLogger', (...args) => constructWithSnakeCase(CSVLogger, args));

  define('Accuracy', (...args) => constructWithSnakeCase(Accuracy, args));
  define('Precision', (...args) => constructWithSnakeCase(Precision, args));
  define('Recall', (...args) => constructWithSnakeCase(Recall, args));
  define('F1Score', (...args) => constructWithSnakeCase(F1Score, args));
  define('ConfusionMatrix', (...args) => constructWithSnakeCase(ConfusionMatrix, args));
  define('MetricCollection', (...args) => constructWithSnakeCase(MetricCollection, args));

  define('save', (model, path) => {
    if (!model || typeof model.stateDict !== 'function') throw new Error('save() requires a model as the first argument');
    if (typeof path !== 'string') throw new Error('save() requires a file path string');
    const tmp = path + '.tmp';
    fs.writeBinary(tmp, serializeCheckpoint({ modelState: model.stateDict() }));
    fs.rename(tmp, path);
  });

  define('load', (model, path) => {
    if (!model || typeof model.loadStateDict !== 'function') throw new Error('load() requires a model as the first argument');
    if (typeof path !== 'string') throw new Error('load() requires a file path string');
    applyCheckpoint(loadCheckpoint(path), model);
    return model;
  });

  define('optim_config', (...args) => {
    const named = takeNamed(args);
    delete named.__named;
    const optimizer = args[0] ?? named.optimizer;
    if (!optimizer) throw new Error('optim_config() requires an optimizer');
    const result = { optimizer };
    const sched = named.lr_scheduler ?? named.lrScheduler;
    if (sched) result.lrScheduler = sched;
    return result;
  });

  define('DataFrame', (...args) => {
    const named = takeNamed(args);
    delete named.__named;
    const colNames = Object.keys(named);
    if (colNames.length === 0) {
      throw new Error('DataFrame() requires named column arrays, e.g. DataFrame(name=[...], age=[...])');
    }
    const n = named[colNames[0]].length;
    const rows = [];
    for (let r = 0; r < n; r++) {
      const row = {};
      for (const c of colNames) row[c] = named[c][r];
      rows.push(row);
    }
    return engine().createDataFrame(rows);
  });

  define('col', name => qcol(name));
  define('lit', value => qlit(value));
  define('expr', sql => qexpr(sql));
  define('avg', column => qavg(column));
  define('count', column => qcount(column));
  define('countStar', () => qcountStar());

  define('load_csv', (...args) => {
    const named = takeNamed(args);
    delete named.__named;
    const path = args[0];
    if (typeof path !== 'string') throw new Error('load_csv() requires a file path string');
    if (_uploadedCsv.has(path)) return engine().sql(`SELECT * FROM ${_uploadedCsv.get(path)}`);
    const sep = named.separator ?? named.sep ?? ',';
    const { rows } = loadCsvRows(path, sep);
    return engine().createDataFrame(rows);
  });

  define('encode', (...args) => {
    const data = args[0];
    if (data instanceof DataFrame) {
      return data.encode(...args.slice(1));
    }
    if (Array.isArray(data)) {
      const classMap = new Map();
      const classes = [];
      const encoded = new Float32Array(data.length);
      for (let i = 0; i < data.length; i++) {
        const key = String(data[i]);
        let idx = classMap.get(key);
        if (idx === undefined) {
          idx = classes.length;
          classMap.set(key, idx);
          classes.push(data[i]);
        }
        encoded[i] = idx;
      }
      return [fw.tensor(encoded, { shape: [encoded.length] }), classes];
    }
    throw new Error('encode() expects a DataFrame or array');
  });

  define('decode', (...args) => {
    const indices = args[0];
    const classes = args[1];
    if (!Array.isArray(classes)) throw new Error('decode() expects (indices, classes) — classes must be an array');
    if (indices instanceof Tensor) {
      return indices.contiguous().data.reduce((result, raw) => {
        const idx = Math.round(raw);
        result.push(idx >= 0 && idx < classes.length ? classes[idx] : `<unknown:${idx}>`);
        return result;
      }, []);
    }
    if (typeof indices === 'number') {
      const idx = Math.round(indices);
      return idx >= 0 && idx < classes.length ? classes[idx] : `<unknown:${idx}>`;
    }
    throw new Error('decode() expects a tensor or number as first argument');
  });

  define('normalize', (...args) => {
    const t = args[0];
    if (!(t instanceof Tensor)) throw new Error('normalize() expects a tensor');
    const named = takeNamed(args);
    delete named.__named;
    const axis = named.axis ?? 0;
    const mu = fw.mean(t, axis, true);
    const diff = fw.sub(t, mu);
    const variance = fw.mean(fw.mul(diff, diff), axis, true);
    const std = fw.sqrt(fw.add(variance, fw.tensor(1e-8)));
    return fw.div(diff, std);
  });

  define('train_test_split', (...args) => {
    const named = takeNamed(args);
    delete named.__named;
    const data = args[0];
    const ratio = named.test_size ?? named.ratio ?? 0.2;
    if (data instanceof Tensor) {
      const n = data.shape[0];
      const splitIdx = Math.round(n * (1 - ratio));
      return [data.slice(0, 0, splitIdx), data.slice(0, splitIdx, n)];
    }
    throw new Error('train_test_split() expects a tensor');
  });
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
  if (typeof named.device === 'string') {
    named.device = DEVICE_BY_NAME[named.device] ?? named.device;
  }
  return fn(...args, named);
}

const DEVICE_BY_NAME = { cpu: CPU_DEVICE, gpu: GPU_DEVICE, wasm: WASM_DEVICE };

function constructWithNamed(Type, args) {
  const named = takeNamed(args);
  delete named.__named;
  if (Type === fw.Softmax || Type === fw.LogSoftmax) return new Type(named.axis ?? args[0] ?? -1);
  if (Type === fw.Conv1d || Type === fw.Conv2d) return new Type(...args, named);
  return new Type(...args, ...Object.values(named));
}

function snakeToCamel(name) {
  return name.replace(/_([a-z])/g, (_, c) => c.toUpperCase());
}

function snakeNamedToCamel(named) {
  const result = {};
  for (const key of Object.keys(named)) {
    result[snakeToCamel(key)] = named[key];
  }
  return result;
}

function constructWithSnakeCase(Type, args) {
  const named = takeNamed(args);
  delete named.__named;
  const opts = snakeNamedToCamel(named);
  if (args.length > 0) return new Type(...args, opts);
  if (Object.keys(opts).length > 0) return new Type(opts);
  return new Type();
}

function constructOptimizer(Type, args) {
  const named = takeNamed(args);
  delete named.__named;
  const params = args[0];
  if (!params) throw new Error(`${Type.name}() requires params as first argument`);
  return new Type(params, snakeNamedToCamel(named));
}

function constructScheduler(Type, args, posNames) {
  const named = takeNamed(args);
  delete named.__named;
  const merged = snakeNamedToCamel(named);
  const positional = [];
  for (let i = 0; i < posNames.length; i++) {
    const name = posNames[i];
    if (i < args.length) positional.push(args[i]);
    else if (merged[name] !== undefined) positional.push(merged[name]);
    else break;
  }
  return new Type(...positional);
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

const TRAINING_SIGNATURES = {
  TensorDataset: [{ name: '...tensors' }],
  DataLoader: [{ name: 'dataset' }, { name: 'batch_size', defaultValue: '1', isOptional: true }, { name: 'shuffle', defaultValue: 'false', isOptional: true }, { name: 'drop_last', defaultValue: 'false', isOptional: true }],
  SGD: [{ name: 'params' }, { name: 'lr', defaultValue: '0.01', isOptional: true }, { name: 'momentum', defaultValue: '0', isOptional: true }, { name: 'weight_decay', defaultValue: '0', isOptional: true }],
  Adam: [{ name: 'params' }, { name: 'lr', defaultValue: '0.001', isOptional: true }, { name: 'betas', isOptional: true }, { name: 'weight_decay', defaultValue: '0', isOptional: true }],
  AdamW: [{ name: 'params' }, { name: 'lr', defaultValue: '0.001', isOptional: true }, { name: 'betas', isOptional: true }, { name: 'weight_decay', defaultValue: '0.01', isOptional: true }],
  StepLR: [{ name: 'optimizer' }, { name: 'step_size' }, { name: 'gamma', defaultValue: '0.1', isOptional: true }],
  CosineAnnealingLR: [{ name: 'optimizer' }, { name: 't_max' }, { name: 'eta_min', defaultValue: '0', isOptional: true }],
  ReduceLROnPlateau: [{ name: 'optimizer' }, { name: 'mode', defaultValue: '"min"', isOptional: true }, { name: 'patience', defaultValue: '10', isOptional: true }, { name: 'factor', defaultValue: '0.1', isOptional: true }],
  Trainer: [{ name: 'max_epochs', defaultValue: '10', isOptional: true }, { name: 'accelerator', defaultValue: '"auto"', isOptional: true }, { name: 'callbacks', isOptional: true }, { name: 'logger', defaultValue: 'true', isOptional: true }],
  EarlyStopping: [{ name: 'monitor' }, { name: 'patience', defaultValue: '3', isOptional: true }, { name: 'mode', defaultValue: '"min"', isOptional: true }],
  ModelCheckpoint: [{ name: 'monitor', isOptional: true }, { name: 'save_top_k', defaultValue: '1', isOptional: true }, { name: 'mode', defaultValue: '"min"', isOptional: true }],
  ProgressCallback: [],
  LearningRateMonitor: [],
  Timer: [],
  GradientAccumulationScheduler: [{ name: 'scheduling' }],
  ConsoleLogger: [],
  CSVLogger: [{ name: 'save_dir', isOptional: true }, { name: 'name', isOptional: true }],
  Accuracy: [{ name: 'task', defaultValue: '"binary"', isOptional: true }, { name: 'num_classes', isOptional: true }, { name: 'top_k', defaultValue: '1', isOptional: true }],
  Precision: [{ name: 'task', defaultValue: '"binary"', isOptional: true }, { name: 'num_classes', isOptional: true }, { name: 'average', defaultValue: '"macro"', isOptional: true }],
  Recall: [{ name: 'task', defaultValue: '"binary"', isOptional: true }, { name: 'num_classes', isOptional: true }, { name: 'average', defaultValue: '"macro"', isOptional: true }],
  F1Score: [{ name: 'task', defaultValue: '"binary"', isOptional: true }, { name: 'num_classes', isOptional: true }, { name: 'average', defaultValue: '"macro"', isOptional: true }],
  ConfusionMatrix: [{ name: 'num_classes' }],
  MetricCollection: [{ name: '...metrics' }],
  optim_config: [{ name: 'optimizer' }, { name: 'lr_scheduler', isOptional: true }],
  load_csv: [{ name: 'path' }, { name: 'separator', defaultValue: '","', isOptional: true }],
  DataFrame: [{ name: 'columns' }],
  col: [{ name: 'name' }],
  lit: [{ name: 'value' }],
  expr: [{ name: 'sql' }],
  avg: [{ name: 'column' }],
  count: [{ name: 'column' }],
  countStar: [],
  encode: [{ name: 'data' }],
  decode: [{ name: 'indices' }, { name: 'classes' }],
  normalize: [{ name: 'tensor' }, { name: 'axis', defaultValue: '0', isOptional: true }],
  train_test_split: [{ name: 'data' }, { name: 'test_size', defaultValue: '0.2', isOptional: true }],
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
  for (const [name, params] of Object.entries(TRAINING_SIGNATURES)) registry.register(name, params);
}
