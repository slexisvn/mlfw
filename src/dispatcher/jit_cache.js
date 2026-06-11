import { buildFunction, indexSelectGatherOpts } from '../compiler/ir/graph/builder.js';
import { GraphModule } from '../compiler/ir/graph/module.js';
import { TensorType, ScalarType } from '../compiler/ir/graph/types.js';
import { lowerGraphToPrimFunc } from '../compiler/passes/lowering/graph_to_tensor.js';
import { BackendPipeline } from '../backend/pipeline.js';
import { RuntimeModule } from '../compiler/runtime/runtime.js';
import { PassManager } from '../compiler/passes/pass_manager.js';
import { DecompositionPass } from '../compiler/passes/decompose/decomposition_pass.js';
import { CanonicalizePass } from '../compiler/passes/canonicalize/canonicalize.js';
import { DCEPass } from '../compiler/passes/simplify/dce.js';
import { reduceInitValue } from '../backend/dtype_map.js';

const _cache = new Map();
const _runtimeModules = new Map();

function _cacheKey(opName, tensorArgs, scalarArgs, target) {
  let key = opName;
  for (let i = 0; i < tensorArgs.length; i++) {
    key += '|' + tensorArgs[i].shape.join(',') + ':' + tensorArgs[i].dtype;
  }
  if (scalarArgs) {
    for (const [k, v] of Object.entries(scalarArgs)) {
      key += '|' + k + '=' + JSON.stringify(v);
    }
  }
  key += '|' + target.name;
  return key;
}

function _getRuntime(targetName) {
  let rt = _runtimeModules.get(targetName);
  if (!rt) {
    rt = new RuntimeModule('jit_' + targetName);
    _runtimeModules.set(targetName, rt);
  }
  return rt;
}

const _REDUCTION_OPS = {
  sum: 'sum', mean: 'mean', max: 'max', min: 'min', prod: 'prod',
};

const _BUILDER_ALIASES = {
  matmul: 'matmul',
  dot: (b, args) => b.dot(args[0], args[1], [args[0].type.rank - 1], [0]),
  clone: (b, args) => b._inferAndBuild('add', [args[0], b.scalarConstant(0, args[0].type.dtype).getResult(0)]),
  transpose: (b, args, s) => {
    const rank = args[0].type.rank;
    const d0 = s?.dim0 ?? 0;
    const d1 = s?.dim1 ?? 1;
    const perm = Array.from({ length: rank }, (_, i) => i);
    perm[d0] = d1;
    perm[d1] = d0;
    return b.transpose(args[0], perm);
  },
  softmax: (b, args, s) => b.softmax(args[0], s?.dim ?? -1),
  log_softmax: (b, args, s) => b.logSoftmax(args[0], s?.dim ?? -1),
  layer_norm: (b, args, s) => b.layernorm(args[0], args[1], args[2], s?.axis ?? -1, s?.eps ?? 1e-5),
  batch_norm: (b, args, s) => b.batchnorm(args[0], args[1], args[2], args[3], args[4], s?.axis ?? 1, s?.eps ?? 1e-5),
  conv2d: (b, args, s) => b.conv(args[0], args[1], s?.strides ?? [1,1], s?.padding ?? [[0,0],[0,0]], { dilation: s?.dilation ?? [1,1], groups: s?.groups ?? 1 }),
  pool2d: (b, args, s) => b.pool2d(args[0], s?.pool_type ?? 'max', s?.kernel_size ?? [2,2], s?.strides ?? [2,2], s?.padding ?? [[0,0],[0,0]]),
  embedding: (b, args) => b.embedding(args[0], args[1]),
  argmax: (b, args, s) => b.argmax(args[0], s?.dim ?? 0, s?.keepdim ?? false),
  argmin: (b, args, s) => b.argmin(args[0], s?.dim ?? 0, s?.keepdim ?? false),
  eq: (b, args) => b.compare(args[0], args[1], 'eq'),
  ne: (b, args) => b.compare(args[0], args[1], 'ne'),
  lt: (b, args) => b.compare(args[0], args[1], 'lt'),
  le: (b, args) => b.compare(args[0], args[1], 'le'),
  gt: (b, args) => b.compare(args[0], args[1], 'gt'),
  ge: (b, args) => b.compare(args[0], args[1], 'ge'),
  clamp: (b, args) => b.clamp(args[1], args[0], args[2]),
  pad: (b, args, s) => b.pad(args[0], args[1], s.low, s.high),
  one_hot: (b, args, s) => b.oneHot(args[0], s.depth, { dtype: ScalarType.F32 }),
  index_select: (b, args, s) => b.gather(args[0], args[1], indexSelectGatherOpts(args[0].type, s.dim ?? 0, args[1].type.rank)),
  gather: (b, args, s) => b.gatherDim(args[0], args[1], s.dim ?? 0),
  scatter_add: (b, args, s) => b.scatterAddDim(args[0], args[1], args[2], s.dim ?? 0),
  cat: (b, args, s) => {
    const rank = args[0].type.rank;
    const dim = (s?.dim ?? 0) < 0 ? rank + (s?.dim ?? 0) : (s?.dim ?? 0);
    return b.concat(args, dim);
  },
  stack: (b, args, s) => {
    const rank = args[0].type.rank;
    const dim = (s?.dim ?? 0) < 0 ? rank + 1 + (s?.dim ?? 0) : (s?.dim ?? 0);
    const expanded = args.map(arg => {
      const newShape = [...arg.type.shape];
      newShape.splice(dim, 0, 1);
      return b.reshape(arg, newShape).getResult(0);
    });
    return b.concat(expanded, dim);
  },
};

let _nextFuncId = 0;

function _buildGraphFunc(opName, tensorArgs, scalarArgs) {
  const inputTypes = tensorArgs.map(t => new TensorType(t.shape, t.dtype));
  const funcName = opName + '_jit_' + (_nextFuncId++);

  const func = buildFunction(funcName, inputTypes, [], (builder, irArgs) => {
    let result;

    if (_REDUCTION_OPS[opName]) {
      const dims = scalarArgs?.dim;
      const rank = irArgs[0].type.rank;
      const dimensions = dims !== undefined && dims !== null
        ? (Array.isArray(dims) ? dims : [dims]).map(d => d < 0 ? rank + d : d)
        : Array.from({ length: rank }, (_, i) => i);
      const initVal = _reductionInit(opName, irArgs[0].type.dtype);
      const initConst = builder.scalarConstant(initVal, irArgs[0].type.dtype);
      result = builder.reduce(irArgs[0], initConst.getResult(0), dimensions, opName);
    } else if (typeof _BUILDER_ALIASES[opName] === 'function') {
      result = _BUILDER_ALIASES[opName](builder, irArgs, scalarArgs);
    } else if (typeof builder[opName] === 'function') {
      result = _callBuilder(builder, opName, irArgs, scalarArgs);
    } else {
      result = builder._inferAndBuild(opName, irArgs, scalarArgs);
    }

    builder.returnOp([result.getResult(0)]);
  });

  const retOp = func.getReturnOp();
  if (retOp && retOp.operands.length > 0) {
    func.outputTypes = Object.freeze([retOp.operands[0].type]);
  }

  return func;
}

function _callBuilder(builder, opName, irArgs, scalarArgs) {
  if (irArgs.length === 1) return builder[opName](irArgs[0]);
  if (irArgs.length === 2) return builder[opName](irArgs[0], irArgs[1]);
  if (irArgs.length === 3) return builder[opName](irArgs[0], irArgs[1], irArgs[2]);
  return builder._inferAndBuild(opName, irArgs, scalarArgs);
}

function _reductionInit(opName, dtype) {
  return reduceInitValue(opName, dtype);
}

export function jitCompile(opName, tensorArgs, scalarArgs, target) {
  const key = _cacheKey(opName, tensorArgs, scalarArgs, target);
  let entry = _cache.get(key);
  if (entry) return entry;

  const func = _buildGraphFunc(opName, tensorArgs, scalarArgs);

  const mod = new GraphModule(opName + '_jit_mod');
  mod.addFunction(func);
  const pm = new PassManager();
  pm.addPass(new DecompositionPass());
  pm.addPass(new CanonicalizePass());
  pm.addPass(new DCEPass());
  pm.run(mod);

  const primFunc = lowerGraphToPrimFunc(func);
  const backend = new BackendPipeline(target);
  const compiled = backend.compile(primFunc);

  const rt = _getRuntime(target.name);
  rt.addCompiledKernel(compiled);

  const retOp = func.getReturnOp();
  const outDtype = retOp && retOp.operands.length > 0 ? retOp.operands[0].type.dtype : null;
  entry = { funcName: compiled.name, runtime: rt, numInputs: tensorArgs.length, outDtype };
  _cache.set(key, entry);
  return entry;
}

export function jitCacheSize() {
  return _cache.size;
}

export function jitCacheClear() {
  _cache.clear();
  _runtimeModules.clear();
}
