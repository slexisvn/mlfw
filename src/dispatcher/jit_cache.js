import { buildFunction } from '../compiler/ir/graph/builder.js';
import { GraphModule } from '../compiler/ir/graph/module.js';
import { TensorType } from '../compiler/ir/graph/types.js';
import { lowerGraphToPrimFunc } from '../compiler/passes/lowering/graph_to_tensor.js';
import { BackendPipeline } from '../backend/pipeline.js';
import { RuntimeModule } from '../runtime/runtime.js';
import { PassManager } from '../compiler/passes/pass_manager.js';
import { DecompositionPass } from '../compiler/passes/decompose/decomposition_pass.js';
import { CanonicalizePass } from '../compiler/passes/canonicalize/canonicalize.js';
import { DCEPass } from '../compiler/passes/simplify/dce.js';
import { Schedule } from '../compiler/schedule/schedule.js';
import { SchedulePolicy } from '../compiler/schedule/rules.js';
import { typedArrayCtor } from '../tensor/types/dtype.js';
import { buildMappedOp } from '../tensor/ops/ir_mapping.js';

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

function _bufferNumel(buf) {
  let n = 1;
  for (const d of buf.shape) n *= (typeof d === 'number' && d > 0 ? d : 1);
  return Math.max(n, 1);
}

function _trialLaunch(rt, compiled, primFunc) {
  const args = [];
  for (const [, buf] of primFunc.bufferMap) {
    args.push(new (typedArrayCtor(buf.dtype))(_bufferNumel(buf)));
  }
  rt.run(compiled.name, ...args);
}

function _compileScheduledGPU(func, target, backend, rt) {
  try {
    const primFunc = lowerGraphToPrimFunc(func, target);
    if (primFunc.shapeParams && primFunc.shapeParams.length > 0) return null;
    new SchedulePolicy(target).applyToAllBlocks(new Schedule(primFunc));
    const compiled = backend.compile(primFunc);
    rt.addCompiledKernel(compiled);
    if (!target.isWebGPU()) _trialLaunch(rt, compiled, primFunc);
    return compiled;
  } catch {
    return null;
  }
}

let _nextFuncId = 0;

function _buildGraphFunc(opName, tensorArgs, scalarArgs) {
  const inputTypes = tensorArgs.map(t => new TensorType(t.shape, t.dtype));
  const funcName = opName + '_jit_' + (_nextFuncId++);

  const func = buildFunction(funcName, inputTypes, [], (builder, irArgs) => {
    let result;

    result = buildMappedOp(builder, opName, irArgs, scalarArgs);

    builder.returnOp([result.getResult(0)]);
  });

  const retOp = func.getReturnOp();
  if (retOp && retOp.operands.length > 0) {
    func.outputTypes = Object.freeze([retOp.operands[0].type]);
  }

  return func;
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

  const rt = _getRuntime(target.name);
  const backend = new BackendPipeline(target);
  const isGPU = typeof target.isGPU === 'function' && target.isGPU();

  let compiled = isGPU ? _compileScheduledGPU(func, target, backend, rt) : null;
  if (!compiled) {
    compiled = backend.compile(lowerGraphToPrimFunc(func, target));
    rt.addCompiledKernel(compiled);
  }

  const retOp = func.getReturnOp();
  const outDtype = retOp && retOp.operands.length > 0 ? retOp.operands[0].type.dtype : null;
  entry = { funcName: compiled.name, runtime: rt, numInputs: tensorArgs.length, outDtype, compiled };
  _cache.set(key, entry);
  return entry;
}

export function jitCacheClear() {
  _cache.clear();
  _runtimeModules.clear();
}
