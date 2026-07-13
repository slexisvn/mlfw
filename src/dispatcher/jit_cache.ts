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
import type { DType, NumericTypedArray } from '../tensor/types/dtype.js';
import { buildMappedOp } from '../tensor/ops/ir_mapping.js';
import type { BuilderLike, GraphOperation as MappedGraphOperation, GraphValue } from '../tensor/ops/ir_mapping.js';

type TensorLike = {
  shape: readonly number[];
  dtype: DType;
};

export type TargetLike = {
  name: string;
  isGPU?: () => boolean;
  isWebGPU: () => boolean;
};

type BufferLike = {
  shape: readonly number[];
  dtype: DType;
};

type PrimFuncLike = {
  bufferMap: Iterable<readonly [unknown, BufferLike]>;
  shapeParams?: readonly unknown[];
};

type CompiledKernel = {
  name: string;
};

type RuntimeLike = {
  run(name: string, ...args: NumericTypedArray[]): unknown;
  addCompiledKernel(compiled: CompiledKernel): void;
};

type ReturnGraphOperation = MappedGraphOperation & {
  operands: readonly GraphValue[];
};

type GraphFunction = {
  outputTypes: readonly unknown[];
  getReturnOp(): ReturnGraphOperation | null;
};

type GraphBuilder = {
  returnOp(values: readonly GraphValue[]): unknown;
};

type ScalarArgs = Record<string, unknown> | null;

export type CacheEntry = {
  funcName: string;
  runtime: RuntimeLike;
  numInputs: number;
  outDtype: DType | null;
  compiled: CompiledKernel;
};

const _cache = new Map<string, CacheEntry>();
const _runtimeModules = new Map<string, RuntimeLike>();
const _lowerGraphToPrimFunc = lowerGraphToPrimFunc as unknown as (func: GraphFunction, target: TargetLike) => PrimFuncLike;
const _BackendPipeline = BackendPipeline as unknown as new (target: TargetLike) => { compile(func: PrimFuncLike): CompiledKernel };

function _cacheKey(opName: string, tensorArgs: readonly TensorLike[], scalarArgs: ScalarArgs, target: TargetLike): string {
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

function _getRuntime(targetName: string): RuntimeLike {
  let rt = _runtimeModules.get(targetName);
  if (!rt) {
    rt = new RuntimeModule('jit_' + targetName) as RuntimeLike;
    _runtimeModules.set(targetName, rt);
  }
  return rt;
}

function _bufferNumel(buf: BufferLike): number {
  let n = 1;
  for (const d of buf.shape) n *= (typeof d === 'number' && d > 0 ? d : 1);
  return Math.max(n, 1);
}

function _trialLaunch(rt: RuntimeLike, compiled: CompiledKernel, primFunc: PrimFuncLike): void {
  const args: NumericTypedArray[] = [];
  for (const [, buf] of primFunc.bufferMap) {
    args.push(new (typedArrayCtor(buf.dtype))(_bufferNumel(buf)));
  }
  rt.run(compiled.name, ...args);
}

function _compileScheduledGPU(func: GraphFunction, target: TargetLike, backend: { compile(func: PrimFuncLike): CompiledKernel }, rt: RuntimeLike): CompiledKernel | null {
  try {
    const primFunc = _lowerGraphToPrimFunc(func, target);
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

function _buildGraphFunc(opName: string, tensorArgs: readonly TensorLike[], scalarArgs: ScalarArgs): GraphFunction {
  const inputTypes = tensorArgs.map(t => new TensorType(t.shape, t.dtype));
  const funcName = opName + '_jit_' + (_nextFuncId++);

  const func = buildFunction(funcName, inputTypes, [], (builder: GraphBuilder, irArgs: readonly GraphValue[]) => {
    let result: MappedGraphOperation;

    result = buildMappedOp(
      builder as unknown as BuilderLike,
      opName,
      irArgs,
      scalarArgs
    );

    builder.returnOp([result.getResult(0)]);
  }) as GraphFunction;

  const retOp = func.getReturnOp();
  if (retOp && retOp.operands.length > 0) {
    func.outputTypes = Object.freeze([retOp.operands[0].type]);
  }

  return func;
}

export function jitCompile(opName: string, tensorArgs: readonly TensorLike[], scalarArgs: ScalarArgs = null, target: TargetLike): CacheEntry {
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
  const backend = new _BackendPipeline(target);
  const isGPU = typeof target.isGPU === 'function' && target.isGPU();

  let compiled = isGPU ? _compileScheduledGPU(func, target, backend, rt) : null;
  if (!compiled) {
    compiled = backend.compile(_lowerGraphToPrimFunc(func, target));
    rt.addCompiledKernel(compiled);
  }

  const retOp = func.getReturnOp();
  const outDtype = retOp && retOp.operands.length > 0 ? retOp.operands[0].type.dtype : null;
  entry = { funcName: compiled.name, runtime: rt, numInputs: tensorArgs.length, outDtype, compiled };
  _cache.set(key, entry);
  return entry;
}

export function jitCacheClear(): void {
  _cache.clear();
  _runtimeModules.clear();
}
