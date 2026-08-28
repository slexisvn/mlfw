export {
  compile, trace, compileWithBackward, CPUTarget, CUDATarget, WasmTarget, WebGPUTarget,
  TraceLevel, randn, zeros, ones, tensor, manual_seed, unseed, noGrad, where,
  sub, mul, sum, printFunction, printModule, dispatcher,
} from '../tools/internals.mjs';

export {
  Linear, ReLU, MSELoss, SGD, Adam, TensorDataset, DataLoader, LightningModule, Trainer,
} from '../tools/internals.mjs';

export {
  CompiledKernel, BackendPipeline, lowerGraphToPrimFunc,
  RuntimeModule, RuntimeTensor, KernelRegistry, constBuffersOf,
  registerBackend, getBackend, hasBackend,
  assignPlanBuffers, computePlanDonations, planMemoryReport,
} from '../tools/internals.mjs';

export {
  DispatchKey, DispatchKeySet, EMPTY_KEY_SET, BACKEND_KEY_SET, AUTOGRAD_KEY_SET,
  FUNCTIONALITY_KEY_SET, backendKeyForDevice, autogradKeyForBackend, computeKeySet,
  KernelFunction, IValue, IValueTag, KernelTable, OperatorEntry, OperatorHandle, Library,
  guardStack, withExcludedKeys, withIncludedKeys, withGuard,
  parseSchema, OperatorSchema, SchemaArg, ArgKind, jitCompile, jitCacheClear,
} from '../tools/internals.mjs';

export {
  Tracer, getActiveTracer, registerTracingDispatch, ShapeEnv, _traceCore,
  foldWeightParams, weightPredicate, MAX_FOLDABLE_ELEMENTS, SymInt,
  BASELINE, DEFAULT_MIN_GAIN, optimizationCandidates, selectWinner,
  candidateByName, gateCacheKey, graphSignature,
} from '../tools/internals.mjs';

export function firstFunction(graphModule) {
  return graphModule.functions().next().value;
}
