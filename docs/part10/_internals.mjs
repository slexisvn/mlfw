import {
  lowerToLIR, FlatIndexSimplifyPass, BackendPipeline, CPUTarget,
} from '../tools/internals.mjs';

export {
  compile, trace, CPUTarget, CUDATarget, WasmTarget, WebGPUTarget,
  TraceLevel, randn, zeros, ones, tensor, manual_seed,
} from '../tools/internals.mjs';

export {
  lowerGraphToPrimFunc, printTensorIR, BackendPipeline, Schedule, resetVarCounter,
  compileGraph, buildFunction, TensorType, ScalarType,
} from '../tools/internals.mjs';

export {
  lowerToLIR, detectAccumulator, ACCUMULATOR_OPS, scanMetadata, flattenIndex,
  computeDynamicStride, computeNumelExpr, verifyLIR, LIRVerificationError,
  buildLirPipeline, LirPassManager, FlatIndexSimplifyPass,
  LIRFunc, LIRFlatLoadNode, LIRFlatStoreNode, LIRAccumulatorNode, LIRBindingsNode,
  LIRMetadata, inferDtype, normalizeDtype, isWasmNativeOp,
} from '../tools/internals.mjs';

export {
  CPUCodegen, WasmCodegen, CUDACodegen, WebGPUCodegen, encodeWat,
  flattenRowMajorIndex, emitSymInt, TargetKind,
  registerCodegen, getCodegenEntry, registerExternalCodegen, getExternalCodegen,
  unregisterExternalCodegen, getCudaIntrin, registerCudaIntrin,
} from '../tools/internals.mjs';

export {
  detectPureMatmul, CUBLAS_PROVIDER, registerExternalCodegenProvider,
  unregisterExternalCodegenProvider, activeExternalCodegenProviders,
  isExternalCodegenEnabled, FuncAttr,
} from '../tools/internals.mjs';

export {
  ForNode, BlockNode, SeqNode, BufferStoreNode, BufferLoadNode,
  VariableNode, IntImmNode, FloatImmNode, MathOpNode, PrimFunc, ForKind, IterVarKind, Buffer,
} from '../tools/internals.mjs';

export { lowerToTir } from '../tools/internals.mjs';

export function toLIR(primFunc, target = CPUTarget(), { simplify = true } = {}) {
  const lir = lowerToLIR(primFunc, target);
  if (simplify) new FlatIndexSimplifyPass().run(lir, { trace: { functionEvent() {} } });
  return lir;
}

export function emit(func, target) {
  return new BackendPipeline(target).compile(func);
}
