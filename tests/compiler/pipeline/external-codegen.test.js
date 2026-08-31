import { describe, it, expect, afterEach } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { GraphModule } from '../../../src/compiler/ir/graph/module.js';
import { CPUTarget, CUDATarget, TargetKind } from '../../../src/compiler/support/target.js';
import { Compiler, CompilerConfig } from '../../../src/compiler/pipeline/compiler.js';
import { BackendPipeline } from '../../../src/backend/pipeline.js';
import { FuncAttr } from '../../../src/compiler/ir/func_attrs.js';
import {
  registerExternalCodegenProvider,
  unregisterExternalCodegenProvider,
  activeExternalCodegenProviders,
  isExternalCodegenEnabled,
  detectPureMatmul,
  CUBLAS_PROVIDER,
} from '../../../src/compiler/pipeline/external_codegen.js';
import { registerExternalCodegen, unregisterExternalCodegen } from '../../../src/backend/codegen_registry.js';
import { lowerGraphToPrimFunc } from '../../../src/compiler/passes/lowering/graph_to_tensor.js';

const PROVIDER = 'test_gemm_library';
const f32 = (shape) => new TensorType(shape, ScalarType.F32);

function matmulModule(name = 'mm') {
  const a = f32([8, 8]);
  const mod = new GraphModule('m');
  mod.addFunction(buildFunction(name, [a, a], [a], (b, args) => {
    b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
  }));
  return mod;
}

function installProvider({ enabled = () => true, targetKind = TargetKind.CPU } = {}) {
  registerExternalCodegenProvider({
    name: PROVIDER,
    enabled,
    annotate: (tirModule) => {
      for (const pf of tirModule) {
        const info = detectPureMatmul(pf);
        if (info) pf.setAttr(FuncAttr.EXTERNAL_CODEGEN, { name: PROVIDER, info });
      }
    },
  });
  registerExternalCodegen(PROVIDER, {
    targetKind,
    runtimeKind: 'js',
    compile: (primFunc, target, info) => ({
      source: '',
      metadata: { kind: 'js', library: PROVIDER, dims: [info.M, info.N, info.K] },
    }),
  });
}

afterEach(() => {
  unregisterExternalCodegenProvider(PROVIDER);
  unregisterExternalCodegen(PROVIDER);
});

describe('external codegen provider registry', () => {
  it('reports cublas as enabled only when the matmul backend selects it', () => {
    const target = CUDATarget();
    expect(isExternalCodegenEnabled(CUBLAS_PROVIDER, new CompilerConfig({ target }), target)).toBe(false);
    expect(isExternalCodegenEnabled(CUBLAS_PROVIDER, new CompilerConfig({ target, matmulBackend: 'cublas' }), target)).toBe(true);
  });

  it('lets a provider opt in through its own predicate', () => {
    const target = CPUTarget();
    installProvider({ enabled: (config) => config.optimization.fastMath === true });
    const off = new CompilerConfig({ target });
    const on = new CompilerConfig({ target, optimization: { fastMath: true } });
    expect(activeExternalCodegenProviders(off, target).map(p => p.name)).not.toContain(PROVIDER);
    expect(activeExternalCodegenProviders(on, target).map(p => p.name)).toContain(PROVIDER);
  });
});

describe('external codegen dispatch', () => {
  it('routes an annotated function to the registered library instead of the target codegen', () => {
    installProvider();
    const target = CPUTarget();
    const func = matmulModule().getFunction('mm');
    const primFunc = lowerGraphToPrimFunc(func, target, null);
    const info = detectPureMatmul(primFunc);
    expect(info).not.toBeNull();
    primFunc.setAttr(FuncAttr.EXTERNAL_CODEGEN, { name: PROVIDER, info });

    const compiled = new BackendPipeline(target).compile(primFunc);
    expect(compiled.metadata.library).toBe(PROVIDER);
    expect(compiled.metadata.dims).toEqual([8, 8, 8]);
    expect(compiled.source).toBe('');
  });

  it('falls back to the target codegen when the library targets another kind', () => {
    installProvider({ targetKind: TargetKind.CUDA });
    const target = CPUTarget();
    const func = matmulModule().getFunction('mm');
    const primFunc = lowerGraphToPrimFunc(func, target, null);
    primFunc.setAttr(FuncAttr.EXTERNAL_CODEGEN, { name: PROVIDER, info: detectPureMatmul(primFunc) });

    const compiled = new BackendPipeline(target).compile(primFunc);
    expect(compiled.metadata.library).toBeUndefined();
    expect(compiled.source.length).toBeGreaterThan(0);
  });

  it('drives the whole compile through the provider without a driver-level mode flag', () => {
    installProvider();
    const result = new Compiler({ target: CPUTarget() }).compile(matmulModule());
    expect(result.succeeded).toBe(true);
    expect(result.module.getKernelMetadata('mm').library).toBe(PROVIDER);
  });

  it('leaves functions untouched when no provider is active', () => {
    const result = new Compiler({ target: CPUTarget() }).compile(matmulModule());
    expect(result.succeeded).toBe(true);
    expect(result.module.getKernelMetadata('mm').library).toBeUndefined();
    expect(result.getSource('mm').length).toBeGreaterThan(0);
  });
});
