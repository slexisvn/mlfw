import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { GraphModule } from '../../../src/compiler/ir/graph/module.js';
import { CPUTarget, CUDATarget, WebGPUTarget } from '../../../src/compiler/support/target.js';
import { CompilerConfig } from '../../../src/compiler/pipeline/compiler.js';
import { countBoundaryClasses, selectGraphSplitStrategy } from '../../../src/compiler/pipeline/graph_split.js';
import { TargetAttr } from '../../../src/compiler/support/target_attrs.js';
import { launchBoundaryClass, countLaunchBoundaries } from '../../../src/compiler/ir/graph/op_traits.js';

const f32 = (shape) => new TensorType(shape, ScalarType.F32);

function moduleOf(name, argTypes, retTypes, build) {
  const mod = new GraphModule('m');
  mod.addFunction(buildFunction(name, argTypes, retTypes, build));
  return mod;
}

function matmulChain(count) {
  const t = f32([8, 8]);
  return moduleOf('f', Array(count + 1).fill(t), [t], (b, args) => {
    let acc = args[0];
    for (let i = 1; i <= count; i++) acc = b.matmul(acc, args[i]).getResult(0);
    b.returnOp([acc]);
  });
}

function elementwiseOnly() {
  const t = f32([8, 8]);
  return moduleOf('f', [t, t], [t], (b, args) => {
    b.returnOp([b.add(args[0], args[1]).getResult(0)]);
  });
}

function strategyFor(module, target, configOpts = {}) {
  const config = new CompilerConfig({ target, ...configOpts });
  return selectGraphSplitStrategy({ config, target, module, boundaries: countBoundaryClasses(module) });
}

describe('launch-boundary op attribute', () => {
  it('classifies boundary ops through the registry, not a name list', () => {
    expect(launchBoundaryClass('dot')).toBe('matmul');
    expect(launchBoundaryClass('cublas_gemm')).toBe('matmul');
    expect(launchBoundaryClass('fused_dot_epilogue')).toBe('matmul');
    expect(launchBoundaryClass('conv')).toBe('conv');
    expect(launchBoundaryClass('quantized_conv')).toBe('conv');
    expect(launchBoundaryClass('scaled_dot_product_attention')).toBe('attention');
    expect(launchBoundaryClass('reduce')).toBe('reduce');
    expect(launchBoundaryClass('add')).toBe(null);
  });

  it('counts boundary classes across a module', () => {
    const mod = matmulChain(2);
    expect(countBoundaryClasses(mod).get('matmul')).toBe(2);
    expect(countBoundaryClasses(elementwiseOnly()).size).toBe(0);
  });

  it('countLaunchBoundaries ignores ops with no boundary class', () => {
    const func = elementwiseOnly().functions().next().value;
    expect(countLaunchBoundaries(func.ops()).size).toBe(0);
  });
});

describe('graph split strategy selection', () => {
  it('selects the matmul-chain strategy once the target threshold is reached', () => {
    const target = CUDATarget();
    expect(strategyFor(matmulChain(1), target)).toBe(null);
    expect(strategyFor(matmulChain(2), target).name).toBe('matmul-chain');
  });

  it('never splits on a target that declares no split thresholds', () => {
    expect(strategyFor(matmulChain(4), CPUTarget())).toBe(null);
  });

  it('honours a per-target threshold override instead of a hardcoded op count', () => {
    const eager = CUDATarget({ attrs: { [TargetAttr.GRAPH_SPLIT]: { matmul: 1 } } });
    const lazy = CUDATarget({ attrs: { [TargetAttr.GRAPH_SPLIT]: { matmul: 5 } } });
    expect(strategyFor(matmulChain(1), eager).name).toBe('matmul-chain');
    expect(strategyFor(matmulChain(4), lazy)).toBe(null);
  });

  it('lets the cublas strategy win over the boundary-count strategies', () => {
    const s = strategyFor(matmulChain(2), CUDATarget(), { matmulBackend: 'cublas' });
    expect(s.name).toBe('cublas');
  });

  it('selects the webgpu strategy from the target, not a driver flag', () => {
    expect(strategyFor(elementwiseOnly(), WebGPUTarget()).name).toBe('webgpu-scan');
    expect(strategyFor(elementwiseOnly(), CPUTarget())).toBe(null);
  });
});

describe('scheduling defaults come from the target', () => {
  it('reads scheduling defaults off target attributes', () => {
    expect(new CompilerConfig({ target: WebGPUTarget() }).scheduling.enabled).toBe(true);
    expect(new CompilerConfig({ target: CUDATarget() }).scheduling.enabled).toBe(false);
    expect(new CompilerConfig({ target: CUDATarget() }).scheduling.gpuTiling).toBe(true);
    expect(new CompilerConfig({ target: CPUTarget() }).scheduling.gpuTiling).toBe(false);
  });

  it('lets explicit config override the target defaults', () => {
    const cfg = new CompilerConfig({ target: CUDATarget(), scheduling: { gpuTiling: false, enabled: true } });
    expect(cfg.scheduling.gpuTiling).toBe(false);
    expect(cfg.scheduling.enabled).toBe(true);
  });
});
