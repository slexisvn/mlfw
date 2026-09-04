import { TargetFeatures, TargetKind } from '../../compiler/support/target.js';
import { TargetAttr, targetAttr } from '../../compiler/support/target_attrs.js';

import { emitTeraModule, teraEntries } from './emit.js';
import { TERAC_DEFAULTS, TeracRuntimeModule } from './runtime_module.js';

import type { TargetOverrides } from '../../compiler/support/target.js';
import type { CompilerConfig, RuntimeModuleLike } from '../../compiler/support/config_types.js';
import type { GraphModule } from '../../compiler/ir/graph/module.js';
import type { TeracOptions } from './runtime_module.js';

export const TERAC_OPTIONS = 'teracOptions';

export type TeracTargetOptions = TeracOptions & TargetOverrides;

export function compileWithTerac(module: GraphModule, config: CompilerConfig): RuntimeModuleLike {
  const options = targetAttr<TeracOptions>(config.target, TERAC_OPTIONS) || {};
  const mlir = emitTeraModule(module);
  return new TeracRuntimeModule(mlir, teraEntries(module), options);
}

export function TeracTarget(options: TeracTargetOptions = {}): TargetFeatures {
  const {
    device, optLevel, build = null, library = null, llvmBin = null, ...overrides
  } = { ...TERAC_DEFAULTS, ...options };
  return new TargetFeatures({
    kind: device === 'cuda' ? TargetKind.CUDA : TargetKind.CPU,
    name: `terac_${device}`,
    ...overrides,
    attrs: {
      [TargetAttr.FUSION]: { enabled: false },
      [TargetAttr.EXTERNAL_COMPILER]: compileWithTerac,
      [TERAC_OPTIONS]: { device, optLevel, build, library, llvmBin },
      ...(overrides.attrs || {}),
    },
  });
}
