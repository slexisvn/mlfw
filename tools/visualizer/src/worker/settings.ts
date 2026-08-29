import { compile } from 'mlfw/index.js';
import { PassContext } from 'mlfw/compiler/passes/pass.js';
import { TARGET_FACTORIES } from './targets.js';
import { SEARCH_BUDGET } from '../catalog/tuning.js';
import type { CompileOptions } from '../protocol.js';

const NON_IDENTIFIER = /[^A-Za-z0-9_$]/g;
const FALLBACK_NAME = 'model';

export type Compilable = Parameters<typeof compile>[0];
type ForwardFn = (...args: unknown[]) => unknown;

function identifier(name: string): string {
  const cleaned = name.replace(NON_IDENTIFIER, '');
  return /^[A-Za-z_$]/.test(cleaned) ? cleaned : FALLBACK_NAME;
}

function wrap(forward: ForwardFn): Compilable {
  const wrapper = class { forward = forward; };
  Object.defineProperty(wrapper, 'name', { value: identifier(forward.name) });
  return new wrapper() as unknown as Compilable;
}

export function asCompilable(model: unknown): Compilable {
  if (model && typeof (model as { forward?: unknown }).forward === 'function') return model as Compilable;
  if (typeof model === 'function') return wrap(model as ForwardFn);
  throw new Error('run(model, inputs): model must be an nn.Module or a function');
}

export function compilerOptions(options: CompileOptions, alsoDisabled: readonly string[] = []): Record<string, unknown> {
  const disabled = [...options.disabledPasses, ...alsoDisabled];
  return {
    target: TARGET_FACTORIES[options.target](),
    verify: options.verify,
    fusion: { enabled: options.fusion, strategy: options.fusionStrategy },
    scheduling: { enabled: options.scheduling, autotune: options.autotune, ...SEARCH_BUDGET },
    optimization: { layout: options.layout },
    passContext: disabled.length > 0 ? new PassContext({ disabledPasses: disabled }) : null,
  };
}
