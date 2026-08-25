import * as mlfw from 'mlfw/index.js';
import { lineFromStack, MODEL_SOURCE_URL } from './source_map.js';

export type ModelRun = {
  model: unknown;
  inputs: unknown[];
  baseLine: number;
};

const RUN_HINT = 'call run(model, inputs) at the end of your code, for example: run(model, [x])';
const PROBE = '__probe(new Error());';

export function frameworkGlobals(): string[] {
  return Object.keys(mlfw).filter(name => name !== 'default').sort();
}

export function evaluateModelSource(source: string): ModelRun {
  const names = frameworkGlobals();
  const values = names.map(name => (mlfw as Record<string, unknown>)[name]);

  let captured: { model: unknown; inputs: unknown[] } | null = null;
  let baseLine = 0;

  const run = (model: unknown, inputs: unknown): void => {
    if (captured) throw new Error('run() was called more than once; a visualized compile takes a single model');
    if (!Array.isArray(inputs)) throw new Error('run(model, inputs): inputs must be an array of tensors');
    captured = { model, inputs };
  };

  const probe = (error: Error): void => {
    baseLine = lineFromStack(error.stack, 0) ?? 0;
  };

  const body = `${PROBE}\n${source}\n;return undefined;\n//# sourceURL=${MODEL_SOURCE_URL}`;
  const factory = new Function(...names, 'run', '__probe', body) as (...args: unknown[]) => void;
  factory(...values, run, probe);

  if (!captured) throw new Error(`no model to compile: ${RUN_HINT}`);
  return { ...(captured as { model: unknown; inputs: unknown[] }), baseLine };
}
