import * as mlfw from 'mlfw/index.js';
import { lineFromStack, MODEL_SOURCE_URL, recordSourceLocations } from './source_map.js';
import { isLayerClass, siteRecording } from './layer_sites.js';

export type ModelRun = {
  model: unknown;
  inputs: unknown[];
};

const RUN_HINT = 'call run(model, inputs) at the end of your code, for example: run(model, [x])';
const PROBE = '__probe(new Error());';

export function frameworkGlobals(): string[] {
  return Object.keys(mlfw).filter(name => name !== 'default').sort();
}

export function evaluateModelSource(source: string, beginRecording: (stop: () => void) => void): ModelRun {
  const names = frameworkGlobals();
  const values = names.map(name => {
    const value = (mlfw as Record<string, unknown>)[name];
    return isLayerClass(value) ? siteRecording(value) : value;
  });

  let captured: { model: unknown; inputs: unknown[] } | null = null;

  const run = (model: unknown, inputs: unknown): void => {
    if (captured) throw new Error('run() was called more than once; a visualized compile takes a single model');
    if (!Array.isArray(inputs)) throw new Error('run(model, inputs): inputs must be an array of tensors');
    captured = { model, inputs };
  };

  const probe = (error: Error): void => {
    beginRecording(recordSourceLocations(lineFromStack(error.stack, 0) ?? 0));
  };

  const body = `${PROBE}\n${source}\n;return undefined;\n//# sourceURL=${MODEL_SOURCE_URL}`;
  const factory = new Function(...names, 'run', '__probe', body) as (...args: unknown[]) => void;
  factory(...values, run, probe);

  if (!captured) throw new Error(`no model to compile: ${RUN_HINT}`);
  return captured as { model: unknown; inputs: unknown[] };
}
