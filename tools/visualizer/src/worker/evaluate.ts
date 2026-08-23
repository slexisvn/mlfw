import * as mlfw from 'mlfw/index.js';

export type ModelRun = {
  model: unknown;
  inputs: unknown[];
};

const RUN_HINT = 'call run(model, inputs) at the end of your code, for example: run(model, [x])';

export function frameworkGlobals(): string[] {
  return Object.keys(mlfw).filter(name => name !== 'default').sort();
}

export function evaluateModelSource(source: string): ModelRun {
  const names = frameworkGlobals();
  const values = names.map(name => (mlfw as Record<string, unknown>)[name]);

  let captured: ModelRun | null = null;
  const run = (model: unknown, inputs: unknown): void => {
    if (captured) throw new Error('run() was called more than once; a visualized compile takes a single model');
    if (!Array.isArray(inputs)) throw new Error('run(model, inputs): inputs must be an array of tensors');
    captured = { model, inputs };
  };

  const body = `${source}\n;return undefined;`;
  const factory = new Function(...names, 'run', body) as (...args: unknown[]) => void;
  factory(...values, run);

  if (!captured) throw new Error(`no model to compile: ${RUN_HINT}`);
  return captured;
}
