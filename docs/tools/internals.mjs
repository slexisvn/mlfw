import './freshness.mjs';
import {
  trace, resetVarCounter, lowerGraphToPrimFunc, BackendPipeline, CPUTarget,
} from '../../dist/internals.node.js';

export * from '../../dist/internals.node.js';

export async function lowerToTir(fn, inputs, target = CPUTarget()) {
  resetVarCounter();
  const graph = await trace(fn, inputs);
  const [name] = graph._functions.keys();
  return lowerGraphToPrimFunc(graph._functions.get(name), target);
}

export function toKernel(primFunc, target = CPUTarget()) {
  const source = new BackendPipeline(target).compile(primFunc).source;
  return { source, call: new Function(`return ${source}`)() };
}
