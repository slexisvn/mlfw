import { deriveSketches } from './derivation.js';

export function getSketchesForBlock(primFunc, blockName, target, blockMap, opts = {}) {
  return deriveSketches(primFunc, blockName, target, opts);
}
