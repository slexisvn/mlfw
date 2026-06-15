import { ANY, STRING, TENSOR, moduleType, functionType } from '../vendor/types.js';
import { HOST_GLOBALS } from '../vendor/typechecker.js';

const SCALAR_KINDS = new Set(['device', 'dtype', 'constant']);

function returnType(builtin) {
  if (!builtin.returns) return ANY;
  if (builtin.returns === 'Tensor') return TENSOR;
  return moduleType(builtin.returns);
}

export function buildBuiltinEnv(languageData) {
  const builtinTypes = new Map();
  const builtinNames = new Set();
  for (const builtin of languageData.builtins ?? []) {
    builtinNames.add(builtin.name);
    builtinTypes.set(builtin.name, SCALAR_KINDS.has(builtin.kind) ? STRING : functionType([], returnType(builtin), true, 0));
  }
  for (const name of languageData.keywords ?? []) builtinNames.add(name);
  for (const name of Object.keys(languageData.pseudoTypes ?? {})) builtinNames.add(name);
  for (const name of HOST_GLOBALS) builtinNames.add(name);
  return { builtinNames, builtinTypes };
}
