import { parse } from './parser.js';
import { buildBuiltinTypes, collectBuiltinNames } from './type_signatures.js';
import { typecheck, HOST_GLOBALS } from './typechecker.js';

export function checkSource(source) {
  const builtinNames = new Set([...collectBuiltinNames(), ...HOST_GLOBALS]);
  const builtinTypes = buildBuiltinTypes();
  return { diagnostics: typecheck(parse(source), { builtinNames, builtinTypes }) };
}
