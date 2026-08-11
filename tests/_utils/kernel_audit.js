import { stripLiterals, countLoops, countTempBuffers } from './kernel_source.js';

const NOISE = [
  [/\+\s*0(?!\.\d*[1-9])\b/, '+0'],
  [/\b0\s*\+/, '0+'],
  [/\*\s*0(?!\.\d*[1-9])\b/, '*0'],
  [/\b0\s*\*\s*(?!\.?\d*[1-9])/, '0*'],
  [/\*\s*1\b(?!\.)/, '*1'],
  [/\b1\s*\*\b(?!\.)/, '1*'],
];

export function arithmeticNoise(src) {
  const s = stripLiterals(src);
  return NOISE.filter(([re]) => re.test(s)).map(([, label]) => label);
}

const count = (src, re) => (stripLiterals(src).match(re) || []).length;

export const extent1Loops = (src) => count(src, /for\s*\(\s*let\s+\w+\s*=\s*0;\s*\w+\s*<\s*1;/g);
export const redundantZeroInits = (src) => count(src, /\w+\[\w+\]\s*=\s*0\.?0?\s*;/g);
export const boundsChecks = (src) => count(src, />=\s*0/g);
export const modulos = (src) => count(src, /[^=!<>]%\s*\d+/g);
export const truncs = (src) => count(src, /Math\.trunc/g);

export function audit(src, name) {
  return {
    name,
    lines: src.split('\n').length,
    loops: countLoops(src),
    tempBuffers: countTempBuffers(src),
    arithmeticNoise: arithmeticNoise(src),
    extent1Loops: extent1Loops(src),
    zeroInits: redundantZeroInits(src),
    boundsChecks: boundsChecks(src),
    modulos: modulos(src),
    truncs: truncs(src),
  };
}
