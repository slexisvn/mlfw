const TYPED_ARRAYS = 'Float32Array|Float64Array|Int32Array|Int16Array|Int8Array|Uint32Array|Uint16Array|Uint8Array';

export function stripLiterals(src) {
  return src
    .replace(/\/\/[^\n]*/g, '')
    .replace(/;;[^\n]*/g, '')
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/`(?:[^`\\]|\\.)*`/g, '``')
    .replace(/"(?:[^"\\]|\\.)*"/g, '""')
    .replace(/'(?:[^'\\]|\\.)*'/g, "''");
}

export function countLoops(src) {
  const s = stripLiterals(src);
  return (s.match(/\bfor\s*\(/g) || []).length + (s.match(/\(loop\s/g) || []).length;
}

export function countTempBuffers(src) {
  const s = stripLiterals(src);
  return (s.match(new RegExp(`new\\s+(?:${TYPED_ARRAYS})\\b`, 'g')) || []).length;
}
