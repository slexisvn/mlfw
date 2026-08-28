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

export function kernelBody(src) {
  const start = src.indexOf('__global__');
  const kernel = start < 0 ? src : src.slice(start);
  const match = kernel.match(/\{([\s\S]*)\}\s*$/);
  return match ? match[1] : null;
}

export function extractDeclaredVars(src) {
  const declared = new Set();
  const paramMatch = src.match(/__global__\s+void\s+\w+\(([^)]*)\)/);
  if (paramMatch) {
    for (const p of paramMatch[1].split(',')) {
      const name = p.trim().split(/\s+/).pop().replace('*', '');
      if (name) declared.add(name);
    }
  }
  for (const m of src.matchAll(/(?:const\s+int|int|float|double|__half)\s+(\w+)\s*(?:=|\[)/g)) {
    declared.add(m[1]);
  }
  for (const m of src.matchAll(/for\s*\(\s*int\s+(\w+)/g)) {
    declared.add(m[1]);
  }
  declared.add('blockIdx');
  declared.add('threadIdx');
  declared.add('blockDim');
  declared.add('gridDim');
  return declared;
}

const C_KEYWORDS = /^(const|int|float|double|void|for|if|else|while|return|__global__|__shared__|__device__|__forceinline__|__half|pragma|unroll|INFINITY|alloca|sizeof)$/;

export function findUndeclaredVars(src) {
  const declared = extractDeclaredVars(src);
  const body = kernelBody(src);
  if (body === null) return [];
  const undeclared = [];
  for (const m of body.matchAll(/([a-zA-Z_]\w*)(?!\s*\()/g)) {
    const name = m[1];
    if (C_KEYWORDS.test(name)) continue;
    if (m.index > 0 && body[m.index - 1] === '.') continue;
    if (declared.has(name) || undeclared.includes(name)) continue;
    undeclared.push(name);
  }
  return undeclared;
}
