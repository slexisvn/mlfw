import { encodeWat } from './wat_encoder.js';

export function buildSnippet(kernel) {
  const { source, name, metadata } = kernel;
  const { bufferOffsets, imports } = metadata;
  const offsets = [...bufferOffsets.entries()];

  const binary = encodeWat(source);

  const lines = [];
  lines.push('(async () => {');
  lines.push('');
  lines.push('const binary = new Uint8Array([' + binary.join(',') + ']);');
  lines.push('');

  if (imports && imports.size > 0) {
    lines.push('const mathImports = {');
    for (const [fn] of imports) {
      if (fn === 'fmod') lines.push('  fmod: (a, b) => a % b,');
      else if (fn === 'rsqrt') lines.push('  rsqrt: x => 1 / Math.sqrt(x),');
      else lines.push(`  ${fn}: Math.${fn},`);
    }
    lines.push('};');
    lines.push('const { instance } = await WebAssembly.instantiate(binary, { math: mathImports });');
  } else {
    lines.push('const { instance } = await WebAssembly.instantiate(binary);');
  }

  lines.push('const memory = instance.exports.memory;');
  lines.push('');

  for (const [bufName, offset] of offsets) {
    lines.push(`// ${bufName}: new Float32Array(memory.buffer, ${offset}, N).set(data);`);
  }

  lines.push('');
  lines.push(`instance.exports.${name}(${offsets.map(([, o]) => o).join(', ')});`);
  lines.push('');

  const last = offsets[offsets.length - 1];
  if (last) {
    lines.push(`const result = new Float32Array(memory.buffer, ${last[1]}, N);`);
    lines.push('console.log(result);');
  }

  lines.push('})();');
  return lines.join('\n');
}
