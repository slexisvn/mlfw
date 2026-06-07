export function buildSnippet(kernel) {
  const { source, name, metadata } = kernel;
  const args = [];
  for (let i = 0; i < metadata.paramCount; i++) args.push(`buf_${i}`);

  const lines = [];
  lines.push(source);
  lines.push('');
  lines.push(`// ${name}(${args.map(a => `/* ${a}: Float32Array */`).join(', ')});`);
  return lines.join('\n');
}
