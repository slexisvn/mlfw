export function buildSnippet(kernel) {
  const { source, name, metadata } = kernel;
  const { blockDim, gridDim, sharedMemBytes, params } = metadata;

  const lines = [];
  lines.push(source);
  lines.push('');
  lines.push('// Launch:');
  lines.push(`//   dim3 block(${blockDim[0]}, ${blockDim[1]}, ${blockDim[2]});`);
  lines.push(`//   dim3 grid(${gridDim[0]}, ${gridDim[1]}, ${gridDim[2]});`);
  if (sharedMemBytes > 0) {
    lines.push(`//   ${name}<<<grid, block, ${sharedMemBytes}>>>(${params.join(', ')});`);
  } else {
    lines.push(`//   ${name}<<<grid, block>>>(${params.join(', ')});`);
  }
  return lines.join('\n');
}
