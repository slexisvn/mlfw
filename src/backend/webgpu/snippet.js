export function buildSnippet(kernel) {
  const { source, name, metadata } = kernel;
  const { bindings, dispatchSize } = metadata;

  const lines = [];
  lines.push('(async () => {');
  lines.push('const adapter = await navigator.gpu.requestAdapter();');
  lines.push('const device = await adapter.requestDevice();');
  lines.push('');
  lines.push('const wgsl = ' + JSON.stringify(source) + ';');
  lines.push('');
  lines.push('const pipeline = device.createComputePipeline({');
  lines.push('  layout: "auto",');
  lines.push('  compute: { module: device.createShaderModule({ code: wgsl }), entryPoint: ' + JSON.stringify(name) + ' }');
  lines.push('});');
  lines.push('');
  lines.push('const entries = [];');
  lines.push('const gpuBuffers = [];');
  lines.push('');

  for (const b of bindings) {
    if (b.name === '_shapes') {
      emitShapeBinding(lines, b);
    } else if (b.packed) {
      emitPackedBinding(lines, b);
    } else if (b.mode === 'read_write') {
      emitOutputBinding(lines, b);
    } else {
      emitInputBinding(lines, b);
    }
    lines.push('');
  }

  lines.push('const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });');
  lines.push('const encoder = device.createCommandEncoder();');
  lines.push('const pass = encoder.beginComputePass();');
  lines.push('pass.setPipeline(pipeline);');
  lines.push('pass.setBindGroup(0, bindGroup);');
  lines.push(`pass.dispatchWorkgroups(${dispatchSize[0]}, ${dispatchSize[1]}, ${dispatchSize[2]});`);
  lines.push('pass.end();');
  lines.push('');

  emitReadback(lines, bindings);

  lines.push('');
  lines.push('gpuBuffers.forEach(b => b.destroy());');
  lines.push('device.destroy();');
  lines.push('})();');
  return lines.join('\n');
}

function emitShapeBinding(lines, b) {
  lines.push(`// binding ${b.index}: uniform (_shapes)`);
  lines.push('{');
  lines.push('  const shapeData = new Uint32Array([/* shape values */]);');
  lines.push('  const size = Math.max(Math.ceil(shapeData.byteLength / 16) * 16, 16);');
  lines.push('  const buf = device.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });');
  lines.push('  device.queue.writeBuffer(buf, 0, shapeData);');
  lines.push('  entries.push({ binding: ' + b.index + ', resource: { buffer: buf } });');
  lines.push('  gpuBuffers.push(buf);');
  lines.push('}');
}

function emitPackedBinding(lines, b) {
  const rw = b.mode === 'read_write';
  const usage = rw
    ? 'GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST'
    : 'GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST';
  lines.push(`// binding ${b.index}: packed ${b.name} (${b.packed.length} buffers, ${b.packedSize} elems)`);
  lines.push('{');
  lines.push(`  const buf = device.createBuffer({ size: ${b.packedSize * 4}, usage: ${usage}, mappedAtCreation: true });`);
  lines.push('  const mapped = new Float32Array(buf.getMappedRange());');
  if (rw) {
    lines.push('  mapped.fill(0);');
  } else {
    for (const entry of b.packed) {
      lines.push(`  // mapped.set(${entry.name}_data, ${entry.offset}); // size=${entry.size}`);
    }
  }
  lines.push('  buf.unmap();');
  lines.push('  entries.push({ binding: ' + b.index + ', resource: { buffer: buf } });');
  lines.push('  gpuBuffers.push(buf);');
  lines.push('}');
}

function emitOutputBinding(lines, b) {
  lines.push(`// binding ${b.index}: output ${b.name}`);
  lines.push('{');
  lines.push('  const size = N * 4; // N = output element count');
  lines.push('  const buf = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, mappedAtCreation: true });');
  lines.push('  new Float32Array(buf.getMappedRange()).fill(0);');
  lines.push('  buf.unmap();');
  lines.push('  entries.push({ binding: ' + b.index + ', resource: { buffer: buf } });');
  lines.push('  gpuBuffers.push(buf);');
  lines.push('}');
}

function emitInputBinding(lines, b) {
  lines.push(`// binding ${b.index}: input ${b.name}`);
  lines.push('{');
  lines.push(`  const data = ${b.name}_data; // Float32Array`);
  lines.push('  const buf = device.createBuffer({ size: Math.max(data.byteLength, 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, mappedAtCreation: true });');
  lines.push('  new Float32Array(buf.getMappedRange()).set(data);');
  lines.push('  buf.unmap();');
  lines.push('  entries.push({ binding: ' + b.index + ', resource: { buffer: buf } });');
  lines.push('  gpuBuffers.push(buf);');
  lines.push('}');
}

function emitReadback(lines, bindings) {
  const outputs = bindings.filter(b => b.mode === 'read_write');
  if (outputs.length === 0) {
    lines.push('device.queue.submit([encoder.finish()]);');
    return;
  }

  for (let i = 0; i < bindings.length; i++) {
    const b = bindings[i];
    if (b.mode !== 'read_write') continue;

    if (b.packed) {
      for (const entry of b.packed) {
        lines.push('{');
        lines.push(`  const rb = device.createBuffer({ size: ${entry.size * 4}, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });`);
        lines.push(`  encoder.copyBufferToBuffer(gpuBuffers[${i}], ${entry.offset * 4}, rb, 0, ${entry.size * 4});`);
        lines.push('  device.queue.submit([encoder.finish()]);');
        lines.push('  await rb.mapAsync(GPUMapMode.READ);');
        lines.push(`  console.log("${entry.name}:", new Float32Array(rb.getMappedRange()));`);
        lines.push('  rb.unmap(); rb.destroy();');
        lines.push('}');
      }
    } else {
      lines.push('{');
      lines.push(`  const src = gpuBuffers[${i}];`);
      lines.push('  const rb = device.createBuffer({ size: src.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });');
      lines.push('  encoder.copyBufferToBuffer(src, 0, rb, 0, src.size);');
      lines.push('  device.queue.submit([encoder.finish()]);');
      lines.push('  await rb.mapAsync(GPUMapMode.READ);');
      lines.push(`  console.log("${b.name}:", new Float32Array(rb.getMappedRange()));`);
      lines.push('  rb.unmap(); rb.destroy();');
      lines.push('}');
    }
  }
}
