import { tensorToContiguous } from '../../dispatcher/jit_dispatch.js';

export function buildWebGPUSnippet(wgsl, entryPoint, meta, paramData, outputShapes) {
  const bindings = meta.bindings;
  const dispatch = meta.dispatchSize;

  const lines = [];
  lines.push('(async () => {');
  lines.push('const adapter = await navigator.gpu.requestAdapter();');
  lines.push('const device = await adapter.requestDevice();');
  lines.push('');
  lines.push('const wgsl = ' + JSON.stringify(wgsl) + ';');
  lines.push('');
  lines.push('const pipeline = device.createComputePipeline({');
  lines.push('  layout: "auto",');
  lines.push('  compute: { module: device.createShaderModule({ code: wgsl }), entryPoint: ' + JSON.stringify(entryPoint) + ' }');
  lines.push('});');
  lines.push('');

  lines.push('const inputData = ' + JSON.stringify(paramData) + ';');
  lines.push('');

  const outputNumels = outputShapes.map(s => s.reduce((a, b) => a * b, 1));
  lines.push('const outputSizes = ' + JSON.stringify(outputNumels) + ';');
  lines.push('const outputShapes = ' + JSON.stringify(outputShapes) + ';');
  lines.push('');

  lines.push('const entries = [];');
  lines.push('const gpuBuffers = [];');
  lines.push('let inputIdx = 0;');
  lines.push('');

  for (let i = 0; i < bindings.length; i++) {
    const b = bindings[i];
    if (b.name === '_shapes') {
      lines.push('// binding ' + b.index + ': uniform shapes');
      lines.push('{');
      lines.push('  const shapeData = new Uint32Array([/* shape values */]);');
      lines.push('  const size = Math.max(Math.ceil(shapeData.byteLength / 16) * 16, 16);');
      lines.push('  const buf = device.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });');
      lines.push('  device.queue.writeBuffer(buf, 0, shapeData);');
      lines.push('  entries.push({ binding: ' + b.index + ', resource: { buffer: buf } });');
      lines.push('  gpuBuffers.push(buf);');
      lines.push('}');
    } else if (b.mode === 'read_write') {
      lines.push('// binding ' + b.index + ': output ' + b.name);
      lines.push('{');
      lines.push('  const n = outputSizes.shift();');
      lines.push('  const buf = device.createBuffer({ size: Math.max(n * 4, 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, mappedAtCreation: true });');
      lines.push('  new Float32Array(buf.getMappedRange()).fill(0);');
      lines.push('  buf.unmap();');
      lines.push('  entries.push({ binding: ' + b.index + ', resource: { buffer: buf } });');
      lines.push('  gpuBuffers.push(buf);');
      lines.push('}');
    } else {
      lines.push('// binding ' + b.index + ': input ' + b.name);
      lines.push('{');
      lines.push('  const data = new Float32Array(inputData[inputIdx++]);');
      lines.push('  const buf = device.createBuffer({ size: Math.max(data.byteLength, 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, mappedAtCreation: true });');
      lines.push('  new Float32Array(buf.getMappedRange()).set(data);');
      lines.push('  buf.unmap();');
      lines.push('  entries.push({ binding: ' + b.index + ', resource: { buffer: buf } });');
      lines.push('  gpuBuffers.push(buf);');
      lines.push('}');
    }
    lines.push('');
  }

  lines.push('const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });');
  lines.push('');
  lines.push('const encoder = device.createCommandEncoder();');
  lines.push('const pass = encoder.beginComputePass();');
  lines.push('pass.setPipeline(pipeline);');
  lines.push('pass.setBindGroup(0, bindGroup);');
  lines.push('pass.dispatchWorkgroups(' + dispatch[0] + ', ' + dispatch[1] + ', ' + dispatch[2] + ');');
  lines.push('pass.end();');
  lines.push('');

  lines.push('const readBufs = [];');
  for (let i = 0; i < bindings.length; i++) {
    const b = bindings[i];
    if (b.mode !== 'read_write') continue;
    lines.push('{');
    lines.push('  const src = gpuBuffers[' + i + '];');
    lines.push('  const rb = device.createBuffer({ size: src.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });');
    lines.push('  encoder.copyBufferToBuffer(src, 0, rb, 0, src.size);');
    lines.push('  readBufs.push(rb);');
    lines.push('}');
  }
  lines.push('');
  lines.push('device.queue.submit([encoder.finish()]);');
  lines.push('');
  lines.push('for (const rb of readBufs) {');
  lines.push('  await rb.mapAsync(GPUMapMode.READ);');
  lines.push('  const result = new Float32Array(rb.getMappedRange());');
  lines.push('  console.log("output:", [...result]);');
  lines.push('  console.log("shape:", outputShapes.shift());');
  lines.push('  rb.unmap();');
  lines.push('}');
  lines.push('');
  lines.push('gpuBuffers.forEach(b => b.destroy());');
  lines.push('device.destroy();');
  lines.push('})();');

  return lines.join('\n');
}

export function extractWebGPUSnippet(compiled, inputs) {
  const kernels = compiled.result.listKernels();
  if (kernels.length === 0) return null;
  const name = kernels[0];
  const kernel = compiled.result.module.kernels.get(name);
  if (!kernel || kernel.metadata.kind !== 'webgpu') return null;

  const meta = kernel.metadata;
  const params = compiled.capturedParams;
  const outputTypes = compiled.outputTypes;

  const allData = [];
  for (let i = 0; i < inputs.length; i++) {
    allData.push(Array.from(tensorToContiguous(inputs[i])));
  }
  for (let i = 0; i < params.length; i++) {
    allData.push(Array.from(tensorToContiguous(params[i])));
  }

  const outputShapes = [];
  for (let i = 0; i < outputTypes.length; i++) {
    outputShapes.push(outputTypes[i].shape);
  }

  return buildWebGPUSnippet(kernel.source, name, meta, allData, outputShapes);
}
