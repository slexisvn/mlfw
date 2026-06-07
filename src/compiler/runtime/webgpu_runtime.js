let _gpuDevice = null;
let _gpuInitPromise = null;
let _bufferUsage = null;
let _mapMode = null;
let _dawnInstance = null;
let _exitRegistered = false;

async function ensureDevice() {
  if (_gpuDevice) return _gpuDevice;
  if (_gpuInitPromise) return _gpuInitPromise;

  _gpuInitPromise = (async () => {
    let gpu = typeof navigator !== 'undefined' && navigator.gpu;
    if (!gpu) {
      try {
        const mod = await import('webgpu');
        _dawnInstance = mod;
        gpu = mod.create([]);
        if (mod.globals) {
          _bufferUsage = mod.globals.GPUBufferUsage;
          _mapMode = mod.globals.GPUMapMode;
        }
      } catch (_) {
        throw new Error('WebGPU not available: install the "webgpu" npm package or run in a browser with WebGPU support');
      }
    }

    if (!_bufferUsage && typeof GPUBufferUsage !== 'undefined') {
      _bufferUsage = GPUBufferUsage;
      _mapMode = GPUMapMode;
    }

    const adapter = await gpu.requestAdapter();
    if (!adapter) throw new Error('WebGPU: no adapter found');

    const adapterLimits = adapter.limits || {};
    const requiredLimits = {};
    const liftKeys = [
      'maxStorageBuffersPerShaderStage',
      'maxStorageBufferBindingSize',
      'maxBufferSize',
      'maxBindingsPerBindGroup',
    ];
    for (const key of liftKeys) {
      if (adapterLimits[key] !== undefined) requiredLimits[key] = adapterLimits[key];
    }
    _gpuDevice = await adapter.requestDevice({ requiredLimits });

    if (!_exitRegistered && typeof process !== 'undefined' && process.on) {
      _exitRegistered = true;
      process.on('exit', () => {
        if (_gpuDevice) {
          _gpuDevice.destroy();
          _gpuDevice = null;
        }
      });
    }

    return _gpuDevice;
  })();

  return _gpuInitPromise;
}

function bufUsage() { return _bufferUsage; }
function mapModeRead() { return _mapMode.READ; }

export function resetDevice() {
  if (_gpuDevice) {
    _gpuDevice.destroy();
    _gpuDevice = null;
  }
  _gpuInitPromise = null;
  _bufferUsage = null;
  _mapMode = null;
  _dawnInstance = null;
}

function createPipeline(device, kernel) {
  const shaderModule = device.createShaderModule({ code: kernel.source });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: shaderModule, entryPoint: kernel.name }
  });
  return { pipeline, shaderModule };
}

export async function instantiateWebGPU(kernel) {
  const device = await ensureDevice();
  const { pipeline } = createPipeline(device, kernel);
  return {
    device,
    pipeline,
    kernel,
    workgroupSize: kernel.metadata.workgroupSize,
    dispatchSize: kernel.metadata.dispatchSize,
    bindings: kernel.metadata.bindings,
  };
}

function buildParamIndex(bindings) {
  const map = new Map();
  let idx = 0;
  for (const b of bindings) {
    if (b.name === '_shapes') continue;
    if (b.packed) {
      for (const entry of b.packed) {
        map.set(entry.name, idx++);
      }
    } else {
      map.set(b.name, idx++);
    }
  }
  return map;
}

export async function runWebGPUKernel(instance, tensorArgs, shapeValues) {
  const { device, pipeline, bindings, dispatchSize } = instance;
  const BU = bufUsage();

  const gpuBuffers = [];
  const bindGroupEntries = [];
  const paramIdx = buildParamIndex(bindings);

  for (let i = 0; i < bindings.length; i++) {
    const binding = bindings[i];

    if (binding.name === '_shapes') {
      const shapeData = new Uint32Array(shapeValues || []);
      const alignedSize = Math.max(Math.ceil(shapeData.byteLength / 16) * 16, 16);
      const uniformBuf = device.createBuffer({
        size: alignedSize,
        usage: BU.UNIFORM | BU.COPY_DST,
      });
      device.queue.writeBuffer(uniformBuf, 0, shapeData);
      gpuBuffers.push(uniformBuf);
      bindGroupEntries.push({ binding: binding.index, resource: { buffer: uniformBuf } });
      continue;
    }

    if (binding.packed) {
      const totalBytes = binding.packedSize * 4;
      const isOutput = binding.mode === 'read_write';
      const usage = isOutput
        ? BU.STORAGE | BU.COPY_SRC | BU.COPY_DST
        : BU.STORAGE | BU.COPY_DST;

      const gpuBuf = device.createBuffer({ size: Math.max(totalBytes, 4), usage, mappedAtCreation: true });
      const mapped = new Float32Array(gpuBuf.getMappedRange());
      for (const entry of binding.packed) {
        const srcIdx = paramIdx.get(entry.name);
        const src = tensorArgs[srcIdx];
        if (src) mapped.set(src, entry.offset);
      }
      gpuBuf.unmap();
      gpuBuffers.push(gpuBuf);
      bindGroupEntries.push({ binding: binding.index, resource: { buffer: gpuBuf } });
      continue;
    }

    const idx = paramIdx.get(binding.name);
    const data = tensorArgs[idx];
    const byteLength = data.byteLength;
    const isOutput = binding.mode === 'read_write';
    const usage = isOutput
      ? BU.STORAGE | BU.COPY_SRC | BU.COPY_DST
      : BU.STORAGE | BU.COPY_DST;

    const gpuBuf = device.createBuffer({ size: Math.max(byteLength, 4), usage, mappedAtCreation: true });
    new Float32Array(gpuBuf.getMappedRange()).set(data);
    gpuBuf.unmap();

    gpuBuffers.push(gpuBuf);
    bindGroupEntries.push({ binding: binding.index, resource: { buffer: gpuBuf } });
  }

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: bindGroupEntries,
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(dispatchSize[0], dispatchSize[1], dispatchSize[2]);
  pass.end();

  const outputReadbacks = [];
  for (let i = 0; i < bindings.length; i++) {
    const binding = bindings[i];
    if (binding.name === '_shapes') continue;
    if (binding.mode !== 'read_write') continue;

    const gpuBuf = gpuBuffers[i];

    if (binding.packed) {
      for (const entry of binding.packed) {
        const byteOff = entry.offset * 4;
        const byteLen = entry.size * 4;
        const readBuf = device.createBuffer({ size: byteLen, usage: BU.MAP_READ | BU.COPY_DST });
        encoder.copyBufferToBuffer(gpuBuf, byteOff, readBuf, 0, byteLen);
        outputReadbacks.push({ readBuf, tensorIdx: paramIdx.get(entry.name), size: entry.size });
      }
    } else {
      const size = gpuBuf.size;
      const readBuf = device.createBuffer({ size, usage: BU.MAP_READ | BU.COPY_DST });
      encoder.copyBufferToBuffer(gpuBuf, 0, readBuf, 0, size);
      outputReadbacks.push({ readBuf, tensorIdx: paramIdx.get(binding.name), size: size / 4 });
    }
  }

  device.queue.submit([encoder.finish()]);

  for (const rb of outputReadbacks) {
    await rb.readBuf.mapAsync(mapModeRead());
    const result = new Float32Array(rb.readBuf.getMappedRange());
    tensorArgs[rb.tensorIdx].set(result.subarray(0, rb.size));
    rb.readBuf.unmap();
    rb.readBuf.destroy();
  }

  for (const buf of gpuBuffers) buf.destroy();
}
