import { wgslType, wgslBytes } from '../../backend/dtype_map.js';
import { bf16ToF32, f32ToBf16 } from '../../tensor/utils/half.js';

let _gpuDevice = null;
let _gpuInitPromise = null;
let _bufferUsage = null;
let _mapMode = null;
let _shaderStage = null;
let _dawnInstance = null;
let _exitRegistered = false;

function wgslViewCtor(dtype) {
  switch (wgslType(dtype)) {
    case 'i32': return Int32Array;
    case 'u32': return Uint32Array;
    case 'f16': return Uint16Array;
    default: return Float32Array;
  }
}

function packTensorInto(view, src, dtype, offset) {
  if (dtype === 'bf16') {
    for (let i = 0; i < src.length; i++) view[offset + i] = bf16ToF32(src[i]);
  } else if (dtype === 'i64') {
    for (let i = 0; i < src.length; i++) view[offset + i] = Number(BigInt.asIntN(32, src[i]));
  } else {
    view.set(src, offset);
  }
}

function unpackTensorFrom(dst, view, dtype, offset, size) {
  if (dtype === 'bf16') {
    for (let i = 0; i < size; i++) dst[i] = f32ToBf16(view[offset + i]);
  } else if (dtype === 'i64') {
    for (let i = 0; i < size; i++) dst[i] = BigInt(view[offset + i]);
  } else {
    dst.set(view.subarray(offset, offset + size));
  }
}

function align4(n) { return Math.ceil(n / 4) * 4; }

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
          _shaderStage = mod.globals.GPUShaderStage;
        }
      } catch (_) {
        throw new Error('WebGPU not available: install the "webgpu" npm package or run in a browser with WebGPU support');
      }
    }

    if (!_bufferUsage && typeof GPUBufferUsage !== 'undefined') {
      _bufferUsage = GPUBufferUsage;
      _mapMode = GPUMapMode;
      _shaderStage = GPUShaderStage;
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
    const requiredFeatures = [];
    if (adapter.features && adapter.features.has('shader-f16')) requiredFeatures.push('shader-f16');
    _gpuDevice = await adapter.requestDevice({ requiredLimits, requiredFeatures });

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
  _shaderStage = null;
  _dawnInstance = null;
}

function bindingBufferType(binding) {
  if (binding.name === '_shapes') return 'uniform';
  if (binding.mode === 'read_write') return 'storage';
  return 'read-only-storage';
}

function createPipeline(device, kernel) {
  const shaderModule = device.createShaderModule({ code: kernel.source });
  const layoutEntries = [];
  for (const binding of kernel.metadata.bindings) {
    layoutEntries.push({
      binding: binding.index,
      visibility: _shaderStage.COMPUTE,
      buffer: { type: bindingBufferType(binding) },
    });
  }
  const bindGroupLayout = device.createBindGroupLayout({ entries: layoutEntries });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
  const pipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module: shaderModule, entryPoint: kernel.name }
  });
  return { pipeline, shaderModule, bindGroupLayout };
}

export async function instantiateWebGPU(kernel) {
  const device = await ensureDevice();
  const { pipeline, bindGroupLayout } = createPipeline(device, kernel);
  return {
    device,
    pipeline,
    bindGroupLayout,
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
  const { device, pipeline, bindGroupLayout, bindings, dispatchSize } = instance;
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

    const elemBytes = wgslBytes(binding.dtype);
    const ViewCtor = wgslViewCtor(binding.dtype);

    if (binding.packed) {
      const totalBytes = align4(binding.packedSize * elemBytes);
      const isOutput = binding.mode === 'read_write';
      const usage = isOutput
        ? BU.STORAGE | BU.COPY_SRC | BU.COPY_DST
        : BU.STORAGE | BU.COPY_DST;

      const gpuBuf = device.createBuffer({ size: Math.max(totalBytes, 4), usage, mappedAtCreation: true });
      const mapped = new ViewCtor(gpuBuf.getMappedRange());
      for (const entry of binding.packed) {
        const srcIdx = paramIdx.get(entry.name);
        const src = tensorArgs[srcIdx];
        if (src) packTensorInto(mapped, src, entry.dtype, entry.offset);
      }
      gpuBuf.unmap();
      gpuBuffers.push(gpuBuf);
      bindGroupEntries.push({ binding: binding.index, resource: { buffer: gpuBuf } });
      continue;
    }

    const idx = paramIdx.get(binding.name);
    const data = tensorArgs[idx];
    const byteLength = align4(data.length * elemBytes);
    const isOutput = binding.mode === 'read_write';
    const usage = isOutput
      ? BU.STORAGE | BU.COPY_SRC | BU.COPY_DST
      : BU.STORAGE | BU.COPY_DST;

    const gpuBuf = device.createBuffer({ size: Math.max(byteLength, 4), usage, mappedAtCreation: true });
    packTensorInto(new ViewCtor(gpuBuf.getMappedRange()), data, binding.dtype, 0);
    gpuBuf.unmap();

    gpuBuffers.push(gpuBuf);
    bindGroupEntries.push({ binding: binding.index, resource: { buffer: gpuBuf } });
  }

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
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
    const elemBytes = wgslBytes(binding.dtype);
    const ViewCtor = wgslViewCtor(binding.dtype);

    if (binding.packed) {
      for (const entry of binding.packed) {
        const byteOff = entry.offset * elemBytes;
        const byteLen = align4(entry.size * elemBytes);
        const readBuf = device.createBuffer({ size: byteLen, usage: BU.MAP_READ | BU.COPY_DST });
        encoder.copyBufferToBuffer(gpuBuf, byteOff, readBuf, 0, byteLen);
        outputReadbacks.push({ readBuf, tensorIdx: paramIdx.get(entry.name), size: entry.size, dtype: entry.dtype, ViewCtor });
      }
    } else {
      const size = gpuBuf.size;
      const readBuf = device.createBuffer({ size, usage: BU.MAP_READ | BU.COPY_DST });
      encoder.copyBufferToBuffer(gpuBuf, 0, readBuf, 0, size);
      outputReadbacks.push({ readBuf, tensorIdx: paramIdx.get(binding.name), size: tensorArgs[paramIdx.get(binding.name)].length, dtype: binding.dtype, ViewCtor });
    }
  }

  device.queue.submit([encoder.finish()]);

  for (const rb of outputReadbacks) {
    await rb.readBuf.mapAsync(mapModeRead());
    const result = new rb.ViewCtor(rb.readBuf.getMappedRange());
    unpackTensorFrom(tensorArgs[rb.tensorIdx], result, rb.dtype, 0, rb.size);
    rb.readBuf.unmap();
    rb.readBuf.destroy();
  }

  for (const buf of gpuBuffers) buf.destroy();
}
