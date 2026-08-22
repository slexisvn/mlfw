import { wgslType, wgslBytes } from '../util/dtype_map.js';
import type { WebGPUBinding } from '../backend/webgpu/codegen.js';
import type { NumericTypedArray } from '../tensor/types/dtype.js';
import type { MutableNumericArray, NumericSettable } from '../tensor/types/options.js';
import type { Tensor } from '../tensor/core/tensor.js';
import type { ExecutionPlan, PlanStepRuntime, RuntimeTensor } from './runtime.js';

type WgslView = Int32Array | Uint32Array | Uint16Array | Float32Array;
type WgslViewCtor = Int32ArrayConstructor | Uint32ArrayConstructor | Uint16ArrayConstructor | Float32ArrayConstructor;

export type WebGPUKernel = {
  name: string;
  source: string;
  metadata: {
    bindings: WebGPUBinding[];
    dispatchSize: number[];
    workgroupSize?: number[];
  };
};

export type WebGPUInstance = {
  device: GPUDevice;
  pipeline: GPUComputePipeline;
  bindGroupLayout: GPUBindGroupLayout;
  kernel: WebGPUKernel;
  workgroupSize?: number[];
  dispatchSize: number[];
  bindings: WebGPUBinding[];
};

type PipelineEntry = { pipeline: GPUComputePipeline; bindGroupLayout: GPUBindGroupLayout };
type DawnModule = { create(args: string[]): GPU; globals?: Record<string, unknown> };
type EagerBufferEntry = { buf: GPUBuffer; bytes: number; dtype: string; persistent: boolean };
type RNNCellLike = {
  x2h: { weight: Tensor; bias: Tensor | null };
  h2h: { weight: Tensor; bias: Tensor | null };
  _webgpuPacked?: Float32Array;
  _webgpuPackedTag?: string;
};
type RNNOpts = { kind?: string; batch: number; hiddenSize: number; seqLen: number; inputSize: number };
type ScanCarry = { initSlot: number; a: number; b: number; finalSlot: number; bytes: number };
type ScanXs = { xsSlot: number; xtSlot: number; stepBytes: number };
type ScanYs = { ytSlot: number; ysSlot: number; stepBytes: number };
type ScanLoop = { loopStart: number; loopEnd: number; T: number; carry: ScanCarry[]; xs: ScanXs[]; ys: ScanYs[] };
type ScanPlan = ExecutionPlan & { scanLoops?: ScanLoop[]; scanLoop?: ScanLoop };
import { bf16ToF32, f32ToBf16 } from '../tensor/utils/half.js';
import { wrapResult } from '../dispatcher/jit_dispatch.js';
import { WEBGPU_DEVICE } from '../tensor/types/device.js';
import { stack } from '../tensor/ops/ops.js';
import { select } from '../tensor/ops/ops.js';

let _gpuDevice: GPUDevice | null = null;
let _gpuInitPromise: Promise<GPUDevice> | null = null;
let _bufferUsage: typeof GPUBufferUsage | null = null;
let _mapMode: typeof GPUMapMode | null = null;
let _shaderStage: typeof GPUShaderStage | null = null;
let _exitRegistered = false;

function wgslViewCtor(dtype: string): WgslViewCtor {
  switch (wgslType(dtype)) {
    case 'i32': return Int32Array;
    case 'u32': return Uint32Array;
    case 'f16': return Uint16Array;
    default: return Float32Array;
  }
}

function packTensorInto(view: WgslView, src: NumericTypedArray, dtype: string, offset: number): void {
  if (dtype === 'bf16') {
    for (let i = 0; i < src.length; i++) (view as MutableNumericArray)[offset + i] = bf16ToF32(src[i] as number);
  } else if (dtype === 'i64') {
    for (let i = 0; i < src.length; i++) (view as MutableNumericArray)[offset + i] = Number(BigInt.asIntN(32, src[i] as bigint));
  } else {
    (view as NumericSettable).set(src, offset);
  }
}

function unpackTensorFrom(dst: NumericTypedArray, view: WgslView, dtype: string, offset: number, size: number): void {
  if (dtype === 'bf16') {
    for (let i = 0; i < size; i++) (dst as MutableNumericArray)[i] = f32ToBf16(view[offset + i]);
  } else if (dtype === 'i64') {
    for (let i = 0; i < size; i++) (dst as MutableNumericArray)[i] = BigInt(view[offset + i]);
  } else {
    (dst as NumericSettable).set(view.subarray(offset, offset + size));
  }
}

function align4(n: number): number { return Math.ceil(n / 4) * 4; }

async function ensureDevice(): Promise<GPUDevice> {
  if (_gpuDevice) return _gpuDevice;
  if (_gpuInitPromise) return _gpuInitPromise;

  _gpuInitPromise = (async () => {
    let gpu: GPU | false | undefined = typeof navigator !== 'undefined' && navigator.gpu;
    if (!gpu) {
      try {
        const mod = await import('webgpu') as unknown as DawnModule;
        gpu = mod.create([]);
        if (mod.globals) {
          _bufferUsage = mod.globals.GPUBufferUsage as typeof GPUBufferUsage;
          _mapMode = mod.globals.GPUMapMode as typeof GPUMapMode;
          _shaderStage = mod.globals.GPUShaderStage as typeof GPUShaderStage;
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

    const adapter = await (gpu as GPU).requestAdapter();
    if (!adapter) throw new Error('WebGPU: no adapter found');

    const adapterLimits = (adapter.limits || {}) as unknown as Record<string, number>;
    const requiredLimits: Record<string, number> = {};
    const liftKeys = [
      'maxStorageBuffersPerShaderStage',
      'maxStorageBufferBindingSize',
      'maxBufferSize',
      'maxBindingsPerBindGroup',
    ];
    for (const key of liftKeys) {
      if (adapterLimits[key] !== undefined) requiredLimits[key] = adapterLimits[key];
    }
    const requiredFeatures: GPUFeatureName[] = [];
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

function bufUsage(): typeof GPUBufferUsage { return _bufferUsage!; }
function mapModeRead(): number { return _mapMode!.READ; }

export function resetDevice(): void {
  if (_gpuDevice) {
    _gpuDevice.destroy();
    _gpuDevice = null;
  }
  _gpuInitPromise = null;
  _bufferUsage = null;
  _mapMode = null;
  _shaderStage = null;
}

function bindingBufferType(binding: WebGPUBinding): GPUBufferBindingType {
  if (binding.name === '_shapes') return 'uniform';
  if (binding.mode === 'read_write') return 'storage';
  return 'read-only-storage';
}

function pipelineParts(device: GPUDevice, kernel: WebGPUKernel): { shaderModule: GPUShaderModule; bindGroupLayout: GPUBindGroupLayout; pipelineLayout: GPUPipelineLayout } {
  const shaderModule = device.createShaderModule({ code: kernel.source });
  const layoutEntries: GPUBindGroupLayoutEntry[] = [];
  for (const binding of kernel.metadata.bindings) {
    layoutEntries.push({
      binding: binding.index,
      visibility: _shaderStage!.COMPUTE,
      buffer: { type: bindingBufferType(binding) },
    });
  }
  const bindGroupLayout = device.createBindGroupLayout({ entries: layoutEntries });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
  return { shaderModule, bindGroupLayout, pipelineLayout };
}

function createPipeline(device: GPUDevice, kernel: WebGPUKernel): PipelineEntry {
  const { shaderModule, bindGroupLayout, pipelineLayout } = pipelineParts(device, kernel);
  const pipeline = device.createComputePipeline({ layout: pipelineLayout, compute: { module: shaderModule, entryPoint: kernel.name } });
  return { pipeline, bindGroupLayout };
}

const _pipelines = new WeakMap<WebGPUKernel, PipelineEntry>();
function pipelineFor(device: GPUDevice, kernel: WebGPUKernel): PipelineEntry {
  let p = _pipelines.get(kernel);
  if (!p) { p = createPipeline(device, kernel); _pipelines.set(kernel, p); }
  return p;
}

export async function instantiateWebGPU(kernel: WebGPUKernel): Promise<WebGPUInstance> {
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

function buildParamIndex(bindings: readonly WebGPUBinding[]): Map<string, number> {
  const map = new Map<string, number>();
  let idx = 0;
  for (const b of bindings) {
    if (b.name === '_shapes') continue;
    if (b.packed) {
      for (const entry of b.packed) {
        map.set(entry.name, entry.argIndex !== undefined ? entry.argIndex : idx++);
      }
    } else {
      map.set(b.name, b.argIndex !== undefined ? b.argIndex : idx++);
    }
  }
  return map;
}

export async function runWebGPUKernel(instance: WebGPUInstance, tensorArgs: readonly NumericTypedArray[], shapeValues: readonly number[] | null): Promise<void> {
  const { device, pipeline, bindGroupLayout, bindings, dispatchSize } = instance;
  const BU = bufUsage();

  const gpuBuffers: GPUBuffer[] = [];
  const bindGroupEntries: GPUBindGroupEntry[] = [];
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

    const elemBytes = wgslBytes(binding.dtype!);
    const ViewCtor = wgslViewCtor(binding.dtype!);

    if (binding.packed) {
      const totalBytes = align4(binding.packedSize! * elemBytes);
      const isOutput = binding.mode === 'read_write';
      const usage = isOutput
        ? BU.STORAGE | BU.COPY_SRC | BU.COPY_DST
        : BU.STORAGE | BU.COPY_DST;

      const gpuBuf = device.createBuffer({ size: Math.max(totalBytes, 4), usage, mappedAtCreation: true });
      const mapped = new ViewCtor(gpuBuf.getMappedRange());
      for (const entry of binding.packed) {
        const srcIdx = paramIdx.get(entry.name)!;
        const src = tensorArgs[srcIdx];
        if (src) packTensorInto(mapped, src, entry.dtype, entry.offset);
      }
      gpuBuf.unmap();
      gpuBuffers.push(gpuBuf);
      bindGroupEntries.push({ binding: binding.index, resource: { buffer: gpuBuf } });
      continue;
    }

    const idx = paramIdx.get(binding.name)!;
    const data = tensorArgs[idx];
    const byteLength = align4(data.length * elemBytes);
    const isOutput = binding.mode === 'read_write';
    const usage = isOutput
      ? BU.STORAGE | BU.COPY_SRC | BU.COPY_DST
      : BU.STORAGE | BU.COPY_DST;

    const gpuBuf = device.createBuffer({ size: Math.max(byteLength, 4), usage, mappedAtCreation: true });
    packTensorInto(new ViewCtor(gpuBuf.getMappedRange()), data, binding.dtype!, 0);
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

  const outputReadbacks: { readBuf: GPUBuffer; tensorIdx: number; size: number; dtype: string; ViewCtor: WgslViewCtor }[] = [];
  for (let i = 0; i < bindings.length; i++) {
    const binding = bindings[i];
    if (binding.name === '_shapes') continue;
    if (binding.mode !== 'read_write') continue;

    const gpuBuf = gpuBuffers[i];
    const elemBytes = wgslBytes(binding.dtype!);
    const ViewCtor = wgslViewCtor(binding.dtype!);

    if (binding.packed) {
      for (const entry of binding.packed) {
        const byteOff = entry.offset * elemBytes;
        const byteLen = align4(entry.size * elemBytes);
        const readBuf = device.createBuffer({ size: byteLen, usage: BU.MAP_READ | BU.COPY_DST });
        encoder.copyBufferToBuffer(gpuBuf, byteOff, readBuf, 0, byteLen);
        outputReadbacks.push({ readBuf, tensorIdx: paramIdx.get(entry.name)!, size: entry.size, dtype: entry.dtype, ViewCtor });
      }
    } else {
      const size = gpuBuf.size;
      const readBuf = device.createBuffer({ size, usage: BU.MAP_READ | BU.COPY_DST });
      encoder.copyBufferToBuffer(gpuBuf, 0, readBuf, 0, size);
      outputReadbacks.push({ readBuf, tensorIdx: paramIdx.get(binding.name)!, size: tensorArgs[paramIdx.get(binding.name)!].length, dtype: binding.dtype!, ViewCtor });
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

export async function prewarmPipelines(device: GPUDevice, kernels: readonly (WebGPUKernel | null)[]): Promise<void> {
  if (typeof device.createComputePipelineAsync !== 'function') return;
  const pending: Promise<unknown>[] = [];
  const seen = new Set<WebGPUKernel>();
  for (const kernel of kernels) {
    if (!kernel || seen.has(kernel) || _pipelines.has(kernel)) continue;
    seen.add(kernel);
    const { shaderModule, bindGroupLayout, pipelineLayout } = pipelineParts(device, kernel);
    pending.push(device.createComputePipelineAsync({ layout: pipelineLayout, compute: { module: shaderModule, entryPoint: kernel.name } })
      .then(pipeline => _pipelines.set(kernel, { pipeline, bindGroupLayout }), () => {}));
  }
  if (pending.length > 0) await Promise.all(pending);
}

export async function runWebGPUPlan(plan: ScanPlan, slots: Array<RuntimeTensor | null>, steps: readonly PlanStepRuntime[]): Promise<void> {
  const device = await ensureDevice();
  const BU = bufUsage();
  await prewarmPipelines(device, steps.map(st => st.kernel as unknown as WebGPUKernel | null));
  const written = new Set<number>();
  for (const st of steps) for (const s of st.outputSlots) written.add(s);

  const groupOf: number[] = new Array(plan.numSlots);
  const groupCount = plan.buffers ? plan.buffers.bufferBytes.length : plan.numSlots;
  for (let s = 0; s < plan.numSlots; s++) groupOf[s] = plan.buffers ? plan.buffers.slotBuffer[s] : s;

  const bufs: (GPUBuffer | null)[] = new Array(plan.numSlots).fill(null);
  const dtypes: string[] = new Array(plan.numSlots).fill('f32');
  const groupBytes: number[] = new Array(groupCount).fill(0);
  for (let s = 0; s < plan.numSlots; s++) {
    const t = slots[s];
    if (!t) continue;
    dtypes[s] = t.dtype || 'f32';
    const bytes = Math.max(align4(t.data.length * wgslBytes(dtypes[s])), 4);
    if (bytes > groupBytes[groupOf[s]]) groupBytes[groupOf[s]] = bytes;
  }

  const groupInit: number[] = new Array(groupCount).fill(-1);
  for (let s = 0; s < plan.numSlots; s++) {
    if (slots[s] && !written.has(s)) groupInit[groupOf[s]] = s;
  }

  const groupBufs: (GPUBuffer | null)[] = new Array(groupCount).fill(null);
  for (let g = 0; g < groupCount; g++) {
    if (groupBytes[g] === 0) continue;
    const init = groupInit[g];
    const buf = device.createBuffer({ size: groupBytes[g], usage: BU.STORAGE | BU.COPY_DST | BU.COPY_SRC, mappedAtCreation: init >= 0 });
    if (init >= 0) {
      packTensorInto(new (wgslViewCtor(dtypes[init]))(buf.getMappedRange()), slots[init]!.data, dtypes[init], 0);
      buf.unmap();
    }
    groupBufs[g] = buf;
  }
  for (let s = 0; s < plan.numSlots; s++) {
    if (slots[s]) bufs[s] = groupBufs[groupOf[s]];
  }

  const PLAN_SUBMIT_CHUNK = 32;
  const uniformBufs: GPUBuffer[] = [];
  const scratchBufs: GPUBuffer[] = [];
  const ctx = { encoder: device.createCommandEncoder(), pending: 0 };
  const maybeFlush = () => { if (++ctx.pending >= PLAN_SUBMIT_CHUNK) { device.queue.submit([ctx.encoder.finish()]); ctx.encoder = device.createCommandEncoder(); ctx.pending = 0; } };

  const encodeStep = (st: PlanStepRuntime): void => {
    const encoder = ctx.encoder;
    const { pipeline, bindGroupLayout } = pipelineFor(device, st.kernel as unknown as WebGPUKernel);
    const ordered = st.inputSlots.concat(st.outputSlots);
    const entries: GPUBindGroupEntry[] = [];
    const writebacks: { slot: number; src: GPUBuffer; srcOff: number; bytes: number }[] = [];
    let argi = 0;
    for (const b of (st.kernel as unknown as WebGPUKernel).metadata.bindings) {
      if (b.name === '_shapes') {
        const shapeData = new Uint32Array(st.shapeValues || []);
        const sz = Math.max(Math.ceil(shapeData.byteLength / 16) * 16, 16);
        const ub = device.createBuffer({ size: sz, usage: BU.UNIFORM | BU.COPY_DST });
        device.queue.writeBuffer(ub, 0, shapeData);
        uniformBufs.push(ub);
        entries.push({ binding: b.index, resource: { buffer: ub } });
      } else if (b.packed) {
        const elemBytes = wgslBytes(b.dtype!);
        const pbuf = device.createBuffer({ size: Math.max(align4(b.packedSize! * elemBytes), 4), usage: BU.STORAGE | BU.COPY_DST | BU.COPY_SRC });
        scratchBufs.push(pbuf);
        const isWrite = b.mode === 'read_write';
        for (const e of b.packed) {
          const slot = ordered[e.argIndex!];
          const bytes = align4(e.size * elemBytes);
          if (isWrite) writebacks.push({ slot, src: pbuf, srcOff: e.offset * elemBytes, bytes });
          else encoder.copyBufferToBuffer(bufs[slot]!, 0, pbuf, e.offset * elemBytes, bytes);
        }
        entries.push({ binding: b.index, resource: { buffer: pbuf } });
      } else {
        entries.push({ binding: b.index, resource: { buffer: bufs[ordered[argi++]]! } });
      }
    }
    const bindGroup = device.createBindGroup({ layout: bindGroupLayout, entries });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    const ds = (st.kernel as unknown as WebGPUKernel).metadata.dispatchSize;
    pass.dispatchWorkgroups(ds[0], ds[1], ds[2]);
    pass.end();
    for (const wb of writebacks) encoder.copyBufferToBuffer(wb.src, wb.srcOff, bufs[wb.slot]!, 0, wb.bytes);
    maybeFlush();
  };

  const scanLoops = plan.scanLoops ? [...plan.scanLoops].sort((a, b) => a.loopStart - b.loopStart) : (plan.scanLoop ? [plan.scanLoop] : null);
  if (scanLoops && scanLoops.length) {
    let cursor = 0;
    for (const sl of scanLoops) {
      for (; cursor < sl.loopStart; cursor++) encodeStep(steps[cursor]);
      for (const c of sl.carry) ctx.encoder.copyBufferToBuffer(bufs[c.initSlot]!, 0, bufs[c.a]!, 0, c.bytes);
      for (let t = 0; t < sl.T; t++) {
        for (const xs of sl.xs) ctx.encoder.copyBufferToBuffer(bufs[xs.xsSlot]!, t * xs.stepBytes, bufs[xs.xtSlot]!, 0, xs.stepBytes);
        for (let i = sl.loopStart; i < sl.loopEnd; i++) encodeStep(steps[i]);
        for (const ys of sl.ys) ctx.encoder.copyBufferToBuffer(bufs[ys.ytSlot]!, 0, bufs[ys.ysSlot]!, t * ys.stepBytes, ys.stepBytes);
        for (const c of sl.carry) { const tmp = bufs[c.a]; bufs[c.a] = bufs[c.b]; bufs[c.b] = tmp; }
      }
      for (const c of sl.carry) ctx.encoder.copyBufferToBuffer(bufs[c.a]!, 0, bufs[c.finalSlot]!, 0, c.bytes);
      cursor = sl.loopEnd;
    }
    for (; cursor < steps.length; cursor++) encodeStep(steps[cursor]);
  } else {
    for (const st of steps) encodeStep(st);
  }
  let encoder = ctx.encoder;

  const argSet = new Set(plan.argSlots);
  const reads: { rb: GPUBuffer; dtype: string; size: number; dst: NumericTypedArray }[] = [];
  for (let s = 0; s < plan.numSlots; s++) {
    if (!bufs[s] || !written.has(s) || !argSet.has(s)) continue;
    const t = slots[s]!;
    const byteLen = Math.max(align4(t.data.length * wgslBytes(dtypes[s])), 4);
    const rb = device.createBuffer({ size: byteLen, usage: BU.MAP_READ | BU.COPY_DST });
    encoder.copyBufferToBuffer(bufs[s]!, 0, rb, 0, byteLen);
    reads.push({ rb, dtype: dtypes[s], size: t.data.length, dst: t.data });
  }

  device.queue.submit([encoder.finish()]);

  for (const r of reads) {
    await r.rb.mapAsync(mapModeRead());
    unpackTensorFrom(r.dst, new (wgslViewCtor(r.dtype))(r.rb.getMappedRange()), r.dtype, 0, r.size);
    r.rb.unmap();
    r.rb.destroy();
  }
  for (const b of groupBufs) if (b) b.destroy();
  for (const u of uniformBufs) u.destroy();
  for (const sb of scratchBufs) sb.destroy();
}

let _eagerEncoder: GPUCommandEncoder | null = null;
let _eagerPending = 0;
const EAGER_SUBMIT_CHUNK = 64;
const _eagerBuffers = new Map<NumericTypedArray, EagerBufferEntry>();
const _eagerTransient: GPUBuffer[] = [];
const _eagerScratch: { buf: GPUBuffer; bytes: number }[] = [];
const _eagerPool = new Map<number, GPUBuffer[]>();

function acquireStorage(bytes: number): GPUBuffer {
  const free = _eagerPool.get(bytes);
  if (free && free.length) return free.pop()!;
  const BU = bufUsage();
  return _gpuDevice!.createBuffer({ size: bytes, usage: BU.STORAGE | BU.COPY_SRC | BU.COPY_DST });
}
function releaseStorage(buf: GPUBuffer, bytes: number): void {
  let free = _eagerPool.get(bytes);
  if (!free) { free = []; _eagerPool.set(bytes, free); }
  free.push(buf);
}

export async function ensureWebGPUEager(): Promise<GPUDevice> {
  await ensureDevice();
  if (!_eagerEncoder) _eagerEncoder = _gpuDevice!.createCommandEncoder();
  return _gpuDevice!;
}

export function webgpuEagerReady(): boolean { return !!_gpuDevice && !!_eagerEncoder; }

function eagerBufferFor(hostArray: NumericTypedArray, dtype: string, asInput: boolean): GPUBuffer {
  const e = _eagerBuffers.get(hostArray);
  if (e) return e.buf;
  const BU = bufUsage();
  const bytes = Math.max(align4(hostArray.length * wgslBytes(dtype)), 4);
  if (asInput) {
    const buf = _gpuDevice!.createBuffer({ size: bytes, usage: BU.STORAGE | BU.COPY_SRC | BU.COPY_DST, mappedAtCreation: true });
    packTensorInto(new (wgslViewCtor(dtype))(buf.getMappedRange()), hostArray, dtype, 0);
    buf.unmap();
    _eagerBuffers.set(hostArray, { buf, bytes, dtype, persistent: asInput });
    return buf;
  }
  const buf = acquireStorage(bytes);
  _eagerBuffers.set(hostArray, { buf, bytes, dtype, persistent: false });
  return buf;
}

export function recordWebGPUEager(compiled: WebGPUKernel, hostArgs: readonly NumericTypedArray[], shapeValues: readonly number[] | undefined): void {
  const { pipeline, bindGroupLayout } = pipelineFor(_gpuDevice!, compiled);
  const { bindings, dispatchSize } = compiled.metadata;
  const paramIdx = buildParamIndex(bindings);
  const BU = bufUsage();
  const entries: GPUBindGroupEntry[] = [];
  const writebacks: { pbuf: GPUBuffer; off: number; sub: GPUBuffer; bytes: number }[] = [];
  for (const binding of bindings) {
    if (binding.name === '_shapes') {
      const shapeData = new Uint32Array(shapeValues || []);
      const sz = Math.max(Math.ceil(shapeData.byteLength / 16) * 16, 16);
      const ub = _gpuDevice!.createBuffer({ size: sz, usage: BU.UNIFORM | BU.COPY_DST });
      _gpuDevice!.queue.writeBuffer(ub, 0, shapeData);
      _eagerTransient.push(ub);
      entries.push({ binding: binding.index, resource: { buffer: ub } });
      continue;
    }
    if (binding.packed) {
      const elemBytes = wgslBytes(binding.dtype!);
      const pbytes = Math.max(align4(binding.packedSize! * elemBytes), 4);
      const pbuf = acquireStorage(pbytes);
      _eagerScratch.push({ buf: pbuf, bytes: pbytes });
      const isWrite = binding.mode === 'read_write';
      for (const e of binding.packed) {
        const sub = eagerBufferFor(hostArgs[paramIdx.get(e.name)!], e.dtype, !isWrite);
        const bytes = align4(e.size * elemBytes);
        if (isWrite) writebacks.push({ pbuf, off: e.offset * elemBytes, sub, bytes });
        else _eagerEncoder!.copyBufferToBuffer(sub, 0, pbuf, e.offset * elemBytes, bytes);
      }
      entries.push({ binding: binding.index, resource: { buffer: pbuf } });
      continue;
    }
    const host = hostArgs[paramIdx.get(binding.name)!];
    const buf = eagerBufferFor(host, binding.dtype!, binding.mode !== 'read_write');
    entries.push({ binding: binding.index, resource: { buffer: buf } });
  }
  const bindGroup = _gpuDevice!.createBindGroup({ layout: bindGroupLayout, entries });
  const pass = _eagerEncoder!.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(dispatchSize[0], dispatchSize[1], dispatchSize[2]);
  pass.end();
  for (const wb of writebacks) _eagerEncoder!.copyBufferToBuffer(wb.pbuf, wb.off, wb.sub, 0, wb.bytes);
  if (++_eagerPending >= EAGER_SUBMIT_CHUNK) {
    _gpuDevice!.queue.submit([_eagerEncoder!.finish()]);
    _eagerEncoder = _gpuDevice!.createCommandEncoder();
    _eagerPending = 0;
  }
}

export async function flushWebGPUEager(): Promise<void> {
  if (!_gpuDevice) return;
  const BU = bufUsage();
  const reads: { rb: GPUBuffer; ha: NumericTypedArray; buf: GPUBuffer; bytes: number; dtype: string }[] = [];
  for (const [ha, e] of _eagerBuffers) {
    if (e.persistent) continue;
    const rb = _gpuDevice.createBuffer({ size: e.bytes, usage: BU.MAP_READ | BU.COPY_DST });
    _eagerEncoder!.copyBufferToBuffer(e.buf, 0, rb, 0, e.bytes);
    reads.push({ rb, ha, buf: e.buf, bytes: e.bytes, dtype: e.dtype });
  }
  _gpuDevice.queue.submit([_eagerEncoder!.finish()]);
  _eagerEncoder = _gpuDevice.createCommandEncoder();
  _eagerPending = 0;
  for (const u of _eagerTransient) u.destroy();
  _eagerTransient.length = 0;
  await Promise.all(reads.map(r => r.rb.mapAsync(mapModeRead())));
  for (const r of reads) {
    unpackTensorFrom(r.ha, new (wgslViewCtor(r.dtype))(r.rb.getMappedRange()), r.dtype, 0, r.ha.length);
    r.rb.unmap();
    r.rb.destroy();
  }
  for (const s of _eagerScratch) releaseStorage(s.buf, s.bytes);
  _eagerScratch.length = 0;
  for (const r of reads) { releaseStorage(r.buf, r.bytes); _eagerBuffers.delete(r.ha); }
}

const _contigKernels = new Map<string, WebGPUKernel>();
function contigKernel(shape: readonly number[], strides: readonly number[], offset: number, dtype: string, numel: number): WebGPUKernel {
  const key = `${shape.join(',')}|${strides.join(',')}|${offset}|${dtype}`;
  let k = _contigKernels.get(key);
  if (k) return k;
  const wt = wgslType(dtype);
  let perDim = '';
  for (let d = shape.length - 1; d >= 0; d--) {
    perDim += `  let idx${d} = rem % ${shape[d]}u; rem = rem / ${shape[d]}u; src = src + idx${d} * ${strides[d]}u;\n`;
  }
  const source = `@group(0) @binding(0) var<storage, read> inp : array<${wt}>;\n`
    + `@group(0) @binding(1) var<storage, read_write> outp : array<${wt}>;\n`
    + `@compute @workgroup_size(64)\n`
    + `fn contig(@builtin(global_invocation_id) gid : vec3<u32>) {\n`
    + `  let i = gid.x;\n  if (i >= ${numel}u) { return; }\n  var rem = i;\n  var src = ${offset}u;\n`
    + perDim
    + `  outp[i] = inp[src];\n}`;
  k = { name: 'contig', source, metadata: { bindings: [{ index: 0, name: 'inp', dtype, mode: 'read' }, { index: 1, name: 'outp', dtype, mode: 'read_write' }], dispatchSize: [Math.ceil(numel / 64), 1, 1] } };
  _contigKernels.set(key, k);
  return k;
}

function webgpuEagerInput(t: Tensor): NumericTypedArray {
  const raw = t._impl.storage.rawData!;
  if (t.isContiguous && t._impl.storageOffset === 0 && raw.length === t.numel) return raw;
  const out = new (raw.constructor as { new (length: number): NumericTypedArray })(t.numel);
  recordWebGPUEager(contigKernel(t.shape, t.strides, t._impl.storageOffset, t.dtype, t.numel), [raw, out], undefined);
  return out;
}

export function webgpuEagerOp(compiled: WebGPUKernel, tensors: readonly Tensor[], outData: NumericTypedArray): void {
  const hostArgs = tensors.map(webgpuEagerInput);
  hostArgs.push(outData);
  recordWebGPUEager(compiled, hostArgs, undefined);
}

const RNN_WG = 256;
const _rnnKernels = new Map<string, WebGPUKernel>();

function lstmKernelWGSL(E: number, H: number, T: number): WebGPUKernel {
  const key = `lstm|${E}|${H}|${T}`;
  let k = _rnnKernels.get(key);
  if (k) return k;
  const G = 4 * H;
  const offWh = G * E, offBx = G * E + G * H, offBh = offBx + G;
  const src =
`fn sig(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
@group(0) @binding(0) var<storage, read> xin : array<f32>;
@group(0) @binding(1) var<storage, read> w : array<f32>;
@group(0) @binding(2) var<storage, read> h0 : array<f32>;
@group(0) @binding(3) var<storage, read> c0 : array<f32>;
@group(0) @binding(4) var<storage, read_write> ys : array<f32>;
@group(0) @binding(5) var<storage, read_write> hn : array<f32>;
@group(0) @binding(6) var<storage, read_write> cn : array<f32>;
var<workgroup> wh : array<f32, ${H}>;
var<workgroup> wc : array<f32, ${H}>;
var<workgroup> wg : array<f32, ${G}>;
@compute @workgroup_size(${RNN_WG})
fn rnn(@builtin(local_invocation_id) lid : vec3<u32>) {
  let tid = lid.x;
  for (var p = tid; p < ${H}u; p = p + ${RNN_WG}u) { wh[p] = h0[p]; wc[p] = c0[p]; }
  workgroupBarrier();
  for (var t = 0u; t < ${T}u; t = t + 1u) {
    for (var j = tid; j < ${G}u; j = j + ${RNN_WG}u) {
      var acc = w[${offBx}u + j] + w[${offBh}u + j];
      let xb = t * ${E}u;
      let wxb = j * ${E}u;
      for (var e = 0u; e < ${E}u; e = e + 1u) { acc = acc + xin[xb + e] * w[wxb + e]; }
      let whb = ${offWh}u + j * ${H}u;
      for (var q = 0u; q < ${H}u; q = q + 1u) { acc = acc + wh[q] * w[whb + q]; }
      wg[j] = acc;
    }
    workgroupBarrier();
    for (var j = tid; j < ${H}u; j = j + ${RNN_WG}u) {
      let ii = sig(wg[j]);
      let ff = sig(wg[${H}u + j]);
      let gg = tanh(wg[${2 * H}u + j]);
      let oo = sig(wg[${3 * H}u + j]);
      let cv = ff * wc[j] + ii * gg;
      wc[j] = cv;
      let nh = oo * tanh(cv);
      wh[j] = nh;
      ys[t * ${H}u + j] = nh;
    }
    workgroupBarrier();
  }
  for (var p = tid; p < ${H}u; p = p + ${RNN_WG}u) { hn[p] = wh[p]; cn[p] = wc[p]; }
}`;
  k = {
    name: 'rnn', source: src, metadata: {
      bindings: [
        { index: 0, name: 'xin', dtype: 'f32', mode: 'read' },
        { index: 1, name: 'w', dtype: 'f32', mode: 'read' },
        { index: 2, name: 'h0', dtype: 'f32', mode: 'read' },
        { index: 3, name: 'c0', dtype: 'f32', mode: 'read' },
        { index: 4, name: 'ys', dtype: 'f32', mode: 'read_write' },
        { index: 5, name: 'hn', dtype: 'f32', mode: 'read_write' },
        { index: 6, name: 'cn', dtype: 'f32', mode: 'read_write' },
      ],
      dispatchSize: [1, 1, 1],
    },
  };
  _rnnKernels.set(key, k);
  return k;
}

function rnnPackedWeights(cell: RNNCellLike, E: number, H: number, G: number): Float32Array {
  const tag = `${E}|${H}|${G}`;
  if (cell._webgpuPacked && cell._webgpuPackedTag === tag) return cell._webgpuPacked;
  const raw = (t: Tensor): NumericTypedArray => (t as Tensor & { contiguous(): Tensor }).contiguous()._impl.storage.rawData!;
  const Wx = raw(cell.x2h.weight), Wh = raw(cell.h2h.weight);
  const bx = cell.x2h.bias ? raw(cell.x2h.bias) : new Float32Array(G);
  const bh = cell.h2h.bias ? raw(cell.h2h.bias) : new Float32Array(G);
  const packed = new Float32Array(G * E + G * H + G + G);
  packed.set(Wx as ArrayLike<number>, 0);
  packed.set(Wh as ArrayLike<number>, G * E);
  packed.set(bx as ArrayLike<number>, G * E + G * H);
  packed.set(bh as ArrayLike<number>, G * E + G * H + G);
  cell._webgpuPacked = packed;
  cell._webgpuPackedTag = tag;
  return packed;
}

function lstmPackedWeights(cell: RNNCellLike, E: number, H: number): Float32Array { return rnnPackedWeights(cell, E, H, 4 * H); }

function gruKernelWGSL(E: number, H: number, T: number): WebGPUKernel {
  const key = `gru|${E}|${H}|${T}`;
  let k = _rnnKernels.get(key);
  if (k) return k;
  const G = 3 * H;
  const offWh = G * E, offBx = G * E + G * H, offBh = offBx + G;
  const src =
`fn sig(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
@group(0) @binding(0) var<storage, read> xin : array<f32>;
@group(0) @binding(1) var<storage, read> w : array<f32>;
@group(0) @binding(2) var<storage, read> h0 : array<f32>;
@group(0) @binding(3) var<storage, read_write> ys : array<f32>;
@group(0) @binding(4) var<storage, read_write> hn : array<f32>;
var<workgroup> wh : array<f32, ${H}>;
var<workgroup> wgx : array<f32, ${G}>;
var<workgroup> wgh : array<f32, ${G}>;
@compute @workgroup_size(${RNN_WG})
fn rnn(@builtin(local_invocation_id) lid : vec3<u32>) {
  let tid = lid.x;
  for (var p = tid; p < ${H}u; p = p + ${RNN_WG}u) { wh[p] = h0[p]; }
  workgroupBarrier();
  for (var t = 0u; t < ${T}u; t = t + 1u) {
    for (var j = tid; j < ${G}u; j = j + ${RNN_WG}u) {
      var ax = w[${offBx}u + j];
      var ah = w[${offBh}u + j];
      let xb = t * ${E}u;
      let wxb = j * ${E}u;
      for (var e = 0u; e < ${E}u; e = e + 1u) { ax = ax + xin[xb + e] * w[wxb + e]; }
      let whb = ${offWh}u + j * ${H}u;
      for (var q = 0u; q < ${H}u; q = q + 1u) { ah = ah + wh[q] * w[whb + q]; }
      wgx[j] = ax;
      wgh[j] = ah;
    }
    workgroupBarrier();
    for (var j = tid; j < ${H}u; j = j + ${RNN_WG}u) {
      let rr = sig(wgx[j] + wgh[j]);
      let zz = sig(wgx[${H}u + j] + wgh[${H}u + j]);
      let nn = tanh(wgx[${2 * H}u + j] + rr * wgh[${2 * H}u + j]);
      let nh = (1.0 - zz) * nn + zz * wh[j];
      wh[j] = nh;
      ys[t * ${H}u + j] = nh;
    }
    workgroupBarrier();
  }
  for (var p = tid; p < ${H}u; p = p + ${RNN_WG}u) { hn[p] = wh[p]; }
}`;
  k = {
    name: 'rnn', source: src, metadata: {
      bindings: [
        { index: 0, name: 'xin', dtype: 'f32', mode: 'read' },
        { index: 1, name: 'w', dtype: 'f32', mode: 'read' },
        { index: 2, name: 'h0', dtype: 'f32', mode: 'read' },
        { index: 3, name: 'ys', dtype: 'f32', mode: 'read_write' },
        { index: 4, name: 'hn', dtype: 'f32', mode: 'read_write' },
      ],
      dispatchSize: [1, 1, 1],
    },
  };
  _rnnKernels.set(key, k);
  return k;
}

export function webgpuRNN(xs: Tensor, cells: readonly RNNCellLike[], opts: RNNOpts, h0: Tensor | null, c0: Tensor | null): (Tensor | null)[] | null {
  if (opts.kind === 'gru') return webgpuGRU(xs, cells, opts, h0);
  if (opts.batch !== 1) return null;
  const H = opts.hiddenSize, T = opts.seqLen, L = cells.length;
  let layerRaw = webgpuEagerInput(xs);
  const hns: Float32Array[] = [], cns: Float32Array[] = [];
  for (let l = 0; l < L; l++) {
    const E = l === 0 ? opts.inputSize : H;
    const packed = lstmPackedWeights(cells[l], E, H);
    let h0Raw: NumericTypedArray, c0Raw: NumericTypedArray;
    if (h0 != null && c0 != null) {
      h0Raw = webgpuEagerInput(select(h0, 0, l));
      c0Raw = webgpuEagerInput(select(c0, 0, l));
    } else {
      h0Raw = new Float32Array(H);
      c0Raw = new Float32Array(H);
    }
    const ys = new Float32Array(T * H), hnA = new Float32Array(H), cnA = new Float32Array(H);
    recordWebGPUEager(lstmKernelWGSL(E, H, T), [layerRaw, packed, h0Raw, c0Raw, ys, hnA, cnA], undefined);
    layerRaw = ys;
    hns.push(hnA);
    cns.push(cnA);
  }
  const out = wrapResult(layerRaw, [T, 1, H], 'f32', WEBGPU_DEVICE);
  const wrapState = (arrs: readonly Float32Array[]): Tensor => L === 1
    ? wrapResult(arrs[0], [1, 1, H], 'f32', WEBGPU_DEVICE)
    : stack(arrs.map(a => wrapResult(a, [1, H], 'f32', WEBGPU_DEVICE)), 0);
  return [out, wrapState(hns), wrapState(cns)];
}

function webgpuGRU(xs: Tensor, cells: readonly RNNCellLike[], opts: RNNOpts, h0: Tensor | null): (Tensor | null)[] | null {
  if (opts.batch !== 1) return null;
  const H = opts.hiddenSize, T = opts.seqLen, L = cells.length;
  let layerRaw = webgpuEagerInput(xs);
  const hns: Float32Array[] = [];
  for (let l = 0; l < L; l++) {
    const E = l === 0 ? opts.inputSize : H;
    const packed = rnnPackedWeights(cells[l], E, H, 3 * H);
    const h0Raw = h0 != null ? webgpuEagerInput(select(h0, 0, l)) : new Float32Array(H);
    const ys = new Float32Array(T * H), hnA = new Float32Array(H);
    recordWebGPUEager(gruKernelWGSL(E, H, T), [layerRaw, packed, h0Raw, ys, hnA], undefined);
    layerRaw = ys;
    hns.push(hnA);
  }
  const out = wrapResult(layerRaw, [T, 1, H], 'f32', WEBGPU_DEVICE);
  const hy = L === 1
    ? wrapResult(hns[0], [1, 1, H], 'f32', WEBGPU_DEVICE)
    : stack(hns.map(a => wrapResult(a, [1, H], 'f32', WEBGPU_DEVICE)), 0);
  return [out, hy];
}
