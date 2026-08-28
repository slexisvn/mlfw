import { GpuCodegenBase } from '../codegen_base.js';
import type { SourceHelper } from '../codegen_base.js';
import { ACCESS_NODE_TYPES, LOOP_NODE_TYPES } from '../../compiler/ir/node_kinds.js';
import { COMPARE_MATH_OPS } from '../../util/dtype_map.js';
import { ForKind } from '../../compiler/ir/tensor/nodes.js';
import { wgslType, wgslBytes, wgslMathFunc, hasWgslMathFunc, cCompareOp } from '../../util/dtype_map.js';
import { flattenRowMajorIndex } from '../index_emit.js';
import { MinHeap } from '../../util/min_heap.js';
import { walk } from '../../compiler/ir/ir_visitor.js';
import { resolveShapeParam } from '../codegen_utils.js';
import { parseThreadAxis } from '../../compiler/analysis/thread_binding.js';
import { allocatedBufferNames, referencedBuffers } from '../../compiler/analysis/tir_queries.js';
import { profileGpuAccesses, loopCarriedIntermediates, extentMismatchBuffers, GpuRaceReason } from '../../compiler/analysis/gpu_race.js';
import type { GpuLaunchDiagnosis } from '../../compiler/analysis/gpu_race.js';


import { LANCZOS_G, LANCZOS_COEFFS, ERF_A, ERF_P } from '../../util/special_math.js';

import { Buffer } from '../../compiler/ir/tensor/buffer.js';
import type { AllocateNode, BlockNode, BufferStoreNode, CallExternNode, ForNode, IfThenElseNode, LetStmtNode, VecCopyNode, WhileNode } from '../../compiler/ir/tensor/nodes.js';
import type { IRStmtNode, LIRAccumulatorNode, LIRBindingsNode, LIRFlatStoreNode } from '../../compiler/ir/lir/nodes.js';
import type { BufferDecl, CodegenFunc } from '../codegen_utils.js';
import type { TargetFeatures } from '../target.js';

const BOOL_OPS = new Set(['!', '&&', '||']);

type WgslNameTable = Readonly<Record<string, string | undefined>>;
type PackedBufEntry = { name: string; offset: number; size: number; dtype: string; argIndex?: number };
type PackGroup = { isWrite: boolean; wt: string; dtype: string; bufs: PackedBufEntry[]; off: number };
type PackedRef = { storage: string; offset: number };
type PoolRef = { pool: string; offset: number };
type PoolDecl = { pool: string; dtype: string; size: number };
type SlotDecl = { name: string; dtype: string; size: number; scalar?: boolean };
type HeapSlot = { decl: SlotDecl; freeAt: number };
type LiveItem = { name: string; size: number; first: number; last: number };
type PlacedItem = { offset: number; size: number; first: number; last: number };
type LivenessResult = { minPos: Map<string, number>; maxPos: Map<string, number> };
type LivenessScope = ReadonlySet<string> | ReadonlyMap<string, Buffer>;
type WorkgroupPool = { offsets: Map<string, PoolRef>; decls: PoolDecl[]; bytes: number };

export type WebGPUBinding = {
  index: number;
  name: string;
  mode: string;
  dtype?: string;
  argIndex?: number;
  packed?: PackedBufEntry[];
  packedSize?: number;
};

function _wgslFloat(v: number): string {
  const s = String(v);
  return /[.e]/.test(s) ? s : `${s}.0`;
}

function _wgslErf(x: string): string {
  const t = `(1.0 / (1.0 + ${_wgslFloat(ERF_P)} * abs(${x})))`;
  const poly = ERF_A.slice().reverse().reduce((acc, c) => `(${_wgslFloat(c)} + ${t} * ${acc})`, '0.0');
  return `((select(-1.0, 1.0, ${x} >= 0.0)) * (1.0 - ${t} * ${poly} * exp(-${x} * ${x})))`;
}

function _wgslLanczosCore(u: string): string {
  const zz = `(${u} - 1.0)`;
  const sum = LANCZOS_COEFFS
    .map((c, i) => (i === 0 ? _wgslFloat(c) : `${_wgslFloat(c)} / (${zz} + ${_wgslFloat(i)})`))
    .join(' + ');
  const t = `(${zz} + ${_wgslFloat(LANCZOS_G + 0.5)})`;
  return `(${_wgslFloat(0.5 * Math.log(2 * Math.PI))} + (${zz} + 0.5) * log(${t}) - ${t} + log(${sum}))`;
}

const _WGSL_PI = _wgslFloat(Math.PI);

const WGSL_LANCZOS_HELPER: SourceHelper = {
  name: '_lanczos',
  deps: [],
  code: `fn _lanczos(u: f32) -> f32 { return ${_wgslLanczosCore('u')}; }`,
};

function _wgslLgamma(x: string): string {
  const reflected = `(log(${_WGSL_PI} / abs(sin(${_WGSL_PI} * ${x}))) - _lanczos(1.0 - ${x}))`;
  return `(select(_lanczos(${x}), ${reflected}, ${x} < 0.5))`;
}

function _wgslGamma(x: string): string {
  const reflected = `(${_WGSL_PI} / (sin(${_WGSL_PI} * ${x}) * exp(_lanczos(1.0 - ${x}))))`;
  return `(select(exp(_lanczos(${x})), ${reflected}, ${x} < 0.5))`;
}

const WGSL_ERF_HELPER: SourceHelper = {
  name: '_erf',
  deps: [],
  code: `fn _erf(x: f32) -> f32 { return ${_wgslErf('x')}; }`,
};

const WGSL_LGAMMA_HELPER: SourceHelper = {
  name: '_lgamma',
  deps: [WGSL_LANCZOS_HELPER],
  code: `fn _lgamma(x: f32) -> f32 { return ${_wgslLgamma('x')}; }`,
};

const WGSL_GAMMA_HELPER: SourceHelper = {
  name: '_gamma',
  deps: [WGSL_LANCZOS_HELPER],
  code: `fn _gamma(x: f32) -> f32 { return ${_wgslGamma('x')}; }`,
};

const WGSL_FLOORMOD_HELPER: SourceHelper = {
  name: 'floormod',
  deps: [],
  code: 'fn floormod(a: i32, b: i32) -> i32 { return ((a % b) + b) % b; }',
};

const WGSL_FLOORDIV_HELPER: SourceHelper = {
  name: 'floordiv',
  deps: [WGSL_FLOORMOD_HELPER],
  code: 'fn floordiv(a: i32, b: i32) -> i32 { return (a - floormod(a, b)) / b; }',
};

const WGSL_MATHOP_HELPERS: Readonly<Record<string, SourceHelper | undefined>> = { '//': WGSL_FLOORDIV_HELPER, '%': WGSL_FLOORMOD_HELPER };

const WGSL_THREAD_TAG_MAP: WgslNameTable = {
  'threadIdx.x': 'local_invocation_id.x',
  'threadIdx.y': 'local_invocation_id.y',
  'threadIdx.z': 'local_invocation_id.z',
  'blockIdx.x': 'workgroup_id.x',
  'blockIdx.y': 'workgroup_id.y',
  'blockIdx.z': 'workgroup_id.z',
};

export type WebGPUKernelConfig = Readonly<{
  name: string;
  source: string;
  workgroupSize: number[];
  dispatchSize: number[];
  sharedMemBytes: number;
  params: string[];
  bindings: WebGPUBinding[];
  launchDiagnosis?: GpuLaunchDiagnosis | null;
}>;

export class WebGPUKernel {
  name: string;
  source: string;
  workgroupSize: number[];
  dispatchSize: number[];
  sharedMemBytes: number;
  params: string[];
  bindings: WebGPUBinding[];
  launchDiagnosis: GpuLaunchDiagnosis | null;

  constructor(config: WebGPUKernelConfig) {
    this.name = config.name;
    this.source = config.source;
    this.workgroupSize = config.workgroupSize;
    this.dispatchSize = config.dispatchSize;
    this.sharedMemBytes = config.sharedMemBytes;
    this.params = config.params;
    this.bindings = config.bindings;
    this.launchDiagnosis = config.launchDiagnosis || null;
  }
}

export class WebGPUCodegen extends GpuCodegenBase {
  _workgroupSize: number[];
  _dispatchSize: number[];
  _defaultDtype: string;
  declare _promotedBuffers: Set<string>;
  declare _promotedBufferDecls: BufferDecl[];
  declare _wgPoolOffsets: Map<string, PoolRef> | null;
  declare _wgPoolDecls: PoolDecl[];
  declare _needsBarriers: boolean;
  declare _serializeThreads: boolean;
  declare _launchDiagnosis: GpuLaunchDiagnosis | null;
  declare _localSlots: Map<string, string> | null;
  declare _slotDecls: SlotDecl[];
  declare _scalarSlotNames: Set<string>;
  declare _crossThread: Set<string> | null;
  declare _crossExtent: Set<string> | null;
  declare _packedMode: boolean;
  declare _packedOffsets: Map<string, PackedRef> | null;

  constructor(target: TargetFeatures) {
    super(target);
    this._workgroupSize = [1, 1, 1];
    this._dispatchSize = [1, 1, 1];
    this._defaultDtype = 'f32';
    this._storeBuffers = new Set();
  }

  generate(func: CodegenFunc): WebGPUKernel {
    this._indent = 0;
    this._lines = [];
    this._threadBindings.clear();
    this._sharedBuffers = [];
    this._workgroupSize = [1, 1, 1];
    this._dispatchSize = [1, 1, 1];
    this._primFunc = func;
    this._storeBuffers = new Set();
    this._promotedBuffers = new Set();
    this._promotedBufferDecls = [];
    this._wgPoolOffsets = null;
    this._wgPoolDecls = [];
    this._needsBarriers = false;
    this._serializeThreads = false;
    this._launchDiagnosis = null;
    this._localSlots = null;
    this._slotDecls = [];
    this._scalarSlotNames = new Set();
    this._crossThread = null;
    this._crossExtent = null;
    this._resetSourceScope();

    const isLIR = func.type === 'LIRFunc';

    if (isLIR) {
      for (const [tag, entries] of func.metadata.threadBindings) {
        this._threadBindings.set(tag, entries);
        for (const entry of entries) {
          if (!entry.isDynamic) this._applyBindingDim(tag, entry.extent);
        }
      }
      this._sharedBuffers = func.metadata.sharedBuffers;
    } else {
      this._scanBindings(func.body);
    }

    this._scanStoreTargets(func.body);
    this._analyzeSharing(func);

    const paramNames: string[] = [];
    const bindings: WebGPUBinding[] = [];
    let bindIdx = 0;
    const needsF16 = this._checkF16Usage(func);

    if (needsF16) this._emit('enable f16;');
    if (needsF16) this._emit('');

    const bufCount = func.bufferMap.size + (func.shapeParams.length > 0 ? 1 : 0);
    const bufList = [...func.bufferMap.values()];
    const canPack = bufCount > 6 && bufList.every(b => b.numel() > 0);
    this._packedMode = canPack;
    this._packedOffsets = null;

    const argIndexOf = new Map<string, number>();
    { let ai = 0; for (const [, buf] of func.bufferMap) argIndexOf.set(buf.name, ai++); }
    {
      let dd = null;
      for (const [, buf] of func.bufferMap) if (this._storeBuffers.has(buf.name)) { dd = buf.dtype; break; }
      this._defaultDtype = dd || (bufList.length ? bufList[0].dtype : 'f32');
    }
    const WT_DTYPE: WgslNameTable = { f32: 'f32', i32: 'i32', u32: 'u32', f16: 'f16' };

    if (canPack) {
      this._packedOffsets = new Map();
      const align4 = (n: number) => Math.ceil(n / 4) * 4;
      const groups = new Map<string, PackGroup>();
      for (const [, buf] of func.bufferMap) {
        paramNames.push(buf.name);
        const isWrite = this._storeBuffers.has(buf.name);
        const wt = wgslType(buf.dtype);
        const key = (isWrite ? 'w:' : 'r:') + wt;
        let g = groups.get(key);
        if (!g) { g = { isWrite, wt, dtype: WT_DTYPE[wt] || buf.dtype, bufs: [], off: 0 }; groups.set(key, g); }
        const numel = buf.numel();
        const storage = (isWrite ? '_pw_' : '_pr_') + wt;
        this._packedOffsets.set(buf.name, { storage, offset: g.off });
        g.bufs.push({ name: buf.name, offset: g.off, size: numel, dtype: buf.dtype, argIndex: argIndexOf.get(buf.name) });
        g.off += align4(numel);
      }
      const ordered = [...groups.values()].sort((a, b) => (a.isWrite === b.isWrite ? a.wt.localeCompare(b.wt) : (a.isWrite ? 1 : -1)));
      for (const g of ordered) {
        const storage = (g.isWrite ? '_pw_' : '_pr_') + g.wt;
        const accessMode = g.isWrite ? 'storage, read_write' : 'storage, read';
        bindings.push({ index: bindIdx, name: storage, mode: g.isWrite ? 'read_write' : 'read', packed: g.bufs, packedSize: g.off, dtype: g.dtype });
        this._emit(`@group(0) @binding(${bindIdx}) var<${accessMode}> ${storage}: array<${g.wt}>;`);
        bindIdx++;
      }
    } else {
      for (const [, buf] of func.bufferMap) {
        paramNames.push(buf.name);
        const mode = this._storeBuffers.has(buf.name) ? 'read_write' : 'read';
        const accessMode = mode === 'read_write' ? 'storage, read_write' : 'storage, read';
        bindings.push({ index: bindIdx, name: buf.name, mode, dtype: buf.dtype, argIndex: argIndexOf.get(buf.name) });
        this._emit(`@group(0) @binding(${bindIdx}) var<${accessMode}> ${buf.name}: array<${wgslType(buf.dtype)}>;`);
        bindIdx++;
      }
    }

    if (func.shapeParams.length > 0) {
      this._emit('');
      this._emit('struct ShapeParams {');
      this._indent++;
      for (const sp of func.shapeParams) {
        paramNames.push(sp.name);
        this._emit(`${sp.name}: u32,`);
      }
      this._indent--;
      this._emit('}');
      bindings.push({ index: bindIdx, name: '_shapes', mode: 'read' });
      this._emit(`@group(0) @binding(${bindIdx}) var<uniform> _shapes: ShapeParams;`);
      bindIdx++;
    }

    this._emit('');

    for (const buf of this._sharedBuffers) {
      const numel = buf.numel();
      this._emit(`var<workgroup> ${buf.name}: array<${wgslType(buf.dtype)}, ${numel > 0 ? numel : 1}>;`);
    }
    for (const d of this._promotedBufferDecls) {
      this._emit(`var<workgroup> ${d.name}: array<${wgslType(d.dtype)}, ${d.size}>;`);
    }
    for (const d of this._wgPoolDecls) {
      this._emit(`var<workgroup> ${d.pool}: array<${wgslType(d.dtype)}, ${d.size > 0 ? d.size : 1}>;`);
    }
    if (this._sharedBuffers.length > 0 || this._promotedBufferDecls.length > 0 || this._wgPoolDecls.length > 0) this._emit('');

    const builtins: string[] = [];
    const hasLocal = !this._serializeThreads && this._hasBindingPrefix('threadIdx');
    const hasWorkgroup = !this._serializeThreads && this._hasBindingPrefix('blockIdx');
    if (hasLocal) builtins.push('@builtin(local_invocation_id) _lid: vec3u');
    if (hasWorkgroup) builtins.push('@builtin(workgroup_id) _wid: vec3u');
    if (!hasLocal && !hasWorkgroup) builtins.push('@builtin(global_invocation_id) _gid: vec3u');

    const maxTotal = this.target.maxThreadsPerBlock || 256;
    while (this._workgroupSize[0] * this._workgroupSize[1] * this._workgroupSize[2] > maxTotal) {
      let maxIdx = 0;
      if (this._workgroupSize[1] > this._workgroupSize[maxIdx]) maxIdx = 1;
      if (this._workgroupSize[2] > this._workgroupSize[maxIdx]) maxIdx = 2;
      this._workgroupSize[maxIdx] = Math.max(1, this._workgroupSize[maxIdx] >> 1);
    }

    const wgx = this._workgroupSize[0];
    const wgy = this._workgroupSize[1];
    const wgz = this._workgroupSize[2];
    this._emit(`@compute @workgroup_size(${wgx}, ${wgy}, ${wgz})`);
    this._emit(`fn ${func.name}(${builtins.join(', ')}) {`);
    this._indent++;

    if (!this._serializeThreads) {
      const declaredVars = new Set<string>();
      for (const [tag, bindings_list] of this._threadBindings) {
        const wgslTag = WGSL_THREAD_TAG_MAP[tag];
        if (!wgslTag) continue;
        for (const info of bindings_list) {
          if (!declaredVars.has(info.varName)) {
            const src = this._wgslBuiltinAccess(tag);
            this._emit(`let ${info.varName}: i32 = i32(${src});`);
            declaredVars.add(info.varName);
          }
        }
      }
    }

    this._assignLocalSlots(func);
    this._emitMissingLocalDecls();
    this._visitNode(func.body);
    this._indent--;
    this._emit('}');

    const t = this.target;
    const workgroupSize = [
      Math.min(this._workgroupSize[0], t.maxBlockDimX),
      Math.min(this._workgroupSize[1], t.maxBlockDimY),
      Math.min(this._workgroupSize[2], t.maxBlockDimZ),
    ];
    const dispatchSize = [
      Math.min(this._dispatchSize[0], t.maxGridDimX),
      Math.min(this._dispatchSize[1], t.maxGridDimY),
      Math.min(this._dispatchSize[2], t.maxGridDimZ),
    ];

    return new WebGPUKernel({
      name: func.name,
      source: [...this._helperPreamble(), ...this._lines].join('\n'),
      workgroupSize,
      dispatchSize,
      sharedMemBytes: this._sharedBuffers.reduce((sum, b) => sum + Math.max(b.sizeInBytes(), 0), 0),
      params: paramNames,
      bindings,
      launchDiagnosis: this._launchDiagnosis,
    });
  }

  _checkF16Usage(func: CodegenFunc): boolean {
    for (const [, buf] of func.bufferMap) {
      if (buf.dtype === 'f16') return true;
    }
    return false;
  }

  _hasBindingPrefix(prefix: string): boolean {
    for (const tag of this._threadBindings.keys()) {
      if (tag.startsWith(prefix)) return true;
    }
    return false;
  }

  _wgslBuiltinAccess(tag: string | null): string {
    if (tag === null) return '_gid.x';
    const idx = tag.indexOf('.');
    if (idx < 0) return '_gid.x';
    const prefix = tag.substring(0, idx);
    const axis = tag.substring(idx + 1);
    if (prefix === 'threadIdx') return `_lid.${axis}`;
    if (prefix === 'blockIdx') return `_wid.${axis}`;
    return `_gid.${axis}`;
  }

  _scanBindings(root: IRStmtNode): void {
    walk(root, (node) => {
      if (node.type === 'ForNode' && node.kind === ForKind.THREAD_BINDING && node.threadTag) {
        const extent = node.extent.type === 'IntImmNode' ? node.extent.value : 0;
        const isDynamic = node.extent.type !== 'IntImmNode';
        const entry = { varName: node.loopVar.name, extent, isDynamic, extentNode: node.extent };
        if (!this._threadBindings.has(node.threadTag)) {
          this._threadBindings.set(node.threadTag, [entry]);
        } else {
          this._threadBindings.get(node.threadTag)!.push(entry);
        }
        if (!isDynamic) this._applyBindingDim(node.threadTag, extent);
      }
      if (node.type === 'AllocateNode' && node.scope === 'shared') {
        this._sharedBuffers.push(node.buffer);
      }
    }, { kinds: 'stmt' });
  }

  _applyBindingDim(tag: string, extent: number): void {
    const p = parseThreadAxis(tag);
    if (!p) return;
    if (p.space === 'thread') this._workgroupSize[p.axis] = Math.max(this._workgroupSize[p.axis], extent);
    else this._dispatchSize[p.axis] = Math.max(this._dispatchSize[p.axis], extent);
  }

  _serialize(reason: string, buffers: ReadonlySet<string>): void {
    this._serializeThreads = true;
    this._workgroupSize = [1, 1, 1];
    this._dispatchSize = [1, 1, 1];
    this._needsBarriers = false;
    this._launchDiagnosis = { reason, buffers: [...buffers] };
  }

  _hasRecurrence(func: CodegenFunc): boolean {
    let found = false;
    walk(func.body, (n) => {
      if (n.type === 'SyncThreadsNode') { found = true; return false; }
    }, { kinds: 'stmt' });
    return found;
  }

  _analyzeSharing(func: CodegenFunc): void {
    const extents = new Set<number>();
    for (const [, entries] of this._threadBindings) {
      for (const e of entries) {
        if (e.extent > 0) extents.add(e.extent);
      }
    }
    const profile = profileGpuAccesses(func, { sharedBuffers: this._sharedBuffers, threadBindings: this._threadBindings });
    const crossThread = this._threadBindings.size > 0 ? loopCarriedIntermediates(profile) : new Set<string>();
    const crossExtent = this._threadBindings.size > 0 ? extentMismatchBuffers(profile) : new Set<string>();
    this._crossThread = crossThread;
    this._crossExtent = crossExtent;
    const hasRecurrence = this._hasRecurrence(func);
    if (!hasRecurrence && extents.size <= 1 && crossThread.size === 0 && crossExtent.size === 0) return;

    const smemLimit = this.target.sharedMemoryBytes || 16384;
    let smemUsed = this._sharedBuffers.reduce((sum, b) => sum + Math.max(b.sizeInBytes(), 0), 0);

    const storageNames = new Set<string>();
    for (const [, buf] of func.bufferMap) storageNames.add(buf.name);

    const candidates = this._collectPromotionCandidates(func, storageNames);
    candidates.sort((a, b) => (crossThread.has(b.name) ? 1 : 0) - (crossThread.has(a.name) ? 1 : 0));

    const dispatchProduct = this._dispatchSize[0] * this._dispatchSize[1] * this._dispatchSize[2];
    if (hasRecurrence) {
      const maxTotal = this.target.maxThreadsPerBlock || 256;
      const wgProduct = this._workgroupSize[0] * this._workgroupSize[1] * this._workgroupSize[2];
      const pool = this._packWorkgroupPool(func, candidates);
      const poolFits = smemUsed + pool.bytes <= smemLimit;
      if (dispatchProduct === 1 && wgProduct <= maxTotal && poolFits) {
        this._needsBarriers = true;
        this._wgPoolOffsets = pool.offsets;
        this._wgPoolDecls = pool.decls;
        for (const c of candidates) this._promotedBuffers.add(c.name);
        return;
      }
      this._serialize(GpuRaceReason.RECURRENCE_EXCEEDS_WORKGROUP, new Set());
    } else if ((crossThread.size > 0 || crossExtent.size > 0) && dispatchProduct > 1) {
      const reason = crossThread.size > 0 ? GpuRaceReason.THREAD_SHARED_INTERMEDIATE : GpuRaceReason.EXTENT_MISMATCH;
      this._serialize(reason, new Set([...crossThread, ...crossExtent]));
    } else {
      this._needsBarriers = true;
    }

    for (const c of candidates) {
      const bytes = c.size * (wgslBytes(c.dtype) || 4);
      if (smemUsed + bytes <= smemLimit) {
        this._promotedBuffers.add(c.name);
        this._promotedBufferDecls.push(c);
        smemUsed += bytes;
      }
    }
  }

  _collectPromotionCandidates(func: CodegenFunc, storageNames: ReadonlySet<string>): BufferDecl[] {
    const candidates: BufferDecl[] = [];
    walk(func.body, (node) => {
      if (node.type === 'AllocateNode' && node.scope !== 'shared' && !storageNames.has(node.buffer.name)) {
        const numel = node.buffer.numel();
        const size = numel > 0 ? numel : this._estimateBufferSize(node.buffer);
        if (size > 0) candidates.push({ name: node.buffer.name, dtype: node.buffer.dtype, size });
      }
    }, { kinds: 'stmt' });

    const refBuffers = new Map<string, Buffer>();
    referencedBuffers(func.body, refBuffers);
    const allocatedNames = new Set<string>();
    allocatedBufferNames(func.body, allocatedNames);
    const candidateNames = new Set(candidates.map(c => c.name));
    for (const [name, buf] of refBuffers) {
      if (storageNames.has(name) || allocatedNames.has(name)) continue;
      if (candidateNames.has(name)) continue;
      const numel = buf.numel();
      const size = numel > 0 ? numel : this._estimateBufferSize(buf);
      if (size > 0) { candidates.push({ name, dtype: buf.dtype, size }); candidateNames.add(name); }
    }
    return candidates;
  }

  _findRecurrenceBody(root: IRStmtNode): IRStmtNode | null {
    let found: IRStmtNode | null = null;
    walk(root, (node) => {
      if (!found && node.type === 'ForNode' && node.kind === ForKind.RECURRENCE) { found = node.body; return false; }
    }, { kinds: 'stmt' });
    return found;
  }

  _namesTouchedOutside(root: IRStmtNode, skip: IRStmtNode | null, names: ReadonlySet<string>): Set<string> {
    const result = new Set<string>();
    walk(root, (node) => {
      if (node === skip) return false;
      const n = node as unknown as { buffer?: Buffer; flushStore?: { buffer?: Buffer }; initLoad?: { buffer?: Buffer } };
      const touch = (buffer: Buffer | null | undefined) => { if (buffer && names.has(buffer.name)) result.add(buffer.name); };
      if (ACCESS_NODE_TYPES.has(node.type)) touch(n.buffer);
      else if (node.type === 'LIRAccumulatorNode') {
        touch(n.flushStore && n.flushStore.buffer);
        touch(n.initLoad && n.initLoad.buffer);
      }
    });
    return result;
  }

  _packWorkgroupPool(func: CodegenFunc, candidates: readonly BufferDecl[]): WorkgroupPool {
    const candNames = new Set(candidates.map(c => c.name));
    const recBody = this._findRecurrenceBody(func.body);
    const { minPos, maxPos } = recBody ? this._livenessWalk(recBody, candNames) : { minPos: new Map<string, number>(), maxPos: new Map<string, number>() };
    const persistent = recBody ? this._namesTouchedOutside(func.body, recBody, candNames) : candNames;
    const byDtype = new Map<string, LiveItem[]>();
    for (const c of candidates) {
      if (!byDtype.has(c.dtype)) byDtype.set(c.dtype, []);
      const carried = persistent.has(c.name) || !minPos.has(c.name);
      const fb = carried ? 0 : minPos.get(c.name) as number;
      const lb = carried ? Number.MAX_SAFE_INTEGER : maxPos.get(c.name) as number;
      byDtype.get(c.dtype)!.push({ name: c.name, size: c.size, first: fb, last: lb });
    }
    const offsets = new Map<string, PoolRef>();
    const decls: PoolDecl[] = [];
    let bytes = 0;
    for (const [dtype, list] of byDtype) {
      list.sort((a, b) => a.first - b.first || b.size - a.size);
      const placed: PlacedItem[] = [];
      const pool = `_wg_${wgslType(dtype)}`;
      let peak = 0;
      for (const it of list) {
        const blocked = placed
          .filter(p => p.first <= it.last && it.first <= p.last)
          .map(p => [p.offset, p.offset + p.size])
          .sort((a, b) => a[0] - b[0]);
        let off = 0;
        for (const [lo, hi] of blocked) {
          if (off + it.size <= lo) break;
          if (off < hi) off = hi;
        }
        placed.push({ offset: off, size: it.size, first: it.first, last: it.last });
        offsets.set(it.name, { pool, offset: off });
        if (off + it.size > peak) peak = off + it.size;
      }
      decls.push({ pool, dtype, size: peak });
      bytes += peak * (wgslBytes(dtype) || 4);
    }
    return { offsets, decls, bytes };
  }

  _emitSync(): void {
    if (this._needsBarriers) {
      this._emit('storageBarrier();');
      this._emit('workgroupBarrier();');
    }
  }

  _visitForNode(node: ForNode): void {
    if (node.kind === ForKind.THREAD_BINDING) {
      if (this._serializeThreads) {
        const varName = node.loopVar.name;
        const extent = this._exprToWGSL(node.extent);
        this._emit(`for (var ${varName}: i32 = 0; ${varName} < ${extent}; ${varName} = ${varName} + 1) {`);
        this._indent++;
        this._visitNode(node.body);
        this._indent--;
        this._emit('}');
        return;
      }
      const extent = node.extent.type === 'IntImmNode' ? node.extent.value : 0;
      const tag = node.threadTag;
      const maxExtent = this._getMaxBindingExtent(tag);
      if (extent > 0 && maxExtent > 0 && extent < maxExtent) {
        const src = this._wgslBuiltinAccess(tag);
        this._emit(`if (i32(${src}) < ${extent}) {`);
        this._indent++;
        this._visitNode(node.body);
        this._indent--;
        this._emit('}');
      } else {
        this._visitNode(node.body);
      }
      if (this._needsBarriers) {
        this._emit('workgroupBarrier();');
      }
      return;
    }
    const varName = node.loopVar.name;
    const extent = this._exprToWGSL(node.extent);
    this._emit(`for (var ${varName}: i32 = 0; ${varName} < ${extent}; ${varName} = ${varName} + 1) {`);
    this._indent++;
    this._visitNode(node.body);
    this._indent--;
    this._emit('}');
  }

  _visitBlockNode(node: BlockNode): void {
    for (const bind of node.iterVars) {
      if (bind.iterVar && bind.binding) {
        this._emit(`let ${bind.iterVar.name}: i32 = ${this._exprToWGSL(bind.binding)};`);
      }
    }
    if (node.initBody) this._visitNode(node.initBody);
    this._visitNode(node.body);
  }

  _emitMissingLocalDecls(): void {
    for (const d of this._slotDecls) {
      if (d.scalar) this._emit(`var ${d.name}: ${wgslType(d.dtype)};`);
      else this._emit(`var ${d.name}: array<${wgslType(d.dtype)}, ${d.size}>;`);
    }
  }

  _collectLocalBuffers(func: CodegenFunc): Map<string, Buffer> {
    const storageNames = new Set<string>();
    for (const [, buf] of func.bufferMap) storageNames.add(buf.name);
    const sharedNames = new Set<string>();
    for (const b of this._sharedBuffers) sharedNames.add(b.name);

    const locals = new Map<string, Buffer>();
    const isLocal = (name: string) =>
      !storageNames.has(name) && !sharedNames.has(name) && !this._promotedBuffers.has(name);

    const refBuffers = new Map<string, Buffer>();
    referencedBuffers(func.body, refBuffers);
    for (const [name, buf] of refBuffers) {
      if (isLocal(name)) locals.set(name, buf);
    }

    walk(func.body, (node) => {
      if (node.type === 'AllocateNode' && node.scope !== 'shared' && node.buffer && isLocal(node.buffer.name)) {
        locals.set(node.buffer.name, node.buffer);
      }
    }, { kinds: 'stmt' });
    return locals;
  }

  _assignLocalSlots(func: CodegenFunc): void {
    const locals = this._collectLocalBuffers(func);
    if (locals.size === 0) return;

    const { minPos, maxPos } = this._computeBufferLiveness(func, locals);

    const order = [...locals.keys()].sort((x, y) =>
      ((minPos.get(x) ?? 0) - (minPos.get(y) ?? 0)) || ((maxPos.get(x) ?? 0) - (maxPos.get(y) ?? 0)));

    this._localSlots = new Map();
    const freeByDtype = new Map<string, MinHeap<HeapSlot>>();
    let slotCounter = 0;
    let scalarCounter = 0;

    const totalThreads = (this._workgroupSize[0] * this._workgroupSize[1] * this._workgroupSize[2])
      * (this._dispatchSize[0] * this._dispatchSize[1] * this._dispatchSize[2]);
    const ct = this._crossThread || new Set<string>();
    const ce = this._crossExtent || new Set<string>();
    const scalarEligible = (name: string, buf: Buffer) => !this._serializeThreads && this._threadBindings.size > 0
      && !ct.has(name) && !ce.has(name) && buf.numel() > 1 && buf.numel() <= totalThreads;

    for (const name of order) {
      const buf = locals.get(name) as Buffer;
      const numel = buf.numel() > 0 ? buf.numel() : this._estimateBufferSize(buf);
      const size = Math.max(numel, 1);
      const mn = minPos.get(name) ?? 0;
      const mx = maxPos.get(name) ?? 0;

      if (scalarEligible(name, buf)) {
        const sname = `_s${scalarCounter++}`;
        this._slotDecls.push({ name: sname, dtype: buf.dtype, size: 1, scalar: true });
        this._localSlots.set(name, sname);
        this._scalarSlotNames.add(sname);
        continue;
      }

      let heap = freeByDtype.get(buf.dtype);
      if (!heap) { heap = new MinHeap<HeapSlot>((a, b) => a.freeAt - b.freeAt); freeByDtype.set(buf.dtype, heap); }

      let slot: HeapSlot | null = null;
      const top = heap.peek();
      if (top && top.freeAt < mn) slot = heap.pop();

      if (slot) {
        if (size > slot.decl.size) slot.decl.size = size;
      } else {
        const decl = { name: `_lt${slotCounter++}`, dtype: buf.dtype, size };
        this._slotDecls.push(decl);
        slot = { decl } as HeapSlot;
      }
      slot.freeAt = mx;
      this._localSlots.set(name, slot.decl.name);
      heap.push(slot);
    }
  }

  _computeBufferLiveness(func: CodegenFunc, locals: LivenessScope): LivenessResult {
    return this._livenessWalk(func.body, locals);
  }

  _livenessWalk(root: IRStmtNode, locals: LivenessScope): LivenessResult {
    const minPos = new Map<string, number>();
    const maxPos = new Map<string, number>();

    let pos = 0;
    let loopDepth = 0;
    let outerSet: Set<string> | null = null;
    let outerStart = 0;

    const touch = (buffer: Buffer | null | undefined) => {
      if (!buffer || !locals.has(buffer.name)) return;
      const name = buffer.name;
      if (!minPos.has(name)) minPos.set(name, pos);
      maxPos.set(name, pos);
      if (outerSet) outerSet.add(name);
    };

    const openedOuterAt = new Set<IRStmtNode>();

    walk(root, {
      pre: (node) => {
        pos++;
        if (LOOP_NODE_TYPES.has(node.type)) {
          if (loopDepth === 0) { outerSet = new Set(); outerStart = pos; openedOuterAt.add(node as IRStmtNode); }
          loopDepth++;
        }
        const n = node as unknown as { buffer?: Buffer; flushStore?: { buffer?: Buffer }; initLoad?: { buffer?: Buffer } };
        if (ACCESS_NODE_TYPES.has(node.type)) touch(n.buffer);
        else if (node.type === 'LIRAccumulatorNode') {
          touch(n.flushStore && n.flushStore.buffer);
          touch(n.initLoad && n.initLoad.buffer);
        }
      },
      post: (node) => {
        if (!LOOP_NODE_TYPES.has(node.type)) return;
        loopDepth--;
        if (!openedOuterAt.delete(node as IRStmtNode)) return;
        for (const name of outerSet as Set<string>) {
          minPos.set(name, Math.min(minPos.get(name) as number, outerStart));
          maxPos.set(name, Math.max(maxPos.get(name) as number, pos));
        }
        outerSet = null;
      },
    });
    return { minPos, maxPos };
  }

  _visitAllocateNode(node: AllocateNode): void {
    if (node.scope !== 'shared') {
      if (this._promotedBuffers.has(node.buffer.name)) return;
      if (this._localSlots && this._localSlots.has(node.buffer.name)) return;
      const numel = node.buffer.numel();
      const size = numel > 0 ? numel : this._estimateBufferSize(node.buffer);
      if (size > 0) {
        this._emit(`var ${node.buffer.name}: array<${wgslType(node.buffer.dtype)}, ${size}>;`);
      }
    }
  }

  _visitIfStmt(node: IfThenElseNode): void {
    this._emit(`if (${this._boolExpr(node.condition)}) {`);
    this._indent++;
    this._visitNode(node.thenBody);
    this._indent--;
    if (node.elseBody) {
      this._emit('} else {');
      this._indent++;
      this._visitNode(node.elseBody);
      this._indent--;
    }
    this._emit('}');
  }

  _visitLetStmtNode(node: LetStmtNode): void {
    const varDtype = node.variable.dtype || this._defaultDtype;
    const dtype = wgslType(varDtype);
    const val = this._numericExpr(node.value, varDtype);
    this._emit(`var ${node.variable.name}: ${dtype} = ${val};`);
    this._visitNode(node.body);
  }

  _visitWhileNode(node: WhileNode): void {
    this._visitNode(node.condBody);
    const condExpr = node.condVar instanceof Buffer
      ? `${this._packedBufAccess(node.condVar.name, '0')} != 0`
      : node.condVar.name;
    this._emit(`while (${condExpr}) {`);
    this._indent++;
    this._visitNode(node.loopBody);
    this._visitNode(node.condBody);
    this._indent--;
    this._emit('}');
  }

  _visitVecCopyNode(node: VecCopyNode): void {
    const dstIdx = this._flatIndex(node.dstBuffer, [node.dstIndex]);
    const srcIdx = this._flatIndex(node.srcBuffer, [node.srcIndex]);
    for (let lane = 0; lane < node.width; lane++) {
      const off = lane === 0 ? '' : ` + ${lane}`;
      const dst = this._packedBufAccess(node.dstBuffer.name, `${dstIdx}${off}`);
      const src = this._packedBufAccess(node.srcBuffer.name, `${srcIdx}${off}`);
      this._emit(`${dst} = ${src};`);
    }
  }

  _visitBufferStoreNode(node: BufferStoreNode): void {
    const idx = this._flatIndex(node.buffer, node.indices);
    const target = this._packedBufAccess(node.buffer.name, idx);
    const val = this._numericExpr(node.value, node.buffer.dtype);
    this._emit(`${target} = ${val};`);
  }

  _visitLIRFlatStore(node: LIRFlatStoreNode): void {
    const idx = this._exprToWGSL(node.offsetExpr);
    const target = this._packedBufAccess(node.buffer.name, idx);
    const val = this._numericExpr(node.value, node.buffer.dtype);
    this._emit(`${target} = ${val};`);
  }

  _visitLIRBindings(node: LIRBindingsNode): void {
    for (const bind of node.bindings) {
      this._emit(`let ${bind.name}: i32 = ${this._numericExpr(bind.expr, 'i32')};`);
    }
    this._visitNode(node.body);
  }

  _visitLIRAccumulator(node: LIRAccumulatorNode): void {
    const accVar = node.localName;
    const dtype = wgslType(node.dtype || this._defaultDtype);
    this._emit(`var ${accVar}: ${dtype} = ${this._exprToWGSL(node.initLoad)};`);

    const varName = node.loopVar.name;
    const extent = this._exprToWGSL(node.extent);
    this._emit(`for (var ${varName}: i32 = 0; ${varName} < ${extent}; ${varName} = ${varName} + 1) {`);
    this._indent++;
    if (node.prologue) this._visitNode(node.prologue);
    const accOp = node.op || '+';
    const accBody = this._exprToWGSL(node.body);
    this._emit(`${accVar} = ${(accOp === 'max' || accOp === 'min') ? `${accOp}(${accVar}, ${accBody})` : `(${accVar} ${accOp} ${accBody})`};`);
    this._indent--;
    this._emit('}');

    const flushIdx = this._exprToWGSL(node.flushStore.offsetExpr);
    const flushTarget = this._packedBufAccess(node.flushStore.buffer.name, flushIdx);
    this._emit(`${flushTarget} = ${accVar};`);
  }

  _isBoolExpr(node: IRStmtNode | null): boolean {
    if (!node) return false;
    if (node.type === 'CompareNode') return true;
    if (node.type === 'MathOpNode' && (BOOL_OPS.has(node.op) || COMPARE_MATH_OPS.has(node.op))) return true;
    return false;
  }

  _numericExpr(node: IRStmtNode | null, dtype?: string): string {
    if (this._isBoolExpr(node)) {
      const wt = dtype ? wgslType(dtype) : 'i32';
      if (wt === 'f32') return `select(0.0, 1.0, ${this._exprToWGSL(node)})`;
      return `select(${wt}(0), ${wt}(1), ${this._exprToWGSL(node)})`;
    }
    return this._exprToWGSL(node);
  }

  _boolExpr(node: IRStmtNode | null): string {
    if (this._isBoolExpr(node)) return this._exprToWGSL(node);
    return `(${this._exprToWGSL(node)} != 0)`;
  }

  _emitCseBinding(name: string, text: string): void {
    this._emit(`let ${name}: i32 = ${text};`);
  }

  _emitExprText(node: IRStmtNode): string {
    return this._exprToWGSL(node);
  }

  _exprToWGSL(node: IRStmtNode | null): string {
    const top = this._beginExpr(node);
    try {
      const bound = this._cseNameFor(node);
      return bound !== null ? bound : this._printExpr(node);
    } finally {
      this._endExpr(top);
    }
  }

  _printExpr(node: IRStmtNode | null): string {
    if (!node) return '0';
    switch (node.type) {
      case 'IntImmNode': return String(node.value);
      case 'FloatImmNode': return this._emitFloatLiteral(node.value);
      case 'VariableNode': return this._resolveVariable(node.name);
      case 'BufferLoadNode': return this._packedBufAccess(node.buffer.name, this._flatIndex(node.buffer, node.indices));
      case 'LIRFlatLoadNode': return this._packedBufAccess(node.buffer.name, this._exprToWGSL(node.offsetExpr));
      case 'MathOpNode': {
        if (!node.b) {
          if (node.op === '!') return `(!${this._boolExpr(node.a)})`;
          return `(${node.op}${this._numericExpr(node.a)})`;
        }
        if (node.op === '&&') return `(${this._boolExpr(node.a)} && ${this._boolExpr(node.b)})`;
        if (node.op === '||') return `(${this._boolExpr(node.a)} || ${this._boolExpr(node.b)})`;
        const a = this._numericExpr(node.a);
        const b = this._numericExpr(node.b);
        const helper = WGSL_MATHOP_HELPERS[node.op];
        if (helper) return `${this._useHelper(helper)}(${a}, ${b})`;
        if (node.op === 'tdiv') return `(${a} / ${b})`;
        if (node.op === 'tmod') return `(${a} % ${b})`;
        return `(${a} ${node.op} ${b})`;
      }
      case 'CompareNode': return `(${this._numericExpr(node.a)} ${cCompareOp(node.direction)} ${this._numericExpr(node.b)})`;
      case 'IfThenElseNode': return `select(${this._exprToWGSL(node.elseBody)}, ${this._exprToWGSL(node.thenBody)}, ${this._boolExpr(node.condition)})`;
      case 'CastNode': return `${wgslType(node.toDtype)}(${this._exprToWGSL(node.expr)})`;
      case 'CallExternNode': return this._emitExternCall(node);
      default: throw new Error(`WebGPU codegen: unhandled expr node '${node.type}'`);
    }
  }

  _resolveVariable(name: string): string {
    if (this._primFunc && this._primFunc.shapeParams) {
      for (const sp of this._primFunc.shapeParams) {
        if (sp.name === name) return `i32(_shapes.${name})`;
      }
    }
    return name;
  }

  _emitFloatLiteral(value: number): string {
    if (value === Infinity) return 'f32(0x1.fffffep+127)';
    if (value === -Infinity) return 'f32(-0x1.fffffep+127)';
    if (Number.isInteger(value)) return `${value}.0`;
    return String(value);
  }

  _emitExternCall(node: CallExternNode): string {
    const n = node.args.length;
    const args = new Array<string>(n);
    for (let i = 0; i < n; i++) args[i] = this._exprToWGSL(node.args[i]);
    const joined = args.join(', ');
    if (node.externName === 'fmod') {
      return `(${args[0]} % ${args[1]})`;
    }
    if (node.externName === 'erf') return `${this._useHelper(WGSL_ERF_HELPER)}(${args[0]})`;
    if (node.externName === 'erfc') return `(1.0 - ${this._useHelper(WGSL_ERF_HELPER)}(${args[0]}))`;
    if (node.externName === 'lgamma') return `${this._useHelper(WGSL_LGAMMA_HELPER)}(${args[0]})`;
    if (node.externName === 'gamma') return `${this._useHelper(WGSL_GAMMA_HELPER)}(${args[0]})`;
    if (node.externName === 'log10') {
      return `(log(${args[0]}) * ${1 / Math.LN10})`;
    }
    const fn = wgslMathFunc(node.externName);
    if (fn === node.externName && !hasWgslMathFunc(node.externName)) {
      throw new Error(`WebGPU codegen: unsupported extern function "${node.externName}"`);
    }
    return `${fn}(${joined})`;
  }

  _packedBufAccess(bufName: string, indexExpr: string): string {
    if (this._wgPoolOffsets && this._wgPoolOffsets.has(bufName)) {
      const info = this._wgPoolOffsets.get(bufName) as PoolRef;
      if (info.offset === 0) return `${info.pool}[${indexExpr}]`;
      return `${info.pool}[${info.offset}u + u32(${indexExpr})]`;
    }
    if (this._packedMode && this._packedOffsets && this._packedOffsets.has(bufName)) {
      const info = this._packedOffsets.get(bufName) as PackedRef;
      if (info.offset === 0) return `${info.storage}[${indexExpr}]`;
      return `${info.storage}[${info.offset}u + u32(${indexExpr})]`;
    }
    if (this._localSlots && this._localSlots.has(bufName)) {
      const sn = this._localSlots.get(bufName) as string;
      if (this._scalarSlotNames.has(sn)) return sn;
      return `${sn}[${indexExpr}]`;
    }
    return `${bufName}[${indexExpr}]`;
  }

  _flatIndex(buffer: Buffer, indices: readonly IRStmtNode[]): string {
    return flattenRowMajorIndex(buffer, indices, (e) => this._exprToWGSL(e), (b, i) => this._computeDynamicStride(b, i), false);
  }

  _resolveShapeParam(buffer: Buffer, dimIdx: number): string {
    return resolveShapeParam(this._primFunc, buffer, dimIdx, (v) => `i32(_shapes.${v.name})`, 'WebGPU', 'wgsl');
  }
}
