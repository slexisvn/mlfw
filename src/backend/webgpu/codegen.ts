import { ForKind } from '../../compiler/ir/tensor/nodes.js';
import { wgslType, wgslBytes, wgslMathFunc, hasWgslMathFunc, cCompareOp } from '../../util/dtype_map.js';
import { flattenRowMajorIndex } from '../index_emit.js';
import { MinHeap } from '../../util/min_heap.js';
import { irChildNodes } from '../../compiler/ir/ir_visitor.js';
import { parseThreadAxis, maxBindingExtent, visitStatements, estimateBufferSize, dynamicDimProduct, resolveShapeParam, walkStmtTree } from '../codegen_utils.js';


import { LANCZOS_G, LANCZOS_COEFFS, ERF_A, ERF_P } from '../../util/special_math.js';

import type { Buffer } from '../../compiler/ir/tensor/buffer.js';
import type { AllocateNode, BlockNode, BufferStoreNode, CallExternNode, ForNode, IfThenElseNode, LetStmtNode, NodeSlots, TirNode, WhileNode } from '../../compiler/ir/tensor/nodes.js';
import type { IRStmtNode, LIRAccumulatorNode, LIRBindingsNode, LIRFlatLoadNode, LIRFlatStoreNode } from '../../compiler/ir/lir/nodes.js';
import type { BufferDecl, CodegenFunc, ThreadBindingEntry } from '../codegen_utils.js';
import type { TargetFeatures } from '../target.js';

const BOOL_OPS = new Set(['!', '&&', '||']);
const CMP_OPS = new Set(['<', '>', '<=', '>=', '==', '!=']);
const COMPARE_DIRS = new Set(['eq', 'ne', 'lt', 'le', 'gt', 'ge']);

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

function _wgslLgamma(x: string): string {
  const reflected = `(log(${_WGSL_PI} / abs(sin(${_WGSL_PI} * ${x}))) - ${_wgslLanczosCore(`(1.0 - ${x})`)})`;
  return `(select(${_wgslLanczosCore(x)}, ${reflected}, ${x} < 0.5))`;
}

function _wgslGamma(x: string): string {
  const reflected = `(${_WGSL_PI} / (sin(${_WGSL_PI} * ${x}) * exp(${_wgslLanczosCore(`(1.0 - ${x})`)})))`;
  return `(select(exp(${_wgslLanczosCore(x)}), ${reflected}, ${x} < 0.5))`;
}

const FULL_WALK_KEYS = ['body', 'loopBody', 'condBody', 'initBody', 'thenBody', 'elseBody', 'value', 'a', 'b', 'condition', 'expr', 'offsetExpr'];
function walkFullChildren(node: IRStmtNode, visit: (child: IRStmtNode) => void): void {
  const slots = node as unknown as NodeSlots;
  for (const k of FULL_WALK_KEYS) if (slots[k]) visit(slots[k] as TirNode);
  if (slots.stmts) for (const s of slots.stmts as TirNode[]) visit(s);
  if (slots.indices) for (const i of slots.indices as TirNode[]) visit(i);
  if (slots.args) for (const a of slots.args as TirNode[]) visit(a);
  if (slots.initLoad) visit(slots.initLoad as TirNode);
  if (slots.flushStore) visit(slots.flushStore as TirNode);
}

const WGSL_THREAD_TAG_MAP: WgslNameTable = {
  'threadIdx.x': 'local_invocation_id.x',
  'threadIdx.y': 'local_invocation_id.y',
  'threadIdx.z': 'local_invocation_id.z',
  'blockIdx.x': 'workgroup_id.x',
  'blockIdx.y': 'workgroup_id.y',
  'blockIdx.z': 'workgroup_id.z',
};

export class WebGPUKernel {
  name: string;
  source: string;
  workgroupSize: number[];
  dispatchSize: number[];
  sharedMemBytes: number;
  params: string[];
  bindings: WebGPUBinding[];

  constructor(name: string, source: string, workgroupSize: number[], dispatchSize: number[], sharedMemBytes: number, params: string[], bindings: WebGPUBinding[]) {
    this.name = name;
    this.source = source;
    this.workgroupSize = workgroupSize;
    this.dispatchSize = dispatchSize;
    this.sharedMemBytes = sharedMemBytes;
    this.params = params;
    this.bindings = bindings;
  }
}

export class WebGPUCodegen {
  target: TargetFeatures;
  _indent: number;
  _lines: string[];
  _threadBindings: Map<string, ThreadBindingEntry[]>;
  _sharedBuffers: Buffer[];
  _workgroupSize: number[];
  _dispatchSize: number[];
  _defaultDtype: string;
  _storeBuffers: Set<string>;
  declare _primFunc: CodegenFunc;
  declare _promotedBuffers: Set<string>;
  declare _promotedBufferDecls: BufferDecl[];
  declare _wgPoolOffsets: Map<string, PoolRef> | null;
  declare _wgPoolDecls: PoolDecl[];
  declare _needsBarriers: boolean;
  declare _serializeThreads: boolean;
  declare _localSlots: Map<string, string> | null;
  declare _slotDecls: SlotDecl[];
  declare _scalarSlotNames: Set<string>;
  declare _crossThread: Set<string> | null;
  declare _crossExtent: Set<string> | null;
  declare _packedMode: boolean;
  declare _packedOffsets: Map<string, PackedRef> | null;

  constructor(target: TargetFeatures) {
    this.target = target;
    this._indent = 0;
    this._lines = [];
    this._threadBindings = new Map();
    this._sharedBuffers = [];
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
    this._localSlots = null;
    this._slotDecls = [];
    this._scalarSlotNames = new Set();
    this._crossThread = null;
    this._crossExtent = null;

    const isLIR = func.type === 'LIRFunc';

    if (isLIR) {
      for (const [tag, entries] of func.metadata.threadBindings as Map<string, ThreadBindingEntry[]>) {
        this._threadBindings.set(tag, entries);
        for (const entry of entries) {
          if (!entry.isDynamic) this._applyBindingDim(tag, entry.extent);
        }
      }
      this._sharedBuffers = func.metadata.sharedBuffers as Buffer[];
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
    this._emitMissingLocalDecls(func);
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

    return new WebGPUKernel(
      func.name,
      this._lines.join('\n'),
      workgroupSize, dispatchSize,
      this._sharedBuffers.reduce((sum, b) => sum + Math.max(b.sizeInBytes(), 0), 0),
      paramNames,
      bindings
    );
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

  _wgslBuiltinAccess(tag: string): string {
    const idx = tag.indexOf('.');
    if (idx < 0) return '_gid.x';
    const prefix = tag.substring(0, idx);
    const axis = tag.substring(idx + 1);
    if (prefix === 'threadIdx') return `_lid.${axis}`;
    if (prefix === 'blockIdx') return `_wid.${axis}`;
    return `_gid.${axis}`;
  }

  _scanStoreTargets(root: IRStmtNode): void {
    const stack: IRStmtNode[] = [root];
    while (stack.length > 0) {
      const node = stack.pop();
      if (!node) continue;
      if (node.type === 'BufferStoreNode' || node.type === 'LIRFlatStoreNode') {
        this._storeBuffers.add(node.buffer.name);
      }
      if (node.type === 'LIRAccumulatorNode' && node.flushStore) {
        this._storeBuffers.add((node.flushStore as LIRFlatStoreNode).buffer.name);
      }
      for (const c of irChildNodes(node)) stack.push(c);
    }
  }

  _scanBindings(root: IRStmtNode): void {
    walkStmtTree(root, (node) => {
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
    });
  }

  _applyBindingDim(tag: string, extent: number): void {
    const p = parseThreadAxis(tag);
    if (!p) return;
    if (p.space === 'thread') this._workgroupSize[p.axis] = Math.max(this._workgroupSize[p.axis], extent);
    else this._dispatchSize[p.axis] = Math.max(this._dispatchSize[p.axis], extent);
  }

  _getMaxBindingExtent(tag: string): number {
    return maxBindingExtent(this._threadBindings, tag);
  }

  _hasRecurrence(func: CodegenFunc): boolean {
    let found = false;
    walkStmtTree(func.body, (n) => {
      if (n.type === 'SyncThreadsNode') { found = true; return false; }
    });
    return found;
  }

  _analyzeSharing(func: CodegenFunc): void {
    const extents = new Set<number>();
    for (const [, entries] of this._threadBindings) {
      for (const e of entries) {
        if (e.extent > 0) extents.add(e.extent);
      }
    }
    const crossThread = this._threadBindings.size > 0 ? this._findCrossThreadBuffers(func) : new Set<string>();
    const crossExtent = this._threadBindings.size > 0 ? this._findCrossExtentBuffers(func) : new Set<string>();
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
      this._serializeThreads = true;
      this._workgroupSize = [1, 1, 1];
      this._dispatchSize = [1, 1, 1];
      this._needsBarriers = false;
    } else {
      this._serializeThreads = (crossThread.size > 0 || crossExtent.size > 0) && dispatchProduct > 1;
      if (this._serializeThreads) {
        this._workgroupSize = [1, 1, 1];
        this._dispatchSize = [1, 1, 1];
      }
      this._needsBarriers = !this._serializeThreads;
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
    walkStmtTree(func.body, (node) => {
      if (node.type === 'AllocateNode' && node.scope !== 'shared' && !storageNames.has(node.buffer.name)) {
        const numel = node.buffer.numel();
        const size = numel > 0 ? numel : this._estimateBufferSize(node.buffer);
        if (size > 0) candidates.push({ name: node.buffer.name, dtype: node.buffer.dtype, size });
      }
    });

    const refBuffers = new Map<string, Buffer>();
    this._scanBufferRefs(func.body, refBuffers);
    const allocatedNames = new Set<string>();
    this._scanAllocateNodes(func.body, allocatedNames);
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
    walkStmtTree(root, (node) => {
      if (!found && node.type === 'ForNode' && node.kind === ForKind.RECURRENCE) { found = node.body; return false; }
    });
    return found;
  }

  _namesTouchedOutside(root: IRStmtNode, skip: IRStmtNode | null, names: ReadonlySet<string>): Set<string> {
    const result = new Set<string>();
    const walk = (node: IRStmtNode | null) => {
      if (!node || node === skip) return;
      if ((node.type === 'BufferLoadNode' || node.type === 'BufferStoreNode' ||
           node.type === 'LIRFlatLoadNode' || node.type === 'LIRFlatStoreNode') && node.buffer
        && names.has(node.buffer.name)) result.add(node.buffer.name);
      if (node.type === 'LIRAccumulatorNode') {
        if (node.flushStore && (node.flushStore as LIRFlatStoreNode).buffer && names.has((node.flushStore as LIRFlatStoreNode).buffer.name)) result.add((node.flushStore as LIRFlatStoreNode).buffer.name);
        if (node.initLoad && (node.initLoad as LIRFlatLoadNode).buffer && names.has((node.initLoad as LIRFlatLoadNode).buffer.name)) result.add((node.initLoad as LIRFlatLoadNode).buffer.name);
      }
      for (const f of ['stmts', 'body', 'initBody', 'condBody', 'loopBody', 'thenBody', 'elseBody', 'value', 'a', 'b', 'condition', 'expr', 'offsetExpr', 'extent', 'indices', 'args']) {
        const c = (node as unknown as NodeSlots)[f];
        if (!c) continue;
        if (Array.isArray(c)) { for (const e of c) walk(e); }
        else if (typeof c === 'object') walk(c);
      }
    };
    walk(root);
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

  _findCrossThreadBuffers(func: CodegenFunc): Set<string> {
    const storageNames = new Set<string>();
    for (const [, buf] of func.bufferMap) storageNames.add(buf.name);
    const result = new Set<string>();

    const indexUsesLoopVar = (indices: readonly IRStmtNode[], loopVars: readonly string[]) => {
      if (loopVars.length === 0) return false;
      const names = new Set<string>();
      for (const idx of indices) this._collectVarNames(idx, names);
      return loopVars.some(v => names.has(v));
    };

    const aliasOf = (expr: IRStmtNode, vars: readonly string[]) => {
      const bv = new Set<string>();
      this._collectVarNames(expr, bv);
      for (const v of bv) if (vars.includes(v)) return true;
      return false;
    };

    const walk = (node: IRStmtNode | null, loopVars: readonly string[]) => {
      if (!node) return;
      let inner = loopVars;
      const isSeqLoop = (node.type === 'ForNode' && node.kind !== ForKind.THREAD_BINDING)
        || node.type === 'WhileNode' || node.type === 'LIRAccumulatorNode';
      if (isSeqLoop && (node as ForNode).loopVar) inner = [...loopVars, (node as ForNode).loopVar.name];

      if (node.type === 'LIRBindingsNode' && node.bindings) {
        for (const b of node.bindings) {
          if (!inner.includes(b.name) && aliasOf(b.expr, inner)) inner = [...inner, b.name];
        }
      }
      if (node.type === 'BlockNode' && node.iterVars) {
        for (const iv of node.iterVars) {
          if (iv.iterVar && iv.binding && !inner.includes(iv.iterVar.name) && aliasOf(iv.binding, inner)) {
            inner = [...inner, iv.iterVar.name];
          }
        }
      }
      if (node.type === 'LetStmtNode' && node.variable && !inner.includes(node.variable.name) && aliasOf(node.value, inner)) {
        inner = [...inner, node.variable.name];
      }

      if (node.type === 'BufferLoadNode' && node.buffer && !storageNames.has(node.buffer.name)
        && indexUsesLoopVar(node.indices || [], inner)) result.add(node.buffer.name);
      if (node.type === 'LIRFlatLoadNode' && node.buffer && !storageNames.has(node.buffer.name)
        && node.offsetExpr && indexUsesLoopVar([node.offsetExpr], inner)) result.add(node.buffer.name);

      walkFullChildren(node, (c) => walk(c, inner));
    };

    walk(func.body, []);
    return result;
  }

  _findCrossExtentBuffers(func: CodegenFunc): Set<string> {
    const storageNames = new Set<string>();
    for (const [, buf] of func.bufferMap) storageNames.add(buf.name);
    const stores = new Map<string, Set<number>>();
    const loads = new Map<string, Set<number>>();
    const threadVarying = new Set<string>();
    const isConst = (v: IRStmtNode | null) => v && (v.type === 'FloatImmNode' || v.type === 'IntImmNode');

    const record = (map: Map<string, Set<number>>, name: string, extent: number) => {
      if (storageNames.has(name)) return;
      let s = map.get(name);
      if (!s) { s = new Set(); map.set(name, s); }
      s.add(extent);
    };

    const walk = (node: IRStmtNode | null, extent: number) => {
      if (!node) return;
      let inner = extent;
      if (node.type === 'ForNode' && node.kind === ForKind.THREAD_BINDING) {
        const e = node.extent && node.extent.type === 'IntImmNode' ? node.extent.value : 0;
        if (e > 0) inner = extent * e;
      }
      if (node.type === 'BufferStoreNode' && node.buffer) { record(stores, node.buffer.name, inner); if (!isConst(node.value)) threadVarying.add(node.buffer.name); }
      if (node.type === 'LIRFlatStoreNode' && node.buffer) { record(stores, node.buffer.name, inner); if (!isConst(node.value)) threadVarying.add(node.buffer.name); }
      if (node.type === 'LIRAccumulatorNode' && node.flushStore && (node.flushStore as LIRFlatStoreNode).buffer) { record(stores, (node.flushStore as LIRFlatStoreNode).buffer.name, inner); threadVarying.add((node.flushStore as LIRFlatStoreNode).buffer.name); }
      if (node.type === 'BufferLoadNode' && node.buffer) record(loads, node.buffer.name, inner);
      if (node.type === 'LIRFlatLoadNode' && node.buffer) record(loads, node.buffer.name, inner);

      walkFullChildren(node, (c) => walk(c, inner));
    };
    walk(func.body, 1);

    const result = new Set<string>();
    for (const [name, loadExtents] of loads) {
      if (!threadVarying.has(name)) continue;
      const storeExtents = stores.get(name);
      if (!storeExtents) continue;
      for (const le of loadExtents) if (!storeExtents.has(le)) { result.add(name); break; }
    }
    return result;
  }

  _collectVarNames(node: IRStmtNode | null, out: Set<string>): void {
    const stack: (IRStmtNode | null)[] = [node];
    while (stack.length > 0) {
      const n = stack.pop();
      if (!n || typeof n !== 'object') continue;
      if (n.type === 'VariableNode') { out.add(n.name); continue; }
      for (const k of ['a', 'b', 'condition', 'thenBody', 'elseBody', 'expr', 'offsetExpr']) {
        if ((n as unknown as NodeSlots)[k]) stack.push((n as unknown as NodeSlots)[k] as TirNode);
      }
      if ((n as unknown as NodeSlots).indices) for (const i of (n as unknown as NodeSlots).indices as TirNode[]) stack.push(i);
      if ((n as unknown as NodeSlots).args) for (const a of (n as unknown as NodeSlots).args as TirNode[]) stack.push(a);
    }
  }

  _emit(line: string): void {
    this._lines.push('  '.repeat(this._indent) + line);
  }

  _visitNode(node: IRStmtNode): void {
    visitStatements(this, node);
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
      const maxExtent = this._getMaxBindingExtent(tag as string);
      if (extent > 0 && maxExtent > 0 && extent < maxExtent) {
        const src = this._wgslBuiltinAccess(tag as string);
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

  _emitMissingLocalDecls(func: CodegenFunc): void;
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
    this._scanBufferRefs(func.body, refBuffers);
    for (const [name, buf] of refBuffers) {
      if (isLocal(name)) locals.set(name, buf);
    }

    walkStmtTree(func.body, (node) => {
      if (node.type === 'AllocateNode' && node.scope !== 'shared' && node.buffer && isLocal(node.buffer.name)) {
        locals.set(node.buffer.name, node.buffer);
      }
    });
    return locals;
  }

  _assignLocalSlots(func: CodegenFunc): void {
    const locals = this._collectLocalBuffers(func);
    if (locals.size === 0) return;

    const { minPos, maxPos } = this._computeBufferLiveness(func, locals);

    const order = [...locals.keys()].sort((x, y) =>
      ((minPos.get(x) as number) - (minPos.get(y) as number)) || ((maxPos.get(x) as number) - (maxPos.get(y) as number)));

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
      const mn = minPos.get(name) as number;
      const mx = maxPos.get(name) as number;

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
    const LOOP_TYPES = new Set(['ForNode', 'WhileNode', 'LIRAccumulatorNode']);
    const CHILD_FIELDS = ['stmts', 'body', 'initBody', 'condBody', 'loopBody', 'thenBody',
      'elseBody', 'value', 'a', 'b', 'condition', 'expr', 'offsetExpr', 'extent', 'indices', 'args'];

    let pos = 0;
    const loopStack: IRStmtNode[] = [];
    let outerSet: Set<string> | null = null;
    let outerStart = 0;

    const touch = (name: string) => {
      if (!locals.has(name)) return;
      if (!minPos.has(name)) minPos.set(name, pos);
      maxPos.set(name, pos);
      if (loopStack.length > 0) outerSet!.add(name);
    };

    const walk = (node: IRStmtNode | null) => {
      if (!node) return;
      pos++;
      const isLoop = LOOP_TYPES.has(node.type);
      let openedOuter = false;
      if (isLoop) {
        if (loopStack.length === 0) { outerSet = new Set(); outerStart = pos; openedOuter = true; }
        loopStack.push(node);
      }
      if ((node.type === 'BufferLoadNode' || node.type === 'BufferStoreNode' ||
           node.type === 'LIRFlatLoadNode' || node.type === 'LIRFlatStoreNode') && node.buffer) {
        touch(node.buffer.name);
      }
      if (node.type === 'LIRAccumulatorNode') {
        if (node.flushStore && (node.flushStore as LIRFlatStoreNode).buffer) touch((node.flushStore as LIRFlatStoreNode).buffer.name);
        if (node.initLoad && (node.initLoad as LIRFlatLoadNode).buffer) touch((node.initLoad as LIRFlatLoadNode).buffer.name);
      }
      for (const f of CHILD_FIELDS) {
        const c = (node as unknown as NodeSlots)[f];
        if (!c) continue;
        if (Array.isArray(c)) { for (const e of c) walk(e); }
        else if (typeof c === 'object') walk(c);
      }
      if (isLoop) {
        loopStack.pop();
        if (openedOuter) {
          const endPos = pos;
          for (const name of outerSet as unknown as Set<string>) {
            minPos.set(name, Math.min(minPos.get(name) as number, outerStart));
            maxPos.set(name, Math.max(maxPos.get(name) as number, endPos));
          }
          outerSet = null;
        }
      }
    };

    walk(root);
    return { minPos, maxPos };
  }

  _scanAllocateNodes(root: IRStmtNode, names: Set<string>): void {
    const stack: IRStmtNode[] = [root];
    while (stack.length > 0) {
      const node = stack.pop();
      if (!node) continue;
      if (node.type === 'AllocateNode') names.add(node.buffer.name);
      for (const c of irChildNodes(node)) stack.push(c);
    }
  }

  _scanBufferRefs(root: IRStmtNode, refs: Map<string, Buffer>): void {
    const stack: IRStmtNode[] = [root];
    while (stack.length > 0) {
      const node = stack.pop();
      if (!node) continue;
      if ((node.type === 'BufferLoadNode' || node.type === 'BufferStoreNode' ||
           node.type === 'LIRFlatLoadNode' || node.type === 'LIRFlatStoreNode') && node.buffer) {
        refs.set(node.buffer.name, node.buffer);
      }
      if (node.type === 'LIRAccumulatorNode') {
        if (node.flushStore && (node.flushStore as LIRFlatStoreNode).buffer) refs.set((node.flushStore as LIRFlatStoreNode).buffer.name, (node.flushStore as LIRFlatStoreNode).buffer);
        if (node.initLoad && (node.initLoad as LIRFlatLoadNode).buffer) refs.set((node.initLoad as LIRFlatLoadNode).buffer.name, (node.initLoad as LIRFlatLoadNode).buffer);
      }
      for (const c of irChildNodes(node)) stack.push(c);
    }
  }

  _estimateBufferSize(buffer: Buffer): number {
    return estimateBufferSize(buffer);
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
    const condExpr = Array.isArray((node.condVar as Buffer).shape)
      ? `${this._packedBufAccess(node.condVar.name, '0')} != 0`
      : node.condVar.name;
    this._emit(`while (${condExpr}) {`);
    this._indent++;
    this._visitNode(node.loopBody);
    this._visitNode(node.condBody);
    this._indent--;
    this._emit('}');
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
    const accOp = node.op || '+';
    const accBody = this._exprToWGSL(node.body);
    this._emit(`${accVar} = ${(accOp === 'max' || accOp === 'min') ? `${accOp}(${accVar}, ${accBody})` : `(${accVar} ${accOp} ${accBody})`};`);
    this._indent--;
    this._emit('}');

    const flushIdx = this._exprToWGSL((node.flushStore as LIRFlatStoreNode).offsetExpr);
    const flushTarget = this._packedBufAccess((node.flushStore as LIRFlatStoreNode).buffer.name, flushIdx);
    this._emit(`${flushTarget} = ${accVar};`);
  }

  _isBoolExpr(node: IRStmtNode | null): boolean {
    if (!node) return false;
    if (node.type === 'CompareNode') return true;
    if (node.type === 'MathOpNode' && (BOOL_OPS.has(node.op) || CMP_OPS.has(node.op))) return true;
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

  _exprToWGSL(node: IRStmtNode | null): string {
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
        if (node.op === '//') return `(${a} / ${b})`;
        if (node.op === '%') return `(${a} % ${b})`;
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
    if (node.externName === 'erf') {
      return _wgslErf(args[0]);
    }
    if (node.externName === 'erfc') {
      return `(1.0 - ${_wgslErf(args[0])})`;
    }
    if (node.externName === 'lgamma') {
      return _wgslLgamma(args[0]);
    }
    if (node.externName === 'gamma') {
      return _wgslGamma(args[0]);
    }
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

  _computeDynamicStride(buffer: Buffer, dimIdx: number): string {
    return dynamicDimProduct(buffer, dimIdx + 1, (b, j) => this._resolveShapeParam(b, j));
  }

  _resolveShapeParam(buffer: Buffer, dimIdx: number): string {
    return resolveShapeParam(this._primFunc, buffer, dimIdx, (v) => `i32(_shapes.${v.name})`, 'WebGPU', 'wgsl');
  }
}
