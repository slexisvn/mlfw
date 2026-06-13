import { ForKind } from '../../compiler/ir/tensor/nodes.js';
import { wgslType, wgslBytes, wgslMathFunc, hasWgslMathFunc } from '../dtype_map.js';
import { inferDtype } from '../../compiler/ir/lir/nodes.js';

const BOOL_OPS = new Set(['!', '&&', '||']);
const CMP_OPS = new Set(['<', '>', '<=', '>=', '==', '!=']);
const COMPARE_DIRS = new Set(['eq', 'ne', 'lt', 'le', 'gt', 'ge']);

const WGSL_THREAD_TAG_MAP = {
  'threadIdx.x': 'local_invocation_id.x',
  'threadIdx.y': 'local_invocation_id.y',
  'threadIdx.z': 'local_invocation_id.z',
  'blockIdx.x': 'workgroup_id.x',
  'blockIdx.y': 'workgroup_id.y',
  'blockIdx.z': 'workgroup_id.z',
};

export class WebGPUKernel {
  constructor(name, source, workgroupSize, dispatchSize, sharedMemBytes, params, bindings) {
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
  constructor(target) {
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

  generate(func) {
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
    this._needsBarriers = false;
    this._serializeThreads = false;
    this._localSlots = null;
    this._slotDecls = [];

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

    const paramNames = [];
    const bindings = [];
    let bindIdx = 0;
    const needsF16 = this._checkF16Usage(func);

    if (needsF16) this._emit('enable f16;');
    if (needsF16) this._emit('');

    const bufCount = func.bufferMap.size + (func.shapeParams.length > 0 ? 1 : 0);
    const bufList = [...func.bufferMap.values()];
    const sameWgslType = bufList.every(b => wgslType(b.dtype) === wgslType(bufList[0].dtype));
    const canPack = bufCount > 6 && bufList.every(b => b.numel() > 0) && sameWgslType;
    this._packedMode = canPack;
    this._packedOffsets = null;

    if (canPack) {
      this._packedOffsets = new Map();
      const readBufs = [], writeBufs = [];
      let readOff = 0, writeOff = 0;
      const align4 = n => Math.ceil(n / 4) * 4;

      for (const [, buf] of func.bufferMap) {
        paramNames.push(buf.name);
        this._defaultDtype = buf.dtype;
        const numel = buf.numel();
        const aligned = align4(numel);
        if (this._storeBuffers.has(buf.name)) {
          this._packedOffsets.set(buf.name, { storage: '_pw', offset: writeOff });
          writeBufs.push({ name: buf.name, offset: writeOff, size: numel, dtype: buf.dtype });
          writeOff += aligned;
        } else {
          this._packedOffsets.set(buf.name, { storage: '_pr', offset: readOff });
          readBufs.push({ name: buf.name, offset: readOff, size: numel, dtype: buf.dtype });
          readOff += aligned;
        }
      }

      if (readBufs.length > 0) {
        bindings.push({ index: bindIdx, name: '_pr', mode: 'read', packed: readBufs, packedSize: readOff, dtype: this._defaultDtype });
        this._emit(`@group(0) @binding(${bindIdx}) var<storage, read> _pr: array<${wgslType(this._defaultDtype)}>;`);
        bindIdx++;
      }
      if (writeBufs.length > 0) {
        bindings.push({ index: bindIdx, name: '_pw', mode: 'read_write', packed: writeBufs, packedSize: writeOff, dtype: this._defaultDtype });
        this._emit(`@group(0) @binding(${bindIdx}) var<storage, read_write> _pw: array<${wgslType(this._defaultDtype)}>;`);
        bindIdx++;
      }
    } else {
      for (const [, buf] of func.bufferMap) {
        paramNames.push(buf.name);
        this._defaultDtype = buf.dtype;
        const mode = this._storeBuffers.has(buf.name) ? 'read_write' : 'read';
        const accessMode = mode === 'read_write' ? 'storage, read_write' : 'storage, read';
        bindings.push({ index: bindIdx, name: buf.name, mode, dtype: buf.dtype });
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
    if (this._sharedBuffers.length > 0 || this._promotedBufferDecls.length > 0) this._emit('');

    const builtins = [];
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
      const declaredVars = new Set();
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

  _checkF16Usage(func) {
    for (const [, buf] of func.bufferMap) {
      if (buf.dtype === 'f16') return true;
    }
    return false;
  }

  _hasBindingPrefix(prefix) {
    for (const tag of this._threadBindings.keys()) {
      if (tag.startsWith(prefix)) return true;
    }
    return false;
  }

  _wgslBuiltinAccess(tag) {
    const idx = tag.indexOf('.');
    if (idx < 0) return '_gid.x';
    const prefix = tag.substring(0, idx);
    const axis = tag.substring(idx + 1);
    if (prefix === 'threadIdx') return `_lid.${axis}`;
    if (prefix === 'blockIdx') return `_wid.${axis}`;
    return `_gid.${axis}`;
  }

  _scanStoreTargets(root) {
    const stack = [root];
    while (stack.length > 0) {
      const node = stack.pop();
      if (!node) continue;
      if (node.type === 'BufferStoreNode' || node.type === 'LIRFlatStoreNode') {
        this._storeBuffers.add(node.buffer.name);
      }
      if (node.type === 'LIRAccumulatorNode' && node.flushStore) {
        this._storeBuffers.add(node.flushStore.buffer.name);
      }
      if (node.body) stack.push(node.body);
      if (node.stmts) for (const s of node.stmts) stack.push(s);
      if (node.thenBody) stack.push(node.thenBody);
      if (node.elseBody) stack.push(node.elseBody);
      if (node.loopBody) stack.push(node.loopBody);
      if (node.condBody) stack.push(node.condBody);
      if (node.initBody) stack.push(node.initBody);
    }
  }

  _scanBindings(root) {
    const stack = [root];
    while (stack.length > 0) {
      const node = stack.pop();
      if (!node) continue;
      if (node.type === 'ForNode' && node.kind === ForKind.THREAD_BINDING && node.threadTag) {
        const extent = node.extent.type === 'IntImmNode' ? node.extent.value : 0;
        const isDynamic = node.extent.type !== 'IntImmNode';
        const entry = { varName: node.loopVar.name, extent, isDynamic, extentNode: node.extent };
        if (!this._threadBindings.has(node.threadTag)) {
          this._threadBindings.set(node.threadTag, [entry]);
        } else {
          this._threadBindings.get(node.threadTag).push(entry);
        }
        if (!isDynamic) this._applyBindingDim(node.threadTag, extent);
      }
      if (node.type === 'AllocateNode' && node.scope === 'shared') {
        this._sharedBuffers.push(node.buffer);
      }
      if (node.body) stack.push(node.body);
      if (node.stmts) for (const s of node.stmts) stack.push(s);
      if (node.thenBody) stack.push(node.thenBody);
      if (node.elseBody) stack.push(node.elseBody);
    }
  }

  _applyBindingDim(tag, extent) {
    const idx = tag.indexOf('.');
    if (idx < 0) return;
    const prefix = tag.substring(0, idx);
    const axis = tag.charCodeAt(idx + 1) - 120;
    if (axis < 0 || axis > 2) return;
    if (prefix === 'threadIdx') this._workgroupSize[axis] = Math.max(this._workgroupSize[axis], extent);
    else if (prefix === 'blockIdx') this._dispatchSize[axis] = Math.max(this._dispatchSize[axis], extent);
  }

  _getMaxBindingExtent(tag) {
    const entries = this._threadBindings.get(tag);
    if (!entries) return 0;
    let max = 0;
    for (const e of entries) if (e.extent > max) max = e.extent;
    return max;
  }

  _analyzeSharing(func) {
    const extents = new Set();
    for (const [, entries] of this._threadBindings) {
      for (const e of entries) {
        if (e.extent > 0) extents.add(e.extent);
      }
    }
    const crossThread = this._threadBindings.size > 0 ? this._findCrossThreadBuffers(func) : new Set();
    if (extents.size <= 1 && crossThread.size === 0) return;

    const dispatchProduct = this._dispatchSize[0] * this._dispatchSize[1] * this._dispatchSize[2];
    this._serializeThreads = crossThread.size > 0 && dispatchProduct > 1;
    if (this._serializeThreads) {
      this._workgroupSize = [1, 1, 1];
      this._dispatchSize = [1, 1, 1];
    }
    this._needsBarriers = !this._serializeThreads;

    const smemLimit = this.target.sharedMemoryBytes || 16384;
    let smemUsed = this._sharedBuffers.reduce((sum, b) => sum + Math.max(b.sizeInBytes(), 0), 0);

    const storageNames = new Set();
    for (const [, buf] of func.bufferMap) storageNames.add(buf.name);

    const candidates = [];

    const stack = [func.body];
    while (stack.length > 0) {
      const node = stack.pop();
      if (!node) continue;
      if (node.type === 'AllocateNode' && node.scope !== 'shared') {
        if (!storageNames.has(node.buffer.name)) {
          const numel = node.buffer.numel();
          const size = numel > 0 ? numel : this._estimateBufferSize(node.buffer);
          if (size > 0) candidates.push({ name: node.buffer.name, dtype: node.buffer.dtype, size });
        }
        stack.push(node.body);
        continue;
      }
      if (node.body) stack.push(node.body);
      if (node.stmts) for (const s of node.stmts) stack.push(s);
      if (node.thenBody) stack.push(node.thenBody);
      if (node.elseBody) stack.push(node.elseBody);
    }

    const refBuffers = new Map();
    this._scanBufferRefs(func.body, refBuffers);
    const allocatedNames = new Set();
    this._scanAllocateNodes(func.body, allocatedNames);
    for (const [name, buf] of refBuffers) {
      if (storageNames.has(name) || allocatedNames.has(name)) continue;
      if (candidates.some(c => c.name === name)) continue;
      const numel = buf.numel();
      const size = numel > 0 ? numel : this._estimateBufferSize(buf);
      if (size > 0) candidates.push({ name, dtype: buf.dtype, size });
    }

    candidates.sort((a, b) => (crossThread.has(b.name) ? 1 : 0) - (crossThread.has(a.name) ? 1 : 0));

    for (const c of candidates) {
      const bytes = c.size * (wgslBytes(c.dtype) || 4);
      if (smemUsed + bytes <= smemLimit) {
        this._promotedBuffers.add(c.name);
        this._promotedBufferDecls.push(c);
        smemUsed += bytes;
      }
    }
  }

  _findCrossThreadBuffers(func) {
    const storageNames = new Set();
    for (const [, buf] of func.bufferMap) storageNames.add(buf.name);
    const result = new Set();

    const indexUsesLoopVar = (indices, loopVars) => {
      if (loopVars.length === 0) return false;
      const names = new Set();
      for (const idx of indices) this._collectVarNames(idx, names);
      return loopVars.some(v => names.has(v));
    };

    const aliasOf = (expr, vars) => {
      const bv = new Set();
      this._collectVarNames(expr, bv);
      for (const v of bv) if (vars.includes(v)) return true;
      return false;
    };

    const walk = (node, loopVars) => {
      if (!node) return;
      let inner = loopVars;
      const isSeqLoop = (node.type === 'ForNode' && node.kind !== ForKind.THREAD_BINDING)
        || node.type === 'WhileNode' || node.type === 'LIRAccumulatorNode';
      if (isSeqLoop && node.loopVar) inner = [...loopVars, node.loopVar.name];

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

      for (const k of ['body', 'loopBody', 'condBody', 'initBody', 'thenBody', 'elseBody', 'value',
        'a', 'b', 'condition', 'expr', 'offsetExpr']) if (node[k]) walk(node[k], inner);
      if (node.stmts) for (const s of node.stmts) walk(s, inner);
      if (node.indices) for (const i of node.indices) walk(i, inner);
      if (node.args) for (const a of node.args) walk(a, inner);
      if (node.initLoad) walk(node.initLoad, inner);
      if (node.flushStore) walk(node.flushStore, inner);
    };

    walk(func.body, []);
    return result;
  }

  _collectVarNames(node, out) {
    const stack = [node];
    while (stack.length > 0) {
      const n = stack.pop();
      if (!n || typeof n !== 'object') continue;
      if (n.type === 'VariableNode') { out.add(n.name); continue; }
      for (const k of ['a', 'b', 'condition', 'thenBody', 'elseBody', 'expr', 'offsetExpr']) {
        if (n[k]) stack.push(n[k]);
      }
      if (n.indices) for (const i of n.indices) stack.push(i);
      if (n.args) for (const a of n.args) stack.push(a);
    }
  }

  _emit(line) {
    this._lines.push('  '.repeat(this._indent) + line);
  }

  _visitNode(node) {
    const stack = [node];
    while (stack.length > 0) {
      const cur = stack.pop();
      if (!cur) continue;
      switch (cur.type) {
        case 'SeqNode':
          for (let i = cur.stmts.length - 1; i >= 0; i--) stack.push(cur.stmts[i]);
          continue;
        case 'AllocateNode':
          this._visitAllocateNode(cur);
          stack.push(cur.body);
          continue;
        case 'ForNode': this._visitForNode(cur); continue;
        case 'BlockNode': this._visitBlockNode(cur); continue;
        case 'IfThenElseNode': this._visitIfStmt(cur); continue;
        case 'LetStmtNode': this._visitLetStmtNode(cur); continue;
        case 'BufferStoreNode': this._visitBufferStoreNode(cur); continue;
        case 'LIRFlatStoreNode': this._visitLIRFlatStore(cur); continue;
        case 'LIRBindingsNode': this._visitLIRBindings(cur); continue;
        case 'LIRAccumulatorNode': this._visitLIRAccumulator(cur); continue;
        case 'WhileNode': this._visitWhileNode(cur); continue;
        case 'EvaluateNode': continue;
        default: continue;
      }
    }
  }

  _visitForNode(node) {
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

  _visitBlockNode(node) {
    for (const bind of node.iterVars) {
      if (bind.iterVar && bind.binding) {
        this._emit(`let ${bind.iterVar.name}: i32 = ${this._exprToWGSL(bind.binding)};`);
      }
    }
    if (node.initBody) this._visitNode(node.initBody);
    this._visitNode(node.body);
  }

  _emitMissingLocalDecls() {
    for (const d of this._slotDecls) {
      this._emit(`var ${d.name}: array<${wgslType(d.dtype)}, ${d.size}>;`);
    }
  }

  _collectLocalBuffers(func) {
    const storageNames = new Set();
    for (const [, buf] of func.bufferMap) storageNames.add(buf.name);
    const sharedNames = new Set();
    for (const b of this._sharedBuffers) sharedNames.add(b.name);

    const locals = new Map();
    const isLocal = (name) =>
      !storageNames.has(name) && !sharedNames.has(name) && !this._promotedBuffers.has(name);

    const refBuffers = new Map();
    this._scanBufferRefs(func.body, refBuffers);
    for (const [name, buf] of refBuffers) {
      if (isLocal(name)) locals.set(name, buf);
    }

    const stack = [func.body];
    while (stack.length > 0) {
      const node = stack.pop();
      if (!node) continue;
      if (node.type === 'AllocateNode' && node.scope !== 'shared' && node.buffer && isLocal(node.buffer.name)) {
        locals.set(node.buffer.name, node.buffer);
      }
      if (node.body) stack.push(node.body);
      if (node.stmts) for (const s of node.stmts) stack.push(s);
      if (node.thenBody) stack.push(node.thenBody);
      if (node.elseBody) stack.push(node.elseBody);
      if (node.loopBody) stack.push(node.loopBody);
      if (node.condBody) stack.push(node.condBody);
      if (node.initBody) stack.push(node.initBody);
    }
    return locals;
  }

  _assignLocalSlots(func) {
    const locals = this._collectLocalBuffers(func);
    if (locals.size === 0) return;

    const { minPos, maxPos } = this._computeBufferLiveness(func, locals);

    const order = [...locals.keys()].sort((x, y) =>
      (minPos.get(x) - minPos.get(y)) || (maxPos.get(x) - maxPos.get(y)));

    this._localSlots = new Map();
    const freeByDtype = new Map();
    let slotCounter = 0;

    for (const name of order) {
      const buf = locals.get(name);
      const numel = buf.numel() > 0 ? buf.numel() : this._estimateBufferSize(buf);
      const size = Math.max(numel, 1);
      const mn = minPos.get(name);
      const mx = maxPos.get(name);

      let heap = freeByDtype.get(buf.dtype);
      if (!heap) { heap = new MinHeap((a, b) => a.freeAt - b.freeAt); freeByDtype.set(buf.dtype, heap); }

      let slot = null;
      const top = heap.peek();
      if (top && top.freeAt < mn) slot = heap.pop();

      if (slot) {
        if (size > slot.decl.size) slot.decl.size = size;
      } else {
        const decl = { name: `_lt${slotCounter++}`, dtype: buf.dtype, size };
        this._slotDecls.push(decl);
        slot = { decl };
      }
      slot.freeAt = mx;
      this._localSlots.set(name, slot.decl.name);
      heap.push(slot);
    }
  }

  _computeBufferLiveness(func, locals) {
    const minPos = new Map();
    const maxPos = new Map();
    const LOOP_TYPES = new Set(['ForNode', 'WhileNode', 'LIRAccumulatorNode']);
    const CHILD_FIELDS = ['stmts', 'body', 'initBody', 'condBody', 'loopBody', 'thenBody',
      'elseBody', 'value', 'a', 'b', 'condition', 'expr', 'offsetExpr', 'extent', 'indices', 'args'];

    let pos = 0;
    const loopStack = [];
    let outerSet = null;
    let outerStart = 0;

    const touch = (name) => {
      if (!locals.has(name)) return;
      if (!minPos.has(name)) minPos.set(name, pos);
      maxPos.set(name, pos);
      if (loopStack.length > 0) outerSet.add(name);
    };

    const walk = (node) => {
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
        if (node.flushStore && node.flushStore.buffer) touch(node.flushStore.buffer.name);
        if (node.initLoad && node.initLoad.buffer) touch(node.initLoad.buffer.name);
      }
      for (const f of CHILD_FIELDS) {
        const c = node[f];
        if (!c) continue;
        if (Array.isArray(c)) { for (const e of c) walk(e); }
        else if (typeof c === 'object') walk(c);
      }
      if (isLoop) {
        loopStack.pop();
        if (openedOuter) {
          const endPos = pos;
          for (const name of outerSet) {
            minPos.set(name, Math.min(minPos.get(name), outerStart));
            maxPos.set(name, Math.max(maxPos.get(name), endPos));
          }
          outerSet = null;
        }
      }
    };

    walk(func.body);
    return { minPos, maxPos };
  }

  _scanAllocateNodes(root, names) {
    const stack = [root];
    while (stack.length > 0) {
      const node = stack.pop();
      if (!node) continue;
      if (node.type === 'AllocateNode') {
        names.add(node.buffer.name);
        stack.push(node.body);
        continue;
      }
      if (node.body) stack.push(node.body);
      if (node.stmts) for (const s of node.stmts) stack.push(s);
      if (node.thenBody) stack.push(node.thenBody);
      if (node.elseBody) stack.push(node.elseBody);
      if (node.loopBody) stack.push(node.loopBody);
      if (node.condBody) stack.push(node.condBody);
      if (node.initBody) stack.push(node.initBody);
    }
  }

  _scanBufferRefs(root, refs) {
    const stack = [root];
    while (stack.length > 0) {
      const node = stack.pop();
      if (!node) continue;
      if ((node.type === 'BufferLoadNode' || node.type === 'BufferStoreNode' ||
           node.type === 'LIRFlatLoadNode' || node.type === 'LIRFlatStoreNode') && node.buffer) {
        refs.set(node.buffer.name, node.buffer);
      }
      if (node.type === 'LIRAccumulatorNode') {
        if (node.flushStore && node.flushStore.buffer) refs.set(node.flushStore.buffer.name, node.flushStore.buffer);
        if (node.initLoad && node.initLoad.buffer) refs.set(node.initLoad.buffer.name, node.initLoad.buffer);
      }
      if (node.value) stack.push(node.value);
      if (node.body) stack.push(node.body);
      if (node.stmts) for (const s of node.stmts) stack.push(s);
      if (node.thenBody) stack.push(node.thenBody);
      if (node.elseBody) stack.push(node.elseBody);
      if (node.loopBody) stack.push(node.loopBody);
      if (node.condBody) stack.push(node.condBody);
      if (node.initBody) stack.push(node.initBody);
      if (node.indices) for (const idx of node.indices) stack.push(idx);
      if (node.a) stack.push(node.a);
      if (node.b) stack.push(node.b);
      if (node.condition) stack.push(node.condition);
      if (node.expr) stack.push(node.expr);
      if (node.args) for (const a of node.args) stack.push(a);
      if (node.offsetExpr) stack.push(node.offsetExpr);
      if (node.extent) stack.push(node.extent);
    }
  }

  _estimateBufferSize(buffer) {
    let n = 1;
    for (const d of buffer.shape) {
      if (typeof d === 'number' && d > 0) n *= d;
      else n *= 1;
    }
    return n;
  }

  _visitAllocateNode(node) {
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

  _visitIfStmt(node) {
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

  _visitLetStmtNode(node) {
    const varDtype = node.variable.dtype || this._defaultDtype;
    const dtype = wgslType(varDtype);
    const val = this._numericExpr(node.value, varDtype);
    this._emit(`var ${node.variable.name}: ${dtype} = ${val};`);
    this._visitNode(node.body);
  }

  _visitWhileNode(node) {
    this._visitNode(node.condBody);
    this._emit(`while (${node.condVar.name}) {`);
    this._indent++;
    this._visitNode(node.loopBody);
    this._visitNode(node.condBody);
    this._indent--;
    this._emit('}');
  }

  _visitBufferStoreNode(node) {
    const idx = this._flatIndex(node.buffer, node.indices);
    const target = this._packedBufAccess(node.buffer.name, idx);
    const val = this._numericExpr(node.value, node.buffer.dtype);
    this._emit(`${target} = ${val};`);
  }

  _visitLIRFlatStore(node) {
    const idx = this._exprToWGSL(node.offsetExpr);
    const target = this._packedBufAccess(node.buffer.name, idx);
    const val = this._numericExpr(node.value, node.buffer.dtype);
    this._emit(`${target} = ${val};`);
  }

  _visitLIRBindings(node) {
    for (const bind of node.bindings) {
      this._emit(`let ${bind.name}: i32 = ${this._numericExpr(bind.expr, 'i32')};`);
    }
    this._visitNode(node.body);
  }

  _visitLIRAccumulator(node) {
    const accVar = node.localName;
    const dtype = wgslType(node.dtype || this._defaultDtype);
    this._emit(`var ${accVar}: ${dtype} = ${this._exprToWGSL(node.initLoad)};`);

    const varName = node.loopVar.name;
    const extent = this._exprToWGSL(node.extent);
    this._emit(`for (var ${varName}: i32 = 0; ${varName} < ${extent}; ${varName} = ${varName} + 1) {`);
    this._indent++;
    this._emit(`${accVar} = (${accVar} + ${this._exprToWGSL(node.body)});`);
    this._indent--;
    this._emit('}');

    const flushIdx = this._exprToWGSL(node.flushStore.offsetExpr);
    const flushTarget = this._packedBufAccess(node.flushStore.buffer.name, flushIdx);
    this._emit(`${flushTarget} = ${accVar};`);
  }

  _isBoolExpr(node) {
    if (!node) return false;
    if (node.type === 'CompareNode') return true;
    if (node.type === 'MathOpNode' && (BOOL_OPS.has(node.op) || CMP_OPS.has(node.op))) return true;
    return false;
  }

  _numericExpr(node, dtype) {
    if (this._isBoolExpr(node)) {
      const wt = dtype ? wgslType(dtype) : 'i32';
      if (wt === 'f32') return `select(0.0, 1.0, ${this._exprToWGSL(node)})`;
      return `select(${wt}(0), ${wt}(1), ${this._exprToWGSL(node)})`;
    }
    return this._exprToWGSL(node);
  }

  _boolExpr(node) {
    if (this._isBoolExpr(node)) return this._exprToWGSL(node);
    return `(${this._exprToWGSL(node)} != 0)`;
  }

  _exprToWGSL(node) {
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
      case 'CompareNode': return `(${this._numericExpr(node.a)} ${node.toC()} ${this._numericExpr(node.b)})`;
      case 'IfThenElseNode': return `select(${this._exprToWGSL(node.elseBody)}, ${this._exprToWGSL(node.thenBody)}, ${this._boolExpr(node.condition)})`;
      case 'CastNode': return `${wgslType(node.toDtype)}(${this._exprToWGSL(node.expr)})`;
      case 'CallExternNode': return this._emitExternCall(node);
      default: return '0';
    }
  }

  _resolveVariable(name) {
    if (this._primFunc && this._primFunc.shapeParams) {
      for (const sp of this._primFunc.shapeParams) {
        if (sp.name === name) return `i32(_shapes.${name})`;
      }
    }
    return name;
  }

  _emitFloatLiteral(value) {
    if (value === Infinity) return 'f32(0x1.fffffep+127)';
    if (value === -Infinity) return 'f32(-0x1.fffffep+127)';
    if (Number.isInteger(value)) return `${value}.0`;
    return String(value);
  }

  _emitExternCall(node) {
    const n = node.args.length;
    const args = new Array(n);
    for (let i = 0; i < n; i++) args[i] = this._exprToWGSL(node.args[i]);
    const joined = args.join(', ');
    if (node.externName === 'fmod') {
      return `(${args[0]} % ${args[1]})`;
    }
    if (node.externName === 'erf') {
      return `((select(-1.0, 1.0, ${args[0]} >= 0.0)) * (1.0 - (1.0 / (1.0 + 0.3275911 * abs(${args[0]}))) * (0.254829592 + (1.0 / (1.0 + 0.3275911 * abs(${args[0]}))) * (-0.284496736 + (1.0 / (1.0 + 0.3275911 * abs(${args[0]}))) * (1.421413741 + (1.0 / (1.0 + 0.3275911 * abs(${args[0]}))) * (-1.453152027 + (1.0 / (1.0 + 0.3275911 * abs(${args[0]}))) * 1.061405429)))) * exp(-${args[0]} * ${args[0]})))`;
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

  _packedBufAccess(bufName, indexExpr) {
    if (this._packedMode && this._packedOffsets && this._packedOffsets.has(bufName)) {
      const info = this._packedOffsets.get(bufName);
      if (info.offset === 0) return `${info.storage}[${indexExpr}]`;
      return `${info.storage}[${info.offset}u + u32(${indexExpr})]`;
    }
    if (this._localSlots && this._localSlots.has(bufName)) {
      return `${this._localSlots.get(bufName)}[${indexExpr}]`;
    }
    return `${bufName}[${indexExpr}]`;
  }

  _flatIndex(buffer, indices) {
    if (indices.length === 0) return '0';
    if (indices.length === 1) return this._exprToWGSL(indices[0]);
    const parts = new Array(indices.length);
    for (let i = 0; i < indices.length; i++) {
      const idx = this._exprToWGSL(indices[i]);
      const stride = buffer.strides[i];
      if (stride === 1) {
        parts[i] = idx;
      } else if (typeof stride === 'number' && stride >= 0) {
        parts[i] = `${idx} * ${stride}`;
      } else {
        parts[i] = `${idx} * ${this._computeDynamicStride(buffer, i)}`;
      }
    }
    return parts.join(' + ');
  }

  _computeDynamicStride(buffer, dimIdx) {
    const parts = [];
    for (let j = dimIdx + 1; j < buffer.shape.length; j++) {
      const d = buffer.shape[j];
      if (typeof d === 'number' && d >= 0) {
        parts.push(String(d));
      } else {
        parts.push(this._resolveShapeParam(buffer, j));
      }
    }
    return parts.length === 0 ? '1' : parts.join(' * ');
  }

  _resolveShapeParam(buffer, dimIdx) {
    if (this._primFunc && this._primFunc.shapeParamMap) {
      const key = `${buffer.name}:${dimIdx}`;
      const v = this._primFunc.shapeParamMap.get(key);
      if (v) return `i32(_shapes.${v.name})`;
    }
    return '1';
  }
}

class MinHeap {
  constructor(compare) {
    this._items = [];
    this._compare = compare;
  }

  get size() { return this._items.length; }

  peek() { return this._items.length > 0 ? this._items[0] : null; }

  push(item) {
    const items = this._items;
    items.push(item);
    let i = items.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this._compare(items[i], items[parent]) >= 0) break;
      [items[i], items[parent]] = [items[parent], items[i]];
      i = parent;
    }
  }

  pop() {
    const items = this._items;
    const top = items[0];
    const last = items.pop();
    if (items.length > 0) {
      items[0] = last;
      let i = 0;
      const n = items.length;
      for (;;) {
        const l = 2 * i + 1;
        const r = 2 * i + 2;
        let smallest = i;
        if (l < n && this._compare(items[l], items[smallest]) < 0) smallest = l;
        if (r < n && this._compare(items[r], items[smallest]) < 0) smallest = r;
        if (smallest === i) break;
        [items[i], items[smallest]] = [items[smallest], items[i]];
        i = smallest;
      }
    }
    return top;
  }
}
