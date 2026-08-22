import { GpuCodegenBase } from '../codegen_base.js';
import { ForKind } from '../../compiler/ir/tensor/nodes.js';
import { cType, cPtrType, cLiteralSuffix, cMathFunc, isDtypeInt, dtypeBytes, cCompareOp } from '../../util/dtype_map.js';
import { flattenRowMajorIndex } from '../index_emit.js';
import { walk } from '../../compiler/ir/ir_visitor.js';
import { resolveShapeParam } from '../codegen_utils.js';
import { parseThreadAxis } from '../../compiler/analysis/thread_binding.js';
import { allocatedBufferNames, referencedBuffers } from '../../compiler/analysis/tir_queries.js';
import { profileGpuAccesses, crossBlockRAWBuffers, threadSharedIntermediates, storedUnderBlockBinding, hasMultiExtentBlockBinding, GpuRaceReason } from '../../compiler/analysis/gpu_race.js';
import type { GpuLaunchDiagnosis } from '../../compiler/analysis/gpu_race.js';
import { getCudaIntrin } from './tensor_intrin.js';
import { FuncAttr } from '../../compiler/ir/func_attrs.js';

import { Buffer } from '../../compiler/ir/tensor/buffer.js';
import type { AllocateNode, BlockNode, BufferStoreNode, CallExternNode, ForNode, IfThenElseNode, LetStmtNode, TirNode, VecCopyNode, WhileNode } from '../../compiler/ir/tensor/nodes.js';
import type { IRStmtNode, LIRAccumulatorNode, LIRBindingsNode, LIRFlatStoreNode } from '../../compiler/ir/lir/nodes.js';
import type { BufferDecl, CodegenFunc } from '../codegen_utils.js';
import type { CudaIntrinInfo } from './tensor_intrin.js';
import type { TargetFeatures } from '../target.js';

const LOCAL_MEMORY_BUDGET_BYTES = 256 * 1024;

type ReductionLoop = { extC: string; extVal: number; varName: string };
type FullReduction = { loops: ReductionLoop[]; block: BlockNode; store: BufferStoreNode; valExpr: TirNode | null; total: number };
type ScratchCandidate = BufferDecl & { bytes: number };
type TensorIntrinAttr = { name: string; info: CudaIntrinInfo };

export type CUDAKernelConfig = Readonly<{
  name: string;
  source: string;
  blockDim: number[];
  gridDim: number[];
  sharedMemBytes: number;
  params: string[];
  outputIndices: number[];
  scratch: BufferDecl[];
  launchDiagnosis?: GpuLaunchDiagnosis | null;
}>;

export class CUDAKernel {
  name: string;
  source: string;
  blockDim: number[];
  gridDim: number[];
  sharedMemBytes: number;
  params: string[];
  outputIndices: number[];
  scratch: BufferDecl[];
  launchDiagnosis: GpuLaunchDiagnosis | null;

  constructor(config: CUDAKernelConfig) {
    this.name = config.name;
    this.source = config.source;
    this.blockDim = config.blockDim;
    this.gridDim = config.gridDim;
    this.sharedMemBytes = config.sharedMemBytes;
    this.params = config.params;
    this.outputIndices = config.outputIndices;
    this.scratch = config.scratch || [];
    this.launchDiagnosis = config.launchDiagnosis || null;
  }
}

export class CUDACodegen extends GpuCodegenBase {
  _blockDim: number[];
  _gridDim: number[];
  _defaultDtype: string;
  _promotedBuffers: Set<string>;
  _promotedBufferDecls: BufferDecl[];
  _declaredLocals: Set<string>;
  _needsBarriers: boolean;
  _globalScratch: BufferDecl[];
  _scratchNames: Set<string>;
  _serializeThreads: boolean;
  _launchDiagnosis: GpuLaunchDiagnosis | null;
  declare _didParallelReduce: boolean;
  declare _primFunc: CodegenFunc;

  constructor(target: TargetFeatures) {
    super(target);
    this._blockDim = [1, 1, 1];
    this._gridDim = [1, 1, 1];
    this._defaultDtype = 'f32';
    this._storeBuffers = new Set();
    this._promotedBuffers = new Set();
    this._promotedBufferDecls = [];
    this._declaredLocals = new Set();
    this._needsBarriers = false;
    this._globalScratch = [];
    this._scratchNames = new Set();
    this._serializeThreads = false;
    this._launchDiagnosis = null;
  }

  generate(func: CodegenFunc): CUDAKernel {
    this._indent = 0;
    this._lines = [];
    this._threadBindings.clear();
    this._sharedBuffers = [];
    this._blockDim = [1, 1, 1];
    this._gridDim = [1, 1, 1];
    this._didParallelReduce = false;
    this._primFunc = func;
    this._storeBuffers = new Set();
    this._promotedBuffers = new Set();
    this._promotedBufferDecls = [];
    this._declaredLocals = new Set();
    this._needsBarriers = false;
    this._globalScratch = [];
    this._scratchNames = new Set();
    this._serializeThreads = false;
    this._launchDiagnosis = null;

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
    this._collectGlobalScratch(func);

    const paramParts: string[] = [];
    const paramNames: string[] = [];
    const outputIndices: number[] = [];
    let bufferIdx = 0;
    for (const [, buf] of func.bufferMap) {
      paramNames.push(buf.name);
      paramParts.push(`${cPtrType(buf.dtype)} ${buf.name}`);
      this._defaultDtype = buf.dtype;
      if (this._storeBuffers.has(buf.name)) outputIndices.push(bufferIdx);
      bufferIdx++;
    }
    for (const s of this._globalScratch) {
      paramNames.push(s.name);
      paramParts.push(`${cPtrType(s.dtype)} ${s.name}`);
    }
    for (const sp of func.shapeParams) {
      paramNames.push(sp.name);
      paramParts.push(`int ${sp.name}`);
    }

    this._emit(`__global__ void ${func.name}(${paramParts.join(', ')}) {`);
    this._indent++;

    const emittedShared = new Set<string>();
    for (const buf of this._sharedBuffers) {
      if (emittedShared.has(buf.name)) continue;
      emittedShared.add(buf.name);
      const numel = buf.numel();
      const al = buf.align16 ? '__align__(16) ' : '';
      this._emit(`__shared__ ${al}${cType(buf.dtype)} ${buf.name}[${numel > 0 ? numel : 1}];`);
    }
    for (const d of this._promotedBufferDecls) {
      this._emit(`__shared__ ${cType(d.dtype)} ${d.name}[${d.size}];`);
    }

    const declaredVars = new Set<string>();
    if (!this._serializeThreads) {
      for (const [tag, bindings] of this._threadBindings) {
        for (const info of bindings) {
          if (!declaredVars.has(info.varName)) {
            this._emit(`const int ${info.varName} = ${tag};`);
            declaredVars.add(info.varName);
          }
        }
      }
    }

    this._emitMissingLocalDecls(func);
    const tensorIntrin = func.getAttr<TensorIntrinAttr>(FuncAttr.TENSOR_INTRIN);
    if (tensorIntrin) {
      const emit = getCudaIntrin(tensorIntrin.name);
      if (!emit) throw new Error(`CUDA codegen: unknown tensor intrinsic '${tensorIntrin.name}'`);
      emit(this, tensorIntrin.info);
    } else {
      this._visitNode(func.body);
    }
    this._indent--;
    this._emit('}');

    const t = this.target;
    const blockDim = this._serializeThreads ? [1, 1, 1] : [
      Math.min(this._blockDim[0], t.maxBlockDimX),
      Math.min(this._blockDim[1], t.maxBlockDimY),
      Math.min(this._blockDim[2], t.maxBlockDimZ),
    ];
    const gridDim = this._serializeThreads ? [1, 1, 1] : [
      Math.min(this._gridDim[0], t.maxGridDimX),
      Math.min(this._gridDim[1], t.maxGridDimY),
      Math.min(this._gridDim[2], t.maxGridDimZ),
    ];

    const blockThreads = blockDim[0] * blockDim[1] * blockDim[2];
    if (blockThreads > t.maxThreadsPerBlock) {
      throw new Error(`[codegen] kernel '${func.name}' block ${blockDim.join('x')} = ${blockThreads} threads exceeds maxThreadsPerBlock ${t.maxThreadsPerBlock}`);
    }

    const sharedBytes = this._sharedBuffers.reduce((sum, b) => sum + Math.max(b.sizeInBytes(), 0), 0)
      + this._promotedBufferDecls.reduce((sum, d) => sum + Math.max(d.size, 0) * dtypeBytes(d.dtype), 0);
    if (sharedBytes > t.sharedMemoryBytes) {
      throw new Error(`[codegen] kernel '${func.name}' shared memory ${sharedBytes} bytes exceeds device limit ${t.sharedMemoryBytes}`);
    }

    return new CUDAKernel({
      name: func.name,
      source: this._lines.join('\n'),
      blockDim,
      gridDim,
      sharedMemBytes: sharedBytes,
      params: paramNames,
      outputIndices,
      scratch: this._globalScratch,
      launchDiagnosis: this._launchDiagnosis,
    });
  }

  _scanBindings(root: IRStmtNode): void {
    walk(root, (node) => {
      if (node.type === 'ForNode' && node.kind === ForKind.THREAD_BINDING && node.threadTag) {
        const extent = node.extent.type === 'IntImmNode' ? node.extent.value : 0;
        const isDynamic = node.extent.type !== 'IntImmNode';
        const entry = { varName: node.loopVar.name, extent, isDynamic, extentNode: node.extent };
        const entries = this._threadBindings.get(node.threadTag);
        if (entries) entries.push(entry);
        else this._threadBindings.set(node.threadTag, [entry]);
        if (!isDynamic) this._applyBindingDim(node.threadTag, extent);
      } else if (node.type === 'AllocateNode' && node.scope === 'shared') {
        this._sharedBuffers.push(node.buffer);
      }
    });
  }

  _applyBindingDim(tag: string, extent: number): void {
    const p = parseThreadAxis(tag);
    if (!p) return;
    if (p.space === 'thread') this._blockDim[p.axis] = Math.max(this._blockDim[p.axis], extent);
    else this._gridDim[p.axis] = Math.max(this._gridDim[p.axis], extent);
  }

  _emitSync(): void {
    this._emit('__syncthreads();');
  }

  _matchFullReduction(node: ForNode): FullReduction | null {
    const loops: ReductionLoop[] = [];
    let cur: IRStmtNode = node;
    while (cur && cur.type === 'ForNode') {
      if (cur.kind !== ForKind.SERIAL) return null;
      const ev = cur.extent.type === 'IntImmNode' ? cur.extent.value : 0;
      if (ev <= 0) return null;
      loops.push({ extC: this._exprToC(cur.extent), extVal: ev, varName: cur.loopVar.name });
      cur = cur.body;
    }
    if (!cur || cur.type !== 'BlockNode') return null;
    const block = cur, store = block.body;
    if (!store || store.type !== 'BufferStoreNode') return null;
    if (block.iterVars && block.iterVars.length !== loops.length) return null;
    if (typeof store.buffer.numel !== 'function' || store.buffer.numel() !== 1) return null;
    const val = store.value;
    if (!val || val.type !== 'MathOpNode' || val.op !== '+') return null;
    const idxStr = store.indices.map(i => this._exprToC(i)).join(',');
    const isOutLoad = (n: TirNode | null) => n && n.type === 'BufferLoadNode' && n.buffer === store.buffer
      && n.indices.map(i => this._exprToC(i)).join(',') === idxStr;
    let valExpr: TirNode | null = null;
    if (isOutLoad(val.a)) valExpr = val.b;
    else if (isOutLoad(val.b)) valExpr = val.a;
    else return null;
    const loopVars = new Set(loops.map(l => l.varName));
    if (idxStr.split(/[^A-Za-z0-9_]/).some(t => loopVars.has(t))) return null;
    const total = loops.reduce((a, l) => a * l.extVal, 1);
    if (total < 2048) return null;
    return { loops, block, store, valExpr, total };
  }

  _emitParallelReduction(node: ForNode, red: FullReduction): void {
    const T = 256;
    this._blockDim = [T, 1, 1];
    this._didParallelReduce = true;
    const ct = cType(red.store.buffer.dtype);
    const outIdx = this._flatIndex(red.store.buffer, red.store.indices);
    this._emit(`__shared__ ${ct} _redsh[${T}];`);
    this._emit(`${ct} _racc = 0;`);
    this._emit(`for (int _rf = threadIdx.x; _rf < ${red.total}; _rf += ${T}) {`);
    this._indent++;
    this._emit('int _rem = _rf;');
    for (let i = red.loops.length - 1; i >= 0; i--) {
      this._emit(`const int ${red.loops[i].varName} = _rem % ${red.loops[i].extC}; _rem /= ${red.loops[i].extC};`);
    }
    for (const bind of red.block.iterVars) {
      if (bind.iterVar && bind.binding) this._emit(`const int ${bind.iterVar.name} = ${this._exprToC(bind.binding)};`);
    }
    this._emit(`_racc = _racc + ${this._exprToC(red.valExpr)};`);
    this._indent--;
    this._emit('}');
    this._emit('_redsh[threadIdx.x] = _racc;');
    this._emit('__syncthreads();');
    this._emit(`for (int _rs = ${T} / 2; _rs > 0; _rs >>= 1) {`);
    this._indent++;
    this._emit('if (threadIdx.x < _rs) _redsh[threadIdx.x] = _redsh[threadIdx.x] + _redsh[threadIdx.x + _rs];');
    this._emit('__syncthreads();');
    this._indent--;
    this._emit('}');
    this._emit(`if (threadIdx.x == 0) ${red.store.buffer.name}[${outIdx}] = _redsh[0];`);
  }

  _visitForNode(node: ForNode): void {
    if (node.kind === ForKind.THREAD_BINDING && this._serializeThreads) {
      const varName = node.loopVar.name;
      const extent = this._exprToC(node.extent);
      this._emit(`for (int ${varName} = 0; ${varName} < ${extent}; ${varName}++) {`);
      this._indent++;
      this._visitNode(node.body);
      this._indent--;
      this._emit('}');
      return;
    }
    if (node.kind === ForKind.THREAD_BINDING) {
      const extent = node.extent.type === 'IntImmNode' ? node.extent.value : 0;
      const tag = node.threadTag;
      const maxExtent = this._getMaxBindingExtent(tag);
      if (extent > 0 && maxExtent > 0 && extent < maxExtent) {
        this._emit(`if (${tag} < ${extent}) {`);
        this._indent++;
        this._visitNode(node.body);
        this._indent--;
        this._emit('}');
      } else {
        this._visitNode(node.body);
      }
      if (this._needsBarriers) {
        this._emit('__syncthreads();');
      }
      return;
    }
    if (this._threadBindings.size === 0 && !this._didParallelReduce) {
      const red = this._matchFullReduction(node);
      if (red) { this._emitParallelReduction(node, red); return; }
    }
    const varName = node.loopVar.name;
    const extent = this._exprToC(node.extent);
    if (node.kind === ForKind.UNROLLED || node.kind === ForKind.VECTORIZED) this._emit(`#pragma unroll`);
    this._emit(`for (int ${varName} = 0; ${varName} < ${extent}; ${varName}++) {`);
    this._indent++;
    this._visitNode(node.body);
    this._indent--;
    this._emit('}');
  }

  _visitBlockNode(node: BlockNode): void {
    for (const bind of node.iterVars) {
      if (bind.iterVar && bind.binding) {
        this._emit(`const int ${bind.iterVar.name} = ${this._exprToC(bind.binding)};`);
      }
    }
    if (node.initBody) this._visitNode(node.initBody);
    this._visitNode(node.body);
  }

  _visitAllocateNode(node: AllocateNode): void {
    if (node.scope !== 'shared') {
      if (this._promotedBuffers.has(node.buffer.name)) return;
      if (this._scratchNames.has(node.buffer.name)) return;
      if (this._declaredLocals.has(node.buffer.name)) return;
      this._declaredLocals.add(node.buffer.name);
      const numel = node.buffer.numel();
      if (numel > 0) {
        const al = node.buffer.align16 ? '__align__(16) ' : '';
        this._emit(`${al}${cType(node.buffer.dtype)} ${node.buffer.name}[${numel}];`);
      } else {
        this._emit(`${cType(node.buffer.dtype)}* ${node.buffer.name} = (${cType(node.buffer.dtype)}*)alloca(${this._dynamicNumel(node.buffer)} * sizeof(${cType(node.buffer.dtype)}));`);
      }
    }
  }

  _visitIfStmt(node: IfThenElseNode): void {
    this._emit(`if (${this._exprToC(node.condition)}) {`);
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
    const dtype = node.variable.dtype || this._defaultDtype;
    this._emit(`${cType(dtype)} ${node.variable.name} = ${this._exprToC(node.value)};`);
    this._visitNode(node.body);
  }

  _visitWhileNode(node: WhileNode): void {
    this._visitNode(node.condBody);
    const condExpr = node.condVar instanceof Buffer ? `${node.condVar.name}[0]` : node.condVar.name;
    this._emit(`while (${condExpr}) {`);
    this._indent++;
    this._visitNode(node.loopBody);
    this._visitNode(node.condBody);
    this._indent--;
    this._emit('}');
  }

  _visitBufferStoreNode(node: BufferStoreNode): void {
    this._emit(`${node.buffer.name}[${this._flatIndex(node.buffer, node.indices)}] = ${this._exprToC(node.value)};`);
  }

  _visitVecCopyNode(node: VecCopyNode): void {
    const vt = `${cType(node.dstBuffer.dtype)}${node.width}`;
    const d = `${node.dstBuffer.name}[${this._flatIndex(node.dstBuffer, [node.dstIndex])}]`;
    const s = `${node.srcBuffer.name}[${this._flatIndex(node.srcBuffer, [node.srcIndex])}]`;
    this._emit(`*reinterpret_cast<${vt}*>(&${d}) = *reinterpret_cast<const ${vt}*>(&${s});`);
  }

  _visitLIRFlatStore(node: LIRFlatStoreNode): void {
    this._emit(`${node.buffer.name}[${this._exprToC(node.offsetExpr)}] = ${this._exprToC(node.value)};`);
  }

  _visitLIRBindings(node: LIRBindingsNode): void {
    for (const bind of node.bindings) {
      this._emit(`const int ${bind.name} = ${this._exprToC(bind.expr)};`);
    }
    this._visitNode(node.body);
  }

  _visitLIRAccumulator(node: LIRAccumulatorNode): void {
    const accVar = node.localName;
    const dtype = node.dtype || this._defaultDtype;
    this._emit(`${cType(dtype)} ${accVar} = ${this._exprToC(node.initLoad)};`);

    const varName = node.loopVar.name;
    const extent = this._exprToC(node.extent);
    this._emit(`for (int ${varName} = 0; ${varName} < ${extent}; ${varName}++) {`);
    this._indent++;
    const accBody = this._exprToC(node.body);
    const accOp = node.op || '+';
    let accRhs: string;
    if (accOp === 'max' || accOp === 'min') {
      if (isDtypeInt(node.dtype)) {
        const cmp = accOp === 'max' ? '>' : '<';
        accRhs = `((${accVar}) ${cmp} (${accBody}) ? (${accVar}) : (${accBody}))`;
      } else {
        accRhs = `${accOp === 'max' ? 'fmaxf' : 'fminf'}(${accVar}, ${accBody})`;
      }
    } else {
      accRhs = `(${accVar} ${accOp} ${accBody})`;
    }
    this._emit(`${accVar} = ${accRhs};`);
    this._indent--;
    this._emit('}');

    this._emit(`${node.flushStore.buffer.name}[${this._exprToC(node.flushStore.offsetExpr)}] = ${accVar};`);
  }

  _exprToC(node: IRStmtNode | null): string {
    if (!node) return '0';
    switch (node.type) {
      case 'IntImmNode': return String(node.value);
      case 'FloatImmNode': return this._emitFloatLiteral(node.value);
      case 'VariableNode': return node.name;
      case 'BufferLoadNode': return `${node.buffer.name}[${this._flatIndex(node.buffer, node.indices)}]`;
      case 'LIRFlatLoadNode': return `${node.buffer.name}[${this._exprToC(node.offsetExpr)}]`;
      case 'MathOpNode': {
        const a = this._exprToC(node.a);
        if (!node.b) return `(${node.op}${a})`;
        const b = this._exprToC(node.b);
        if (node.op === '//') return `(((${a}) - ((((${a}) % (${b})) + (${b})) % (${b}))) / (${b}))`;
        if (node.op === '%') return `((((${a}) % (${b})) + (${b})) % (${b}))`;
        if (node.op === 'tdiv') return `(${a} / ${b})`;
        if (node.op === 'tmod') return `(${a} % ${b})`;
        return `(${a} ${node.op} ${b})`;
      }
      case 'CompareNode': return `(${this._exprToC(node.a)} ${cCompareOp(node.direction)} ${this._exprToC(node.b)})`;
      case 'IfThenElseNode': return `(${this._exprToC(node.condition)} ? ${this._exprToC(node.thenBody)} : ${this._exprToC(node.elseBody)})`;
      case 'CastNode': return `((${cType(node.toDtype)})(${this._exprToC(node.expr)}))`;
      case 'CallExternNode': return this._emitExternCall(node);
      default: throw new Error(`CUDA codegen: unhandled expr node '${node.type}'`);
    }
  }

  _emitFloatLiteral(value: number): string {
    if (value === Infinity) return 'INFINITY';
    if (value === -Infinity) return '(-INFINITY)';
    const suffix = cLiteralSuffix(this._defaultDtype);
    const s = String(value);
    const mantissa = /[.eEnN]/.test(s) ? s : s + '.0';
    return `${mantissa}${suffix}`;
  }

  _emitExternCall(node: CallExternNode): string {
    const n = node.args.length;
    const args = new Array<string>(n);
    for (let i = 0; i < n; i++) args[i] = this._exprToC(node.args[i]);
    const joined = args.join(', ');
    const dtype = node.dtype || this._defaultDtype;
    if (node.externName === 'rsqrt') return `${cMathFunc('rsqrt', dtype)}(${joined})`;
    if (node.externName === 'sign') {
      const v = args[0];
      const zero = `0.0${cLiteralSuffix(dtype)}`;
      return `((${v} > ${zero}) - (${v} < ${zero}))`;
    }
    if (node.externName === 'min' || node.externName === 'max') {
      if (isDtypeInt(dtype)) {
        const op = node.externName === 'min' ? '<' : '>';
        return `((${args[0]}) ${op} (${args[1]}) ? (${args[0]}) : (${args[1]}))`;
      }
      return `${cMathFunc(node.externName, dtype)}(${joined})`;
    }
    const fn = cMathFunc(node.externName, dtype);
    return `${fn}(${joined})`;
  }

  _flatIndex(buffer: Buffer, indices: readonly IRStmtNode[]): string {
    return flattenRowMajorIndex(buffer, indices, (e) => this._exprToC(e), (b, i) => this._computeDynamicStride(b, i), false);
  }

  _serialize(reason: string, buffers: ReadonlySet<string>): void {
    this._serializeThreads = true;
    this._needsBarriers = false;
    this._launchDiagnosis = { reason, buffers: [...buffers] };
  }

  _promoteCrossThreadToShared(func: CodegenFunc, crossThread: ReadonlySet<string>): boolean {
    const storageNames = new Set<string>();
    for (const [, buf] of func.bufferMap) storageNames.add(buf.name);
    const refBuffers = new Map<string, Buffer>();
    referencedBuffers(func.body, refBuffers);

    const limit = this.target.sharedMemoryBytes || 49152;
    const alreadyShared = new Set(this._sharedBuffers.map(b => b.name));
    let used = this._sharedBuffers.reduce((sum, b) => sum + Math.max(b.sizeInBytes(), 0), 0);
    const decls: BufferDecl[] = [];
    for (const name of crossThread) {
      if (storageNames.has(name) || this._promotedBuffers.has(name) || alreadyShared.has(name)) continue;
      const buf = refBuffers.get(name);
      if (!buf) return false;
      const numel = buf.numel();
      const size = numel > 0 ? numel : this._estimateBufferSize(buf);
      if (size <= 0) return false;
      used += size * dtypeBytes(buf.dtype);
      if (used > limit) return false;
      decls.push({ name, dtype: buf.dtype, size });
    }
    for (const d of decls) {
      this._promotedBuffers.add(d.name);
      this._promotedBufferDecls.push(d);
    }
    return true;
  }

  _analyzeSharing(func: CodegenFunc): void {
    if (func.hasAttr(FuncAttr.TENSOR_INTRIN)) { this._needsBarriers = false; return; }
    if (func.getAttr(FuncAttr.GPU_REGISTER_BLOCKED)) {
      this._needsBarriers = false;
      return;
    }
    const profile = profileGpuAccesses(func, { sharedBuffers: this._sharedBuffers, threadBindings: this._threadBindings });
    const gridThreads = this._gridDim[0] * this._gridDim[1] * this._gridDim[2];
    const crossBlock = crossBlockRAWBuffers(profile);
    if (gridThreads > 1 && crossBlock.size > 0) {
      this._serialize(GpuRaceReason.CROSS_BLOCK_RAW, crossBlock);
      return;
    }
    if (this._threadBindings.size > 0) {
      const blockThreads = this._blockDim[0] * this._blockDim[1] * this._blockDim[2];
      const crossThread = threadSharedIntermediates(profile);
      if (blockThreads * gridThreads > 1 && crossThread.size > 0) {
        if (!storedUnderBlockBinding(profile, crossThread) && this._promoteCrossThreadToShared(func, crossThread)) {
          this._needsBarriers = true;
          return;
        }
        this._serialize(GpuRaceReason.THREAD_SHARED_INTERMEDIATE, crossThread);
        return;
      }
    }
    const multiExtent = hasMultiExtentBlockBinding(this._threadBindings);
    if (multiExtent.blockSpace) {
      this._serialize(GpuRaceReason.MULTI_EXTENT_BLOCK_BINDING, new Set());
      return;
    }
    if (!multiExtent.threadSpace) return;

    this._needsBarriers = true;

    const storageNames = new Set<string>();
    for (const [, buf] of func.bufferMap) storageNames.add(buf.name);

    walk(func.body, (node) => {
      if (node.type !== 'AllocateNode' || node.scope === 'shared' || storageNames.has(node.buffer.name)) return;
      const numel = node.buffer.numel();
      const size = numel > 0 ? numel : this._estimateBufferSize(node.buffer);
      if (size <= 0) return;
      this._promotedBuffers.add(node.buffer.name);
      this._promotedBufferDecls.push({ name: node.buffer.name, dtype: node.buffer.dtype, size });
    });

    const refBuffers = new Map<string, Buffer>();
    referencedBuffers(func.body, refBuffers);
    const allocatedNames = new Set<string>();
    allocatedBufferNames(func.body, allocatedNames);
    for (const [name, buf] of refBuffers) {
      if (storageNames.has(name) || allocatedNames.has(name)) continue;
      if (this._promotedBuffers.has(name)) continue;
      const numel = buf.numel();
      const size = numel > 0 ? numel : this._estimateBufferSize(buf);
      if (size > 0) {
        this._promotedBuffers.add(name);
        this._promotedBufferDecls.push({ name, dtype: buf.dtype, size });
      }
    }
  }

  _emitMissingLocalDecls(func: CodegenFunc): void {
    const storageNames = new Set<string>();
    for (const [, buf] of func.bufferMap) storageNames.add(buf.name);

    const allocatedNames = new Set<string>();
    allocatedBufferNames(func.body, allocatedNames);

    const refBuffers = new Map<string, Buffer>();
    referencedBuffers(func.body, refBuffers);

    for (const [name, buf] of refBuffers) {
      if (storageNames.has(name) || allocatedNames.has(name)) continue;
      if (this._promotedBuffers.has(name)) continue;
      if (this._scratchNames.has(name)) continue;
      if (this._declaredLocals.has(name)) continue;
      const numel = buf.numel();
      const size = numel > 0 ? numel : this._estimateBufferSize(buf);
      if (size > 0) {
        this._declaredLocals.add(name);
        this._emit(`${cType(buf.dtype)} ${name}[${size}];`);
      }
    }
  }

  _collectGlobalScratch(func: CodegenFunc): void {
    const serialized = this._serializeThreads;
    if (this._threadBindings.size > 0 && !serialized) return;
    const THRESHOLD = 32768;
    const storageNames = new Set<string>();
    for (const [, buf] of func.bufferMap) storageNames.add(buf.name);
    const candidates: ScratchCandidate[] = [];
    const seen = new Set<string>();
    const consider = (name: string, buf: Buffer) => {
      if (!buf || storageNames.has(name) || this._promotedBuffers.has(name) || this._scratchNames.has(name) || seen.has(name)) return;
      const numel = typeof buf.numel === 'function' ? buf.numel() : 0;
      const size = numel > 0 ? numel : this._estimateBufferSize(buf);
      if (size <= 0) return;
      seen.add(name);
      candidates.push({ name, dtype: buf.dtype, size, bytes: size * dtypeBytes(buf.dtype) });
    };
    walk(func.body, (n) => {
      if (n.type === 'AllocateNode' && n.scope !== 'shared') consider(n.buffer.name, n.buffer);
    });
    const refBuffers = new Map<string, Buffer>();
    referencedBuffers(func.body, refBuffers);
    for (const [name, buf] of refBuffers) consider(name, buf);

    const offload = (c: ScratchCandidate) => { this._scratchNames.add(c.name); this._globalScratch.push({ name: c.name, dtype: c.dtype, size: c.size }); };
    if (!serialized) {
      for (const c of candidates) if (c.size > THRESHOLD) offload(c);
      return;
    }
    let localBytes = candidates.reduce((s, c) => s + c.bytes, 0);
    if (localBytes <= LOCAL_MEMORY_BUDGET_BYTES) return;
    for (const c of [...candidates].sort((a, b) => b.bytes - a.bytes)) {
      if (localBytes <= LOCAL_MEMORY_BUDGET_BYTES) break;
      offload(c);
      localBytes -= c.bytes;
    }
  }

  _resolveShapeParam(buffer: Buffer, dimIdx: number): string {
    return resolveShapeParam(this._primFunc, buffer, dimIdx, (v) => v.name, 'CUDA', 'c');
  }
}
