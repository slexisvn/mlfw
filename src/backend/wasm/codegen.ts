import { ForKind } from '../../compiler/ir/tensor/nodes.js';
import { wasmType, wasmLoad, wasmStore, wasmBytes, isDtypeFloat, wasmSimdEntry, wasmVecOp } from '../../util/dtype_map.js';
import type { SimdInfo } from '../../util/dtype_map.js';
import { inferDtype } from '../../compiler/ir/lir/nodes.js';
import { HALF_WASM_CONSTANTS } from '../../tensor/utils/half.js';
import { irChildNodes } from '../../compiler/ir/ir_visitor.js';
import { resolveShapeParam, isZeroFillBody } from '../codegen_utils.js';

import type { Buffer } from '../../compiler/ir/tensor/buffer.js';
import type { BlockNode, BufferLoadNode, BufferStoreNode, CallExternNode, CastNode, CompareNode, ForNode, IfThenElseNode, MathOpNode, NodeSlots, SeqNode, TirNode, WhileNode } from '../../compiler/ir/tensor/nodes.js';
import type { IRStmtNode, LIRAccumulatorNode, LIRBindingsNode, LIRFlatStoreNode, LIRFunc } from '../../compiler/ir/lir/nodes.js';
import type { CodegenFunc } from '../codegen_utils.js';
import type { TargetFeatures } from '../target.js';

const _HALF_DTYPES = new Set(['f16', 'bf16']);

type SimdOpKey = keyof SimdInfo;
type InstrTable = Readonly<Record<string, string | undefined>>;
type VectorMode = {
  dtype: string;
  lanes: number;
  loopVar: string;
  simd: SimdInfo;
  laneVars: Set<string>;
  addrLocal?: string | null;
  vecLets?: Set<string>;
  _addrEmitted?: boolean;
  _addrKey?: string | null;
};
type WasmAcc = { local: string; bufName: string; indices?: TirNode[] };
type WasmAccPattern = { buf: Buffer; indices: TirNode[]; outerIndices: TirNode[] };
type BindingRef = { name: string; expr: IRStmtNode };

export type WasmParallelInfo = { extent: number; outputIndices: number[]; poolSafe: boolean };
export type WasmCodegenResult = {
  name: string;
  wat: string;
  memoryPages: number;
  bufferOffsets: Map<string, number>;
  imports: Map<string, string>;
  params: string[];
  parallel: WasmParallelInfo | null;
};

export class WasmCodegen {
  target: TargetFeatures;
  _lines: string[];
  _indent: number;
  _locals: Map<string, string>;
  _imports: Map<string, string>;
  _bufferOffsets: Map<string, number>;
  _totalMemBytes: number;
  _defaultDtype: string;
  _vectorMode: VectorMode | null;
  _vecTmpCounter: number;
  _loopVarStack: string[];
  declare _primFunc: CodegenFunc;
  declare _wasmAcc: WasmAcc | null;
  declare _waccCounter: number;
  declare _hasParallel: boolean;
  declare _parallelExtent: number;
  declare _intMinMaxEmitDepth: number;
  declare _intDivEmitDepth: number;

  constructor(target: TargetFeatures) {
    this.target = target;
    this._lines = [];
    this._indent = 0;
    this._locals = new Map();
    this._imports = new Map();
    this._bufferOffsets = new Map();
    this._totalMemBytes = 0;
    this._defaultDtype = 'f32';
    this._vectorMode = null;
    this._vecTmpCounter = 0;
    this._loopVarStack = [];
  }

  generate(func: CodegenFunc): WasmCodegenResult {
    this._lines = [];
    this._indent = 0;
    this._locals.clear();
    this._imports.clear();
    this._bufferOffsets.clear();
    this._totalMemBytes = 0;
    this._primFunc = func;
    this._wasmAcc = null;
    this._waccCounter = 0;
    this._hasParallel = false;
    this._parallelExtent = 0;
    this._vectorMode = null;
    this._vecTmpCounter = 0;

    const isLIR = func.type === 'LIRFunc';

    if (isLIR) {
      for (const [name, offset] of func.metadata.memoryLayout.bufferOffsets) {
        this._bufferOffsets.set(name, offset);
      }
      this._totalMemBytes = func.metadata.memoryLayout.totalBytes;
      for (const [name, info] of func.metadata.externCalls) {
        const sig = this._mathImportSig(name, info.argCount);
        this._imports.set(name, sig);
      }
      for (const [name, dtype] of func.metadata.locals) {
        this._ensureLocal(name, wasmType(dtype));
      }
    } else {
      this._layoutBuffers(func);
      this._scanMathImports(func.body);
    }

    this._scanParallel(func.body);

    const paramNames: string[] = [];
    for (const [, buf] of func.bufferMap) {
      paramNames.push(buf.name);
    }
    for (const sp of func.shapeParams) {
      paramNames.push(sp.name);
    }

    this._emit('(module');
    this._indent++;

    const memPages = Math.max(1, Math.ceil(this._totalMemBytes / 65536));
    const maxPages = Math.max(256, memPages);
    this._emit(`(memory (export "memory") ${memPages} ${maxPages})`);

    for (const [name, sig] of this._imports) {
      this._emit(`(import "math" "${name}" (func $math_${name} ${sig}))`);
    }

    const paramDecls: string[] = [];
    for (const [, buf] of func.bufferMap) {
      paramDecls.push('(param i32)');
    }
    const shapeParamEntries: string[] = [];
    for (const sp of func.shapeParams) {
      paramDecls.push('(param i32)');
      this._ensureLocal(sp.name, 'i32');
      shapeParamEntries.push(sp.name);
    }
    if (this._hasParallel) {
      paramDecls.push('(param i32)');
      paramDecls.push('(param i32)');
      this._ensureLocal('_par_start', 'i32');
      this._ensureLocal('_par_end', 'i32');
      paramNames.push('_par_start');
      paramNames.push('_par_end');
    }
    this._emit('(func (export "' + func.name + '") ' + paramDecls.join(' '));
    this._indent++;

    if (!isLIR) {
      this._prescanLocals(func.body);
      this._vecTmpCounter = 0;
    }

    if (this.target.supportsSimd()) {
      this._prescanVecLocalsAll(func.body);
      this._vecTmpCounter = 0;
    }

    this._fixLetStmtLocals(func.body);

    this._prescanIntMinMax(func.body);

    this._ensureHalfScratch(func);

    const localDecls: string[] = [];
    for (const [name, type] of this._locals) {
      localDecls.push('(local $' + name + ' ' + type + ')');
    }
    if (localDecls.length > 0) this._emit(localDecls.join(' '));

    const bufCount = func.bufferMap.size;
    for (let i = 0; i < shapeParamEntries.length; i++) {
      this._emit('(local.get ' + (bufCount + i) + ')');
      this._emit('local.set $' + shapeParamEntries[i]);
    }
    if (this._hasParallel) {
      const parIdx = bufCount + shapeParamEntries.length;
      this._emit('(local.get ' + parIdx + ')');
      this._emit('local.set $_par_start');
      this._emit('(local.get ' + (parIdx + 1) + ')');
      this._emit('local.set $_par_end');
    }

    this._visitNode(func.body);

    this._indent--;
    this._emit(')');
    this._indent--;
    this._emit(')');

    return {
      name: func.name,
      wat: this._lines.join('\n'),
      memoryPages: memPages,
      bufferOffsets: new Map(this._bufferOffsets),
      imports: this._imports,
      params: paramNames,
      parallel: this._hasParallel
        ? { extent: this._parallelExtent, outputIndices: this._findOutputIndices(func), poolSafe: this._isParallelSafe(func) }
        : null,
    };
  }

  _isParallelSafe(func: CodegenFunc): boolean {
    const parallels: ForNode[] = [];
    const findStack: IRStmtNode[] = [func.body];
    while (findStack.length > 0) {
      const node = findStack.pop();
      if (!node) continue;
      if (node.type === 'ForNode' && node.kind === ForKind.PARALLEL) parallels.push(node);
      const slots = node as unknown as NodeSlots;
      if (slots.body) findStack.push(slots.body as TirNode);
      if (slots.stmts) for (const s of slots.stmts as TirNode[]) findStack.push(s);
      if (slots.thenBody) findStack.push(slots.thenBody as TirNode);
      if (slots.elseBody) findStack.push(slots.elseBody as TirNode);
      if (slots.loopBody) findStack.push(slots.loopBody as TirNode);
    }
    if (parallels.length !== 1) return false;
    const parallelNode = parallels[0];

    const topStmts = func.body && (func.body as SeqNode).stmts ? (func.body as SeqNode).stmts : [func.body];
    if (!topStmts.includes(parallelNode)) return false;

    const collectStores = (root: IRStmtNode, into: Set<IRStmtNode>) => {
      const stack: IRStmtNode[] = [root];
      while (stack.length > 0) {
        const node = stack.pop();
        if (!node) continue;
        if (node.type === 'BufferStoreNode' || node.type === 'LIRFlatStoreNode') into.add(node);
        if (node.type === 'LIRAccumulatorNode' && node.flushStore) into.add(node.flushStore);
        const slots = node as unknown as NodeSlots;
        if (slots.body) stack.push(slots.body as TirNode);
        if (slots.stmts) for (const s of slots.stmts as TirNode[]) stack.push(s);
        if (slots.thenBody) stack.push(slots.thenBody as TirNode);
        if (slots.elseBody) stack.push(slots.elseBody as TirNode);
        if (slots.loopBody) stack.push(slots.loopBody as TirNode);
      }
    };

    const allStores = new Set<IRStmtNode>();
    const innerStores = new Set<IRStmtNode>();
    collectStores(func.body, allStores);
    collectStores(parallelNode.body, innerStores);
    for (const st of allStores) {
      if (!innerStores.has(st)) return false;
    }
    return true;
  }

  _ensureLocal(name: string, type: string): void {
    if (!this._locals.has(name)) {
      this._locals.set(name, type);
    }
  }

  _emit(line: string): void {
    this._lines.push('  '.repeat(this._indent) + line);
  }

  _emitLoadOp(dtype: string): void {
    if (_HALF_DTYPES.has(dtype)) { this._emitHalfDecode(dtype); return; }
    this._emit(wasmLoad(dtype));
  }

  _emitStoreOp(dtype: string): void {
    if (_HALF_DTYPES.has(dtype)) { this._emitHalfEncode(dtype); return; }
    this._emit(wasmStore(dtype));
  }

  _emitHalfDecode(dtype: string): void {
    this._emit('i32.load16_u');
    this._emit('local.set $_half_i');
    if (dtype === 'bf16') {
      this._emit('(local.get $_half_i)');
      this._emit('(i32.const 16)');
      this._emit('i32.shl');
      this._emit('f32.reinterpret_i32');
      return;
    }
    this._emit('(local.get $_half_i)');
    this._emit('(i32.const 32767)');
    this._emit('i32.and');
    this._emit('(i32.const 13)');
    this._emit('i32.shl');
    this._emit('f32.reinterpret_i32');
    this._emit('(f32.const ' + HALF_WASM_CONSTANTS.F16_MAGIC_MUL + ')');
    this._emit('f32.mul');
    this._emit('local.set $_half_f');
    this._emit('(local.get $_half_f)');
    this._emit('i32.reinterpret_f32');
    this._emit('local.set $_half_i2');
    this._emit('(local.get $_half_i2)');
    this._emit('(i32.const 2139095040)');
    this._emit('i32.or');
    this._emit('(local.get $_half_i2)');
    this._emit('(local.get $_half_f)');
    this._emit('(f32.const 65536)');
    this._emit('f32.ge');
    this._emit('select');
    this._emit('(local.get $_half_i)');
    this._emit('(i32.const 32768)');
    this._emit('i32.and');
    this._emit('(i32.const 16)');
    this._emit('i32.shl');
    this._emit('i32.or');
    this._emit('f32.reinterpret_i32');
  }

  _emitHalfEncode(dtype: string): void {
    this._emit('local.set $_half_f');
    this._emit('(local.get $_half_f)');
    this._emit('i32.reinterpret_f32');
    this._emit('local.set $_half_i');
    if (dtype === 'bf16') {
      this._emit('(local.get $_half_i)');
      this._emit('(i32.const 16)');
      this._emit('i32.shr_u');
      this._emit('(i32.const 1)');
      this._emit('i32.and');
      this._emit('(i32.const 32767)');
      this._emit('i32.add');
      this._emit('(local.get $_half_i)');
      this._emit('i32.add');
      this._emit('(i32.const 16)');
      this._emit('i32.shr_u');
      this._emit('i32.store16');
      return;
    }
    this._emit('(local.get $_half_i)');
    this._emit('(i32.const 2147483647)');
    this._emit('i32.and');
    this._emit('local.set $_half_i2');
    this._emit('(i32.const 32256)');
    this._emit('(i32.const 31744)');
    this._emit('(local.get $_half_i2)');
    this._emit('(i32.const 2139095040)');
    this._emit('i32.gt_s');
    this._emit('select');
    this._emit('(local.get $_half_i2)');
    this._emit('f32.reinterpret_i32');
    this._emit('(f32.const 0.5)');
    this._emit('f32.add');
    this._emit('i32.reinterpret_f32');
    this._emit('(i32.const 1056964608)');
    this._emit('i32.sub');
    this._emit('(local.get $_half_i2)');
    this._emit('(i32.const ' + HALF_WASM_CONSTANTS.F16_ADD_BIAS + ')');
    this._emit('i32.add');
    this._emit('(local.get $_half_i2)');
    this._emit('(i32.const 13)');
    this._emit('i32.shr_u');
    this._emit('(i32.const 1)');
    this._emit('i32.and');
    this._emit('i32.add');
    this._emit('(i32.const 13)');
    this._emit('i32.shr_u');
    this._emit('(local.get $_half_i2)');
    this._emit('(i32.const 947912704)');
    this._emit('i32.lt_s');
    this._emit('select');
    this._emit('(local.get $_half_i2)');
    this._emit('(i32.const 1199570944)');
    this._emit('i32.ge_s');
    this._emit('select');
    this._emit('(local.get $_half_i)');
    this._emit('(i32.const 16)');
    this._emit('i32.shr_u');
    this._emit('(i32.const 32768)');
    this._emit('i32.and');
    this._emit('i32.or');
    this._emit('i32.store16');
  }

  _ensureHalfScratch(func: CodegenFunc): void {
    let needs = false;
    for (const [, buf] of func.bufferMap) {
      if (_HALF_DTYPES.has(buf.dtype)) { needs = true; break; }
    }
    if (!needs && (func as LIRFunc).metadata && (func as LIRFunc).metadata.locals) {
      for (const [, dtype] of (func as LIRFunc).metadata.locals) {
        if (_HALF_DTYPES.has(dtype)) { needs = true; break; }
      }
    }
    if (!needs) needs = this._treeHasHalf(func.body);
    if (needs) {
      this._ensureLocal('_half_f', 'f32');
      this._ensureLocal('_half_i', 'i32');
      this._ensureLocal('_half_i2', 'i32');
    }
  }

  _treeHasHalf(root: IRStmtNode): boolean {
    const stack: IRStmtNode[] = [root];
    while (stack.length > 0) {
      const n = stack.pop();
      if (!n || typeof n !== 'object') continue;
      if ((n.type === 'BufferLoadNode' || n.type === 'BufferStoreNode') && n.buffer && _HALF_DTYPES.has(n.buffer.dtype)) return true;
      if ((n.type === 'LIRFlatLoadNode' || n.type === 'LIRFlatStoreNode') && _HALF_DTYPES.has(n.dtype)) return true;
      const slots = n as unknown as NodeSlots;
      if (slots.body) stack.push(slots.body as TirNode);
      if (slots.value && typeof slots.value === 'object') stack.push(slots.value as TirNode);
      if (slots.stmts) for (const s of slots.stmts as TirNode[]) stack.push(s);
      if (slots.a) stack.push(slots.a as TirNode);
      if (slots.b) stack.push(slots.b as TirNode);
      if (slots.expr) stack.push(slots.expr as TirNode);
      if (slots.thenBody) stack.push(slots.thenBody as TirNode);
      if (slots.elseBody) stack.push(slots.elseBody as TirNode);
      if (slots.initBody) stack.push(slots.initBody as TirNode);
      if (slots.condition) stack.push(slots.condition as TirNode);
      if (slots.offsetExpr) stack.push(slots.offsetExpr as TirNode);
      if (slots.args) for (const a of slots.args as TirNode[]) stack.push(a);
      if (slots.indices) for (const idx of slots.indices as TirNode[]) stack.push(idx);
    }
    return false;
  }

  _visitNode(startNode: IRStmtNode): void {
    let node: IRStmtNode = startNode;
    while (node) {
      switch (node.type) {
        case 'SeqNode':
          for (let i = 0; i < node.stmts.length - 1; i++) this._visitNode(node.stmts[i]);
          node = node.stmts[node.stmts.length - 1];
          continue;
        case 'AllocateNode':
          node = node.body;
          continue;
        case 'LetStmtNode': {
          const vm = this._vectorMode;
          if (vm && vm.simd && this._dependsOnVecVar(node.value)) {
            const vecName = node.variable.name + '_vlet';
            this._ensureLocal(vecName, 'v128');
            if (!vm.vecLets) vm.vecLets = new Set();
            vm.vecLets.add(node.variable.name);
            this._emitVecExpr(node.value);
            this._emit('local.set $' + vecName);
            node = node.body;
            continue;
          }
          const varDtype = inferDtype(node.value) || node.variable.dtype || this._defaultDtype;
          this._locals.set(node.variable.name, wasmType(varDtype));
          this._emitCoercedTo(node.value, this._numPrefix(varDtype));
          this._emit(`local.set $${node.variable.name}`);
          node = node.body;
          continue;
        }
        case 'ForNode': this._visitFor(node); return;
        case 'BlockNode': this._visitBlock(node); return;
        case 'IfThenElseNode': this._visitIf(node); return;
        case 'BufferStoreNode':
          if (this._vectorMode) { this._emitVecStore(node); return; }
          this._visitStore(node); return;
        case 'LIRFlatStoreNode':
          if (this._vectorMode) { this._emitVecFlatStore(node); return; }
          this._visitLIRFlatStore(node); return;
        case 'LIRBindingsNode': this._visitLIRBindings(node); return;
        case 'LIRAccumulatorNode': this._visitLIRAccumulator(node); return;
        case 'WhileNode': this._visitWhile(node); return;
        case 'EvaluateNode': return;
        case 'SyncThreadsNode': return;
        default: throw new Error(`WASM codegen: unhandled statement node '${node.type}'`);
      }
    }
  }

  _findOutputIndices(func: CodegenFunc): number[] {
    const storeNames = new Set<string>();
    const stack: IRStmtNode[] = [func.body];
    while (stack.length > 0) {
      const node = stack.pop();
      if (!node) continue;
      if ((node.type === 'BufferStoreNode' || node.type === 'LIRFlatStoreNode') && node.buffer) {
        storeNames.add(node.buffer.name);
      }
      if (node.type === 'LIRAccumulatorNode' && node.flushStore && node.flushStore.buffer) {
        storeNames.add(node.flushStore.buffer.name);
      }
      const slots = node as unknown as NodeSlots;
      if (slots.body) stack.push(slots.body as TirNode);
      if (slots.stmts) for (const s of slots.stmts as TirNode[]) stack.push(s);
      if (slots.thenBody) stack.push(slots.thenBody as TirNode);
      if (slots.elseBody) stack.push(slots.elseBody as TirNode);
      if (slots.loopBody) stack.push(slots.loopBody as TirNode);
    }
    const indices: number[] = [];
    let idx = 0;
    for (const [, buf] of func.bufferMap) {
      if (storeNames.has(buf.name)) indices.push(idx);
      idx++;
    }
    return indices;
  }

  _scanParallel(root: IRStmtNode): void {
    const stack: IRStmtNode[] = [root];
    while (stack.length > 0) {
      const node = stack.pop();
      if (!node) continue;
      if (node.type === 'ForNode' && node.kind === ForKind.PARALLEL) {
        this._hasParallel = true;
        this._parallelExtent = node.extent && node.extent.type === 'IntImmNode' ? node.extent.value : 0;
        return;
      }
      const slots = node as unknown as NodeSlots;
      if (slots.body) stack.push(slots.body as TirNode);
      if (slots.stmts) for (const s of slots.stmts as TirNode[]) stack.push(s);
      if (slots.thenBody) stack.push(slots.thenBody as TirNode);
      if (slots.elseBody) stack.push(slots.elseBody as TirNode);
    }
  }

  _visitFor(node: ForNode): void {
    const varName = node.loopVar.name;
    const extent = this._constExtent(node.extent);

    if (node.kind === ForKind.PARALLEL) {
      if (extent !== null && this._parallelExtent && extent !== this._parallelExtent) {
        this._emitForLoop(varName, node.extent, node.body);
        return;
      }
      this._emit('(local.get $_par_start)');
      this._emit('local.set $' + varName);
      this._emit('(block $break_' + varName);
      this._indent++;
      this._emit('(loop $loop_' + varName);
      this._indent++;
      this._emit('(local.get $' + varName + ')');
      this._emit('(local.get $_par_end)');
      this._emit('i32.ge_s');
      this._emit('br_if $break_' + varName);
      this._visitNode(node.body);
      this._emit('(local.get $' + varName + ')');
      this._emit('(i32.const 1)');
      this._emit('i32.add');
      this._emit('local.set $' + varName);
      this._emit('br $loop_' + varName);
      this._indent--;
      this._emit(')');
      this._indent--;
      this._emit(')');
      return;
    }

    if (node.kind === ForKind.VECTORIZED && this.target.supportsSimd() && extent !== null) {
      this._visitVectorizedFor(node);
      return;
    }

    if ((node.kind === ForKind.UNROLLED || node.kind === ForKind.VECTORIZED) && extent !== null && extent <= 32 && !this._isZeroFillBody(node.body)) {
      for (let i = 0; i < extent; i++) {
        this._emit('(i32.const ' + i + ')');
        this._emit('local.set $' + varName);
        this._visitNode(node.body);
      }
      return;
    }

    const accInfo = this._detectWasmAcc(node);
    if (accInfo) {
      const accLocal = '_wacc_' + (this._waccCounter = (this._waccCounter || 0) + 1);
      this._ensureLocal(accLocal, wasmType(accInfo.buf.dtype));
      this._emitAddr(accInfo.buf, accInfo.outerIndices);
      this._emitLoadOp(accInfo.buf.dtype);
      this._emit('local.set $' + accLocal);
      this._wasmAcc = { local: accLocal, bufName: accInfo.buf.name, indices: accInfo.indices };
      this._emitForLoop(varName, node.extent, node.body);
      this._emitAddr(accInfo.buf, accInfo.outerIndices);
      this._emit('(local.get $' + accLocal + ')');
      this._emitStoreOp(accInfo.buf.dtype);
      this._wasmAcc = null;
      return;
    }

    this._emitForLoop(varName, node.extent, node.body);
  }

  _visitLIRFlatStore(node: LIRFlatStoreNode): void {
    if (this._wasmAcc && node.buffer.name === this._wasmAcc.bufName) {
      this._emitCoercedTo(node.value, this._numPrefix(node.dtype));
      this._emit('local.set $' + this._wasmAcc.local);
      return;
    }
    this._emitFlatAddr(node.buffer, node.offsetExpr);
    this._emitCoercedTo(node.value, this._numPrefix(node.dtype));
    this._emitStoreOp(node.dtype);
  }

  _visitLIRBindings(node: LIRBindingsNode): void {
    for (const bind of node.bindings) {
      this._emitExpr(bind.expr);
      this._emit(`local.set $${bind.name}`);
    }
    this._visitNode(node.body);
  }

  _vecAccumOperandsUnitStride(node: LIRAccumulatorNode): boolean {
    const vecVar = node.loopVar && node.loopVar.name;
    if (!vecVar) return false;
    const usesVar = (expr: IRStmtNode | null) => {
      const st: (IRStmtNode | null)[] = [expr];
      while (st.length > 0) {
        const m = st.pop();
        if (!m || typeof m !== 'object') continue;
        if (m.type === 'VariableNode' && m.name === vecVar) return true;
        const slots = m as unknown as NodeSlots;
        if (slots.a) st.push(slots.a as TirNode);
        if (slots.b) st.push(slots.b as TirNode);
        if (slots.expr) st.push(slots.expr as TirNode);
        if (slots.args) for (const x of slots.args as TirNode[]) st.push(x);
        if (slots.indices) for (const x of slots.indices as TirNode[]) st.push(x);
        if (slots.offsetExpr) st.push(slots.offsetExpr as TirNode);
      }
      return false;
    };
    const stridedMul = (expr: IRStmtNode) => {
      const st: IRStmtNode[] = [expr];
      while (st.length > 0) {
        const m = st.pop();
        if (!m || typeof m !== 'object') continue;
        if (m.type === 'MathOpNode' && m.op === '*' && (usesVar(m.a) || usesVar(m.b))) return true;
        const slots = m as unknown as NodeSlots;
        if (slots.a) st.push(slots.a as TirNode);
        if (slots.b) st.push(slots.b as TirNode);
        if (slots.expr) st.push(slots.expr as TirNode);
        if (slots.args) for (const x of slots.args as TirNode[]) st.push(x);
      }
      return false;
    };
    const stack: IRStmtNode[] = [node.body];
    while (stack.length > 0) {
      const n = stack.pop();
      if (!n || typeof n !== 'object') continue;
      if (n.type === 'BufferLoadNode' && Array.isArray(n.indices)) {
        for (let i = 0; i < n.indices.length - 1; i++) {
          if (usesVar(n.indices[i])) return false;
        }
      }
      if (n.type === 'LIRFlatLoadNode' && n.offsetExpr && stridedMul(n.offsetExpr)) return false;
      const slots = n as unknown as NodeSlots;
      if (slots.a) stack.push(slots.a as TirNode);
      if (slots.b) stack.push(slots.b as TirNode);
      if (slots.expr) stack.push(slots.expr as TirNode);
      if (slots.args) for (const x of slots.args as TirNode[]) stack.push(x);
      if (slots.body) stack.push(slots.body as TirNode);
    }
    return true;
  }

  _accumInstr(op: string, dtype: string): string {
    if (op === '*') return 'mul';
    if (op === 'max' || op === 'min') {
      if (isDtypeFloat(dtype)) return op;
      throw new Error(`wasm accumulator: integer ${op} reduction not supported (dtype ${dtype})`);
    }
    return 'add';
  }

  _visitLIRAccumulator(node: LIRAccumulatorNode): void {
    const accLocal = node.localName;
    const dtype = node.dtype;
    this._ensureLocal(accLocal, wasmType(dtype));

    const extent = this._constExtent(node.extent);
    const accOp = node.op || '+';
    const simdEntry = accOp === '+' && extent !== null && node.loopKind === ForKind.VECTORIZED
      && this.target.supportsSimd() ? wasmSimdEntry(dtype) : null;
    const lanes = simdEntry ? this.target.vectorWidth : 0;

    if (simdEntry && (extent as number) >= lanes && this._vecAccumOperandsUnitStride(node)) {
      this._visitVecAccumulator(node, simdEntry, lanes, extent as number);
      return;
    }

    this._emitExpr(node.initLoad);
    this._emit('local.set $' + accLocal);

    const prevAcc = this._wasmAcc;
    this._wasmAcc = {
      local: accLocal,
      bufName: node.flushStore.buffer.name,
    };

    const varName = node.loopVar.name;
    this._emit('(i32.const 0)');
    this._emit('local.set $' + varName);
    this._emit('(block $break_' + varName);
    this._indent++;
    this._emit('(loop $loop_' + varName);
    this._indent++;
    this._emit('(local.get $' + varName + ')');
    this._emitExpr(node.extent);
    this._emit('i32.ge_s');
    this._emit('br_if $break_' + varName);

    this._emit('(local.get $' + accLocal + ')');
    this._emitCoercedTo(node.body, this._numPrefix(dtype));
    this._emit(this._numPrefix(dtype) + '.' + this._accumInstr(accOp, dtype));
    this._emit('local.set $' + accLocal);

    this._emit('(local.get $' + varName + ')');
    this._emit('(i32.const 1)');
    this._emit('i32.add');
    this._emit('local.set $' + varName);
    this._emit('br $loop_' + varName);
    this._indent--;
    this._emit(')');
    this._indent--;
    this._emit(')');

    this._emitFlatAddr(node.flushStore.buffer, node.flushStore.offsetExpr);
    this._emit('(local.get $' + accLocal + ')');
    this._emitStoreOp(node.flushStore.dtype);

    this._wasmAcc = prevAcc;
  }

  _visitVecAccumulator(node: LIRAccumulatorNode, simdEntry: SimdInfo, lanes: number, extent: number): void {
    const accLocal = node.localName;
    const dtype = node.dtype;
    const varName = node.loopVar.name;
    const isFloat = isDtypeFloat(dtype);
    const scalarAdd = isFloat ? 'f32.add' : 'i32.add';
    const vecAdd = wasmVecOp(dtype, 'add');
    const mainExtent = Math.floor(extent / lanes) * lanes;
    const tailStart = mainExtent;

    const vaccLocal = accLocal + '_vec';
    this._ensureLocal(vaccLocal, 'v128');

    this._emitExpr(node.initLoad);
    this._emit('local.set $' + accLocal);

    this._emit(isFloat ? '(f32.const 0)' : '(i32.const 0)');
    this._emit(simdEntry.splat);
    this._emit('local.set $' + vaccLocal);

    const prevAcc = this._wasmAcc;
    this._wasmAcc = {
      local: accLocal,
      bufName: node.flushStore.buffer.name,
    };

    this._vectorMode = { dtype, lanes, loopVar: varName, simd: simdEntry, laneVars: this._computeLaneVars(node) };
    this._emit('(i32.const 0)');
    this._emit('local.set $' + varName);
    this._emit('(block $vbreak_' + varName);
    this._indent++;
    this._emit('(loop $vloop_' + varName);
    this._indent++;
    this._emit('(local.get $' + varName + ')');
    this._emit('(i32.const ' + mainExtent + ')');
    this._emit('i32.ge_s');
    this._emit('br_if $vbreak_' + varName);

    this._emit('(local.get $' + vaccLocal + ')');
    this._emitVecExpr(node.body);
    this._emit(vecAdd as string);
    this._emit('local.set $' + vaccLocal);

    this._emit('(local.get $' + varName + ')');
    this._emit('(i32.const ' + lanes + ')');
    this._emit('i32.add');
    this._emit('local.set $' + varName);
    this._emit('br $vloop_' + varName);
    this._indent--;
    this._emit(')');
    this._indent--;
    this._emit(')');
    this._vectorMode = null;

    this._emit('(local.get $' + accLocal + ')');
    for (let l = 0; l < lanes; l++) {
      this._emit('(local.get $' + vaccLocal + ')');
      this._emit(simdEntry.extractLane + ' ' + l);
      this._emit(scalarAdd);
    }
    this._emit('local.set $' + accLocal);

    if (tailStart < extent) {
      this._emit('(i32.const ' + tailStart + ')');
      this._emit('local.set $' + varName);
      this._emit('(block $tbreak_' + varName);
      this._indent++;
      this._emit('(loop $tloop_' + varName);
      this._indent++;
      this._emit('(local.get $' + varName + ')');
      this._emit('(i32.const ' + extent + ')');
      this._emit('i32.ge_s');
      this._emit('br_if $tbreak_' + varName);

      this._emit('(local.get $' + accLocal + ')');
      this._emitCoerced(node.body, isFloat);
      this._emit(scalarAdd);
      this._emit('local.set $' + accLocal);

      this._emit('(local.get $' + varName + ')');
      this._emit('(i32.const 1)');
      this._emit('i32.add');
      this._emit('local.set $' + varName);
      this._emit('br $tloop_' + varName);
      this._indent--;
      this._emit(')');
      this._indent--;
      this._emit(')');
    }

    this._emitFlatAddr(node.flushStore.buffer, node.flushStore.offsetExpr);
    this._emit('(local.get $' + accLocal + ')');
    this._emitStoreOp(node.flushStore.dtype);

    this._wasmAcc = prevAcc;
  }

  _emitFlatAddr(buffer: Buffer, offsetExpr: IRStmtNode | null): void {
    const baseOffset = this._bufferOffsets.get(buffer.name) || 0;
    const bytes = wasmBytes(buffer.dtype || 'f32');

    if (!offsetExpr || (offsetExpr.type === 'IntImmNode' && offsetExpr.value === 0)) {
      this._emit(`(i32.const ${baseOffset})`);
      return;
    }

    const vm = this._vectorMode;
    const offKey = bytes + '#' + this._exprKey(offsetExpr);
    if (vm && vm.addrLocal && this._exprKey(offsetExpr) !== '?') {
      if (!vm._addrEmitted || vm._addrKey !== offKey) {
        this._emitExpr(offsetExpr);
        this._emit(`(i32.const ${bytes})`);
        this._emit('i32.mul');
        this._emit('local.set $' + vm.addrLocal);
        vm._addrEmitted = true;
        vm._addrKey = offKey;
      }
      this._emit('(local.get $' + vm.addrLocal + ')');
      if (baseOffset > 0) {
        this._emit(`(i32.const ${baseOffset})`);
        this._emit('i32.add');
      }
      return;
    }

    this._emitExpr(offsetExpr);
    this._emit(`(i32.const ${bytes})`);
    this._emit('i32.mul');
    if (baseOffset > 0) {
      this._emit(`(i32.const ${baseOffset})`);
      this._emit('i32.add');
    }
  }

  _emitForLoop(varName: string, extent: IRStmtNode, body: IRStmtNode): void {
    this._emit('(i32.const 0)');
    this._emit('local.set $' + varName);
    this._emit('(block $break_' + varName);
    this._indent++;
    this._emit('(loop $loop_' + varName);
    this._indent++;
    this._emit('(local.get $' + varName + ')');
    this._emitExpr(extent);
    this._emit('i32.ge_s');
    this._emit('br_if $break_' + varName);
    this._loopVarStack.push(varName);
    this._visitNode(body);
    this._loopVarStack.pop();
    this._emit('(local.get $' + varName + ')');
    this._emit('(i32.const 1)');
    this._emit('i32.add');
    this._emit('local.set $' + varName);
    this._emit('br $loop_' + varName);
    this._indent--;
    this._emit(')');
    this._indent--;
    this._emit(')');
  }

  _visitBlock(node: BlockNode): void {
    for (const bind of node.iterVars) {
      if (bind.iterVar && bind.binding) {
        this._emitExpr(bind.binding);
        this._emit(`local.set $${bind.iterVar.name}`);
        const vm = this._vectorMode;
        if (vm && vm.laneVars && this._dependsOnVecVar(bind.binding)) {
          vm.laneVars.add(bind.iterVar.name);
        }
      }
    }
    if (node.initBody) {
      const reductionVar = this._loopVarStack.length > 0 ? this._loopVarStack[this._loopVarStack.length - 1] : null;
      if (reductionVar) {
        this._emit('(local.get $' + reductionVar + ')');
        this._emit('i32.eqz');
        this._emit('(if');
        this._indent++;
        this._emit('(then');
        this._indent++;
        this._visitNode(node.initBody);
        this._indent--;
        this._emit(')');
        this._indent--;
        this._emit(')');
      } else {
        this._visitNode(node.initBody);
      }
    }
    this._visitNode(node.body);
  }

  _visitStore(node: BufferStoreNode): void {
    if (this._wasmAcc && this._isAccTarget(node.buffer, node.indices)) {
      this._emitCoercedTo(node.value, this._numPrefix(node.buffer.dtype));
      this._emit('local.set $' + this._wasmAcc.local);
      return;
    }
    this._emitAddr(node.buffer, node.indices);
    this._emitCoercedTo(node.value, this._numPrefix(node.buffer.dtype));
    this._emitStoreOp(node.buffer.dtype);
  }

  _visitIf(node: IfThenElseNode): void {
    this._emitExpr(node.condition);
    this._emit('(if');
    this._indent++;
    this._emit('(then');
    this._indent++;
    this._visitNode(node.thenBody);
    this._indent--;
    this._emit(')');
    if (node.elseBody) {
      this._emit('(else');
      this._indent++;
      this._visitNode(node.elseBody);
      this._indent--;
      this._emit(')');
    }
    this._indent--;
    this._emit(')');
  }

  _visitWhile(node: WhileNode): void {
    this._visitNode(node.condBody);
    this._emit('(block $wbreak');
    this._indent++;
    this._emit('(loop $wloop');
    this._indent++;
    this._emitAddr(node.condVar as Buffer, []);
    this._emitLoadOp(node.condVar.dtype);
    this._emit('i32.eqz');
    this._emit('br_if $wbreak');
    this._visitNode(node.loopBody);
    this._visitNode(node.condBody);
    this._emit('br $wloop');
    this._indent--;
    this._emit(')');
    this._indent--;
    this._emit(')');
  }

  _emitAddr(buffer: Buffer, indices: readonly IRStmtNode[]): void {
    const baseOffset = this._bufferOffsets.get(buffer.name) || 0;
    const bytes = wasmBytes(buffer.dtype);

    if (indices.length === 0) {
      this._emit(`(i32.const ${baseOffset})`);
      return;
    }

    const vm = this._vectorMode;
    if (vm && vm.addrLocal && indices.length > 0) {
      const offKey = bytes + '#' + indices.map(ix => this._exprKey(ix)).join(',');
      if (!vm._addrEmitted || vm._addrKey !== offKey) {
        this._emitFlatIndex(buffer, indices);
        this._emit(`(i32.const ${bytes})`);
        this._emit('i32.mul');
        this._emit('local.set $' + vm.addrLocal);
        vm._addrEmitted = true;
        vm._addrKey = offKey;
      }
      this._emit('(local.get $' + vm.addrLocal + ')');
      if (baseOffset > 0) {
        this._emit(`(i32.const ${baseOffset})`);
        this._emit('i32.add');
      }
      return;
    }

    this._emitFlatIndex(buffer, indices);
    this._emit(`(i32.const ${bytes})`);
    this._emit('i32.mul');
    if (baseOffset > 0) {
      this._emit(`(i32.const ${baseOffset})`);
      this._emit('i32.add');
    }
  }

  _emitFlatIndex(buffer: Buffer, indices: readonly IRStmtNode[]): void {
    if (indices.length === 1) {
      this._emitExpr(indices[0]);
      return;
    }
    let first = true;
    for (let i = 0; i < indices.length; i++) {
      this._emitExpr(indices[i]);
      const stride = buffer.strides[i];
      if (typeof stride === 'number' && stride >= 0) {
        if (stride !== 1) {
          this._emit(`(i32.const ${stride})`);
          this._emit('i32.mul');
        }
      } else {
        this._emitDynamicStride(buffer, i);
        this._emit('i32.mul');
      }
      if (!first) this._emit('i32.add');
      first = false;
    }
  }

  _emitDynamicStride(buffer: Buffer, dimIdx: number): void {
    let count = 0;
    for (let j = dimIdx + 1; j < buffer.shape.length; j++) {
      const d = buffer.shape[j];
      if (typeof d === 'number' && d >= 0) {
        this._emit(`(i32.const ${d})`);
      } else {
        const spName = this._resolveShapeParam(buffer, j);
        this._emit(`(local.get $${spName})`);
      }
      if (count > 0) this._emit('i32.mul');
      count++;
    }
    if (count === 0) this._emit('(i32.const 1)');
  }

  _resolveShapeParam(buffer: Buffer, dimIdx: number): string {
    return resolveShapeParam(this._primFunc, buffer, dimIdx, (v) => v.name, 'WASM', 'wat');
  }

  _emitExpr(node: IRStmtNode | null): void {
    if (!node) { this._emit('(i32.const 0)'); return; }

    switch (node.type) {
      case 'IntImmNode':
        this._emit(`(i32.const ${node.value})`);
        break;
      case 'FloatImmNode':
        this._emit(`(f32.const ${node.value})`);
        break;
      case 'VariableNode':
        this._emit(`(local.get $${node.name})`);
        break;
      case 'BufferLoadNode':
        if (this._wasmAcc && this._isAccTarget(node.buffer, node.indices)) {
          this._emit('(local.get $' + this._wasmAcc.local + ')');
        } else {
          this._emitAddr(node.buffer, node.indices);
          this._emitLoadOp(node.buffer.dtype);
        }
        break;
      case 'LIRFlatLoadNode':
        if (this._wasmAcc && node.buffer.name === this._wasmAcc.bufName) {
          this._emit('(local.get $' + this._wasmAcc.local + ')');
        } else {
          this._emitFlatAddr(node.buffer, node.offsetExpr);
          this._emitLoadOp(node.dtype);
        }
        break;
      case 'MathOpNode':
        this._emitMathOp(node);
        break;
      case 'CompareNode':
        this._emitCompare(node);
        break;
      case 'CastNode':
        this._emitCast(node);
        break;
      case 'CallExternNode':
        this._emitCallExtern(node);
        break;
      case 'IfThenElseNode': {
        const resultDtype = node._dtype || inferDtype(node.thenBody);
        const resultType = this._numPrefix(resultDtype);
        this._emitExpr(node.condition);
        if (isDtypeFloat(this._wasmExprDtype(node.condition))) {
          this._emit('(f32.const 0)');
          this._emit('f32.ne');
        }
        this._emit('(if (result ' + resultType + ')');
        this._indent++;
        this._emit('(then');
        this._indent++;
        this._emitCoercedTo(node.thenBody, resultType);
        this._indent--;
        this._emit(')');
        this._emit('(else');
        this._indent++;
        this._emitCoercedTo(node.elseBody, resultType);
        this._indent--;
        this._emit(')');
        this._indent--;
        this._emit(')');
        break;
      }
      default:
        this._emit('(i32.const 0)');
        break;
    }
  }

  _numPrefix(dtype: string): string {
    if (dtype === 'f64') return 'f64';
    if (dtype === 'i64') return 'i64';
    if (isDtypeFloat(dtype)) return 'f32';
    return 'i32';
  }

  _exprPrefix(node: IRStmtNode | null): string {
    return this._numPrefix((node && node._dtype) || inferDtype(node));
  }

  _convertTo(fromPrefix: string, toPrefix: string): void {
    if (fromPrefix === toPrefix) return;
    if (toPrefix === 'f64') {
      if (fromPrefix === 'f32') this._emit('f64.promote_f32');
      else if (fromPrefix === 'i32') this._emit('f64.convert_i32_s');
      else if (fromPrefix === 'i64') this._emit('f64.convert_i64_s');
    } else if (toPrefix === 'f32') {
      if (fromPrefix === 'f64') this._emit('f32.demote_f64');
      else if (fromPrefix === 'i32') this._emit('f32.convert_i32_s');
      else if (fromPrefix === 'i64') this._emit('f32.convert_i64_s');
    } else if (toPrefix === 'i32') {
      if (fromPrefix === 'f64') this._emit('i32.trunc_f64_s');
      else if (fromPrefix === 'f32') this._emit('i32.trunc_f32_s');
      else if (fromPrefix === 'i64') this._emit('i32.wrap_i64');
    } else if (toPrefix === 'i64') {
      if (fromPrefix === 'i32') this._emit('i64.extend_i32_s');
      else if (fromPrefix === 'f32') this._emit('i64.trunc_f32_s');
      else if (fromPrefix === 'f64') this._emit('i64.trunc_f64_s');
    }
  }

  _emitCoercedTo(child: IRStmtNode | null, targetPrefix: string): void {
    this._emitExpr(child);
    this._convertTo(this._exprPrefix(child), targetPrefix);
  }

  _emitCoerced(child: IRStmtNode | null, targetFloat: boolean): void {
    this._emitCoercedTo(child, targetFloat ? 'f32' : 'i32');
  }

  _emitMathOp(node: MathOpNode): void {
    const pa = this._exprPrefix(node.a);
    const pb = node.b ? this._exprPrefix(node.b) : pa;
    const prefix = this._joinPrefix(pa, pb);

    if (!node.b) {
      if (node.op === '-') {
        if (prefix === 'f32' || prefix === 'f64') {
          this._emitCoercedTo(node.a, prefix);
          this._emit(prefix + '.neg');
        } else {
          this._emit('(' + prefix + '.const 0)');
          this._emitCoercedTo(node.a, prefix);
          this._emit(prefix + '.sub');
        }
      } else if (node.op === '!') {
        this._emitExpr(node.a);
        this._emit('i32.eqz');
      }
      return;
    }

    if (node.op === '&&') {
      this._emitExpr(node.a);
      this._emitExpr(node.b);
      this._emit('i32.and');
      return;
    }
    if (node.op === '||') {
      this._emitExpr(node.a);
      this._emitExpr(node.b);
      this._emit('i32.or');
      return;
    }

    const isFloat = prefix === 'f32' || prefix === 'f64';
    if (!isFloat) {
      if (node.op === '/' || node.op === '//') { this._emitIntDiv(node, prefix); return; }
      if (node.op === '%') { this._emitIntRem(node, prefix); return; }
    }

    this._emitCoercedTo(node.a, prefix);
    this._emitCoercedTo(node.b, prefix);

    switch (node.op) {
      case '+': this._emit(`${prefix}.add`); break;
      case '-': this._emit(`${prefix}.sub`); break;
      case '*': this._emit(`${prefix}.mul`); break;
      case '/': this._emit(isFloat ? `${prefix}.div` : `${prefix}.div_s`); break;
      case '%': this._emit(`${prefix}.rem_s`); break;
      case '//': this._emit(`${prefix}.div_s`); break;
      case '<': this._emit(isFloat ? `${prefix}.lt` : `${prefix}.lt_s`); break;
      case '>': this._emit(isFloat ? `${prefix}.gt` : `${prefix}.gt_s`); break;
      case '<=': this._emit(isFloat ? `${prefix}.le` : `${prefix}.le_s`); break;
      case '>=': this._emit(isFloat ? `${prefix}.ge` : `${prefix}.ge_s`); break;
      default: this._emit(`${prefix}.add`); break;
    }
  }

  _joinPrefix(pa: string, pb: string): string {
    if (pa === 'f64' || pb === 'f64') return 'f64';
    if (pa === 'f32' || pb === 'f32') return 'f32';
    if (pa === 'i64' || pb === 'i64') return 'i64';
    return 'i32';
  }

  _emitCompare(node: CompareNode): void {
    const pa = this._exprPrefix(node.a);
    const pb = this._exprPrefix(node.b);
    const prefix = this._joinPrefix(pa, pb);
    const isFloat = prefix === 'f32' || prefix === 'f64';
    this._emitCoercedTo(node.a, prefix);
    this._emitCoercedTo(node.b, prefix);
    const ops: InstrTable = { eq: 'eq', ne: 'ne', lt: isFloat ? 'lt' : 'lt_s', le: isFloat ? 'le' : 'le_s', gt: isFloat ? 'gt' : 'gt_s', ge: isFloat ? 'ge' : 'ge_s' };
    const cmp = ops[node.direction];
    if (!cmp) throw new Error(`WASM codegen: unhandled compare direction '${node.direction}'`);
    this._emit(prefix + '.' + cmp);
  }

  _emitCast(node: CastNode): void {
    this._emitExpr(node.expr);
    this._convertTo(this._numPrefix(node.fromDtype), this._numPrefix(node.toDtype));
  }

  _emitCallExtern(node: CallExternNode): void {
    if ((node.externName === 'min' || node.externName === 'max') && !isDtypeFloat(node.dtype)) {
      this._emitIntMinMax(node);
      return;
    }
    if (node.externName === 'abs' && !isDtypeFloat(node.dtype)) {
      this._emitIntAbs(node);
      return;
    }

    const NATIVE = new Set(['sqrt', 'abs', 'ceil', 'floor', 'min', 'max', 'rsqrt']);
    const prefix = node.externName === 'abs' || NATIVE.has(node.externName)
      ? (this._numPrefix(node.dtype) === 'i32' ? 'f32' : this._numPrefix(node.dtype))
      : 'f32';

    if (node.externName === 'rsqrt') {
      this._emit('(' + prefix + '.const 1)');
      for (const arg of node.args) this._emitCoercedTo(arg, prefix);
      this._emit(prefix + '.sqrt');
      this._emit(prefix + '.div');
      return;
    }

    for (const arg of node.args) this._emitCoercedTo(arg, prefix);

    switch (node.externName) {
      case 'sqrt': this._emit(prefix + '.sqrt'); break;
      case 'abs': this._emit(prefix + '.abs'); break;
      case 'ceil': this._emit(prefix + '.ceil'); break;
      case 'floor': this._emit(prefix + '.floor'); break;
      case 'min': this._emit(prefix + '.min'); break;
      case 'max': this._emit(prefix + '.max'); break;
      default:
        if (this._imports.has(node.externName)) {
          this._emit(`call $math_${node.externName}`);
          if (this._numPrefix(node.dtype) === 'f64') this._emit('f64.promote_f32');
        }
        break;
    }
  }

  _emitIntAbs(node: CallExternNode): void {
    const depth = this._intMinMaxEmitDepth || 0;
    const t = '_iabs' + depth;
    this._intMinMaxEmitDepth = depth + 1;
    this._emitExpr(node.args[0]);
    this._emit('local.set $' + t);
    this._intMinMaxEmitDepth = depth;
    this._emit('(i32.const 0)');
    this._emit('(local.get $' + t + ')');
    this._emit('i32.sub');
    this._emit('(local.get $' + t + ')');
    this._emit('(local.get $' + t + ')');
    this._emit('(i32.const 0)');
    this._emit('i32.lt_s');
    this._emit('select');
  }

  _emitIntMinMax(node: CallExternNode): void {
    const depth = this._intMinMaxEmitDepth || 0;
    const aLocal = '_immm_a' + depth;
    const bLocal = '_immm_b' + depth;
    this._intMinMaxEmitDepth = depth + 1;
    this._emitExpr(node.args[0]);
    this._emit('local.set $' + aLocal);
    this._emitExpr(node.args[1]);
    this._emit('local.set $' + bLocal);
    this._intMinMaxEmitDepth = depth;
    this._emit('(local.get $' + aLocal + ')');
    this._emit('(local.get $' + bLocal + ')');
    this._emit('(local.get $' + aLocal + ')');
    this._emit('(local.get $' + bLocal + ')');
    this._emit(node.externName === 'min' ? 'i32.lt_s' : 'i32.gt_s');
    this._emit('select');
  }

  _isIntDivNode(n: IRStmtNode): boolean {
    if (n.type !== 'MathOpNode' || !n.b) return false;
    if (n.op !== '/' && n.op !== '//' && n.op !== '%') return false;
    const prefix = this._joinPrefix(this._exprPrefix(n.a), this._exprPrefix(n.b));
    return prefix === 'i32' || prefix === 'i64';
  }

  _emitIntDiv(node: MathOpNode, prefix: string): void {
    const depth = this._intDivEmitDepth || 0;
    const a = '_idiv_a' + depth, b = '_idiv_b' + depth;
    this._intDivEmitDepth = depth + 1;
    this._emitCoercedTo(node.a, prefix); this._emit('local.set $' + a);
    this._emitCoercedTo(node.b, prefix); this._emit('local.set $' + b);
    this._intDivEmitDepth = depth;
    const MIN = prefix === 'i64' ? '-9223372036854775808' : '-2147483648';
    this._emit('(local.get $' + a + ')');
    this._emit('(' + prefix + '.const 1)');
    this._emit('(local.get $' + b + ')');
    this._emit('(local.get $' + b + ')'); this._emit(prefix + '.eqz');
    this._emit('(local.get $' + a + ')'); this._emit('(' + prefix + '.const ' + MIN + ')'); this._emit(prefix + '.eq');
    this._emit('(local.get $' + b + ')'); this._emit('(' + prefix + '.const -1)'); this._emit(prefix + '.eq');
    this._emit('i32.and');
    this._emit('i32.or');
    this._emit('select');
    this._emit(prefix + '.div_s');
    this._emit('local.set $' + a);
    this._emit('(' + prefix + '.const 0)');
    this._emit('(local.get $' + a + ')');
    this._emit('(local.get $' + b + ')'); this._emit(prefix + '.eqz');
    this._emit('select');
  }

  _emitIntRem(node: MathOpNode, prefix: string): void {
    const depth = this._intDivEmitDepth || 0;
    const a = '_idiv_a' + depth, b = '_idiv_b' + depth;
    this._intDivEmitDepth = depth + 1;
    this._emitCoercedTo(node.a, prefix); this._emit('local.set $' + a);
    this._emitCoercedTo(node.b, prefix); this._emit('local.set $' + b);
    this._intDivEmitDepth = depth;
    this._emit('(local.get $' + a + ')');
    this._emit('(' + prefix + '.const 1)');
    this._emit('(local.get $' + b + ')');
    this._emit('(local.get $' + b + ')'); this._emit(prefix + '.eqz');
    this._emit('select');
    this._emit(prefix + '.rem_s');
    this._emit('local.set $' + a);
    this._emit('(' + prefix + '.const 0)');
    this._emit('(local.get $' + a + ')');
    this._emit('(local.get $' + b + ')'); this._emit(prefix + '.eqz');
    this._emit('select');
  }

  _prescanIntMinMax(root: IRStmtNode): void {
    const visit = (n: IRStmtNode | null, depth: number, divDepth: number) => {
      if (!n || typeof n !== 'object') return;
      let childDepth = depth;
      let childDivDepth = divDepth;
      if (n.type === 'CallExternNode' && (n.externName === 'min' || n.externName === 'max') && !isDtypeFloat(n.dtype)) {
        this._ensureLocal('_immm_a' + depth, 'i32');
        this._ensureLocal('_immm_b' + depth, 'i32');
        childDepth = depth + 1;
      } else if (n.type === 'CallExternNode' && n.externName === 'abs' && !isDtypeFloat(n.dtype)) {
        this._ensureLocal('_iabs' + depth, 'i32');
        childDepth = depth + 1;
      } else if (this._isIntDivNode(n)) {
        const t = wasmType(this._joinPrefix(this._exprPrefix((n as MathOpNode).a), this._exprPrefix((n as MathOpNode).b)));
        this._ensureLocal('_idiv_a' + divDepth, t);
        this._ensureLocal('_idiv_b' + divDepth, t);
        childDivDepth = divDepth + 1;
      }
      const slots = n as unknown as NodeSlots;
      if (slots.body) visit(slots.body as TirNode, childDepth, childDivDepth);
      if (slots.value && typeof slots.value === 'object') visit(slots.value as TirNode, childDepth, childDivDepth);
      if (slots.a) visit(slots.a as TirNode, childDepth, childDivDepth);
      if (slots.b) visit(slots.b as TirNode, childDepth, childDivDepth);
      if (slots.expr) visit(slots.expr as TirNode, childDepth, childDivDepth);
      if (slots.condition) visit(slots.condition as TirNode, childDepth, childDivDepth);
      if (slots.offsetExpr) visit(slots.offsetExpr as TirNode, childDepth, childDivDepth);
      if (slots.thenBody) visit(slots.thenBody as TirNode, childDepth, childDivDepth);
      if (slots.elseBody) visit(slots.elseBody as TirNode, childDepth, childDivDepth);
      if (slots.initBody) visit(slots.initBody as TirNode, childDepth, childDivDepth);
      if (slots.stmts) for (const s of slots.stmts as TirNode[]) visit(s, childDepth, childDivDepth);
      if (slots.args) for (const x of slots.args as TirNode[]) visit(x, childDepth, childDivDepth);
      if (slots.indices) for (const x of slots.indices as TirNode[]) visit(x, childDepth, childDivDepth);
      if ((n as LIRBindingsNode).bindings) for (const x of (n as LIRBindingsNode).bindings) visit(x.expr, childDepth, childDivDepth);
      if ((n as BlockNode).iterVars) for (const x of (n as BlockNode).iterVars) if (x.binding) visit(x.binding, childDepth, childDivDepth);
    };
    visit(root, 0, 0);
  }

  _mathImportSig(name: string, argc: number): string {
    const params = Array(argc).fill('(param f32)').join(' ');
    return `${params} (result f32)`;
  }

  _constExtent(node: IRStmtNode): number | null {
    return node.type === 'IntImmNode' ? node.value : null;
  }

  _isZeroFillBody(body: IRStmtNode): boolean {
    return isZeroFillBody(body);
  }

  _collectBindings(root: IRStmtNode, out: BindingRef[]): void {
    const stack: IRStmtNode[] = [root];
    while (stack.length > 0) {
      const n = stack.pop();
      if (!n || typeof n !== 'object') continue;
      if (n.type === 'BlockNode' && n.iterVars) {
        for (const bind of n.iterVars) {
          if (bind.iterVar && bind.binding) out.push({ name: bind.iterVar.name, expr: bind.binding });
        }
      }
      if (n.type === 'LIRBindingsNode' && n.bindings) {
        for (const bind of n.bindings) out.push({ name: bind.name, expr: bind.expr });
      }
      const slots = n as unknown as NodeSlots;
      if (slots.body) stack.push(slots.body as TirNode);
      if (slots.stmts) for (const s of slots.stmts as TirNode[]) stack.push(s);
      if (slots.thenBody) stack.push(slots.thenBody as TirNode);
      if (slots.elseBody) stack.push(slots.elseBody as TirNode);
      if (slots.initBody) stack.push(slots.initBody as TirNode);
      if (slots.loopBody) stack.push(slots.loopBody as TirNode);
    }
  }

  _computeLaneVars(node: ForNode | LIRAccumulatorNode): Set<string> {
    const laneVars = new Set([node.loopVar.name]);
    const bindings: BindingRef[] = [];
    this._collectBindings(node.body, bindings);
    const varsIn = (expr: IRStmtNode) => {
      const names: string[] = [];
      const st: IRStmtNode[] = [expr];
      while (st.length > 0) {
        const m = st.pop();
        if (!m || typeof m !== 'object') continue;
        if (m.type === 'VariableNode') names.push(m.name);
        const slots = m as unknown as NodeSlots;
        if (slots.a) st.push(slots.a as TirNode);
        if (slots.b) st.push(slots.b as TirNode);
        if (slots.expr) st.push(slots.expr as TirNode);
        if (slots.args) for (const x of slots.args as TirNode[]) st.push(x);
      }
      return names;
    };
    let changed = true;
    while (changed) {
      changed = false;
      for (const b of bindings) {
        if (laneVars.has(b.name)) continue;
        if (varsIn(b.expr).some(nm => laneVars.has(nm))) {
          laneVars.add(b.name);
          changed = true;
        }
      }
    }
    return laneVars;
  }

  _vecLoadsContiguous(root: IRStmtNode, laneVars: ReadonlySet<string>): boolean {
    const usesLane = (expr: IRStmtNode | null) => {
      const st: (IRStmtNode | null)[] = [expr];
      while (st.length > 0) {
        const m = st.pop();
        if (!m || typeof m !== 'object') continue;
        if (m.type === 'VariableNode' && laneVars.has(m.name)) return true;
        const slots = m as unknown as NodeSlots;
        if (slots.a) st.push(slots.a as TirNode);
        if (slots.b) st.push(slots.b as TirNode);
        if (slots.expr) st.push(slots.expr as TirNode);
        if (slots.args) for (const x of slots.args as TirNode[]) st.push(x);
        if (slots.indices) for (const x of slots.indices as TirNode[]) st.push(x);
        if (slots.offsetExpr) st.push(slots.offsetExpr as TirNode);
      }
      return false;
    };
    const stridedMul = (expr: IRStmtNode) => {
      const st: IRStmtNode[] = [expr];
      while (st.length > 0) {
        const m = st.pop();
        if (!m || typeof m !== 'object') continue;
        if (m.type === 'MathOpNode' && m.op === '*' && (usesLane(m.a) || usesLane(m.b))) return true;
        const slots = m as unknown as NodeSlots;
        if (slots.a) st.push(slots.a as TirNode);
        if (slots.b) st.push(slots.b as TirNode);
        if (slots.expr) st.push(slots.expr as TirNode);
        if (slots.args) for (const x of slots.args as TirNode[]) st.push(x);
      }
      return false;
    };
    const stack: IRStmtNode[] = [root];
    while (stack.length > 0) {
      const n = stack.pop();
      if (!n || typeof n !== 'object') continue;
      if (n.type === 'BufferLoadNode' && Array.isArray(n.indices)) {
        for (let i = 0; i < n.indices.length - 1; i++) {
          if (usesLane(n.indices[i])) return false;
        }
      }
      if (n.type === 'LIRFlatLoadNode' && n.offsetExpr && stridedMul(n.offsetExpr)) return false;
      const slots = n as unknown as NodeSlots;
      if (slots.a) stack.push(slots.a as TirNode);
      if (slots.b) stack.push(slots.b as TirNode);
      if (slots.expr) stack.push(slots.expr as TirNode);
      if (slots.args) for (const x of slots.args as TirNode[]) stack.push(x);
      if (slots.value && typeof slots.value === 'object') stack.push(slots.value as TirNode);
      if (slots.body) stack.push(slots.body as TirNode);
      if (slots.stmts) for (const s of slots.stmts as TirNode[]) stack.push(s);
      if (slots.thenBody) stack.push(slots.thenBody as TirNode);
      if (slots.elseBody) stack.push(slots.elseBody as TirNode);
      if (slots.loopBody) stack.push(slots.loopBody as TirNode);
    }
    return true;
  }

  _vecStoresLaneIndexed(root: IRStmtNode, laneVars: ReadonlySet<string>): boolean {
    const stack: IRStmtNode[] = [root];
    while (stack.length > 0) {
      const node = stack.pop();
      if (!node) continue;
      let addr: TirNode[] | IRStmtNode | null = null;
      if (node.type === 'BufferStoreNode') addr = node.indices;
      else if (node.type === 'LIRFlatStoreNode') addr = node.offsetExpr;
      if (addr !== null && addr !== undefined) {
        const names: string[] = [];
        const st: IRStmtNode[] = Array.isArray(addr) ? [...addr] : [addr];
        while (st.length > 0) {
          const m = st.pop();
          if (!m || typeof m !== 'object') continue;
          if (m.type === 'VariableNode') names.push(m.name);
          const slots = m as unknown as NodeSlots;
          if (slots.a) st.push(slots.a as TirNode);
          if (slots.b) st.push(slots.b as TirNode);
          if (slots.expr) st.push(slots.expr as TirNode);
          if (slots.args) for (const x of slots.args as TirNode[]) st.push(x);
          if (slots.indices) for (const x of slots.indices as TirNode[]) st.push(x);
          if (slots.offsetExpr) st.push(slots.offsetExpr as TirNode);
        }
        if (!names.some((nm) => laneVars.has(nm))) return false;
      }
      const slots = node as unknown as NodeSlots;
      if (slots.body) stack.push(slots.body as TirNode);
      if (slots.stmts) for (const s of slots.stmts as TirNode[]) stack.push(s);
      if (slots.thenBody) stack.push(slots.thenBody as TirNode);
      if (slots.elseBody) stack.push(slots.elseBody as TirNode);
      if (slots.loopBody) stack.push(slots.loopBody as TirNode);
    }
    return true;
  }

  _visitVectorizedFor(node: ForNode): void {
    const varName = node.loopVar.name;
    const extent = this._constExtent(node.extent) as number;
    const dtype = this._inferBodyDtype(node.body) || this._defaultDtype;
    const lanes = this.target.vectorWidth;
    const simdEntry = wasmSimdEntry(dtype);

    if (!simdEntry || extent < lanes || this._treeHasHalf(node.body)) {
      this._emitForLoop(varName, node.extent, node.body);
      return;
    }

    const laneVars = this._computeLaneVars(node);
    if (!this._vecStoresLaneIndexed(node.body, laneVars) || !this._vecLoadsContiguous(node.body, laneVars)) {
      this._emitForLoop(varName, node.extent, node.body);
      return;
    }

    const mainExtent = Math.floor(extent / lanes) * lanes;
    const tailExtent = extent - mainExtent;
    const bytes = wasmBytes(dtype);
    const bufAccessCount = this._countBufAccesses(node.body);
    const useAddrCSE = bufAccessCount >= 2;
    const addrLocal = useAddrCSE ? '_vaddr_' + varName : null;

    if (mainExtent > 0) {
      this._vectorMode = { dtype, lanes, loopVar: varName, simd: simdEntry, addrLocal, laneVars };
      if (mainExtent === lanes) {
        this._emit('(i32.const 0)');
        this._emit('local.set $' + varName);
        this._emitVecAddrReset();
        this._visitNode(node.body);
      } else {
        this._emit('(i32.const 0)');
        this._emit('local.set $' + varName);
        this._emit('(block $vbreak_' + varName);
        this._indent++;
        this._emit('(loop $vloop_' + varName);
        this._indent++;
        this._emit('(local.get $' + varName + ')');
        this._emit('(i32.const ' + mainExtent + ')');
        this._emit('i32.ge_s');
        this._emit('br_if $vbreak_' + varName);
        this._emitVecAddrReset();
        this._visitNode(node.body);
        this._emit('(local.get $' + varName + ')');
        this._emit('(i32.const ' + lanes + ')');
        this._emit('i32.add');
        this._emit('local.set $' + varName);
        this._emit('br $vloop_' + varName);
        this._indent--;
        this._emit(')');
        this._indent--;
        this._emit(')');
      }
      this._vectorMode = null;
    }

    if (tailExtent > 0) {
      for (let i = mainExtent; i < extent; i++) {
        this._emit('(i32.const ' + i + ')');
        this._emit('local.set $' + varName);
        this._visitNode(node.body);
      }
    }
  }

  _emitVecAddrReset(): void {
    if (this._vectorMode && this._vectorMode.addrLocal) {
      this._vectorMode._addrEmitted = false;
      this._vectorMode._addrKey = null;
    }
  }

  _countBufAccesses(root: IRStmtNode): number {
    let count = 0;
    const stack: IRStmtNode[] = [root];
    while (stack.length > 0) {
      const n = stack.pop();
      if (!n || typeof n !== 'object') continue;
      if (n.type === 'BufferLoadNode' || n.type === 'BufferStoreNode' ||
          n.type === 'LIRFlatLoadNode' || n.type === 'LIRFlatStoreNode') {
        count++;
      }
      const slots = n as unknown as NodeSlots;
      if (slots.body) stack.push(slots.body as TirNode);
      if (slots.value && typeof slots.value === 'object') stack.push(slots.value as TirNode);
      if (slots.a) stack.push(slots.a as TirNode);
      if (slots.b) stack.push(slots.b as TirNode);
      if (slots.stmts) for (const s of slots.stmts as TirNode[]) stack.push(s);
      if (slots.args) for (const a of slots.args as TirNode[]) stack.push(a);
      if (slots.indices) for (const idx of slots.indices as TirNode[]) stack.push(idx);
      if (slots.thenBody) stack.push(slots.thenBody as TirNode);
      if (slots.elseBody) stack.push(slots.elseBody as TirNode);
      if (slots.expr) stack.push(slots.expr as TirNode);
      if (slots.condition) stack.push(slots.condition as TirNode);
      if (slots.offsetExpr) stack.push(slots.offsetExpr as TirNode);
    }
    return count;
  }

  _inferBodyDtype(body: IRStmtNode): string | null {
    const stack: IRStmtNode[] = [body];
    while (stack.length > 0) {
      const n = stack.pop();
      if (!n) continue;
      if (n.type === 'BufferStoreNode' && n.buffer) return n.buffer.dtype;
      if (n.type === 'LIRFlatStoreNode') return n.dtype || this._defaultDtype;
      if (n.type === 'BufferLoadNode' && n.buffer) return n.buffer.dtype;
      if (n.type === 'LIRFlatLoadNode') return n.dtype || this._defaultDtype;
      const slots = n as unknown as NodeSlots;
      if (slots.body) stack.push(slots.body as TirNode);
      if (slots.stmts) for (const s of slots.stmts as TirNode[]) stack.push(s);
      if (slots.thenBody) stack.push(slots.thenBody as TirNode);
      if (slots.elseBody) stack.push(slots.elseBody as TirNode);
      if (slots.value && typeof slots.value === 'object') stack.push(slots.value as TirNode);
    }
    return null;
  }

  _emitVecStore(node: BufferStoreNode): void {
    const vm = this._vectorMode as VectorMode;
    this._emitAddr(node.buffer, node.indices);
    this._emitVecExpr(node.value);
    this._emit(vm.simd.vecStore);
  }

  _emitVecFlatStore(node: LIRFlatStoreNode): void {
    const vm = this._vectorMode as VectorMode;
    this._emitFlatAddr(node.buffer, node.offsetExpr);
    this._emitVecExpr(node.value);
    this._emit(vm.simd.vecStore);
  }

  _dependsOnVecVar(exprOrList: IRStmtNode | readonly IRStmtNode[] | null): boolean {
    const vm = this._vectorMode;
    if (!vm) return true;
    const laneVars = vm.laneVars;
    const stack: (IRStmtNode | null)[] = Array.isArray(exprOrList) ? [...exprOrList] : [exprOrList as IRStmtNode];
    while (stack.length > 0) {
      const n = stack.pop();
      if (!n || typeof n !== 'object') continue;
      if (n.type === 'VariableNode' && (n.name === vm.loopVar || (laneVars && laneVars.has(n.name)))) return true;
      const slots = n as unknown as NodeSlots;
      if (slots.a) stack.push(slots.a as TirNode);
      if (slots.b) stack.push(slots.b as TirNode);
      if (slots.expr) stack.push(slots.expr as TirNode);
      if (slots.args) for (const x of slots.args as TirNode[]) stack.push(x);
      if (slots.indices) for (const x of slots.indices as TirNode[]) stack.push(x);
      if (slots.offsetExpr) stack.push(slots.offsetExpr as TirNode);
    }
    return false;
  }

  _emitVecExpr(node: IRStmtNode | null): void {
    if (!node) { this._emit('(i32.const 0)'); return; }
    const vm = this._vectorMode as VectorMode;
    const dtype = vm.dtype;

    switch (node.type) {
      case 'BufferLoadNode':
        if (this._dependsOnVecVar(node.indices)) {
          this._emitAddr(node.buffer, node.indices);
          this._emit(vm.simd.vecLoad);
        } else {
          this._emitAddr(node.buffer, node.indices);
          this._emit(wasmLoad(node.buffer.dtype));
          this._emit(vm.simd.splat);
        }
        break;
      case 'LIRFlatLoadNode':
        if (this._dependsOnVecVar(node.offsetExpr)) {
          this._emitFlatAddr(node.buffer, node.offsetExpr);
          this._emit(vm.simd.vecLoad);
        } else {
          this._emitFlatAddr(node.buffer, node.offsetExpr);
          this._emit(wasmLoad(node.dtype));
          this._emit(vm.simd.splat);
        }
        break;
      case 'FloatImmNode':
        this._emit('(f32.const ' + node.value + ')');
        this._emit(vm.simd.splat);
        break;
      case 'IntImmNode':
        if (isDtypeFloat(dtype)) {
          this._emit('(f32.const ' + node.value + ')');
          this._emit(vm.simd.splat);
        } else {
          this._emit('(i32.const ' + node.value + ')');
          this._emit(vm.simd.splat);
        }
        break;
      case 'VariableNode':
        if (vm.vecLets && vm.vecLets.has(node.name)) {
          this._emit('(local.get $' + node.name + '_vlet)');
        } else if (node.name === vm.loopVar) {
          this._emit('(local.get $' + node.name + ')');
        } else {
          const localType = this._locals.get(node.name);
          if (localType === 'f32') {
            this._emit('(local.get $' + node.name + ')');
            this._emit(vm.simd.splat);
          } else {
            this._emit('(local.get $' + node.name + ')');
            if (isDtypeFloat(dtype)) this._emit('f32.convert_i32_s');
            this._emit(vm.simd.splat);
          }
        }
        break;
      case 'MathOpNode':
        this._emitVecMathOp(node);
        break;
      case 'CompareNode':
        this._emitVecCompare(node);
        break;
      case 'CallExternNode':
        this._emitVecCallExtern(node);
        break;
      case 'CastNode':
        this._emitVecExpr(node.expr);
        if (isDtypeFloat(node.toDtype) && this._isVecMaskExpr(node.expr)) {
          this._emit('(f32.const 1)');
          this._emit('f32x4.splat');
          this._emit('v128.and');
        }
        break;
      case 'IfThenElseNode':
        this._emitVecSelect(node);
        break;
      default:
        this._emitExpr(node);
        break;
    }
  }

  _isVecMaskExpr(node: IRStmtNode | null): boolean {
    if (!node) return false;
    if (node.type === 'CompareNode') return true;
    if (node.type === 'MathOpNode' && (node.op === '&&' || node.op === '||' || node.op === '!')) return true;
    return false;
  }

  _emitVecMathOp(node: MathOpNode): void {
    const vm = this._vectorMode as VectorMode;
    const dtype = vm.dtype;

    if (!node.b) {
      if (node.op === '-') {
        const negOp = wasmVecOp(dtype, 'neg');
        if (negOp) {
          this._emitVecExpr(node.a);
          this._emit(negOp as string);
        } else {
          this._emit('(i32.const 0)');
          this._emit(vm.simd.splat);
          this._emitVecExpr(node.a);
          this._emit(wasmVecOp(dtype, 'sub') as string);
        }
      } else if (node.op === '!') {
        this._emitVecExpr(node.a);
        this._emit('v128.not');
      }
      return;
    }

    if (node.op === '&&') {
      this._emitVecExpr(node.a);
      this._emitVecExpr(node.b);
      this._emit('v128.and');
      return;
    }
    if (node.op === '||') {
      this._emitVecExpr(node.a);
      this._emitVecExpr(node.b);
      this._emit('v128.or');
      return;
    }

    const OP_MAP: InstrTable = { '+': 'add', '-': 'sub', '*': 'mul', '/': 'div' };
    const opName = OP_MAP[node.op];
    if (opName) {
      const vecInstr = wasmVecOp(dtype, opName as SimdOpKey);
      if (vecInstr) {
        this._emitVecExpr(node.a);
        this._emitVecExpr(node.b);
        this._emit(vecInstr as string);
        return;
      }
    }

    const CMP_MAP: InstrTable = { '<': 'lt', '>': 'gt', '<=': 'le', '>=': 'ge' };
    const cmpName = CMP_MAP[node.op];
    if (cmpName) {
      const vecInstr = wasmVecOp(dtype, cmpName as SimdOpKey);
      if (vecInstr) {
        this._emitVecExpr(node.a);
        this._emitVecExpr(node.b);
        this._emit(vecInstr as string);
        return;
      }
    }

    this._emitExpr(node);
    this._emit(vm.simd.splat);
  }

  _emitVecCompare(node: CompareNode): void {
    const vm = this._vectorMode as VectorMode;
    const dtype = vm.dtype;
    const vecInstr = wasmVecOp(dtype, node.direction as SimdOpKey);
    if (vecInstr) {
      this._emitVecExpr(node.a);
      this._emitVecExpr(node.b);
      this._emit(vecInstr as string);
    } else {
      this._emitExpr(node);
      this._emit(vm.simd.splat);
    }
  }

  _emitVecCallExtern(node: CallExternNode): void {
    const vm = this._vectorMode as VectorMode;
    const dtype = vm.dtype;
    const vecInstr = wasmVecOp(dtype, node.externName as SimdOpKey);

    if (vecInstr) {
      if (node.externName === 'min' || node.externName === 'max') {
        this._emitVecExpr(node.args[0]);
        this._emitVecExpr(node.args[1]);
      } else {
        this._emitVecExpr(node.args[0]);
      }
      this._emit(vecInstr as string);
      return;
    }

    if (node.externName === 'rsqrt') {
      const sqrtOp = wasmVecOp(dtype, 'sqrt');
      if (sqrtOp) {
        this._emit('(f32.const 1)');
        this._emit(vm.simd.splat);
        this._emitVecExpr(node.args[0]);
        this._emit(sqrtOp as string);
        this._emit(wasmVecOp(dtype, 'div') as string);
        return;
      }
    }

    this._emitScalarizeFallback(node);
  }

  _emitVecSelect(node: IfThenElseNode): void {
    const vm = this._vectorMode as VectorMode;
    this._emitVecExpr(node.thenBody);
    this._emitVecExpr(node.elseBody);
    this._emitVecExpr(node.condition);
    this._emit(vm.simd.bitselect);
  }

  _emitScalarizeFallback(node: CallExternNode): void {
    const vm = this._vectorMode as VectorMode;
    const lanes = vm.lanes;
    const extractLane = vm.simd.extractLane;
    const replaceLane = vm.simd.replaceLane;
    const splat = vm.simd.splat;

    const tmpName = '_vtmp_' + (this._vecTmpCounter++);
    this._ensureLocal(tmpName, 'v128');

    const laneResults: string[] = [];
    for (let l = 0; l < lanes; l++) {
      const ln = '_vl_' + tmpName + '_' + l;
      this._ensureLocal(ln, wasmType(vm.dtype));
      laneResults.push(ln);
    }

    this._emitVecExpr(node.args[0]);
    this._emit('local.set $' + tmpName);

    for (let l = 0; l < lanes; l++) {
      this._emit('(local.get $' + tmpName + ')');
      this._emit(extractLane + ' ' + l);
      if (node.args.length > 1) {
        const arg2Tmp = '_vtmp2_' + tmpName;
        if (l === 0) {
          this._ensureLocal(arg2Tmp, 'v128');
          const prevVM = this._vectorMode;
          this._vectorMode = vm;
          this._emitVecExpr(node.args[1]);
          this._vectorMode = prevVM;
          this._emit('local.set $' + arg2Tmp);
        }
        this._emit('(local.get $' + arg2Tmp + ')');
        this._emit(extractLane + ' ' + l);
      }
      if (this._imports.has(node.externName)) {
        this._emit('call $math_' + node.externName);
      }
      this._emit('local.set $' + laneResults[l]);
    }

    this._emit('(local.get $' + laneResults[lanes - 1] + ')');
    this._emit(splat);
    for (let l = lanes - 2; l >= 0; l--) {
      this._emit('(local.get $' + laneResults[l] + ')');
      this._emit(replaceLane + ' ' + l);
    }
  }

  _prescanVecLocalsAll(root: IRStmtNode): void {
    const stack: IRStmtNode[] = [root];
    while (stack.length > 0) {
      const n = stack.pop();
      if (!n) continue;
      if (n.type === 'ForNode' && n.kind === ForKind.VECTORIZED) {
        this._prescanVecLocals(n.body);
        if (this._countBufAccesses(n.body) >= 2) {
          this._ensureLocal('_vaddr_' + n.loopVar.name, 'i32');
        }
        this._prescanVecLets(n);
      }
      if (n.type === 'LIRAccumulatorNode' && n.loopKind === ForKind.VECTORIZED) {
        this._ensureLocal(n.localName + '_vec', 'v128');
        this._prescanVecLocals(n.body);
      }
      const slots = n as unknown as NodeSlots;
      if (slots.body) stack.push(slots.body as TirNode);
      if (slots.stmts) for (const s of slots.stmts as TirNode[]) stack.push(s);
      if (slots.thenBody) stack.push(slots.thenBody as TirNode);
      if (slots.elseBody) stack.push(slots.elseBody as TirNode);
      if (slots.initBody) stack.push(slots.initBody as TirNode);
      if (slots.loopBody) stack.push(slots.loopBody as TirNode);
    }
  }

  _prescanVecLets(forNode: ForNode): void {
    const laneVars = this._computeLaneVars(forNode);
    const dependsOn = (expr: IRStmtNode) => {
      const st: IRStmtNode[] = [expr];
      while (st.length > 0) {
        const m = st.pop();
        if (!m || typeof m !== 'object') continue;
        if (m.type === 'VariableNode' && laneVars.has(m.name)) return true;
        const slots = m as unknown as NodeSlots;
        if (slots.a) st.push(slots.a as TirNode);
        if (slots.b) st.push(slots.b as TirNode);
        if (slots.expr) st.push(slots.expr as TirNode);
        if (slots.args) for (const x of slots.args as TirNode[]) st.push(x);
        if (slots.indices) for (const x of slots.indices as TirNode[]) st.push(x);
        if (slots.offsetExpr) st.push(slots.offsetExpr as TirNode);
      }
      return false;
    };
    const stack: IRStmtNode[] = [forNode.body];
    while (stack.length > 0) {
      const n = stack.pop();
      if (!n || typeof n !== 'object') continue;
      if (n.type === 'LetStmtNode' && n.variable && dependsOn(n.value)) {
        this._ensureLocal(n.variable.name + '_vlet', 'v128');
      }
      const slots = n as unknown as NodeSlots;
      if (slots.body) stack.push(slots.body as TirNode);
      if (slots.stmts) for (const s of slots.stmts as TirNode[]) stack.push(s);
      if (slots.thenBody) stack.push(slots.thenBody as TirNode);
      if (slots.elseBody) stack.push(slots.elseBody as TirNode);
      if (slots.initBody) stack.push(slots.initBody as TirNode);
    }
  }

  _prescanVecLocals(body: IRStmtNode): void {
    const stack: IRStmtNode[] = [body];
    while (stack.length > 0) {
      const n = stack.pop();
      if (!n) continue;
      if (n.type === 'CallExternNode' && n.externName) {
        const vecInstr = wasmVecOp(this._defaultDtype, n.externName as SimdOpKey);
        if (!vecInstr && n.externName !== 'rsqrt') {
          const tmpName = '_vtmp_' + (this._vecTmpCounter);
          this._ensureLocal(tmpName, 'v128');
          const lanes = this.target.vectorWidth;
          for (let l = 0; l < lanes; l++) {
            this._ensureLocal('_vl_' + tmpName + '_' + l, wasmType(this._defaultDtype));
          }
          if (n.args.length > 1) {
            this._ensureLocal('_vtmp2_' + tmpName, 'v128');
          }
          this._vecTmpCounter++;
        }
      }
      const slots = n as unknown as NodeSlots;
      if (slots.body) stack.push(slots.body as TirNode);
      if (slots.stmts) for (const s of slots.stmts as TirNode[]) stack.push(s);
      if (slots.value && typeof slots.value === 'object' && (slots.value as TirNode).type) stack.push(slots.value as TirNode);
      if (slots.a && typeof slots.a === 'object') stack.push(slots.a as TirNode);
      if (slots.b && typeof slots.b === 'object') stack.push(slots.b as TirNode);
      if (slots.args) for (const a of slots.args as TirNode[]) if (typeof a === 'object') stack.push(a);
      if (slots.thenBody) stack.push(slots.thenBody as TirNode);
      if (slots.elseBody) stack.push(slots.elseBody as TirNode);
    }
  }

  _layoutBuffers(primFunc: CodegenFunc): void {
    let offset = 0;
    const align = 16;
    const place = (buf: Buffer) => {
      offset = Math.ceil(offset / align) * align;
      this._bufferOffsets.set(buf.name, offset);
      const numel = buf.numel();
      const isDynamic = buf.shape.some(d => typeof d !== 'number' || d < 0);
      if (!isDynamic && numel > 0) {
        offset += numel * wasmBytes(buf.dtype);
      } else {
        offset += 65536;
      }
    };
    for (const [, buf] of primFunc.bufferMap) place(buf);

    const tempBuffers = new Map<string, Buffer>();
    this._collectBuffers(primFunc.body, tempBuffers);
    for (const [name, buf] of tempBuffers) {
      if (this._bufferOffsets.has(name)) continue;
      place(buf);
    }

    this._totalMemBytes = offset;
  }

  _collectBuffers(node: IRStmtNode, result: Map<string, Buffer>): void {
    const stack: IRStmtNode[] = [node];
    while (stack.length > 0) {
      const n = stack.pop();
      if (!n) continue;
      if ((n.type === 'BufferStoreNode' || n.type === 'BufferLoadNode') && n.buffer) {
        result.set(n.buffer.name, n.buffer);
      }
      if ((n as BlockNode).reads) for (const r of (n as BlockNode).reads) if (r.buffer) result.set(r.buffer.name, r.buffer);
      if ((n as BlockNode).writes) for (const w of (n as BlockNode).writes) if (w.buffer) result.set(w.buffer.name, w.buffer);
      for (const c of irChildNodes(n)) stack.push(c);
    }
  }

  _scanMathImports(node: IRStmtNode): void {
    const stack: IRStmtNode[] = [node];
    while (stack.length > 0) {
      const n = stack.pop();
      if (!n || typeof n !== 'object') continue;
      if (n.type === 'CallExternNode' && n.externName) {
        const name = n.externName;
        if (name !== 'sqrt' && name !== 'min' && name !== 'max') {
          const sig = this._mathImportSig(name, n.args.length);
          this._imports.set(name, sig);
        }
      }
      const slots = n as unknown as NodeSlots;
      if (slots.body) stack.push(slots.body as TirNode);
      if (slots.stmts) for (const s of slots.stmts as TirNode[]) stack.push(s);
      if (slots.thenBody) stack.push(slots.thenBody as TirNode);
      if (slots.elseBody) stack.push(slots.elseBody as TirNode);
      if (slots.initBody) stack.push(slots.initBody as TirNode);
      if (slots.value && typeof slots.value === 'object' && (slots.value as TirNode).type) stack.push(slots.value as TirNode);
      if (slots.a && typeof slots.a === 'object') stack.push(slots.a as TirNode);
      if (slots.b && typeof slots.b === 'object') stack.push(slots.b as TirNode);
      if (slots.expr && typeof slots.expr === 'object') stack.push(slots.expr as TirNode);
      if (slots.args) for (const a of slots.args as TirNode[]) if (typeof a === 'object') stack.push(a);
      if (slots.indices) for (const idx of slots.indices as TirNode[]) if (typeof idx === 'object') stack.push(idx);
      if (slots.condition && typeof slots.condition === 'object') stack.push(slots.condition as TirNode);
    }
  }

  _prescanLocals(node: IRStmtNode): void {
    this._waccCounter = 0;
    const stack: IRStmtNode[] = [node];
    while (stack.length > 0) {
      const n = stack.pop();
      if (!n) continue;
      if (n.type === 'ForNode') {
        this._ensureLocal(n.loopVar.name, 'i32');
        const accDtype = this._accPatternDtype(n);
        if (accDtype) {
          this._ensureLocal('_wacc_' + (++this._waccCounter), wasmType(accDtype));
        }
        if (n.kind === ForKind.VECTORIZED && this.target.supportsSimd()) {
          this._prescanVecLocals(n.body);
        }
      }
      if (n.type === 'BlockNode') {
        for (const bind of n.iterVars) {
          if (bind.iterVar) this._ensureLocal(bind.iterVar.name, 'i32');
        }
      }
      if (n.type === 'LetStmtNode' && n.variable) {
        const letDtype = inferDtype(n.value) || n.variable.dtype || this._defaultDtype;
        this._ensureLocal(n.variable.name, wasmType(letDtype));
      }
      const slots = n as unknown as NodeSlots;
      if (slots.body) stack.push(slots.body as TirNode);
      if (slots.stmts) for (const s of slots.stmts as TirNode[]) stack.push(s);
      if (slots.thenBody) stack.push(slots.thenBody as TirNode);
      if (slots.elseBody) stack.push(slots.elseBody as TirNode);
      if (slots.initBody) stack.push(slots.initBody as TirNode);
    }
    this._waccCounter = 0;
  }

  _wasmExprDtype(node: IRStmtNode | null): string {
    if (!node) return 'i32';
    if (node.type === 'CompareNode') return 'i32';
    if (node.type === 'MathOpNode') {
      if (node.op === '!' || node.op === '&&' || node.op === '||') return 'i32';
      if (node.op === '<' || node.op === '>' || node.op === '<=' || node.op === '>=') return 'i32';
    }
    if (node.type === 'VariableNode') {
      const localType = this._locals.get(node.name);
      if (localType) return localType === 'f32' ? 'f32' : 'i32';
    }
    return inferDtype(node);
  }

  _fixLetStmtLocals(node: IRStmtNode): void {
    const stack: IRStmtNode[] = [node];
    while (stack.length > 0) {
      const n = stack.pop();
      if (!n) continue;
      if (n.type === 'LetStmtNode' && n.variable && n.value) {
        const valDtype = inferDtype(n.value);
        if (valDtype) this._locals.set(n.variable.name, wasmType(valDtype));
      }
      const slots = n as unknown as NodeSlots;
      if (slots.body) stack.push(slots.body as TirNode);
      if (slots.stmts) for (const s of slots.stmts as TirNode[]) stack.push(s);
      if (slots.thenBody) stack.push(slots.thenBody as TirNode);
      if (slots.elseBody) stack.push(slots.elseBody as TirNode);
      if (slots.initBody) stack.push(slots.initBody as TirNode);
    }
  }

  _accPatternDtype(forNode: ForNode): string | null {
    let block = forNode.body;
    if (!block || block.type !== 'BlockNode') return null;
    const inner = block.body;
    if (!inner || inner.type !== 'BufferStoreNode') return null;
    const val = inner.value;
    if (!val || val.type !== 'MathOpNode' || val.op !== '+') return null;
    if (val.a && val.a.type === 'BufferLoadNode' && val.a.buffer.name === inner.buffer.name) return inner.buffer.dtype;
    if (val.b && val.b.type === 'BufferLoadNode' && val.b.buffer.name === inner.buffer.name) return inner.buffer.dtype;
    return null;
  }

  _detectWasmAcc(forNode: ForNode): WasmAccPattern | null {
    let block = forNode.body;
    if (!block || block.type !== 'BlockNode') return null;
    const inner = block.body;
    if (!inner || inner.type !== 'BufferStoreNode') return null;
    const store = inner;
    const val = store.value;
    if (!val || val.type !== 'MathOpNode' || val.op !== '+') return null;
    let loadSide: BufferLoadNode | null = null;
    if (val.a && val.a.type === 'BufferLoadNode' && val.a.buffer.name === store.buffer.name) loadSide = val.a;
    else if (val.b && val.b.type === 'BufferLoadNode' && val.b.buffer.name === store.buffer.name) loadSide = val.b;
    if (!loadSide) return null;
    const storeKey = this._indicesKey(store.buffer.name, store.indices);
    const loadKey = this._indicesKey(loadSide.buffer.name, loadSide.indices);
    if (storeKey !== loadKey) return null;
    const outerIndices = store.indices.map(idx => {
      if (idx.type !== 'VariableNode') return idx;
      for (const bind of block.iterVars) {
        if (bind.iterVar && bind.iterVar.name === idx.name && bind.binding) return bind.binding;
      }
      return idx;
    });
    return { buf: store.buffer, indices: store.indices, outerIndices };
  }

  _isAccTarget(buffer: Buffer, indices: readonly IRStmtNode[]): boolean {
    if (!this._wasmAcc) return false;
    if (buffer.name !== this._wasmAcc.bufName) return false;
    if (!this._wasmAcc.indices) return false;
    return this._indicesKey(buffer.name, indices) === this._indicesKey(this._wasmAcc.bufName, this._wasmAcc.indices);
  }

  _indicesKey(bufName: string, indices: readonly IRStmtNode[]): string {
    const parts: string[] = [];
    for (let i = 0; i < indices.length; i++) {
      parts.push(this._exprKey(indices[i]));
    }
    return bufName + ':' + parts.join(',');
  }

  _exprKey(node: IRStmtNode | null): string {
    if (!node) return '?';
    if (node.type === 'VariableNode') return '$' + node.name;
    if (node.type === 'IntImmNode') return String(node.value);
    if (node.type === 'MathOpNode') return '(' + this._exprKey(node.a) + node.op + (node.b ? this._exprKey(node.b) : '') + ')';
    return '?';
  }
}
