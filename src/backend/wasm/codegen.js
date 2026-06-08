import { ForKind } from '../../compiler/ir/tensor/nodes.js';
import { wasmType, wasmLoad, wasmStore, wasmBytes, isDtypeFloat, isDtypeInt } from '../dtype_map.js';
import { inferDtype } from '../../compiler/ir/lir/nodes.js';

export class WasmCodegen {
  constructor(target) {
    this.target = target;
    this._lines = [];
    this._indent = 0;
    this._locals = new Map();
    this._localCounter = 0;
    this._imports = new Map();
    this._bufferOffsets = new Map();
    this._totalMemBytes = 0;
    this._defaultDtype = 'f32';
  }

  generate(func) {
    this._lines = [];
    this._indent = 0;
    this._locals.clear();
    this._localCounter = 0;
    this._imports.clear();
    this._bufferOffsets.clear();
    this._totalMemBytes = 0;
    this._primFunc = func;
    this._wasmAcc = null;
    this._waccCounter = 0;
    this._hasParallel = false;
    this._parallelExtent = 0;

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

    const paramNames = [];
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

    const paramDecls = [];
    for (const [, buf] of func.bufferMap) {
      paramDecls.push('(param i32)');
    }
    const shapeParamEntries = [];
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
    }

    this._fixLetStmtLocals(func.body);

    const localDecls = [];
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
      parallel: this._hasParallel ? { extent: this._parallelExtent, outputIndices: this._findOutputIndices(func) } : null,
    };
  }

  _ensureLocal(name, type) {
    if (!this._locals.has(name)) {
      this._locals.set(name, type);
    }
  }

  _emit(line) {
    this._lines.push('  '.repeat(this._indent) + line);
  }

  _visitNode(startNode) {
    let node = startNode;
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
          const varDtype = inferDtype(node.value) || node.variable.dtype || this._defaultDtype;
          this._locals.set(node.variable.name, wasmType(varDtype));
          this._emitCoerced(node.value, isDtypeFloat(varDtype));
          this._emit(`local.set $${node.variable.name}`);
          node = node.body;
          continue;
        }
        case 'ForNode': this._visitFor(node); return;
        case 'BlockNode': this._visitBlock(node); return;
        case 'IfThenElseNode': this._visitIf(node); return;
        case 'BufferStoreNode': this._visitStore(node); return;
        case 'LIRFlatStoreNode': this._visitLIRFlatStore(node); return;
        case 'LIRBindingsNode': this._visitLIRBindings(node); return;
        case 'LIRAccumulatorNode': this._visitLIRAccumulator(node); return;
        case 'WhileNode': this._visitWhile(node); return;
        case 'EvaluateNode': return;
        default: return;
      }
    }
  }

  _findOutputIndices(func) {
    const storeNames = new Set();
    const stack = [func.body];
    while (stack.length > 0) {
      const node = stack.pop();
      if (!node) continue;
      if ((node.type === 'BufferStoreNode' || node.type === 'LIRFlatStoreNode') && node.buffer) {
        storeNames.add(node.buffer.name);
      }
      if (node.type === 'LIRAccumulatorNode' && node.flushStore && node.flushStore.buffer) {
        storeNames.add(node.flushStore.buffer.name);
      }
      if (node.body) stack.push(node.body);
      if (node.stmts) for (const s of node.stmts) stack.push(s);
      if (node.thenBody) stack.push(node.thenBody);
      if (node.elseBody) stack.push(node.elseBody);
      if (node.loopBody) stack.push(node.loopBody);
    }
    const indices = [];
    let idx = 0;
    for (const [, buf] of func.bufferMap) {
      if (storeNames.has(buf.name)) indices.push(idx);
      idx++;
    }
    return indices;
  }

  _scanParallel(root) {
    const stack = [root];
    while (stack.length > 0) {
      const node = stack.pop();
      if (!node) continue;
      if (node.type === 'ForNode' && node.kind === ForKind.PARALLEL) {
        this._hasParallel = true;
        this._parallelExtent = node.extent && node.extent.type === 'IntImmNode' ? node.extent.value : 0;
        return;
      }
      if (node.body) stack.push(node.body);
      if (node.stmts) for (const s of node.stmts) stack.push(s);
      if (node.thenBody) stack.push(node.thenBody);
      if (node.elseBody) stack.push(node.elseBody);
    }
  }

  _visitFor(node) {
    const varName = node.loopVar.name;
    const extent = this._constExtent(node.extent);

    if (node.kind === ForKind.PARALLEL) {
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
      this._emit(wasmLoad(accInfo.buf.dtype));
      this._emit('local.set $' + accLocal);
      this._wasmAcc = { local: accLocal, bufName: accInfo.buf.name, indices: accInfo.indices };
      this._emitForLoop(varName, node.extent, node.body);
      this._emitAddr(accInfo.buf, accInfo.outerIndices);
      this._emit('(local.get $' + accLocal + ')');
      this._emit(wasmStore(accInfo.buf.dtype));
      this._wasmAcc = null;
      return;
    }

    this._emitForLoop(varName, node.extent, node.body);
  }

  _visitLIRFlatStore(node) {
    if (this._wasmAcc && node.buffer.name === this._wasmAcc.bufName) {
      this._emitCoerced(node.value, isDtypeFloat(node.dtype));
      this._emit('local.set $' + this._wasmAcc.local);
      return;
    }
    this._emitFlatAddr(node.buffer, node.offsetExpr);
    this._emitCoerced(node.value, isDtypeFloat(node.dtype));
    this._emit(wasmStore(node.dtype));
  }

  _visitLIRBindings(node) {
    for (const bind of node.bindings) {
      this._emitExpr(bind.expr);
      this._emit(`local.set $${bind.name}`);
    }
    this._visitNode(node.body);
  }

  _visitLIRAccumulator(node) {
    const accLocal = node.localName;
    const dtype = node.dtype;
    this._ensureLocal(accLocal, wasmType(dtype));

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
    this._emitCoerced(node.body, isDtypeFloat(dtype));
    this._emit(isDtypeFloat(dtype) ? 'f32.add' : 'i32.add');
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
    this._emit(wasmStore(node.flushStore.dtype));

    this._wasmAcc = prevAcc;
  }

  _emitFlatAddr(buffer, offsetExpr) {
    const baseOffset = this._bufferOffsets.get(buffer.name) || 0;
    const bytes = wasmBytes(buffer.dtype || 'f32');

    if (!offsetExpr || (offsetExpr.type === 'IntImmNode' && offsetExpr.value === 0)) {
      this._emit(`(i32.const ${baseOffset})`);
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

  _emitForLoop(varName, extent, body) {
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
    this._visitNode(body);
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

  _visitBlock(node) {
    for (const bind of node.iterVars) {
      if (bind.iterVar && bind.binding) {
        this._emitExpr(bind.binding);
        this._emit(`local.set $${bind.iterVar.name}`);
      }
    }
    if (node.initBody) this._visitNode(node.initBody);
    this._visitNode(node.body);
  }

  _visitStore(node) {
    if (this._wasmAcc && this._isAccTarget(node.buffer, node.indices)) {
      this._emitCoerced(node.value, isDtypeFloat(node.buffer.dtype));
      this._emit('local.set $' + this._wasmAcc.local);
      return;
    }
    this._emitAddr(node.buffer, node.indices);
    this._emitCoerced(node.value, isDtypeFloat(node.buffer.dtype));
    this._emit(wasmStore(node.buffer.dtype));
  }

  _visitIf(node) {
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

  _visitWhile(node) {
    this._visitNode(node.condBody);
    this._emit('(block $wbreak');
    this._indent++;
    this._emit('(loop $wloop');
    this._indent++;
    this._emit(`(local.get $${node.condVar.name})`);
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

  _emitAddr(buffer, indices) {
    const baseOffset = this._bufferOffsets.get(buffer.name) || 0;
    const bytes = wasmBytes(buffer.dtype);

    if (indices.length === 0) {
      this._emit(`(i32.const ${baseOffset})`);
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

  _emitFlatIndex(buffer, indices) {
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

  _emitDynamicStride(buffer, dimIdx) {
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

  _resolveShapeParam(buffer, dimIdx) {
    if (this._primFunc && this._primFunc.shapeParamMap) {
      const key = `${buffer.name}:${dimIdx}`;
      const v = this._primFunc.shapeParamMap.get(key);
      if (v) return v.name;
    }
    return '_ds_0';
  }

  _emitExpr(node) {
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
          this._emit(wasmLoad(node.buffer.dtype));
        }
        break;
      case 'LIRFlatLoadNode':
        if (this._wasmAcc && node.buffer.name === this._wasmAcc.bufName) {
          this._emit('(local.get $' + this._wasmAcc.local + ')');
        } else {
          this._emitFlatAddr(node.buffer, node.offsetExpr);
          this._emit(wasmLoad(node.dtype));
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
        const resultIsFloat = isDtypeFloat(resultDtype);
        const resultType = resultIsFloat ? 'f32' : 'i32';
        this._emitExpr(node.condition);
        if (isDtypeFloat(this._wasmExprDtype(node.condition))) {
          this._emit('(f32.const 0)');
          this._emit('f32.ne');
        }
        this._emit('(if (result ' + resultType + ')');
        this._indent++;
        this._emit('(then');
        this._indent++;
        this._emitCoerced(node.thenBody, resultIsFloat);
        this._indent--;
        this._emit(')');
        this._emit('(else');
        this._indent++;
        this._emitCoerced(node.elseBody, resultIsFloat);
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

  _emitCoerced(child, targetFloat) {
    this._emitExpr(child);
    const childDtype = (child && child._dtype) || inferDtype(child);
    if (targetFloat && childDtype === 'i32') {
      this._emit('f32.convert_i32_s');
    }
  }

  _emitMathOp(node) {
    const da = (node.a && node.a._dtype) || inferDtype(node.a);
    const db = node.b ? ((node.b._dtype) || inferDtype(node.b)) : da;
    const isFloat = isDtypeFloat(da) || isDtypeFloat(db);
    const prefix = isFloat ? 'f32' : 'i32';

    if (!node.b) {
      if (node.op === '-') {
        if (isFloat) {
          this._emitCoerced(node.a, true);
          this._emit('f32.neg');
        } else {
          this._emit('(i32.const 0)');
          this._emitExpr(node.a);
          this._emit('i32.sub');
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

    this._emitCoerced(node.a, isFloat);
    this._emitCoerced(node.b, isFloat);

    switch (node.op) {
      case '+': this._emit(`${prefix}.add`); break;
      case '-': this._emit(`${prefix}.sub`); break;
      case '*': this._emit(`${prefix}.mul`); break;
      case '/': this._emit(prefix === 'f32' ? 'f32.div' : 'i32.div_s'); break;
      case '%': this._emit('i32.rem_s'); break;
      case '//': this._emit('i32.div_s'); break;
      case '<': this._emit(prefix === 'f32' ? 'f32.lt' : 'i32.lt_s'); break;
      case '>': this._emit(prefix === 'f32' ? 'f32.gt' : 'i32.gt_s'); break;
      case '<=': this._emit(prefix === 'f32' ? 'f32.le' : 'i32.le_s'); break;
      case '>=': this._emit(prefix === 'f32' ? 'f32.ge' : 'i32.ge_s'); break;
      default: this._emit(`${prefix}.add`); break;
    }
  }

  _emitCompare(node) {
    const da = (node.a && node.a._dtype) || inferDtype(node.a);
    const db = (node.b && node.b._dtype) || inferDtype(node.b);
    const isFloat = isDtypeFloat(da) || isDtypeFloat(db);
    const prefix = isFloat ? 'f32' : 'i32';
    this._emitCoerced(node.a, isFloat);
    this._emitCoerced(node.b, isFloat);
    const ops = { eq: 'eq', ne: 'ne', lt: prefix === 'f32' ? 'lt' : 'lt_s', le: prefix === 'f32' ? 'le' : 'le_s', gt: prefix === 'f32' ? 'gt' : 'gt_s', ge: prefix === 'f32' ? 'ge' : 'ge_s' };
    this._emit(prefix + '.' + (ops[node.direction] || 'eq'));
  }

  _emitCast(node) {
    this._emitExpr(node.expr);
    const fromFloat = isDtypeFloat(node.fromDtype);
    const toFloat = isDtypeFloat(node.toDtype);
    if (fromFloat && !toFloat) this._emit('i32.trunc_f32_s');
    else if (!fromFloat && toFloat) this._emit('f32.convert_i32_s');
  }

  _emitCallExtern(node) {
    for (const arg of node.args) this._emitExpr(arg);

    switch (node.externName) {
      case 'sqrt': this._emit('f32.sqrt'); break;
      case 'abs': this._emit('f32.abs'); break;
      case 'ceil': this._emit('f32.ceil'); break;
      case 'floor': this._emit('f32.floor'); break;
      case 'min': this._emit('f32.min'); break;
      case 'max': this._emit('f32.max'); break;
      case 'rsqrt':
        this._emit('f32.sqrt');
        this._emit('(f32.const 1)');
        this._emit('f32.div');
        break;
      default:
        if (this._imports.has(node.externName)) {
          this._emit(`call $math_${node.externName}`);
        }
        break;
    }
  }

  _mathImportSig(name, argc) {
    const params = Array(argc).fill('(param f32)').join(' ');
    return `${params} (result f32)`;
  }

  _constExtent(node) {
    return node.type === 'IntImmNode' ? node.value : null;
  }

  _isZeroFillBody(body) {
    let cur = body;
    while (cur) {
      if (cur.type === 'ForNode') { cur = cur.body; continue; }
      if (cur.type === 'BlockNode') { cur = cur.body; continue; }
      if (cur.type === 'BufferStoreNode' || cur.type === 'LIRFlatStoreNode') {
        const val = cur.value;
        return (val.type === 'FloatImmNode' && val.value === 0) || (val.type === 'IntImmNode' && val.value === 0);
      }
      return false;
    }
    return false;
  }

  _layoutBuffers(primFunc) {
    let offset = 0;
    const align = 16;
    this._dynamicBuffers = new Set();
    for (const [, buf] of primFunc.bufferMap) {
      offset = Math.ceil(offset / align) * align;
      this._bufferOffsets.set(buf.name, offset);
      const numel = buf.numel();
      if (numel > 0) {
        offset += numel * wasmBytes(buf.dtype);
      } else {
        this._dynamicBuffers.add(buf.name);
        offset += 65536;
      }
    }

    const tempBuffers = new Map();
    this._collectBuffers(primFunc.body, tempBuffers);
    for (const [name, buf] of tempBuffers) {
      if (this._bufferOffsets.has(name)) continue;
      offset = Math.ceil(offset / align) * align;
      this._bufferOffsets.set(name, offset);
      const numel = buf.numel();
      if (numel > 0) {
        offset += numel * wasmBytes(buf.dtype);
      } else {
        this._dynamicBuffers.add(buf.name);
        offset += 65536;
      }
    }

    this._totalMemBytes = offset;
  }

  _collectBuffers(node, result) {
    const stack = [node];
    while (stack.length > 0) {
      const n = stack.pop();
      if (!n) continue;
      if ((n.type === 'BufferStoreNode' || n.type === 'BufferLoadNode') && n.buffer) {
        result.set(n.buffer.name, n.buffer);
      }
      if (n.body) stack.push(n.body);
      if (n.stmts) for (const s of n.stmts) stack.push(s);
      if (n.thenBody) stack.push(n.thenBody);
      if (n.elseBody) stack.push(n.elseBody);
      if (n.initBody) stack.push(n.initBody);
      if (n.value && typeof n.value === 'object' && n.value.type) stack.push(n.value);
      if (n.reads) for (const r of n.reads) if (r.buffer) result.set(r.buffer.name, r.buffer);
      if (n.writes) for (const w of n.writes) if (w.buffer) result.set(w.buffer.name, w.buffer);
    }
  }

  _scanMathImports(node) {
    const stack = [node];
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
      if (n.body) stack.push(n.body);
      if (n.stmts) for (const s of n.stmts) stack.push(s);
      if (n.thenBody) stack.push(n.thenBody);
      if (n.elseBody) stack.push(n.elseBody);
      if (n.initBody) stack.push(n.initBody);
      if (n.value && typeof n.value === 'object' && n.value.type) stack.push(n.value);
      if (n.a && typeof n.a === 'object') stack.push(n.a);
      if (n.b && typeof n.b === 'object') stack.push(n.b);
      if (n.expr && typeof n.expr === 'object') stack.push(n.expr);
      if (n.args) for (const a of n.args) if (typeof a === 'object') stack.push(a);
      if (n.indices) for (const idx of n.indices) if (typeof idx === 'object') stack.push(idx);
      if (n.condition && typeof n.condition === 'object') stack.push(n.condition);
    }
  }

  _prescanLocals(node) {
    this._waccCounter = 0;
    const stack = [node];
    while (stack.length > 0) {
      const n = stack.pop();
      if (!n) continue;
      if (n.type === 'ForNode') {
        this._ensureLocal(n.loopVar.name, 'i32');
        const accDtype = this._accPatternDtype(n);
        if (accDtype) {
          this._ensureLocal('_wacc_' + (++this._waccCounter), wasmType(accDtype));
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
      if (n.body) stack.push(n.body);
      if (n.stmts) for (const s of n.stmts) stack.push(s);
      if (n.thenBody) stack.push(n.thenBody);
      if (n.elseBody) stack.push(n.elseBody);
      if (n.initBody) stack.push(n.initBody);
    }
    this._waccCounter = 0;
  }

  _wasmExprDtype(node) {
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

  _fixLetStmtLocals(node) {
    const stack = [node];
    while (stack.length > 0) {
      const n = stack.pop();
      if (!n) continue;
      if (n.type === 'LetStmtNode' && n.variable && n.value) {
        const valDtype = inferDtype(n.value);
        if (valDtype) this._locals.set(n.variable.name, wasmType(valDtype));
      }
      if (n.body) stack.push(n.body);
      if (n.stmts) for (const s of n.stmts) stack.push(s);
      if (n.thenBody) stack.push(n.thenBody);
      if (n.elseBody) stack.push(n.elseBody);
      if (n.initBody) stack.push(n.initBody);
    }
  }

  _accPatternDtype(forNode) {
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

  _inferDtype(node) {
    return (node && node._dtype) || inferDtype(node);
  }

  _detectWasmAcc(forNode) {
    let block = forNode.body;
    if (!block || block.type !== 'BlockNode') return null;
    const inner = block.body;
    if (!inner || inner.type !== 'BufferStoreNode') return null;
    const store = inner;
    const val = store.value;
    if (!val || val.type !== 'MathOpNode' || val.op !== '+') return null;
    let loadSide = null;
    if (val.a && val.a.type === 'BufferLoadNode' && val.a.buffer.name === store.buffer.name) loadSide = val.a;
    else if (val.b && val.b.type === 'BufferLoadNode' && val.b.buffer.name === store.buffer.name) loadSide = val.b;
    if (!loadSide) return null;
    const storeKey = this._indicesKey(store.buffer, store.indices);
    const loadKey = this._indicesKey(loadSide.buffer, loadSide.indices);
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

  _isAccTarget(buffer, indices) {
    if (!this._wasmAcc) return false;
    if (buffer.name !== this._wasmAcc.bufName) return false;
    return this._indicesKey(buffer, indices) === this._indicesKey({ name: this._wasmAcc.bufName, shape: buffer.shape, strides: buffer.strides }, this._wasmAcc.indices);
  }

  _indicesKey(buffer, indices) {
    const parts = [];
    for (let i = 0; i < indices.length; i++) {
      parts.push(this._exprKey(indices[i]));
    }
    return buffer.name + ':' + parts.join(',');
  }

  _exprKey(node) {
    if (!node) return '?';
    if (node.type === 'VariableNode') return '$' + node.name;
    if (node.type === 'IntImmNode') return String(node.value);
    if (node.type === 'MathOpNode') return '(' + this._exprKey(node.a) + node.op + (node.b ? this._exprKey(node.b) : '') + ')';
    return '?';
  }
}
