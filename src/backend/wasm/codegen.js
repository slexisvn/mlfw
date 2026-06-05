import { ForKind } from '../../compiler/ir/tensor/nodes.js';
import { wasmType, wasmLoad, wasmStore, wasmBytes, isDtypeFloat, isDtypeInt } from '../dtype_map.js';

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

  generate(primFunc) {
    this._lines = [];
    this._indent = 0;
    this._locals.clear();
    this._localCounter = 0;
    this._imports.clear();
    this._bufferOffsets.clear();
    this._totalMemBytes = 0;
    this._primFunc = primFunc;

    this._layoutBuffers(primFunc);
    this._scanMathImports(primFunc.body);

    const paramNames = [];
    for (const [, buf] of primFunc.bufferMap) {
      paramNames.push(buf.name);
    }
    for (const sp of primFunc.shapeParams) {
      paramNames.push(sp.name);
    }

    this._emit('(module');
    this._indent++;

    const memPages = Math.max(1, Math.ceil(this._totalMemBytes / 65536));
    this._emit(`(memory (export "memory") ${memPages} 256)`);

    for (const [name, sig] of this._imports) {
      this._emit(`(import "math" "${name}" (func $math_${name} ${sig}))`);
    }

    const paramDecls = [];
    for (const [, buf] of primFunc.bufferMap) {
      paramDecls.push('(param i32)');
    }
    const shapeParamEntries = [];
    for (const sp of primFunc.shapeParams) {
      paramDecls.push('(param i32)');
      this._ensureLocal(sp.name, 'i32');
      shapeParamEntries.push(sp.name);
    }
    this._emit('(func (export "' + primFunc.name + '") ' + paramDecls.join(' '));
    this._indent++;

    const localDecls = [];
    this._prescanLocals(primFunc.body);
    for (const [name, type] of this._locals) {
      localDecls.push('(local $' + name + ' ' + type + ')');
    }
    if (localDecls.length > 0) this._emit(localDecls.join(' '));

    const bufCount = primFunc.bufferMap.size;
    for (let i = 0; i < shapeParamEntries.length; i++) {
      this._emit('(local.get ' + (bufCount + i) + ')');
      this._emit('local.set $' + shapeParamEntries[i]);
    }

    this._visitNode(primFunc.body);

    this._indent--;
    this._emit(')');
    this._indent--;
    this._emit(')');

    return {
      name: primFunc.name,
      wat: this._lines.join('\n'),
      memoryPages: memPages,
      bufferOffsets: new Map(this._bufferOffsets),
      imports: this._imports,
      params: paramNames,
    };
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

  _mathImportSig(name, argc) {
    const params = Array(argc).fill('(param f32)').join(' ');
    return `${params} (result f32)`;
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
        this._ensureLocal(n.variable.name, wasmType(n.variable.dtype || this._defaultDtype));
      }
      if (n.body) stack.push(n.body);
      if (n.stmts) for (const s of n.stmts) stack.push(s);
      if (n.thenBody) stack.push(n.thenBody);
      if (n.elseBody) stack.push(n.elseBody);
      if (n.initBody) stack.push(n.initBody);
    }
    this._waccCounter = 0;
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
        case 'LetStmtNode':
          this._emitExpr(node.value);
          this._emit(`local.set $${node.variable.name}`);
          node = node.body;
          continue;
        case 'ForNode': this._visitFor(node); return;
        case 'BlockNode': this._visitBlock(node); return;
        case 'IfThenElseNode': this._visitIf(node); return;
        case 'BufferStoreNode': this._visitStore(node); return;
        case 'WhileNode': this._visitWhile(node); return;
        case 'EvaluateNode': return;
        default: return;
      }
    }
  }

  _visitFor(node) {
    const varName = node.loopVar.name;
    const extent = this._constExtent(node.extent);

    if ((node.kind === ForKind.UNROLLED || node.kind === ForKind.VECTORIZED) && extent !== null && extent <= 32) {
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
        const resultIsFloat = isDtypeFloat(this._inferDtype(node.thenBody));
        const resultType = resultIsFloat ? 'f32' : 'i32';
        this._emitExpr(node.condition);
        const condDtype = this._inferDtype(node.condition);
        if (isDtypeFloat(condDtype)) {
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
    if (targetFloat && this._inferDtype(child) === 'i32') {
      this._emit('f32.convert_i32_s');
    }
  }

  _emitMathOp(node) {
    const da = this._inferDtype(node.a);
    const db = node.b ? this._inferDtype(node.b) : da;
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
      }
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
    const da = this._inferDtype(node.a);
    const db = this._inferDtype(node.b);
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

  _inferDtype(node) {
    if (!node) return this._defaultDtype;
    if (node.type === 'IntImmNode') return 'i32';
    if (node.type === 'FloatImmNode') return 'f32';
    if (node.type === 'BufferLoadNode') return node.buffer.dtype;
    if (node.type === 'CastNode') return node.toDtype;
    if (node.type === 'CallExternNode') return node.dtype || this._defaultDtype;
    if (node.type === 'CompareNode') return 'i32';
    if (node.type === 'VariableNode') {
      const d = node.dtype || 'i32';
      if (d === 'int32' || d === 'index') return 'i32';
      if (d === 'float32') return 'f32';
      return d;
    }
    if (node.type === 'MathOpNode') {
      const da = this._inferDtype(node.a);
      if (isDtypeFloat(da)) return da;
      if (node.b) {
        const db = this._inferDtype(node.b);
        if (isDtypeFloat(db)) return db;
      }
      return da;
    }
    if (node.type === 'IfThenElseNode') return this._inferDtype(node.thenBody);
    return this._defaultDtype;
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

  _constExtent(node) {
    return node.type === 'IntImmNode' ? node.value : null;
  }
}
