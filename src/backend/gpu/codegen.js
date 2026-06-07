import { ForKind } from '../../compiler/ir/tensor/nodes.js';
import { cType, cPtrType, cLiteralSuffix, cMathFunc } from '../dtype_map.js';
import { inferDtype } from '../../compiler/ir/lir/nodes.js';

export class GPUKernel {
  constructor(name, source, blockDim, gridDim, sharedMemBytes, params) {
    this.name = name;
    this.source = source;
    this.blockDim = blockDim;
    this.gridDim = gridDim;
    this.sharedMemBytes = sharedMemBytes;
    this.params = params;
  }
}

export class GPUCodegen {
  constructor(target) {
    this.target = target;
    this._indent = 0;
    this._lines = [];
    this._threadBindings = new Map();
    this._sharedBuffers = [];
    this._blockDim = [1, 1, 1];
    this._gridDim = [1, 1, 1];
    this._defaultDtype = 'f32';
  }

  generate(func) {
    this._indent = 0;
    this._lines = [];
    this._threadBindings.clear();
    this._sharedBuffers = [];
    this._blockDim = [1, 1, 1];
    this._gridDim = [1, 1, 1];
    this._primFunc = func;

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

    const paramParts = [];
    const paramNames = [];
    for (const [, buf] of func.bufferMap) {
      paramNames.push(buf.name);
      paramParts.push(`${cPtrType(buf.dtype)} ${buf.name}`);
      this._defaultDtype = buf.dtype;
    }
    for (const sp of func.shapeParams) {
      paramNames.push(sp.name);
      paramParts.push(`int ${sp.name}`);
    }

    this._emit(`__global__ void ${func.name}(${paramParts.join(', ')}) {`);
    this._indent++;

    for (const buf of this._sharedBuffers) {
      const numel = buf.numel();
      this._emit(`__shared__ ${cType(buf.dtype)} ${buf.name}[${numel > 0 ? numel : 1}];`);
    }

    const declaredVars = new Set();
    for (const [tag, bindings] of this._threadBindings) {
      for (const info of bindings) {
        if (!declaredVars.has(info.varName)) {
          this._emit(`const int ${info.varName} = ${tag};`);
          declaredVars.add(info.varName);
        }
      }
    }

    this._visitNode(func.body);
    this._indent--;
    this._emit('}');

    const t = this.target;
    const blockDim = [
      Math.min(this._blockDim[0], t.maxBlockDimX),
      Math.min(this._blockDim[1], t.maxBlockDimY),
      Math.min(this._blockDim[2], t.maxBlockDimZ),
    ];
    const gridDim = [
      Math.min(this._gridDim[0], t.maxGridDimX),
      Math.min(this._gridDim[1], t.maxGridDimY),
      Math.min(this._gridDim[2], t.maxGridDimZ),
    ];

    return new GPUKernel(
      func.name,
      this._lines.join('\n'),
      blockDim, gridDim,
      this._sharedBuffers.reduce((sum, b) => sum + Math.max(b.sizeInBytes(), 0), 0),
      paramNames
    );
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
    if (prefix === 'threadIdx') this._blockDim[axis] = Math.max(this._blockDim[axis], extent);
    else if (prefix === 'blockIdx') this._gridDim[axis] = Math.max(this._gridDim[axis], extent);
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
      this._visitNode(node.body);
      return;
    }
    const varName = node.loopVar.name;
    const extent = this._exprToC(node.extent);
    if (node.kind === ForKind.UNROLLED) this._emit(`#pragma unroll`);
    this._emit(`for (int ${varName} = 0; ${varName} < ${extent}; ${varName}++) {`);
    this._indent++;
    this._visitNode(node.body);
    this._indent--;
    this._emit('}');
  }

  _visitBlockNode(node) {
    for (const bind of node.iterVars) {
      if (bind.iterVar && bind.binding) {
        this._emit(`const int ${bind.iterVar.name} = ${this._exprToC(bind.binding)};`);
      }
    }
    if (node.initBody) this._visitNode(node.initBody);
    this._visitNode(node.body);
  }

  _visitAllocateNode(node) {
    if (node.scope !== 'shared') {
      const numel = node.buffer.numel();
      if (numel > 0) {
        this._emit(`${cType(node.buffer.dtype)} ${node.buffer.name}[${numel}];`);
      } else {
        this._emit(`${cType(node.buffer.dtype)}* ${node.buffer.name} = (${cType(node.buffer.dtype)}*)alloca(${this._dynamicNumel(node.buffer)} * sizeof(${cType(node.buffer.dtype)}));`);
      }
    }
  }

  _visitIfStmt(node) {
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

  _visitLetStmtNode(node) {
    this._emit(`${cType(this._defaultDtype)} ${node.variable.name} = ${this._exprToC(node.value)};`);
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
    this._emit(`${node.buffer.name}[${this._flatIndex(node.buffer, node.indices)}] = ${this._exprToC(node.value)};`);
  }

  _visitLIRFlatStore(node) {
    this._emit(`${node.buffer.name}[${this._exprToC(node.offsetExpr)}] = ${this._exprToC(node.value)};`);
  }

  _visitLIRBindings(node) {
    for (const bind of node.bindings) {
      this._emit(`const int ${bind.name} = ${this._exprToC(bind.expr)};`);
    }
    this._visitNode(node.body);
  }

  _visitLIRAccumulator(node) {
    const accVar = node.localName;
    const dtype = node.dtype || this._defaultDtype;
    this._emit(`${cType(dtype)} ${accVar} = ${this._exprToC(node.initLoad)};`);

    const varName = node.loopVar.name;
    const extent = this._exprToC(node.extent);
    this._emit(`for (int ${varName} = 0; ${varName} < ${extent}; ${varName}++) {`);
    this._indent++;
    this._emit(`${accVar} = (${accVar} + ${this._exprToC(node.body)});`);
    this._indent--;
    this._emit('}');

    this._emit(`${node.flushStore.buffer.name}[${this._exprToC(node.flushStore.offsetExpr)}] = ${accVar};`);
  }

  _exprToC(node) {
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
        if (node.op === '//') return `(${a} / ${b})`;
        return `(${a} ${node.op} ${b})`;
      }
      case 'CompareNode': return `(${this._exprToC(node.a)} ${node.toC()} ${this._exprToC(node.b)})`;
      case 'IfThenElseNode': return `(${this._exprToC(node.condition)} ? ${this._exprToC(node.thenBody)} : ${this._exprToC(node.elseBody)})`;
      case 'CastNode': return `((${cType(node.toDtype)})(${this._exprToC(node.expr)}))`;
      case 'CallExternNode': return this._emitExternCall(node);
      default: return '0';
    }
  }

  _emitFloatLiteral(value) {
    if (value === Infinity) return 'INFINITY';
    if (value === -Infinity) return '(-INFINITY)';
    const suffix = cLiteralSuffix(this._defaultDtype);
    return `${value}${suffix}`;
  }

  _emitExternCall(node) {
    const n = node.args.length;
    const args = new Array(n);
    for (let i = 0; i < n; i++) args[i] = this._exprToC(node.args[i]);
    const joined = args.join(', ');
    const dtype = node.dtype || this._defaultDtype;
    if (node.externName === 'rsqrt') return `${cMathFunc('rsqrt', dtype) || 'rsqrtf'}(${joined})`;
    if (node.externName === 'sign') {
      const v = args[0];
      const zero = `0.0${cLiteralSuffix(dtype)}`;
      return `((${v} > ${zero}) - (${v} < ${zero}))`;
    }
    const fn = cMathFunc(node.externName, dtype);
    return `${fn}(${joined})`;
  }

  _flatIndex(buffer, indices) {
    if (indices.length === 0) return '0';
    if (indices.length === 1) return this._exprToC(indices[0]);
    const parts = new Array(indices.length);
    for (let i = 0; i < indices.length; i++) {
      const idx = this._exprToC(indices[i]);
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

  _dynamicNumel(buffer) {
    const parts = [];
    for (let d = 0; d < buffer.shape.length; d++) {
      const dim = buffer.shape[d];
      if (typeof dim === 'number' && dim >= 0) {
        parts.push(String(dim));
      } else {
        parts.push(this._resolveShapeParam(buffer, d));
      }
    }
    return parts.length === 0 ? '1' : parts.join(' * ');
  }

  _resolveShapeParam(buffer, dimIdx) {
    if (this._primFunc && this._primFunc.shapeParamMap) {
      const key = `${buffer.name}:${dimIdx}`;
      const v = this._primFunc.shapeParamMap.get(key);
      if (v) return v.name;
    }
    return '1';
  }
}
