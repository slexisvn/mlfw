import { ForKind } from '../../ir/tensor/nodes.js';
import { jsTypedArray, isJSMathFunc, isDtypeInt } from '../dtype_map.js';

export class CPUCodegen {
  constructor(target) {
    this.target = target;
    this._indent = 0;
    this._lines = [];
    this._loopStack = [];
  }

  generate(primFunc) {
    this._indent = 0;
    this._lines = [];

    const paramBuffers = new Set();
    const paramNames = [];
    for (const [, buf] of primFunc.bufferMap) {
      paramNames.push(buf.name);
      paramBuffers.add(buf.name);
    }
    for (const sp of primFunc.shapeParams) {
      paramNames.push(sp.name);
    }
    this._paramBuffers = paramBuffers;

    const usedBuffers = new Map();
    const allocatedBuffers = new Set();
    this._scanTree(primFunc.body, usedBuffers, allocatedBuffers);

    this._emit(`function ${primFunc.name}(${paramNames.join(', ')}) {`);
    this._indent++;

    for (const [bufName, buf] of usedBuffers) {
      if (!paramBuffers.has(bufName) && !allocatedBuffers.has(bufName)) {
        const numel = buf.numel();
        if (numel > 0) {
          this._emit(`const ${bufName} = new ${jsTypedArray(buf.dtype)}(${numel});`);
        } else if (numel < 0) {
          const parts = [];
          for (let d = 0; d < buf.shape.length; d++) {
            parts.push(typeof buf.shape[d] === 'number' && buf.shape[d] >= 0 ? String(buf.shape[d]) : `/* dyn */1`);
          }
          this._emit(`const ${bufName} = new ${jsTypedArray(buf.dtype)}(${parts.join(' * ')});`);
        }
      }
    }

    this._visitNode(primFunc.body);
    this._indent--;
    this._emit('}');

    return this._lines.join('\n');
  }

  _scanTree(node, usedBuffers, allocatedBuffers) {
    if (!node || typeof node !== 'object') return;
    switch (node.type) {
      case 'BufferStoreNode':
      case 'BufferLoadNode':
        if (node.buffer) usedBuffers.set(node.buffer.name, node.buffer);
        break;
      case 'AllocateNode':
        if (node.buffer) allocatedBuffers.add(node.buffer.name);
        break;
    }
    if (node.body) this._scanTree(node.body, usedBuffers, allocatedBuffers);
    if (node.value && typeof node.value === 'object' && node.value.type) this._scanTree(node.value, usedBuffers, allocatedBuffers);
    if (node.stmts) for (const s of node.stmts) this._scanTree(s, usedBuffers, allocatedBuffers);
    if (node.thenBody) this._scanTree(node.thenBody, usedBuffers, allocatedBuffers);
    if (node.elseBody) this._scanTree(node.elseBody, usedBuffers, allocatedBuffers);
    if (node.initBody) this._scanTree(node.initBody, usedBuffers, allocatedBuffers);
    if (node.condition && typeof node.condition === 'object' && node.condition.type) this._scanTree(node.condition, usedBuffers, allocatedBuffers);
    if (node.a && typeof node.a === 'object' && node.a.type) this._scanTree(node.a, usedBuffers, allocatedBuffers);
    if (node.b && typeof node.b === 'object' && node.b.type) this._scanTree(node.b, usedBuffers, allocatedBuffers);
    if (node.expr && typeof node.expr === 'object' && node.expr.type) this._scanTree(node.expr, usedBuffers, allocatedBuffers);
    if (node.args) for (const a of node.args) { if (typeof a === 'object' && a !== null && a.type) this._scanTree(a, usedBuffers, allocatedBuffers); }
    if (node.indices) for (const idx of node.indices) { if (typeof idx === 'object' && idx !== null && idx.type) this._scanTree(idx, usedBuffers, allocatedBuffers); }
    if (node.reads) for (const r of node.reads) { if (r.buffer) usedBuffers.set(r.buffer.name, r.buffer); }
    if (node.writes) for (const w of node.writes) { if (w.buffer) usedBuffers.set(w.buffer.name, w.buffer); }
    if (node.iterVars) for (const iv of node.iterVars) { if (iv.binding && typeof iv.binding === 'object' && iv.binding.type) this._scanTree(iv.binding, usedBuffers, allocatedBuffers); }
  }

  _emit(line) {
    this._lines.push('  '.repeat(this._indent) + line);
  }

  _visitNode(node) {
    if (!node) return;
    switch (node.type) {
      case 'SeqNode': for (const s of node.stmts) this._visitNode(s); return;
      case 'ForNode': this._visitForNode(node); return;
      case 'BlockNode': this._visitBlockNode(node); return;
      case 'AllocateNode': this._visitAllocateNode(node); return;
      case 'IfThenElseNode': this._visitIfThenElseStmt(node); return;
      case 'LetStmtNode': this._visitLetStmtNode(node); return;
      case 'BufferStoreNode': this._visitBufferStoreNode(node); return;
      case 'WhileNode': this._visitWhileNode(node); return;
      case 'EvaluateNode': return;
      default: return;
    }
  }

  _visitForNode(node) {
    const varName = node.loopVar.name;
    const extent = this._exprToJS(node.extent);

    if (node.kind === ForKind.UNROLLED) {
      const constExtent = node.extent.type === 'IntImmNode' ? node.extent.value : null;
      if (constExtent && constExtent <= 16) {
        for (let i = 0; i < constExtent; i++) {
          this._emit(`{ const ${varName} = ${i};`);
          this._indent++;
          this._loopStack.push(varName);
          this._visitNode(node.body);
          this._loopStack.pop();
          this._indent--;
          this._emit('}');
        }
        return;
      }
    }

    this._emit(`for (let ${varName} = 0; ${varName} < ${extent}; ${varName}++) {`);
    this._indent++;
    this._loopStack.push(varName);
    this._visitNode(node.body);
    this._loopStack.pop();
    this._indent--;
    this._emit('}');
  }

  _visitBlockNode(node) {
    for (const bind of node.iterVars) {
      if (bind.iterVar && bind.binding) {
        this._emit(`const ${bind.iterVar.name} = ${this._exprToJS(bind.binding)};`);
      }
    }
    if (node.initBody) {
      const reductionVar = this._loopStack.length > 0 ? this._loopStack[this._loopStack.length - 1] : null;
      if (reductionVar) {
        this._emit(`if (${reductionVar} === 0) {`);
        this._indent++;
        this._visitNode(node.initBody);
        this._indent--;
        this._emit('}');
      } else {
        this._visitNode(node.initBody);
      }
    }
    this._visitNode(node.body);
  }

  _visitAllocateNode(node) {
    const buf = node.buffer;
    const numel = buf.numel();
    if (numel > 0) {
      this._emit(`const ${buf.name} = new ${jsTypedArray(buf.dtype)}(${numel});`);
    }
    this._visitNode(node.body);
  }

  _visitIfThenElseStmt(node) {
    this._emit(`if (${this._exprToJS(node.condition)}) {`);
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
    this._emit(`const ${node.variable.name} = ${this._exprToJS(node.value)};`);
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
    this._emit(`${node.buffer.name}[${this._flatIndex(node.buffer, node.indices)}] = ${this._exprToJS(node.value)};`);
  }

  _exprToJS(node) {
    if (!node) return '0';
    switch (node.type) {
      case 'IntImmNode': return String(node.value);
      case 'FloatImmNode': return String(node.value);
      case 'VariableNode': return node.name;
      case 'BufferLoadNode': return `${node.buffer.name}[${this._flatIndex(node.buffer, node.indices)}]`;
      case 'MathOpNode': {
        const a = this._exprToJS(node.a);
        if (!node.b) return `(${node.op}${a})`;
        const b = this._exprToJS(node.b);
        if (node.op === '%') return `((${a} % ${b} + ${b}) % ${b})`;
        if (node.op === '//') return `((${a} / ${b}) | 0)`;
        return `(${a} ${node.op} ${b})`;
      }
      case 'CompareNode': return `(${this._exprToJS(node.a)} ${node.toJS()} ${this._exprToJS(node.b)})`;
      case 'IfThenElseNode': return `(${this._exprToJS(node.condition)} ? ${this._exprToJS(node.thenBody)} : ${this._exprToJS(node.elseBody)})`;
      case 'CastNode': return this._emitCastJS(node);
      case 'CallExternNode': return this._emitExternCall(node);
      default: return '0';
    }
  }

  _emitCastJS(node) {
    const inner = this._exprToJS(node.expr);
    if (node.toDtype === 'bool') return `(${inner} ? 1 : 0)`;
    if (isDtypeInt(node.toDtype)) return `(${inner} | 0)`;
    return `(+${inner})`;
  }

  _emitExternCall(node) {
    const jsArgs = new Array(node.args.length);
    for (let i = 0; i < node.args.length; i++) jsArgs[i] = this._exprToJS(node.args[i]);
    const joined = jsArgs.join(', ');
    if (isJSMathFunc(node.externName)) return `Math.${node.externName}(${joined})`;
    if (node.externName === 'rsqrt') return `(1.0 / Math.sqrt(${joined}))`;
    if (node.externName === 'fmod') return `((${jsArgs[0]} % ${jsArgs[1]} + ${jsArgs[1]}) % ${jsArgs[1]})`;
    return `${node.externName}(${joined})`;
  }

  _flatIndex(buffer, indices) {
    if (indices.length === 0) return '0';
    if (indices.length === 1) return this._exprToJS(indices[0]);
    const parts = new Array(indices.length);
    for (let i = 0; i < indices.length; i++) {
      const idx = this._exprToJS(indices[i]);
      parts[i] = buffer.strides[i] === 1 ? idx : `${idx} * ${buffer.strides[i]}`;
    }
    return parts.join(' + ');
  }
}
