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
    this._primFunc = primFunc;

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
            parts.push(typeof buf.shape[d] === 'number' && buf.shape[d] >= 0 ? String(buf.shape[d]) : this._resolveShapeParam(buf, d));
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

  _scanTree(root, usedBuffers, allocatedBuffers) {
    const stack = [root];
    while (stack.length > 0) {
      const node = stack.pop();
      if (!node || typeof node !== 'object') continue;
      switch (node.type) {
        case 'BufferStoreNode':
        case 'BufferLoadNode':
          if (node.buffer) usedBuffers.set(node.buffer.name, node.buffer);
          break;
        case 'AllocateNode':
          if (node.buffer) allocatedBuffers.add(node.buffer.name);
          break;
      }
      if (node.body) stack.push(node.body);
      if (node.value && typeof node.value === 'object' && node.value.type) stack.push(node.value);
      if (node.stmts) for (const s of node.stmts) stack.push(s);
      if (node.thenBody) stack.push(node.thenBody);
      if (node.elseBody) stack.push(node.elseBody);
      if (node.initBody) stack.push(node.initBody);
      if (node.condition && typeof node.condition === 'object' && node.condition.type) stack.push(node.condition);
      if (node.a && typeof node.a === 'object' && node.a.type) stack.push(node.a);
      if (node.b && typeof node.b === 'object' && node.b.type) stack.push(node.b);
      if (node.expr && typeof node.expr === 'object' && node.expr.type) stack.push(node.expr);
      if (node.args) for (const a of node.args) { if (typeof a === 'object' && a !== null && a.type) stack.push(a); }
      if (node.indices) for (const idx of node.indices) { if (typeof idx === 'object' && idx !== null && idx.type) stack.push(idx); }
      if (node.reads) for (const r of node.reads) { if (r.buffer) usedBuffers.set(r.buffer.name, r.buffer); }
      if (node.writes) for (const w of node.writes) { if (w.buffer) usedBuffers.set(w.buffer.name, w.buffer); }
      if (node.iterVars) for (const iv of node.iterVars) { if (iv.binding && typeof iv.binding === 'object' && iv.binding.type) stack.push(iv.binding); }
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
        case 'AllocateNode': {
          const buf = node.buffer;
          const numel = buf.numel();
          if (numel > 0) {
            this._emit(`const ${buf.name} = new ${jsTypedArray(buf.dtype)}(${numel});`);
          } else if (numel < 0) {
            this._emit(`const ${buf.name} = new ${jsTypedArray(buf.dtype)}(${this._dynamicNumel(buf)});`);
          }
          node = node.body;
          continue;
        }
        case 'LetStmtNode':
          this._emit(`const ${node.variable.name} = ${this._exprToJS(node.value)};`);
          node = node.body;
          continue;
        case 'ForNode': this._visitForNode(node); return;
        case 'BlockNode': this._visitBlockNode(node); return;
        case 'IfThenElseNode': this._visitIfThenElseStmt(node); return;
        case 'BufferStoreNode': this._visitBufferStoreNode(node); return;
        case 'WhileNode': this._visitWhileNode(node); return;
        case 'EvaluateNode': return;
        default: return;
      }
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

  _exprToJS(root) {
    if (!root) return '0';
    const vals = [];
    const work = [{ node: root, phase: 0 }];

    while (work.length > 0) {
      const top = work[work.length - 1];
      const node = top.node;

      if (!node) { work.pop(); vals.push('0'); continue; }

      switch (node.type) {
        case 'IntImmNode': work.pop(); vals.push(String(node.value)); continue;
        case 'FloatImmNode': work.pop(); vals.push(String(node.value)); continue;
        case 'VariableNode': work.pop(); vals.push(node.name); continue;
        case 'BufferLoadNode': work.pop(); vals.push(`${node.buffer.name}[${this._flatIndex(node.buffer, node.indices)}]`); continue;

        case 'MathOpNode':
          if (top.phase === 0) { top.phase = 1; work.push({ node: node.a, phase: 0 }); }
          else if (top.phase === 1 && node.b) { top.phase = 2; work.push({ node: node.b, phase: 0 }); }
          else {
            work.pop();
            if (!node.b) { vals.push(`(${node.op}${vals.pop()})`); }
            else {
              const b = vals.pop(), a = vals.pop();
              if (node.op === '%') vals.push(`((${a} % ${b} + ${b}) % ${b})`);
              else if (node.op === '//') vals.push(`((${a} / ${b}) | 0)`);
              else vals.push(`(${a} ${node.op} ${b})`);
            }
          }
          continue;

        case 'CompareNode':
          if (top.phase === 0) { top.phase = 1; work.push({ node: node.a, phase: 0 }); }
          else if (top.phase === 1) { top.phase = 2; work.push({ node: node.b, phase: 0 }); }
          else { work.pop(); const b = vals.pop(), a = vals.pop(); vals.push(`(${a} ${node.toJS()} ${b})`); }
          continue;

        case 'IfThenElseNode':
          if (top.phase === 0) { top.phase = 1; work.push({ node: node.condition, phase: 0 }); }
          else if (top.phase === 1) { top.phase = 2; work.push({ node: node.thenBody, phase: 0 }); }
          else if (top.phase === 2) { top.phase = 3; work.push({ node: node.elseBody, phase: 0 }); }
          else { work.pop(); const e = vals.pop(), t = vals.pop(), c = vals.pop(); vals.push(`(${c} ? ${t} : ${e})`); }
          continue;

        case 'CastNode':
          if (top.phase === 0) { top.phase = 1; work.push({ node: node.expr, phase: 0 }); }
          else {
            work.pop(); const inner = vals.pop();
            if (node.toDtype === 'bool') vals.push(`(${inner} ? 1 : 0)`);
            else if (isDtypeInt(node.toDtype)) vals.push(`(${inner} | 0)`);
            else vals.push(`(+${inner})`);
          }
          continue;

        case 'CallExternNode':
          if (top.phase < node.args.length) { const p = top.phase; top.phase++; work.push({ node: node.args[p], phase: 0 }); }
          else {
            work.pop();
            const args = [];
            for (let i = 0; i < node.args.length; i++) args.unshift(vals.pop());
            const joined = args.join(', ');
            if (isJSMathFunc(node.externName)) vals.push(`Math.${node.externName}(${joined})`);
            else if (node.externName === 'rsqrt') vals.push(`(1.0 / Math.sqrt(${joined}))`);
            else if (node.externName === 'fmod') vals.push(`((${args[0]} % ${args[1]} + ${args[1]}) % ${args[1]})`);
            else vals.push(`${node.externName}(${joined})`);
          }
          continue;

        default: work.pop(); vals.push('0'); continue;
      }
    }

    return vals.length > 0 ? vals[0] : '0';
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

  _flatIndex(buffer, indices) {
    if (indices.length === 0) return '0';
    if (indices.length === 1) return this._exprToJS(indices[0]);
    const parts = new Array(indices.length);
    for (let i = 0; i < indices.length; i++) {
      const idx = this._exprToJS(indices[i]);
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
      if (v) return v.name;
    }
    return '1';
  }
}
