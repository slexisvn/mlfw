import { ForKind } from '../../compiler/ir/tensor/nodes.js';
import { jsTypedArray, isJSMathFunc, isDtypeInt } from '../dtype_map.js';

import '../../tensor/utils/half.js';

const _HALF_EXPAND = { f16: '__mlfw_f16_to_f32', bf16: '__mlfw_bf16_to_f32' };
const _HALF_ROUND = { f16: '__mlfw_f32_to_f16', bf16: '__mlfw_f32_to_bf16' };

export class CPUCodegen {
  constructor(target) {
    this.target = target;
    this._indent = 0;
    this._lines = [];
    this._loopStack = [];
  }

  generate(func) {
    this._indent = 0;
    this._lines = [];
    this._aliases = new Map();
    this._accTarget = null;
    this._accVar = null;
    this._accCounter = 0;

    const isLIR = func.type === 'LIRFunc';

    const paramBuffers = new Set();
    const paramNames = [];
    for (const [, buf] of func.bufferMap) {
      paramNames.push(buf.name);
      paramBuffers.add(buf.name);
    }
    for (const sp of func.shapeParams) {
      paramNames.push(sp.name);
    }
    this._paramBuffers = paramBuffers;

    let usedBuffers, allocatedBuffers, zeroBuffers, constantBuffers;

    if (isLIR) {
      usedBuffers = func.metadata.usedBuffers;
      allocatedBuffers = func.metadata.allocatedBuffers;
      zeroBuffers = func.metadata.zeroBuffers;
      constantBuffers = func.metadata.constantBuffers;
    } else {
      usedBuffers = new Map();
      allocatedBuffers = new Set();
      this._scanTree(func.body, usedBuffers, allocatedBuffers);
      zeroBuffers = this._findZeroOnlyBuffers(func.body, paramBuffers);
      constantBuffers = this._constantBuffers;
    }

    this._zeroBuffers = zeroBuffers;
    this._constantBuffers = constantBuffers;
    this._localBuffers = new Set();
    this._primFunc = func;

    this._emit(`function ${func.name}(${paramNames.join(', ')}) {`);
    this._indent++;

    for (const [bufName, buf] of usedBuffers) {
      if (zeroBuffers.has(bufName)) continue;
      if (constantBuffers.has(bufName)) continue;
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
    for (const [, buf] of usedBuffers) {
      if (!paramBuffers.has(buf.name)) {
        this._localBuffers.add(buf.name);
      }
    }

    this._visitNode(func.body);
    this._indent--;
    this._emit('}');

    return this._cleanupSource(this._lines.join('\n'));
  }

  _emit(line) {
    this._lines.push('  '.repeat(this._indent) + line);
  }

  _wrapLoad(dtype, expr) {
    const fn = _HALF_EXPAND[dtype];
    return fn ? `${fn}(${expr})` : expr;
  }

  _wrapStoreVal(dtype, expr) {
    const fn = _HALF_ROUND[dtype];
    if (fn) return `${fn}(${expr})`;
    if (dtype === 'i64') return `BigInt(${expr})`;
    return expr;
  }

  _zeroLit(dtype) {
    return dtype === 'i64' ? '0n' : '0';
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
        case 'LIRFlatStoreNode': this._visitLIRFlatStore(node); return;
        case 'LIRBindingsNode': this._visitLIRBindings(node); return;
        case 'LIRAccumulatorNode': this._visitLIRAccumulator(node); return;
        case 'WhileNode': this._visitWhileNode(node); return;
        case 'EvaluateNode': return;
        default: return;
      }
    }
  }

  _visitForNode(node) {
    if (this._isRedundantZeroFill(node)) return;

    const varName = node.loopVar.name;

    if (node.extent.type === 'IntImmNode' && node.extent.value === 1) {
      this._aliases.set(varName, '0');
      this._visitNode(node.body);
      return;
    }

    const extent = this._exprToJS(node.extent);

    if (node.kind === ForKind.UNROLLED) {
      const constExtent = node.extent.type === 'IntImmNode' ? node.extent.value : null;
      if (constExtent && constExtent <= 32 && !this._isZeroFillBody(node.body)) {
        for (let i = 0; i < constExtent; i++) {
          this._emit('{ const ' + varName + ' = ' + i + ';');
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

    const accInfo = this._detectReductionAcc(node);
    if (accInfo) {
      const accVar = '_acc_' + (this._accCounter = (this._accCounter || 0) + 1);
      const prevAcc = this._accTarget;
      const prevVar = this._accVar;
      this._accTarget = accInfo;
      this._accVar = accVar;
      this._emit('let ' + accVar + ' = ' + this._wrapLoad(accInfo.dtype, accInfo.bufName + '[' + accInfo.idxExpr + ']') + ';');
      this._emit('for (let ' + varName + ' = 0; ' + varName + ' < ' + extent + '; ' + varName + '++) {');
      this._indent++;
      this._loopStack.push(varName);
      this._visitNode(node.body);
      this._loopStack.pop();
      this._indent--;
      this._emit('}');
      this._emit(accInfo.bufName + '[' + accInfo.idxExpr + '] = ' + this._wrapStoreVal(accInfo.dtype, accVar) + ';');
      this._accTarget = prevAcc;
      this._accVar = prevVar;
      return;
    }

    this._emit('for (let ' + varName + ' = 0; ' + varName + ' < ' + extent + '; ' + varName + '++) {');
    this._indent++;
    this._loopStack.push(varName);
    this._visitNode(node.body);
    this._loopStack.pop();
    this._indent--;
    this._emit('}');
  }

  _visitLIRFlatStore(node) {
    if (this._zeroBuffers && this._zeroBuffers.has(node.buffer.name)) return;
    if (this._constantBuffers && this._constantBuffers.has(node.buffer.name)) return;
    if (this._accTarget && node.buffer.name === this._accTarget.bufName
        && this._exprToJS(node.offsetExpr) === this._accTarget.idxExpr) {
      this._emit(this._accVar + ' = ' + this._exprToJS(node.value) + ';');
      return;
    }
    this._emit(node.buffer.name + '[' + this._exprToJS(node.offsetExpr) + '] = ' + this._wrapStoreVal(node.dtype || node.buffer.dtype, this._exprToJS(node.value)) + ';');
  }

  _visitLIRBindings(node) {
    for (const bind of node.bindings) {
      const expr = this._exprToJS(bind.expr);
      this._aliases.set(bind.name, expr);
    }
    this._visitNode(node.body);
  }

  _visitLIRAccumulator(node) {
    const accVar = node.localName;
    const initExpr = this._exprToJS(node.initLoad);
    this._emit('let ' + accVar + ' = ' + initExpr + ';');

    const prevAcc = this._accTarget;
    const prevVar = this._accVar;
    this._accTarget = {
      bufName: node.flushStore.buffer.name,
      idxExpr: this._exprToJS(node.flushStore.offsetExpr),
    };
    this._accVar = accVar;

    const varName = node.loopVar.name;
    const extent = this._exprToJS(node.extent);
    this._emit('for (let ' + varName + ' = 0; ' + varName + ' < ' + extent + '; ' + varName + '++) {');
    this._indent++;
    this._loopStack.push(varName);
    this._emit(accVar + ' = (' + accVar + ' + ' + this._exprToJS(node.body) + ');');
    this._loopStack.pop();
    this._indent--;
    this._emit('}');

    this._emit(node.flushStore.buffer.name + '[' + this._accTarget.idxExpr + '] = ' + this._wrapStoreVal(node.flushStore.dtype || node.flushStore.buffer.dtype, accVar) + ';');
    this._accTarget = prevAcc;
    this._accVar = prevVar;
  }

  _detectReductionAcc(forNode) {
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

    for (const bind of block.iterVars) {
      if (bind.iterVar && bind.binding) {
        this._aliases.set(bind.iterVar.name, this._exprToJS(bind.binding));
      }
    }

    const storeIdx = this._flatIndex(store.buffer, store.indices);
    const loadIdx = this._flatIndex(loadSide.buffer, loadSide.indices);
    if (storeIdx !== loadIdx) return null;

    const loopVar = forNode.loopVar.name;
    if (storeIdx.includes(loopVar)) return null;
    return { bufName: store.buffer.name, idxExpr: storeIdx, dtype: store.buffer.dtype };
  }

  _visitBlockNode(node) {
    for (const bind of node.iterVars) {
      if (bind.iterVar && bind.binding) {
        const expr = this._exprToJS(bind.binding);
        this._aliases.set(bind.iterVar.name, expr);
      }
    }

    if (node.initBody) {
      const reductionVar = this._loopStack.length > 0 ? this._loopStack[this._loopStack.length - 1] : null;
      if (reductionVar) {
        this._emit('if (' + reductionVar + ' === 0) {');
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
    this._emit(`while (${node.condVar.name}[0]) {`);
    this._indent++;
    this._visitNode(node.loopBody);
    this._visitNode(node.condBody);
    this._indent--;
    this._emit('}');
  }

  _visitBufferStoreNode(node) {
    if (this._zeroBuffers && this._zeroBuffers.has(node.buffer.name)) return;
    if (this._constantBuffers && this._constantBuffers.has(node.buffer.name)) return;
    if (this._accTarget && node.buffer.name === this._accTarget.bufName
        && this._flatIndex(node.buffer, node.indices) === this._accTarget.idxExpr) {
      this._emit(this._accVar + ' = ' + this._exprToJS(node.value) + ';');
      return;
    }
    this._emit(node.buffer.name + '[' + this._flatIndex(node.buffer, node.indices) + '] = ' + this._wrapStoreVal(node.buffer.dtype, this._exprToJS(node.value)) + ';');
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
        case 'VariableNode': work.pop(); vals.push(this._aliases.get(node.name) || node.name); continue;
        case 'BufferLoadNode': {
          work.pop();
          if (this._zeroBuffers && this._zeroBuffers.has(node.buffer.name)) {
            vals.push(this._zeroLit(node.buffer.dtype));
          } else if (this._constantBuffers && this._constantBuffers.has(node.buffer.name)) {
            const c = this._constantBuffers.get(node.buffer.name);
            vals.push(node.buffer.dtype === 'i64' ? `BigInt(${c})` : c);
          } else if (this._accTarget && node.buffer.name === this._accTarget.bufName && this._flatIndex(node.buffer, node.indices) === this._accTarget.idxExpr) {
            vals.push(this._accVar);
          } else {
            vals.push(this._wrapLoad(node.buffer.dtype, node.buffer.name + '[' + this._flatIndex(node.buffer, node.indices) + ']'));
          }
          continue;
        }

        case 'LIRFlatLoadNode': {
          work.pop();
          if (this._zeroBuffers && this._zeroBuffers.has(node.buffer.name)) {
            vals.push(this._zeroLit(node.dtype || node.buffer.dtype));
          } else if (this._constantBuffers && this._constantBuffers.has(node.buffer.name)) {
            const c = String(this._constantBuffers.get(node.buffer.name));
            vals.push((node.dtype || node.buffer.dtype) === 'i64' ? `BigInt(${c})` : c);
          } else if (this._accTarget && node.buffer.name === this._accTarget.bufName
                     && this._exprToJS(node.offsetExpr) === this._accTarget.idxExpr) {
            vals.push(this._accVar);
          } else {
            vals.push(this._wrapLoad(node.dtype || node.buffer.dtype, node.buffer.name + '[' + this._exprToJS(node.offsetExpr) + ']'));
          }
          continue;
        }

        case 'MathOpNode':
          if (top.phase === 0) { top.phase = 1; work.push({ node: node.a, phase: 0 }); }
          else if (top.phase === 1 && node.b) { top.phase = 2; work.push({ node: node.b, phase: 0 }); }
          else {
            work.pop();
            if (!node.b) { vals.push(`(${node.op}${vals.pop()})`); }
            else {
              const b = vals.pop(), a = vals.pop();
              if ((node.op === '+' || node.op === '-') && b === '0') { vals.push(a); }
              else if (node.op === '+' && a === '0') { vals.push(b); }
              else if (node.op === '*' && (a === '0' || b === '0')) { vals.push('0'); }
              else if (node.op === '*' && b === '1') { vals.push(a); }
              else if (node.op === '*' && a === '1') { vals.push(b); }
              else if (node.op === '%') vals.push(`((${a} % ${b} + ${b}) % ${b})`);
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
            else if (_HALF_EXPAND[node.toDtype]) vals.push(`${_HALF_EXPAND[node.toDtype]}(${_HALF_ROUND[node.toDtype]}(${inner}))`);
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
            else if (node.externName === 'exp2') vals.push(`Math.pow(2, ${joined})`);
            else if (node.externName === 'erf') vals.push(`((x_erf => { const t = 1.0 / (1.0 + 0.3275911 * Math.abs(x_erf)); const p = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))); return (x_erf >= 0 ? 1 : -1) * (1.0 - p * Math.exp(-x_erf * x_erf)); })(${joined}))`);
            else if (node.externName === 'log10') vals.push(`(Math.log(${joined}) * ${1 / Math.LN10})`);
            else throw new Error(`CPU codegen: unsupported extern function "${node.externName}"`);
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
    const parts = [];
    for (let i = 0; i < indices.length; i++) {
      const idx = this._exprToJS(indices[i]);
      if (idx === '0') continue;
      const stride = buffer.strides[i];
      if (stride === 1) {
        parts.push(idx);
      } else if (typeof stride === 'number' && stride >= 0) {
        parts.push(`${idx} * ${stride}`);
      } else {
        parts.push(`${idx} * ${this._computeDynamicStride(buffer, i)}`);
      }
    }
    return parts.length === 0 ? '0' : parts.join(' + ');
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

  _cleanupSource(src) {
    const lines = src.split('\n');

    const allocName = new Array(lines.length).fill(null);
    const counts = new Map();
    for (let i = 0; i < lines.length; i++) {
      const m = lines[i].match(/^\s*const (\w+) = new \w+Array\(\d+\);\s*$/);
      if (m) allocName[i] = m[1];
      const ids = lines[i].match(/[A-Za-z_]\w*/g);
      if (ids) for (const id of ids) counts.set(id, (counts.get(id) || 0) + 1);
    }

    const out = [];
    for (let i = 0; i < lines.length; i++) {
      if (allocName[i] !== null && counts.get(allocName[i]) === 1) continue; // dead alloc
      const line = lines[i];
      if (/^\s*\}\s*$/.test(line) && out.length > 0 && /^\s*for\s*\(.*\{\s*$/.test(out[out.length - 1])) {
        out.pop();
        continue;
      }
      out.push(line);
    }
    return out.join('\n');
  }

  _isRedundantZeroFill(node) {
    let cur = node.body;
    while (cur) {
      if (cur.type === 'ForNode') {
        if (cur.extent.type === 'IntImmNode' && cur.extent.value === 1) {
          cur = cur.body;
          continue;
        }
        return this._isRedundantZeroFill(cur);
      }
      if (cur.type === 'BlockNode') { cur = cur.body; continue; }
      if (cur.type === 'BufferStoreNode') {
        const val = cur.value;
        const isZero = (val.type === 'FloatImmNode' && val.value === 0) || (val.type === 'IntImmNode' && val.value === 0);
        return isZero;
      }
      if (cur.type === 'LIRFlatStoreNode') {
        const val = cur.value;
        const isZero = (val.type === 'FloatImmNode' && val.value === 0) || (val.type === 'IntImmNode' && val.value === 0);
        return isZero;
      }
      return false;
    }
    return false;
  }

  _isZeroFillBody(body) {
    let cur = body;
    while (cur) {
      if (cur.type === 'ForNode') {
        cur = cur.body;
        continue;
      }
      if (cur.type === 'BlockNode') { cur = cur.body; continue; }
      if (cur.type === 'BufferStoreNode' || cur.type === 'LIRFlatStoreNode') {
        const val = cur.value;
        return (val.type === 'FloatImmNode' && val.value === 0) || (val.type === 'IntImmNode' && val.value === 0);
      }
      return false;
    }
    return false;
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

  _findZeroOnlyBuffers(root, paramBuffers) {
    const bufWrites = new Map();
    const stack = [root];
    while (stack.length > 0) {
      const node = stack.pop();
      if (!node || typeof node !== 'object') continue;
      if (node.type === 'BufferStoreNode') {
        const name = node.buffer.name;
        if (!paramBuffers.has(name)) {
          if (!bufWrites.has(name)) bufWrites.set(name, []);
          bufWrites.get(name).push(node.value);
        }
      }
      if (node.body) stack.push(node.body);
      if (node.stmts) for (const s of node.stmts) stack.push(s);
      if (node.thenBody) stack.push(node.thenBody);
      if (node.elseBody) stack.push(node.elseBody);
      if (node.initBody) stack.push(node.initBody);
    }
    const result = new Set();
    this._constantBuffers = new Map();
    for (const [name, writes] of bufWrites) {
      if (writes.length === 0) continue;
      if (writes.every(v =>
        (v.type === 'FloatImmNode' && v.value === 0) ||
        (v.type === 'IntImmNode' && v.value === 0)
      )) {
        result.add(name);
        continue;
      }
      const first = writes[0];
      const isConst = (first.type === 'FloatImmNode' || first.type === 'IntImmNode');
      if (isConst && writes.every(v => v.type === first.type && v.value === first.value)) {
        this._constantBuffers.set(name, String(first.value));
      }
    }
    return result;
  }
}
