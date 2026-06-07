import { inferDtype } from './nodes.js';

export class LIRPrinter {
  constructor() {
    this._indent = 0;
    this._lines = [];
  }

  print(node) {
    this._indent = 0;
    this._lines = [];
    this.visit(node);
    return this._lines.join('\n');
  }

  visit(node) {
    if (!node || typeof node !== 'object') return;
    const method = `visit${node.type}`;
    if (typeof this[method] === 'function') {
      this[method](node);
    } else {
      this._emit(`[${node.type}]`);
    }
  }

  visitLIRFunc(node) {
    const params = [];
    for (const [, buf] of node.bufferMap) params.push(buf.name);
    for (const sp of node.shapeParams) params.push(sp.name);

    this._emit(`lir_func ${node.name}(${params.join(', ')}) {`);
    this._indent++;

    if (node.metadata.locals.size > 0) {
      const localDecls = [];
      for (const [name, dtype] of node.metadata.locals) {
        localDecls.push(`${name}: ${dtype}`);
      }
      this._emit(`locals: ${localDecls.join(', ')}`);
    }

    if (node.metadata.externCalls.size > 0) {
      const imports = [];
      for (const [name, info] of node.metadata.externCalls) {
        imports.push(`${name}(${info.argCount})`);
      }
      this._emit(`imports: ${imports.join(', ')}`);
    }

    if (node.metadata.memoryLayout.totalBytes > 0) {
      this._emit(`memory: ${node.metadata.memoryLayout.totalBytes} bytes`);
    }

    this.visit(node.body);
    this._indent--;
    this._emit('}');
  }

  visitLIRFlatStoreNode(node) {
    this._emit(`${node.buffer.name}[${this.exprStr(node.offsetExpr)}] = ${this.exprStr(node.value)}`);
  }

  visitLIRFlatLoadNode(node) {
    return `${node.buffer.name}[${this.exprStr(node.offsetExpr)}]`;
  }

  visitLIRAccumulatorNode(node) {
    this._emit(`acc ${node.localName}: ${node.dtype} = load(${node.buffer ? node.buffer.name : '?'})`);
    this._emit(`for ${node.loopVar.name} in [0, ${this.exprStr(node.extent)}) @${node.loopKind} {`);
    this._indent++;
    this._emit(`${node.localName} += ${this.exprStr(node.body)}`);
    this._indent--;
    this._emit('}');
    this._emit(`flush ${node.localName} -> ${this.exprStr(node.flushStore)}`);
  }

  visitLIRBindingsNode(node) {
    for (const bind of node.bindings) {
      this._emit(`bind ${bind.name} = ${this.exprStr(bind.expr)}`);
    }
    this.visit(node.body);
  }

  visitForNode(node) {
    const tag = node.threadTag ? ` @${node.threadTag}` : '';
    this._emit(`for ${node.loopVar.name} in [0, ${this.exprStr(node.extent)}) @${node.kind}${tag} {`);
    this._indent++;
    this.visit(node.body);
    this._indent--;
    this._emit('}');
  }

  visitSeqNode(node) {
    for (const s of node.stmts) this.visit(s);
  }

  visitLetStmtNode(node) {
    this._emit(`let ${node.variable.name} = ${this.exprStr(node.value)}`);
    this.visit(node.body);
  }

  visitAllocateNode(node) {
    this._emit(`allocate ${node.buffer.name}[${node.buffer.shape.join(',')}] @${node.scope}`);
    this.visit(node.body);
  }

  visitIfThenElseNode(node) {
    this._emit(`if (${this.exprStr(node.condition)}) {`);
    this._indent++;
    this.visit(node.thenBody);
    this._indent--;
    if (node.elseBody) {
      this._emit('} else {');
      this._indent++;
      this.visit(node.elseBody);
      this._indent--;
    }
    this._emit('}');
  }

  visitWhileNode(node) {
    this._emit(`while (${node.condVar.name}) {`);
    this._indent++;
    this.visit(node.condBody);
    this.visit(node.loopBody);
    this._indent--;
    this._emit('}');
  }

  visitEvaluateNode(node) {
    this._emit(`eval ${this.exprStr(node.value)}`);
  }

  exprStr(node) {
    if (!node) return '?';

    switch (node.type) {
      case 'IntImmNode': return String(node.value);
      case 'FloatImmNode': return `${node.value}f`;
      case 'VariableNode': return node.name;
      case 'LIRFlatLoadNode':
        return `${node.buffer.name}[${this.exprStr(node.offsetExpr)}]`;
      case 'MathOpNode':
        if (!node.b) return `(${node.op}${this.exprStr(node.a)})`;
        return `(${this.exprStr(node.a)} ${node.op} ${this.exprStr(node.b)})`;
      case 'CompareNode':
        return `(${this.exprStr(node.a)} ${node.direction} ${this.exprStr(node.b)})`;
      case 'CastNode':
        return `cast<${node.toDtype}>(${this.exprStr(node.expr)})`;
      case 'CallExternNode':
        return `${node.externName}(${node.args.map(a => this.exprStr(a)).join(', ')})`;
      case 'IfThenElseNode':
        return `(${this.exprStr(node.condition)} ? ${this.exprStr(node.thenBody)} : ${this.exprStr(node.elseBody)})`;
      case 'LIRFlatStoreNode':
        return `${node.buffer.name}[${this.exprStr(node.offsetExpr)}]`;
      default:
        return `[${node.type}]`;
    }
  }

  _emit(line) {
    this._lines.push('  '.repeat(this._indent) + line);
  }
}

export function printLIR(node) {
  return new LIRPrinter().print(node);
}
