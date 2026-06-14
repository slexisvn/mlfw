import { TensorType, typeToString } from './types.js';


export class IRPrinter {
  constructor(options = {}) {
    this.indent = 0;
    this.indentStr = options.indentStr || '  ';
    this.valueNames = new Map();
    this._nextValueId = 0;
  }

  printModule(module) {
    const lines = [];
    lines.push(`module @${module.name} {`);
    this.indent++;
    let first = true;
    for (const func of module) {
      if (!first) lines.push('');
      first = false;
      this.printFunction(func, lines);
    }
    this.indent--;
    lines.push('}');
    return lines.join('\n');
  }

  printFunction(func, lines = []) {
    this.valueNames.clear();
    this._nextValueId = 0;

    const args = func.args.map((arg, i) => {
      const name = this._nameValue(arg);
      return `${name}: ${typeToString(arg.type)}`;
    });

    const resultTypes = func.outputTypes.map(t => typeToString(t)).join(', ');
    lines.push(`${this._indentPrefix()}func @${func.name}(${args.join(', ')}) -> (${resultTypes}) {`);
    this.indent++;

    for (const block of func.body) {
      this.printBlock(block, lines, block === func.entryBlock);
    }

    this.indent--;
    lines.push(`${this._indentPrefix()}}`);
    return lines.join('\n');
  }

  printBlock(block, lines, isEntry = false) {
    if (!isEntry && block.arguments.length > 0) {
      const args = block.arguments.map(arg => {
        const name = this._nameValue(arg);
        return `${name}: ${typeToString(arg.type)}`;
      });
      lines.push(`${this._indentPrefix()}^bb(${args.join(', ')}):`);
    }

    for (const op of block) {
      this.printOperation(op, lines);
    }
  }

  printOperation(op, lines = []) {
    const prefix = this._indentPrefix();
    let line = prefix;

    if (op.numResults > 0) {
      const resultNames = [];
      for (let i = 0; i < op.numResults; i++) {
        resultNames.push(this._nameValue(op.getResult(i)));
      }
      line += resultNames.join(', ') + ' = ';
    }

    line += op.opName;

    if (op.numOperands > 0) {
      const operandStrs = [];
      for (let i = 0; i < op.numOperands; i++) {
        operandStrs.push(this._valueName(op.getOperand(i)));
      }
      line += '(' + operandStrs.join(', ') + ')';
    } else {
      line += '()';
    }

    if (op.attributes.size > 0) {
      const attrStrs = [];
      for (const [key, val] of sortedEntries(op.attributes)) {
        attrStrs.push(`${key} = ${formatAttrValue(val)}`);
      }
      line += ' {' + attrStrs.join(', ') + '}';
    }

    if (op.numResults > 0) {
      const types = [];
      for (let i = 0; i < op.numResults; i++) {
        types.push(typeToString(op.getResult(i).type));
      }
      line += ' : ' + types.join(', ');
    }

    lines.push(line);

    for (let i = 0; i < op.regions.length; i++) {
      const region = op.regions[i];
      lines.push(`${prefix}{`);
      this.indent++;
      for (const block of region) {
        if (block.arguments.length > 0) {
          const args = block.arguments.map(arg => {
            const name = this._nameValue(arg);
            return `${name}: ${typeToString(arg.type)}`;
          });
          lines.push(`${this._indentPrefix()}^bb(${args.join(', ')}):`);
        }
        for (const innerOp of block) {
          this.printOperation(innerOp, lines);
        }
      }
      this.indent--;
      lines.push(`${prefix}}`);
    }

    return lines.join('\n');
  }

  _nameValue(value) {
    if (this.valueNames.has(value)) return this.valueNames.get(value);
    const name = `%${this._nextValueId++}`;
    this.valueNames.set(value, name);
    return name;
  }

  _valueName(value) {
    if (this.valueNames.has(value)) return this.valueNames.get(value);
    return `%?`;
  }

  _indentPrefix() {
    let s = '';
    for (let i = 0; i < this.indent; i++) s += this.indentStr;
    return s;
  }
}

function sortedEntries(map) {
  const entries = [...map.entries()];
  entries.sort((a, b) => a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0);
  return entries;
}

function formatAttrValue(val) {
  if (val === null || val === undefined) return 'null';
  if (typeof val === 'number') {
    if (Number.isFinite(val)) return String(val);
    if (val === Infinity) return 'inf';
    if (val === -Infinity) return '-inf';
    return 'nan';
  }
  if (typeof val === 'boolean') return String(val);
  if (typeof val === 'string') return `"${val}"`;
  if (val instanceof TensorType) return typeToString(val);
  if (Array.isArray(val)) {
    if (val.length === 0) return '[]';
    if (Array.isArray(val[0])) {
      return '[' + val.map(v => formatAttrValue(v)).join(', ') + ']';
    }
    return '[' + val.map(v => formatAttrValue(v)).join(', ') + ']';
  }
  if (typeof val === 'object' && val.constructor === Object) {
    const entries = Object.entries(val).sort((a, b) => a[0].localeCompare(b[0]));
    return '{' + entries.map(([k, v]) => `${k}: ${formatAttrValue(v)}`).join(', ') + '}';
  }
  return String(val);
}

export function printModule(module) {
  return new IRPrinter().printModule(module);
}

export function printFunction(func) {
  const printer = new IRPrinter();
  const lines = [];
  printer.printFunction(func, lines);
  return lines.join('\n');
}

export function printOperation(op) {
  const printer = new IRPrinter();
  const lines = [];
  printer.printOperation(op, lines);
  return lines.join('\n');
}
