import { TensorType, typeToString } from './types.js';
import { dtypeKeys, jsTypedArray } from '../../../util/dtype_map.js';
import type { Block as BlockType } from './block.js';
import type { AttrValue } from './types.js';
import type { Value } from './value.js';
import type { Block } from './block.js';
import type { Operation } from './operation.js';
import type { GraphFunction } from './function.js';
import type { GraphModule } from './module.js';

export type IRPrinterOptions = Readonly<{ indentStr?: string }>;


export class IRPrinter {
  indent: number;
  indentStr: string;
  valueNames: Map<Value, string>;
  private _nextValueId: number;

  constructor(options: IRPrinterOptions = {}) {
    this.indent = 0;
    this.indentStr = options.indentStr || '  ';
    this.valueNames = new Map();
    this._nextValueId = 0;
  }

  printModule(module: GraphModule): string {
    const lines: string[] = [];
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

  printFunction(func: GraphFunction, lines: string[] | null = null): string | undefined {
    const ownLines = lines === null;
    if (ownLines) lines = [];
    const out = lines as string[];
    this.valueNames.clear();
    this._nextValueId = 0;
    for (const block of func.body) this._nameBlockValues(block);

    const args = func.args.map((arg, i) => {
      const name = this._nameValue(arg);
      return `${name}: ${typeToString(arg.type)}`;
    });

    const resultTypes = func.outputTypes.map(t => typeToString(t)).join(', ');
    out.push(`${this._indentPrefix()}func @${func.name}(${args.join(', ')}) -> (${resultTypes}) {`);
    this.indent++;

    for (const block of func.body) {
      this.printBlock(block, out, block === func.entryBlock);
    }

    this.indent--;
    out.push(`${this._indentPrefix()}}`);
    return ownLines ? out.join('\n') : undefined;
  }

  printBlock(block: Block, lines: string[], isEntry = false): void {
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

  printOperation(op: Operation, lines: string[] | null = null): string | undefined {
    const ownLines = lines === null;
    if (ownLines) lines = [];
    const out = lines as string[];
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

    out.push(line);

    for (let i = 0; i < op.regions.length; i++) {
      const region = op.regions[i];
      out.push(`${prefix}{`);
      this.indent++;
      for (const block of region) {
        if (block.arguments.length > 0) {
          const args = block.arguments.map(arg => {
            const name = this._nameValue(arg);
            return `${name}: ${typeToString(arg.type)}`;
          });
          out.push(`${this._indentPrefix()}^bb(${args.join(', ')}):`);
        }
        for (const innerOp of block) {
          this.printOperation(innerOp, out);
        }
      }
      this.indent--;
      out.push(`${prefix}}`);
    }

    return ownLines ? out.join('\n') : undefined;
  }

  _nameBlockValues(block: BlockType): void {
    for (const arg of block.arguments) this._nameValue(arg);
    for (const op of block) {
      for (let i = 0; i < op.numResults; i++) this._nameValue(op.getResult(i));
      for (const region of op.regions) {
        for (const inner of region.blocks) this._nameBlockValues(inner);
      }
    }
  }

  _nameValue(value: Value): string {
    if (this.valueNames.has(value)) return this.valueNames.get(value) as string;
    const name = `%${this._nextValueId++}`;
    this.valueNames.set(value, name);
    return name;
  }

  _valueName(value: Value): string {
    if (this.valueNames.has(value)) return this.valueNames.get(value) as string;
    return `%?`;
  }

  _indentPrefix(): string {
    let s = '';
    for (let i = 0; i < this.indent; i++) s += this.indentStr;
    return s;
  }
}

function sortedEntries(map: ReadonlyMap<string, AttrValue>): [string, AttrValue][] {
  const entries = [...map.entries()];
  entries.sort((a, b) => a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0);
  return entries;
}

const DENSE_DTYPE_BY_ARRAY: Record<string, string> = {};
for (const dtype of dtypeKeys()) {
  const name = jsTypedArray(dtype);
  if (!(name in DENSE_DTYPE_BY_ARRAY)) DENSE_DTYPE_BY_ARRAY[name] = dtype;
}

function formatAttrValue(val: AttrValue | undefined): string {
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
    const arr = val as readonly AttrValue[];
    if (arr.length === 0) return '[]';
    if (Array.isArray(arr[0])) {
      return '[' + arr.map(v => formatAttrValue(v)).join(', ') + ']';
    }
    return '[' + arr.map(v => formatAttrValue(v)).join(', ') + ']';
  }
  if (ArrayBuffer.isView(val) && !(val instanceof DataView)) {
    const view = val as unknown as { constructor: { name: string }; length: number; [index: number]: number };
    const dtype = DENSE_DTYPE_BY_ARRAY[view.constructor.name] || 'f32';
    const items: string[] = [];
    for (let i = 0; i < view.length; i++) items.push(formatAttrValue(view[i]));
    return `dense<${dtype}>[${items.join(', ')}]`;
  }
  if (typeof val === 'object' && val.constructor === Object) {
    const entries = Object.entries(val as unknown as Record<string, AttrValue>).sort((a, b) => a[0].localeCompare(b[0]));
    return '{' + entries.map(([k, v]) => `${k}: ${formatAttrValue(v)}`).join(', ') + '}';
  }
  return String(val);
}

export function printModule(module: GraphModule): string {
  return new IRPrinter().printModule(module);
}

export function printFunction(func: GraphFunction): string {
  const printer = new IRPrinter();
  const lines: string[] = [];
  printer.printFunction(func, lines);
  return lines.join('\n');
}

export function printOperation(op: Operation): string {
  const printer = new IRPrinter();
  const lines: string[] = [];
  printer.printOperation(op, lines);
  return lines.join('\n');
}
