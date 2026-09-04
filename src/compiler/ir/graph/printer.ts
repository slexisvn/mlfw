import { TensorType, typeToString, isFloatType, isBoolType } from './types.js';
import { formatLocation } from '../location.js';
import { dtypeKeys, jsTypedArray } from '../../../util/dtype_map.js';
import {
  GENERIC_REGIONS, customFormOf, flattenPairs, formatFloatLiteral, formatNonFiniteAttr,
  groupSpans, isNumberArray, isNumberPairs, qualify,
} from './mlir_format.js';
import type { MlirAttrBinding, MlirOpForm, MlirRegionForm } from './mlir_format.js';
import type { AttrValue, IRType, ScalarDType } from './types.js';
import type { Value } from './value.js';
import type { Block } from './block.js';
import type { Operation } from './operation.js';
import type { GraphFunction } from './function.js';
import type { GraphModule } from './module.js';

export type IRPrinterOptions = Readonly<{ indentStr?: string; locations?: boolean }>;

export class IRPrinter {
  indent: number;
  indentStr: string;
  locations: boolean;
  valueNames: Map<Value, string>;
  private _nextValueId: number;
  private _forms: Map<Operation, MlirOpForm | null>;
  private _elided: Set<Operation>;

  constructor(options: IRPrinterOptions = {}) {
    this.indent = 0;
    this.indentStr = options.indentStr || '  ';
    this.locations = options.locations === true;
    this.valueNames = new Map();
    this._nextValueId = 0;
    this._forms = new Map();
    this._elided = new Set();
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
    this._forms.clear();
    this._elided.clear();
    for (const block of func.body) this._collectElided(block);
    for (const arg of func.args) this._nameValue(arg);
    for (const block of func.body) this._nameBlockValues(block);
    for (const op of this._elided) this._nameValue(op.getResult(0));

    const args = func.args.map((arg) => `${this._valueName(arg)}: ${typeToString(arg.type)}`);
    const resultTypes = func.outputTypes.map((t) => typeToString(t)).join(', ');
    const funcAttrs = this._attributeDict(func.attributes, null);
    const trailing = funcAttrs === '' ? '' : ` attributes ${funcAttrs}`;
    out.push(`${this._indentPrefix()}func.func @${func.name}(${args.join(', ')}) -> (${resultTypes})${trailing} {`);
    this.indent++;

    let index = 0;
    for (const block of func.body) {
      this.printBlock(block, out, block === func.entryBlock, index++);
    }

    this.indent--;
    out.push(`${this._indentPrefix()}}`);
    return ownLines ? out.join('\n') : undefined;
  }

  printBlock(block: Block, lines: string[], isEntry = false, index = 0): void {
    if (!isEntry && (block.arguments.length > 0 || index > 0)) {
      lines.push(this._indentPrefix() + this._blockLabel(block, index));
    }
    for (const op of block) {
      if (this._elided.has(op)) continue;
      this.printOperation(op, lines);
    }
  }

  printOperation(op: Operation, lines: string[] | null = null): string | undefined {
    const ownLines = lines === null;
    if (ownLines) lines = [];
    const out = lines as string[];
    const form = this._formOf(op);
    if (form) this._printCustom(op, form, out);
    else this._printGeneric(op, out);
    return ownLines ? out.join('\n') : undefined;
  }

  _printCustom(op: Operation, form: MlirOpForm, out: string[]): void {
    let line = this._indentPrefix() + this._resultPrefix(op) + form.mnemonic;
    const parts: string[] = [];
    if (form.keyword) {
      parts.push(`${form.keyword.toMlir.get(op.getAttr<string>(form.keyword.ir) as string)},`);
    }
    if (form.groups) parts.push(...this._groupClauses(op, form));
    else {
      const operands: string[] = [];
      for (let i = 0; i < op.numOperands; i++) {
        if (i === form.seedOperand) continue;
        operands.push(this._valueName(op.getOperand(i)));
      }
      if (operands.length > 0) parts.push(operands.join(', '));
    }
    const attrs = this._attrDict(op, form);
    if (attrs !== '') parts.push(attrs);
    if (form.types === 'elements') parts.push(this._densePayload(op));
    if (parts.length > 0) line += ' ' + parts.join(' ');
    line += this._customTypeSuffix(op, form);
    if (!form.regions) {
      out.push(line + this._locSuffix(op));
      return;
    }
    this._printRegions(op, form.regions, line + this._locSuffix(op), '', out);
  }

  _groupClauses(op: Operation, form: MlirOpForm): string[] {
    const groups = form.groups as NonNullable<MlirOpForm['groups']>;
    const bounds = groupSpans(op, groups) as number[];
    const clauses: string[] = [];
    for (let g = 0; g < groups.length; g++) {
      const start = bounds[g];
      const end = bounds[g + 1];
      if (start === end && groups[g].optional) continue;
      const names: string[] = [];
      const types: string[] = [];
      for (let i = start; i < end; i++) {
        names.push(this._valueName(op.getOperand(i)));
        types.push(typeToString(op.getOperand(i).type));
      }
      if (groups[g].keyword === null) {
        clauses.push(names.join(', '));
        continue;
      }
      const spelt = groups[g].types && names.length > 0
        ? `${names.join(', ')} : ${types.join(', ')}`
        : names.join(', ');
      clauses.push(`${groups[g].keyword}(${spelt})`);
    }
    return clauses;
  }

  _printRegions(op: Operation, form: MlirRegionForm, head: string, tail: string, out: string[]): void {
    const prefix = this._indentPrefix();
    out.push(head + form.open);
    for (let i = 0; i < op.regions.length; i++) {
      if (i > 0) out.push(prefix + (form.repeat ? form.separators[0] : form.separators[i - 1]));
      this.indent += form.labelDepth;
      const blocks = op.getRegion(i).blocks;
      for (let b = 0; b < blocks.length; b++) {
        if (b > 0 || blocks[b].arguments.length > 0) {
          out.push(this._indentPrefix() + this._blockLabel(blocks[b], b));
        }
        this.indent++;
        for (const inner of blocks[b]) {
          if (!this._elided.has(inner)) this.printOperation(inner, out);
        }
        this.indent--;
      }
      this.indent -= form.labelDepth;
    }
    out.push(prefix + form.close + tail);
  }

  _printGeneric(op: Operation, out: string[]): void {
    const prefix = this._indentPrefix();
    const operands: string[] = [];
    for (let i = 0; i < op.numOperands; i++) operands.push(this._valueName(op.getOperand(i)));
    const head = `${prefix}${this._resultPrefix(op)}"${qualify(op.opName)}"(${operands.join(', ')})`;
    const attrs = this._attrDict(op, null);
    const tail = (attrs === '' ? '' : ` ${attrs}`)
      + ` : (${operands.map((_, i) => typeToString(op.getOperand(i).type)).join(', ')}) -> `
      + this._resultTypeList(op) + this._locSuffix(op);

    if (op.regions.length === 0) {
      out.push(head + tail);
      return;
    }
    this._printRegions(op, GENERIC_REGIONS, head, tail, out);
  }

  _blockLabel(block: Block, index: number): string {
    if (block.arguments.length === 0) return `^bb${index}:`;
    const args = block.arguments.map((arg) => `${this._valueName(arg)}: ${typeToString(arg.type)}`);
    return `^bb${index}(${args.join(', ')}):`;
  }

  _resultPrefix(op: Operation): string {
    if (op.numResults === 0) return '';
    const names: string[] = [];
    for (let i = 0; i < op.numResults; i++) names.push(this._valueName(op.getResult(i)));
    return names.join(', ') + ' = ';
  }

  _resultTypeList(op: Operation): string {
    const types: string[] = [];
    for (let i = 0; i < op.numResults; i++) types.push(typeToString(op.getResult(i).type));
    return types.length === 1 ? types[0] : `(${types.join(', ')})`;
  }

  _operandTypeList(op: Operation, skip: number): string {
    const types: string[] = [];
    for (let i = 0; i < op.numOperands; i++) {
      if (i !== skip) types.push(typeToString(op.getOperand(i).type));
    }
    return types.join(', ');
  }

  _customTypeSuffix(op: Operation, form: MlirOpForm): string {
    switch (form.types) {
      case 'elements':
        return '';
      case 'result':
        return ` : ${typeToString(op.getResult(0).type)}`;
      case 'resultList':
        return ` -> (${op.results.map((r) => typeToString(r.type)).join(', ')})`;
      case 'functional':
        return ` : (${this._operandTypeList(op, form.seedOperand)}) -> ${this._resultTypeList(op)}`;
      case 'operandToResult':
        return ` : ${typeToString(op.getOperand(0).type)} -> ${typeToString(op.getResult(0).type)}`;
      case 'operandsToResult':
        return ` : ${this._operandTypeList(op, form.seedOperand)} -> ${typeToString(op.getResult(0).type)}`;
      case 'firstAndResult':
        return ` : ${typeToString(op.getOperand(0).type)}, ${typeToString(op.getResult(0).type)}`;
      default:
        return op.numOperands === 0 ? '' : ` : ${this._operandTypeList(op, form.seedOperand)}`;
    }
  }

  _densePayload(op: Operation): string {
    const type = op.getResult(0).type as TensorType;
    return `${denseLiteral(op.getAttr('value'), type)} : ${typeToString(type)}`;
  }

  _attrDict(op: Operation, form: MlirOpForm | null): string {
    return this._attributeDict(op.attributes, form);
  }

  _attributeDict(attributes: ReadonlyMap<string, AttrValue>, form: MlirOpForm | null): string {
    const entries: [string, string][] = [];
    for (const [key, value] of attributes) {
      if (form && form.consumedAttrs.has(key)) continue;
      const binding = form ? form.attrByIr.get(key) : undefined;
      const name = binding ? binding.mlir : key;
      const text = value === true
        ? name
        : `${name} = ${binding ? formatBoundAttr(value, binding) : formatAttrValue(value)}`;
      entries.push([name, text]);
    }
    if (entries.length === 0) return '';
    entries.sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
    return '{' + entries.map((entry) => entry[1]).join(', ') + '}';
  }

  _locSuffix(op: Operation): string {
    return this.locations && op.loc !== null ? ` loc(${formatLocation(op.loc)})` : '';
  }

  _formOf(op: Operation): MlirOpForm | null {
    let form = this._forms.get(op);
    if (form === undefined) {
      form = customFormOf(op);
      this._forms.set(op, form);
    }
    return form;
  }

  _collectElided(block: Block): void {
    const seedUses = new Map<Operation, number>();
    this._countSeedUses(block, seedUses);
    for (const [producer, count] of seedUses) {
      if (producer.getResult(0).useCount === count) this._elided.add(producer);
    }
  }

  _countSeedUses(block: Block, seedUses: Map<Operation, number>): void {
    for (const op of block) {
      const form = this._formOf(op);
      const seed = form && form.seedOperand >= 0 ? op.getOperand(form.seedOperand).definingOp : null;
      if (seed) seedUses.set(seed, (seedUses.get(seed) || 0) + 1);
      for (const region of op.regions) {
        for (const inner of region.blocks) this._countSeedUses(inner, seedUses);
      }
    }
  }

  _nameBlockValues(block: Block): void {
    for (const arg of block.arguments) this._nameValue(arg);
    for (const op of block) {
      if (this._elided.has(op)) continue;
      for (let i = 0; i < op.numResults; i++) this._nameValue(op.getResult(i));
      const form = this._formOf(op);
      if (form && !form.regions) continue;
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

const DENSE_DTYPE_BY_ARRAY: Record<string, ScalarDType> = {};
for (const dtype of dtypeKeys()) {
  const name = jsTypedArray(dtype);
  if (!(name in DENSE_DTYPE_BY_ARRAY)) DENSE_DTYPE_BY_ARRAY[name] = dtype as ScalarDType;
}

type NumericView = ArrayBufferView & { constructor: { name: string }; length: number; [index: number]: number };

function isNumericView(value: unknown): value is NumericView {
  return ArrayBuffer.isView(value) && !(value instanceof DataView);
}

export function formatScalar(value: number, dtype: ScalarDType): string {
  if (isBoolType(dtype)) return value ? 'true' : 'false';
  if (isFloatType(dtype)) return formatFloatLiteral(value, dtype);
  return String(value);
}

function denseElements(view: NumericView, shape: readonly unknown[], dtype: ScalarDType, offset: number, stride: number): string {
  if (shape.length === 0) return formatScalar(view[offset], dtype);
  const extent = shape[0] as number;
  const inner = stride / Math.max(extent, 1);
  const parts: string[] = [];
  for (let i = 0; i < extent; i++) {
    parts.push(denseElements(view, shape.slice(1), dtype, offset + i * inner, inner));
  }
  return `[${parts.join(', ')}]`;
}

export function denseLiteral(value: AttrValue | undefined, type: TensorType): string {
  if (isNumericView(value)) {
    return `dense<${denseElements(value, type.shape, type.dtype, 0, value.length)}>`;
  }
  return `dense<${formatScalar(Number(value), type.dtype)}>`;
}

function formatBoundAttr(value: AttrValue, binding: MlirAttrBinding): string {
  if (binding.kind === 'i64pairs' && isNumberPairs(value)) {
    return formatI64Array(flattenPairs(value));
  }
  if (binding.kind === 'i64array' && isNumberArray(value)) return formatI64Array(value);
  if (binding.kind === 'i64' && typeof value === 'number') return `${value} : i64`;
  return formatAttrValue(value);
}

function formatI64Array(values: readonly number[]): string {
  return values.length === 0 ? 'array<i64>' : `array<i64: ${values.join(', ')}>`;
}

export function formatAttrValue(val: AttrValue | undefined): string {
  if (val === null || val === undefined) return 'unit';
  if (typeof val === 'number') return Number.isFinite(val) ? String(val) : formatNonFiniteAttr(val);
  if (typeof val === 'boolean') return String(val);
  if (typeof val === 'string') return JSON.stringify(val);
  if (val instanceof TensorType) return typeToString(val);
  if (Array.isArray(val)) return '[' + (val as readonly AttrValue[]).map((v) => formatAttrValue(v)).join(', ') + ']';
  if (isNumericView(val)) {
    const dtype = DENSE_DTYPE_BY_ARRAY[val.constructor.name] || 'f32';
    const type = new TensorType([val.length], dtype);
    return `${denseLiteral(val as unknown as AttrValue, type)} : ${typeToString(type)}`;
  }
  if (typeof val === 'object' && val.constructor === Object) {
    const entries = Object.entries(val as unknown as Record<string, AttrValue>).sort((a, b) => a[0].localeCompare(b[0]));
    return '{' + entries.map(([k, v]) => `${k} = ${formatAttrValue(v)}`).join(', ') + '}';
  }
  if (typeof (val as IRType).equals === 'function') return typeToString(val as IRType);
  return String(val);
}

export function printModule(module: GraphModule, options: IRPrinterOptions = {}): string {
  return new IRPrinter(options).printModule(module);
}

export function printFunction(func: GraphFunction, options: IRPrinterOptions = {}): string {
  const printer = new IRPrinter(options);
  const lines: string[] = [];
  printer.printFunction(func, lines);
  return lines.join('\n');
}

export function printOperation(op: Operation, options: IRPrinterOptions = {}): string {
  const printer = new IRPrinter(options);
  const lines: string[] = [];
  printer.printOperation(op, lines);
  return lines.join('\n');
}
