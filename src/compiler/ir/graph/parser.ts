import { Block, Region } from './block.js';
import { Operation } from './operation.js';
import { GraphFunction } from './function.js';
import { GraphModule } from './module.js';
import { TensorType, TupleType, TokenType, FunctionType, Layout, DYNAMIC, dtypeFromString, isFloatType, layoutFromString } from './types.js';
import { SymInt } from '../sym_int.js';
import { jsTypedArray } from '../../../util/dtype_map.js';
import {
  GENERIC_REGIONS, combinerScalarType, derivedAttrValue, isNumberArray, mlirFormOfMnemonic,
  nestPairs, parseFloatLiteral, seedConstantAttrs,
  unqualify,
} from './mlir_format.js';
import type { MlirOpForm, MlirRegionForm } from './mlir_format.js';
import type { AttrValue, Dim, IRType, ScalarDType } from './types.js';
import type { Value } from './value.js';
import { parseLocation } from '../location.js';
import type { Location } from '../location.js';

const TYPED_ARRAY_BY_NAME: Record<string, new (values: number[]) => ArrayBufferView> = {
  Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Uint8Array, Uint16Array, Uint32Array,
} as unknown as Record<string, new (values: number[]) => ArrayBufferView>;

const SYM_BINARY: Record<string, (a: SymInt | number, b: SymInt | number) => SymInt | number> = {
  '+': SymInt.add, '-': SymInt.sub, '*': SymInt.mul, '/': SymInt.div, '%': SymInt.mod,
};

export class IRParseError extends Error {
  line: number;

  constructor(message: string, line: number) {
    super(line >= 0 ? `line ${line + 1}: ${message}` : message);
    this.name = 'IRParseError';
    this.line = line;
  }
}

class Scanner {
  text: string;
  pos: number;
  line: number;

  constructor(text: string, line: number) {
    this.text = text;
    this.pos = 0;
    this.line = line;
  }

  fail(message: string): never {
    throw new IRParseError(`${message} (at column ${this.pos + 1} of '${this.text.trim()}')`, this.line);
  }

  skipSpace(): void {
    while (this.pos < this.text.length && (this.text[this.pos] === ' ' || this.text[this.pos] === '\t')) this.pos++;
  }

  get atEnd(): boolean {
    this.skipSpace();
    return this.pos >= this.text.length;
  }

  peek(): string {
    this.skipSpace();
    return this.text[this.pos] || '';
  }

  eat(token: string): boolean {
    this.skipSpace();
    if (!this.text.startsWith(token, this.pos)) return false;
    this.pos += token.length;
    return true;
  }

  expect(token: string): void {
    if (!this.eat(token)) this.fail(`expected '${token}'`);
  }

  readWhile(accept: (ch: string) => boolean): string {
    const start = this.pos;
    while (this.pos < this.text.length && accept(this.text[this.pos])) this.pos++;
    return this.text.slice(start, this.pos);
  }

  readGroup(open: string, close: string): string {
    this.expect(open);
    const start = this.pos;
    let depth = 1;
    while (this.pos < this.text.length) {
      const ch = this.text[this.pos];
      if (ch === '"') {
        this.pos++;
        while (this.pos < this.text.length && this.text[this.pos] !== '"') {
          if (this.text[this.pos] === '\\') this.pos++;
          this.pos++;
        }
      } else if (ch === open) {
        depth++;
      } else if (ch === close) {
        depth--;
        if (depth === 0) {
          const inner = this.text.slice(start, this.pos);
          this.pos++;
          return inner;
        }
      }
      this.pos++;
    }
    this.fail(`unbalanced '${open}'`);
  }

  readString(): string {
    this.expect('"');
    const start = this.pos - 1;
    while (this.pos < this.text.length && this.text[this.pos] !== '"') {
      if (this.text[this.pos] === '\\') this.pos++;
      this.pos++;
    }
    this.expect('"');
    return JSON.parse(this.text.slice(start, this.pos)) as string;
  }

  identifier(): string {
    this.skipSpace();
    const name = this.readWhile((ch) => /[A-Za-z0-9_.$]/.test(ch));
    if (name === '') this.fail('expected an identifier');
    return name;
  }

  valueName(): string {
    this.expect('%');
    const name = this.identifier();
    return this.eat('#') ? `%${name}#${this.identifier()}` : `%${name}`;
  }
}

function splitTopLevel(text: string, separator: string): string[] {
  const parts: string[] = [];
  let depth = 0;
  let start = 0;
  let inString = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i++;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === '<' || ch === '(' || ch === '[' || ch === '{') depth++;
    else if (ch === '>' || ch === ')' || ch === ']' || ch === '}') depth--;
    else if (depth === 0 && ch === separator) {
      parts.push(text.slice(start, i));
      start = i + 1;
    }
  }
  parts.push(text.slice(start));
  return parts;
}

function parseSymExpr(sc: Scanner): SymInt | number {
  for (const fn of ['ceildiv', 'max', 'min'] as const) {
    if (sc.eat(`${fn}(`)) {
      const a = parseSymExpr(sc);
      sc.expect(',');
      const b = parseSymExpr(sc);
      sc.expect(')');
      return fn === 'max' ? SymInt.max(a, b) : fn === 'min' ? SymInt.min(a, b) : SymInt.ceilDiv(a, b);
    }
  }
  if (sc.eat('(')) {
    if (sc.eat('-')) {
      const inner = parseSymExpr(sc);
      sc.expect(')');
      return SymInt.neg(inner);
    }
    const a = parseSymExpr(sc);
    const op = sc.peek();
    const apply = SYM_BINARY[op];
    if (!apply) sc.fail(`unknown symbolic operator '${op}'`);
    sc.expect(op);
    const b = parseSymExpr(sc);
    sc.expect(')');
    return apply(a, b);
  }
  sc.skipSpace();
  if (/[-0-9]/.test(sc.peek())) {
    const digits = sc.readWhile((ch) => /[-0-9.]/.test(ch));
    return Number(digits);
  }
  return SymInt.var(sc.identifier());
}

function parseDim(text: string, line: number): Dim {
  const trimmed = text.trim();
  if (trimmed === '?') return DYNAMIC;
  if (trimmed.startsWith('[')) {
    const sc = new Scanner(trimmed.slice(1, -1), line);
    const sym = parseSymExpr(sc);
    return sym as Dim;
  }
  const value = Number(trimmed);
  if (trimmed === '' || !Number.isFinite(value)) throw new IRParseError(`invalid dimension '${trimmed}'`, line);
  return value;
}

function parseTensorBody(inner: string, line: number): TensorType {
  const parts = splitTopLevel(inner, ',');
  const segments = splitTopLevel(parts[0], 'x');
  const spelling = segments[segments.length - 1].trim();
  const dtype = dtypeFromString(spelling);
  if (!dtype) {
    throw new IRParseError(`'tensor<${inner}>' does not end in a known dtype (got '${spelling}')`, line);
  }
  const dimParts = segments.slice(0, -1);
  if (dimParts.length === 1 && dimParts[0].trim() === '') dimParts.pop();
  const shape: Dim[] = dimParts.map((d) => parseDim(d, line));
  let layout: Layout | null = null;
  if (parts.length > 1) layout = layoutFromString(parts[1].trim());
  return new TensorType(shape, dtype, layout);
}

function parseType(sc: Scanner): IRType {
  if (sc.eat('tensor<')) {
    sc.pos--;
    return parseTensorBody(sc.readGroup('<', '>'), sc.line);
  }
  if (sc.eat('tuple<')) {
    sc.pos--;
    const inner = sc.readGroup('<', '>');
    if (inner.trim() === '') return new TupleType([]);
    return new TupleType(splitTopLevel(inner, ',').map((part) => parseType(new Scanner(part, sc.line))));
  }
  if (sc.eat('token')) return new TokenType();
  if (sc.peek() === '(') {
    const inputs = parseTypeList(sc.readGroup('(', ')'), sc.line);
    sc.expect('->');
    const outputs = parseTypeList(sc.readGroup('(', ')'), sc.line);
    return new FunctionType(inputs, outputs);
  }
  sc.fail('expected a type');
}

function parseTypeList(text: string, line: number): IRType[] {
  if (text.trim() === '') return [];
  return splitTopLevel(text, ',').map((part) => parseType(new Scanner(part, line)));
}

function parseGroupedTypes(sc: Scanner): IRType[] {
  if (sc.peek() === '(') return parseTypeList(sc.readGroup('(', ')'), sc.line);
  return [parseType(sc)];
}

function readNumericLiteral(sc: Scanner): string {
  sc.skipSpace();
  const sign = sc.eat('-') ? '-' : '';
  if (sc.text.startsWith('0x', sc.pos) || sc.text.startsWith('0X', sc.pos)) {
    sc.pos += 2;
    return `${sign}0x${sc.readWhile((ch) => /[0-9a-fA-F]/.test(ch))}`;
  }
  const digits = sc.readWhile((ch) => /[0-9.]/.test(ch));
  const mark = sc.pos;
  if (!sc.eat('e') && !sc.eat('E')) return sign + digits;
  const exponentSign = sc.eat('-') ? '-' : (sc.eat('+') ? '+' : '');
  const exponent = sc.readWhile((ch) => /[0-9]/.test(ch));
  if (exponent === '') {
    sc.pos = mark;
    return sign + digits;
  }
  return `${sign}${digits}e${exponentSign}${exponent}`;
}

function parseScalarLiteral(sc: Scanner, dtype: ScalarDType | null): number {
  if (sc.eat('true')) return 1;
  if (sc.eat('false')) return 0;
  const text = readNumericLiteral(sc);
  if (text === '') sc.fail('expected a numeric literal');
  const typed = sc.eat(':') ? dtypeFromString(sc.identifier()) : null;
  const effective = typed || dtype;
  if (effective && isFloatType(effective)) return parseFloatLiteral(text, effective);
  if (text.includes('0x')) return parseFloatLiteral(text, 'f64');
  return Number(text);
}

function parseDenseBody(text: string, type: TensorType, line: number): AttrValue {
  const trimmed = text.trim();
  if (!trimmed.startsWith('[')) {
    return parseScalarLiteral(new Scanner(trimmed, line), type.dtype);
  }
  const values: number[] = [];
  const collect = (body: string): void => {
    const inner = body.trim();
    if (!inner.startsWith('[')) {
      values.push(parseScalarLiteral(new Scanner(inner, line), type.dtype));
      return;
    }
    const stripped = inner.slice(1, -1);
    if (stripped.trim() === '') return;
    for (const part of splitTopLevel(stripped, ',')) collect(part);
  };
  collect(trimmed);
  const Ctor = TYPED_ARRAY_BY_NAME[jsTypedArray(type.dtype)];
  if (!Ctor) throw new IRParseError(`no typed array for dtype '${type.dtype}'`, line);
  return new Ctor(values) as unknown as AttrValue;
}

function parseDenseAttr(sc: Scanner): { value: AttrValue; type: TensorType } {
  sc.expect('dense');
  const body = sc.readGroup('<', '>');
  sc.expect(':');
  const type = parseType(sc);
  if (!(type instanceof TensorType)) sc.fail('dense attribute needs a tensor type');
  return { value: parseDenseBody(body, type, sc.line), type };
}

function parseAttrValue(sc: Scanner): AttrValue {
  sc.skipSpace();
  if (sc.eat('unit')) return null;
  if (sc.eat('true')) return true;
  if (sc.eat('false')) return false;
  if (sc.peek() === '"') return sc.readString();
  if (sc.text.startsWith('array<', sc.pos)) {
    sc.expect('array');
    const body = sc.readGroup('<', '>');
    const colon = body.indexOf(':');
    if (colon < 0) return [];
    return splitTopLevel(body.slice(colon + 1), ',').map((v) => Number(v.trim()));
  }
  if (sc.text.startsWith('dense<', sc.pos)) return parseDenseAttr(sc).value;
  if (sc.peek() === '[') {
    const body = sc.readGroup('[', ']');
    if (body.trim() === '') return [];
    return splitTopLevel(body, ',').map((part) => parseAttrValue(new Scanner(part, sc.line)));
  }
  if (sc.peek() === '{') {
    const body = sc.readGroup('{', '}');
    const record: Record<string, AttrValue> = {};
    for (const [key, value] of attrEntries(body, sc.line)) record[key] = value;
    return record as unknown as AttrValue;
  }
  if (sc.text.startsWith('tensor<', sc.pos) || sc.text.startsWith('tuple<', sc.pos) || sc.text.startsWith('token', sc.pos)) {
    return parseType(sc) as unknown as AttrValue;
  }
  return parseScalarLiteral(sc, null);
}

function* attrEntries(body: string, line: number): Generator<[string, AttrValue]> {
  if (body.trim() === '') return;
  for (const entry of splitTopLevel(body, ',')) {
    const idx = entry.indexOf('=');
    if (idx < 0) {
      yield [entry.trim(), true];
      continue;
    }
    yield [entry.slice(0, idx).trim(), parseAttrValue(new Scanner(entry.slice(idx + 1), line))];
  }
}

function parseAttrs(body: string, line: number): Map<string, AttrValue> {
  const attrs = new Map<string, AttrValue>();
  for (const [key, value] of attrEntries(body, line)) attrs.set(key, value);
  return attrs;
}

type Line = { text: string; indent: number; no: number };

type BlockRecord = { argNames: string[]; argTypes: IRType[]; ops: OpRecord[] };
type OpRecord = {
  line: Line;
  opName: string;
  form: MlirOpForm | null;
  resultNames: string[];
  resultTypes: IRType[];
  attrs: Map<string, AttrValue>;
  operandNames: string[];
  regions: BlockRecord[][];
  deps: Set<string>;
  loc: Location | null;
};

const LOCATION_MARKER = ' loc(';
const FUNCTION_KEYWORD = 'func.func';
const FUNCTION_ATTRIBUTES = 'attributes';

export function splitTrailingLocation(text: string): { body: string; loc: string | null } {
  let depth = 0;
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '\\') i++;
      else if (ch === '"') quoted = false;
      continue;
    }
    if (ch === '"') { quoted = true; continue; }
    if (ch === '(' || ch === '[' || ch === '{') { depth++; continue; }
    if (ch === ')' || ch === ']' || ch === '}') { depth--; continue; }
    if (depth !== 0 || !text.startsWith(LOCATION_MARKER, i)) continue;
    if (!text.endsWith(')')) break;
    return { body: text.slice(0, i), loc: text.slice(i + LOCATION_MARKER.length, -1) };
  }
  return { body: text, loc: null };
}

function readBlockArgs(sc: Scanner, line: number): { names: string[]; types: IRType[] } {
  const names: string[] = [];
  const types: IRType[] = [];
  if (sc.peek() !== '(') return { names, types };
  const inner = sc.readGroup('(', ')');
  if (inner.trim() === '') return { names, types };
  for (const part of splitTopLevel(inner, ',')) {
    const argSc = new Scanner(part, line);
    names.push(argSc.valueName());
    argSc.expect(':');
    types.push(parseType(argSc));
  }
  return { names, types };
}

const CUSTOM_REGION_OPEN = ' {';

function leadingMnemonic(text: string): string {
  const assign = text.indexOf(' = ');
  const head = (assign < 0 ? text : text.slice(assign + ' = '.length)).trimStart();
  const end = head.search(/[^A-Za-z0-9_.]/);
  return end < 0 ? head : head.slice(0, end);
}

function openingRegionForm(line: Line): MlirRegionForm | null {
  if (line.text.endsWith(GENERIC_REGIONS.open)) return GENERIC_REGIONS;
  if (!line.text.endsWith(CUSTOM_REGION_OPEN)) return null;
  const mnemonic = leadingMnemonic(line.text);
  const form = mlirFormOfMnemonic(mnemonic);
  if (!form || !form.regions) {
    throw new IRParseError(`'${mnemonic}' opens a body but its assembly has none`, line.no);
  }
  return form.regions;
}

class RecordReader {
  lines: Line[];
  index: number;

  constructor(lines: Line[]) {
    this.lines = lines;
    this.index = 0;
  }

  get current(): Line | null {
    return this.index < this.lines.length ? this.lines[this.index] : null;
  }

  readBlocks(minIndent: number, closerIndent = -1): BlockRecord[] {
    const blocks: BlockRecord[] = [];
    for (;;) {
      const line = this.current;
      if (!line || line.indent < minIndent) break;
      if (line.indent <= closerIndent && line.text.startsWith('}')) break;
      if (line.text.startsWith('^')) {
        const sc = new Scanner(line.text, line.no);
        sc.expect('^');
        sc.identifier();
        const { names, types } = readBlockArgs(sc, line.no);
        sc.expect(':');
        blocks.push({ argNames: names, argTypes: types, ops: [] });
        this.index++;
        continue;
      }
      if (blocks.length === 0) blocks.push({ argNames: [], argTypes: [], ops: [] });
      blocks[blocks.length - 1].ops.push(this.readOp());
    }
    if (blocks.length === 0) blocks.push({ argNames: [], argTypes: [], ops: [] });
    return blocks;
  }

  readOp(): OpRecord {
    const source = this.current as Line;
    this.index++;
    const regions: BlockRecord[][] = [];
    let text = source.text;
    const regionForm = openingRegionForm(source);
    if (regionForm) {
      text = text.slice(0, -regionForm.open.length);
      for (let i = 0; ; i++) {
        regions.push(this.readBlocks(source.indent, source.indent));
        const closer = this.current;
        if (!closer || closer.indent !== source.indent || !closer.text.startsWith('}')) {
          throw new IRParseError(`unterminated region on '${source.text}'`, closer ? closer.no : source.no);
        }
        this.index++;
        const separator = regionForm.repeat ? regionForm.separators[0] : regionForm.separators[i];
        if (separator !== undefined && closer.text === separator) continue;
        if (!closer.text.startsWith(regionForm.close)) {
          throw new IRParseError(`unterminated region on '${source.text}'`, closer.no);
        }
        text += closer.text.slice(regionForm.close.length);
        break;
      }
    }
    const { body, loc: locText } = splitTrailingLocation(text);
    const line: Line = { text: body, indent: source.indent, no: source.no };
    const record = parseOpBody(line, regions);
    record.loc = locText === null ? null : parseLocation(locText);
    for (const region of regions) collectRegionDeps(region, record.deps);
    return record;
  }
}

function readResultNames(sc: Scanner, line: Line): string[] {
  const names: string[] = [];
  if (!line.text.startsWith('%')) return names;
  const assign = line.text.indexOf(' = ');
  if (assign < 0) return names;
  for (const part of splitTopLevel(line.text.slice(0, assign), ',')) {
    const partSc = new Scanner(part, line.no);
    const name = partSc.valueName();
    if (!partSc.eat(':')) {
      names.push(name);
      continue;
    }
    const count = Number(partSc.identifier());
    if (!Number.isInteger(count) || count < 1) {
      throw new IRParseError(`'${name}' declares a result group of ${count}`, line.no);
    }
    for (let i = 0; i < count; i++) names.push(`${name}#${i}`);
  }
  sc.pos = assign + ' = '.length;
  return names;
}

function parseOpBody(line: Line, regions: BlockRecord[][]): OpRecord {
  const sc = new Scanner(line.text, line.no);
  const resultNames = readResultNames(sc, line);
  const record: OpRecord = {
    line, opName: '', form: null, resultNames, resultTypes: [], attrs: new Map(),
    operandNames: [], regions, deps: new Set(), loc: null,
  };
  if (sc.peek() === '"') parseGenericOp(sc, record);
  else parseCustomOp(sc, record);
  if (record.resultTypes.length !== resultNames.length) {
    throw new IRParseError(
      `'${record.opName}' names ${resultNames.length} results but declares ${record.resultTypes.length} result types`,
      line.no);
  }
  for (const name of record.operandNames) record.deps.add(name);
  return record;
}

function parseGenericOp(sc: Scanner, record: OpRecord): void {
  record.opName = unqualify(sc.readString());
  const operandBody = sc.readGroup('(', ')');
  if (operandBody.trim() !== '') {
    for (const part of splitTopLevel(operandBody, ',')) {
      record.operandNames.push(new Scanner(part, record.line.no).valueName());
    }
  }
  if (sc.peek() === '{') record.attrs = parseAttrs(sc.readGroup('{', '}'), record.line.no);
  sc.expect(':');
  parseTypeList(sc.readGroup('(', ')'), record.line.no);
  sc.expect('->');
  record.resultTypes = parseGroupedTypes(sc);
}

function parseCustomOp(sc: Scanner, record: OpRecord): void {
  const mnemonic = sc.identifier();
  const form = mlirFormOfMnemonic(mnemonic);
  if (!form) throw new IRParseError(`unknown operation '${mnemonic}'`, record.line.no);
  record.opName = form.opName;
  record.form = form;

  if (form.keyword) {
    const spelling = sc.identifier();
    const irKind = form.keyword.toIr.get(spelling);
    if (irKind === undefined) throw new IRParseError(`'${mnemonic}' has no kind '${spelling}'`, record.line.no);
    record.attrs.set(form.keyword.ir, irKind);
    sc.expect(',');
  }
  if (form.groups) parseOperandGroups(sc, form, record);
  else {
    while (sc.peek() === '%') {
      record.operandNames.push(sc.valueName());
      if (!sc.eat(',')) break;
    }
  }
  if (sc.peek() === '{') {
    for (const [key, value] of attrEntries(sc.readGroup('{', '}'), record.line.no)) {
      const binding = form.attrByMlir.get(key);
      record.attrs.set(binding ? binding.ir : key,
                       binding && binding.kind === 'i64pairs' && isNumberArray(value)
                         ? nestPairs(value) as unknown as AttrValue
                         : value);
    }
  }
  for (const entry of form.fixed) record.attrs.set(entry.ir, entry.value);
  if (form.types === 'elements') {
    const dense = parseDenseAttr(sc);
    record.attrs.set('value', dense.value);
    record.resultTypes = [dense.type];
  } else if (form.types === 'resultList') {
    sc.expect('->');
    record.resultTypes = parseTypeList(sc.readGroup('(', ')'), record.line.no);
  } else if (sc.eat(':')) {
    record.resultTypes = parseCustomTypes(sc, form);
  }
  for (const entry of form.derived) {
    const value = derivedAttrValue(entry.from, record.resultTypes[0]);
    if (value !== undefined) record.attrs.set(entry.ir, value);
  }
}

function parseOperandGroups(sc: Scanner, form: MlirOpForm, record: OpRecord): void {
  const groups = form.groups as NonNullable<MlirOpForm['groups']>;
  for (const group of groups) {
    if (group.keyword === null) {
      const count = group.size.kind === 'fixed' ? group.size.count : 0;
      for (let i = 0; i < count; i++) {
        record.operandNames.push(sc.valueName());
        if (i + 1 < count) sc.expect(',');
      }
      continue;
    }
    if (!sc.eat(group.keyword)) {
      if (group.optional) {
        if (group.size.kind === 'attr') record.attrs.set(group.size.attr, 0);
        continue;
      }
      sc.fail(`expected '${group.keyword}'`);
    }
    const body = sc.readGroup('(', ')');
    const split = group.types ? splitTopLevel(body, ':') : [body];
    const names = split[0] && split[0].trim() !== ''
      ? splitTopLevel(split[0], ',').map((part) => new Scanner(part, record.line.no).valueName())
      : [];
    if (group.types) {
      const types = split.length > 1 ? parseTypeList(split.slice(1).join(':'), record.line.no) : [];
      if (types.length !== names.length) {
        throw new IRParseError(
          `'${form.mnemonic}' clause '${group.keyword}' names ${names.length} operands but declares ${types.length} types`,
          record.line.no);
      }
    }
    for (const name of names) record.operandNames.push(name);
    if (group.size.kind === 'attr') record.attrs.set(group.size.attr, names.length);
  }
}

function parseCustomTypes(sc: Scanner, form: MlirOpForm): IRType[] {
  switch (form.types) {
    case 'result':
      return [parseType(sc)];
    case 'functional': {
      parseTypeList(sc.readGroup('(', ')'), sc.line);
      sc.expect('->');
      return parseGroupedTypes(sc);
    }
    case 'operandToResult':
    case 'operandsToResult': {
      const rest = sc.text.slice(sc.pos);
      const arrow = rest.indexOf('->');
      if (arrow < 0) sc.fail("expected '->'");
      parseTypeList(rest.slice(0, arrow), sc.line);
      sc.pos += arrow + '->'.length;
      return [parseType(sc)];
    }
    case 'firstAndResult': {
      parseType(sc);
      sc.expect(',');
      return [parseType(sc)];
    }
    default:
      parseTypeList(sc.text.slice(sc.pos), sc.line);
      return [];
  }
}

function collectRegionDeps(blocks: readonly BlockRecord[], deps: Set<string>): void {
  for (const block of blocks) {
    for (const op of block.ops) {
      for (const name of op.deps) deps.add(name);
    }
  }
}

function dependencyOrder(ops: readonly OpRecord[]): OpRecord[] {
  const producer = new Map<string, OpRecord>();
  for (const op of ops) {
    for (const name of op.resultNames) producer.set(name, op);
  }
  const ordered: OpRecord[] = [];
  const state = new Map<OpRecord, number>();
  const visit = (op: OpRecord): void => {
    const seen = state.get(op);
    if (seen === 2) return;
    if (seen === 1) throw new IRParseError(`'${op.opName}' participates in a value dependency cycle`, op.line.no);
    state.set(op, 1);
    for (const name of op.deps) {
      const dep = producer.get(name);
      if (dep && dep !== op) visit(dep);
    }
    state.set(op, 2);
    ordered.push(op);
  };
  for (const op of ops) visit(op);
  return ordered;
}

type BuiltOp = { op: Operation; seed: Operation | null };

class Materializer {
  scopes: Map<string, Value>[];

  constructor(values: Map<string, Value>) {
    this.scopes = [values];
  }

  bind(name: string, value: Value, line: number): void {
    const scope = this.scopes[this.scopes.length - 1];
    if (scope.has(name)) throw new IRParseError(`value '${name}' is defined twice`, line);
    scope.set(name, value);
  }

  resolve(name: string, line: number): Value {
    for (let i = this.scopes.length - 1; i >= 0; i--) {
      const value = this.scopes[i].get(name);
      if (value) return value;
    }
    throw new IRParseError(`use of undefined value '${name}'`, line);
  }

  fillBlock(block: Block, record: BlockRecord): void {
    const built = new Map<OpRecord, BuiltOp>();
    for (const record0 of dependencyOrder(record.ops)) {
      built.set(record0, this.buildOp(record0));
    }
    for (const op of record.ops) {
      const entry = built.get(op) as BuiltOp;
      if (entry.seed) block.pushOp(entry.seed);
      block.pushOp(entry.op);
    }
  }

  buildOp(record: OpRecord): BuiltOp {
    const operands = record.operandNames.map((name) => this.resolve(name, record.line.no));
    const regions = record.regions.map((blocks) => {
      const region = new Region();
      this.scopes.push(new Map());
      for (const blockRecord of blocks) {
        const block = new Block(blockRecord.argTypes);
        region.addBlock(block);
        for (let i = 0; i < blockRecord.argNames.length; i++) {
          this.bind(blockRecord.argNames[i], block.arguments[i], record.line.no);
        }
        this.fillBlock(block, blockRecord);
      }
      this.scopes.pop();
      return region;
    });
    const seed = this.buildElided(record, operands, regions);
    const op = new Operation(record.opName, operands, record.resultTypes, record.attrs, regions);
    op.loc = record.loc;
    for (let i = 0; i < record.resultNames.length; i++) {
      this.bind(record.resultNames[i], op.getResult(i), record.line.no);
    }
    return { op, seed };
  }

  buildElided(record: OpRecord, operands: Value[], regions: Region[]): Operation | null {
    const form = record.form;
    if (!form || !form.combiner || operands.length === 0) return null;
    const scalarType = combinerScalarType(operands[0].type);
    if (!scalarType) throw new IRParseError(`'${record.opName}' accumulates into a non-tensor`, record.line.no);

    let seed: Operation | null = null;
    if (form.seedOperand >= 0) {
      const irKind = record.attrs.get(form.keyword!.ir) as string;
      const seedSpec = seedConstantAttrs(irKind, operands[0].type);
      if (!seedSpec) throw new IRParseError(`'${record.opName}' has no accumulator seed`, record.line.no);
      seed = new Operation('constant', [], [seedSpec.type], seedSpec.attrs, []);
      operands.splice(form.seedOperand, 0, seed.getResult(0));
    }

    const combiner = new Region();
    combiner.addBlock(new Block([scalarType, scalarType]));
    regions.push(combiner);
    return seed;
  }
}

function toLines(text: string, indentWidth: number): Line[] {
  const lines: Line[] = [];
  const raw = text.split('\n');
  for (let i = 0; i < raw.length; i++) {
    const stripped = raw[i].replace(/\s+$/, '');
    if (stripped.trim() === '') continue;
    const leading = stripped.length - stripped.trimStart().length;
    lines.push({ text: stripped.trimStart(), indent: Math.floor(leading / indentWidth), no: i });
  }
  return lines;
}

function parseSignature(line: Line): { name: string; inputTypes: IRType[]; outputTypes: IRType[]; argNames: string[]; attrs: Map<string, AttrValue> } {
  const sc = new Scanner(line.text, line.no);
  sc.expect(FUNCTION_KEYWORD);
  sc.expect('@');
  const name = sc.identifier();
  const argBody = sc.readGroup('(', ')');
  const argNames: string[] = [];
  const inputTypes: IRType[] = [];
  if (argBody.trim() !== '') {
    for (const part of splitTopLevel(argBody, ',')) {
      const argSc = new Scanner(part, line.no);
      argNames.push(argSc.valueName());
      argSc.expect(':');
      inputTypes.push(parseType(argSc));
    }
  }
  const outputTypes = sc.eat('->') ? parseGroupedTypes(sc) : [];
  const attrs = sc.eat(FUNCTION_ATTRIBUTES)
    ? parseAttrs(sc.readGroup('{', '}'), line.no)
    : new Map<string, AttrValue>();
  sc.expect('{');
  return { name, inputTypes, outputTypes, argNames, attrs };
}

function buildFunctionFrom(lines: Line[], start: number, indentWidth: number): { func: GraphFunction; next: number } {
  const header = lines[start];
  const { name, inputTypes, outputTypes, argNames, attrs } = parseSignature(header);
  const func = new GraphFunction(name, inputTypes, outputTypes);
  for (const [key, value] of attrs) func.setAttr(key, value);

  const bodyIndent = header.indent + 1;
  let end = start + 1;
  while (end < lines.length && !(lines[end].indent === header.indent && lines[end].text === '}')) end++;
  if (end >= lines.length) throw new IRParseError(`unterminated function '${name}'`, header.no);

  const values = new Map<string, Value>();
  for (let i = 0; i < argNames.length; i++) values.set(argNames[i], func.args[i]);

  const blocks = new RecordReader(lines.slice(start + 1, end)).readBlocks(bodyIndent);
  if (blocks.length > 1) {
    throw new IRParseError(`function '${name}' declares more than one top-level block`, header.no);
  }
  if (blocks[0].argNames.length > 0) {
    throw new IRParseError(`function '${name}' entry block cannot carry an explicit label`, header.no);
  }
  new Materializer(values).fillBlock(func.entryBlock, blocks[0]);

  return { func, next: end + 1 };
}

export type ParseOptions = Readonly<{ indentWidth?: number }>;

export function parseModule(text: string, { indentWidth = 2 }: ParseOptions = {}): GraphModule {
  const lines = toLines(text, indentWidth);
  if (lines.length === 0) throw new IRParseError('empty input', -1);

  const header = new Scanner(lines[0].text, lines[0].no);
  header.expect('module');
  header.expect('@');
  const module = new GraphModule(header.identifier());
  header.expect('{');

  let i = 1;
  while (i < lines.length && !(lines[i].indent === 0 && lines[i].text === '}')) {
    if (!lines[i].text.startsWith(FUNCTION_KEYWORD)) {
      throw new IRParseError(`expected a function, got '${lines[i].text}'`, lines[i].no);
    }
    const { func, next } = buildFunctionFrom(lines, i, indentWidth);
    module.addFunction(func);
    i = next;
  }
  return module;
}

export function parseFunction(text: string, { indentWidth = 2 }: ParseOptions = {}): GraphFunction {
  const lines = toLines(text, indentWidth);
  if (lines.length === 0) throw new IRParseError('empty input', -1);
  if (!lines[0].text.startsWith(FUNCTION_KEYWORD)) {
    throw new IRParseError(`expected a function, got '${lines[0].text}'`, lines[0].no);
  }
  return buildFunctionFrom(lines, 0, indentWidth).func;
}
