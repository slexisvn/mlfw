import { Block, Region } from './block.js';
import { Operation } from './operation.js';
import { GraphFunction } from './function.js';
import { GraphModule } from './module.js';
import { TensorType, TupleType, TokenType, FunctionType, Layout, ScalarType, DYNAMIC } from './types.js';
import { SymInt } from '../../analysis/sym_int.js';
import { jsTypedArray } from '../../../util/dtype_map.js';
import type { AttrValue, Dim, IRType, ScalarDType } from './types.js';
import type { Value } from './value.js';
import { parseLocation } from '../location.js';
import type { Location } from '../location.js';

const SCALAR_TYPES = new Set<string>(Object.values(ScalarType));

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
        while (this.pos < this.text.length && this.text[this.pos] !== '"') this.pos++;
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

  identifier(): string {
    this.skipSpace();
    const name = this.readWhile((ch) => /[A-Za-z0-9_.$]/.test(ch));
    if (name === '') this.fail('expected an identifier');
    return name;
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
      if (ch === '"') inString = false;
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
  const dtype = segments[segments.length - 1].trim();
  if (!SCALAR_TYPES.has(dtype)) {
    throw new IRParseError(`'tensor<${inner}>' does not end in a known dtype (got '${dtype}')`, line);
  }
  const dimParts = segments.slice(0, -1);
  if (dimParts.length === 1 && dimParts[0].trim() === '') dimParts.pop();
  const shape: Dim[] = dimParts.map((d) => parseDim(d, line));
  let layout: Layout | null = null;
  if (parts.length > 1) {
    const order = parts[1].trim();
    layout = new Layout(order.slice(1, -1).split(',').map((v) => Number(v.trim())));
  }
  return new TensorType(shape, dtype as ScalarDType, layout);
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

function parseAttrValue(sc: Scanner): AttrValue {
  sc.skipSpace();
  if (sc.eat('null')) return null;
  if (sc.eat('true')) return true;
  if (sc.eat('false')) return false;
  if (sc.eat('-inf')) return -Infinity;
  if (sc.eat('inf')) return Infinity;
  if (sc.eat('nan')) return NaN;
  if (sc.peek() === '"') {
    sc.expect('"');
    const value = sc.readWhile((ch) => ch !== '"');
    sc.expect('"');
    return value;
  }
  if (sc.eat('dense<')) {
    sc.pos -= 'dense<'.length;
    sc.expect('dense');
    const dtype = sc.readGroup('<', '>').trim();
    const body = sc.readGroup('[', ']');
    const values = body.trim() === '' ? [] : splitTopLevel(body, ',').map((v) => Number(parseAttrValue(new Scanner(v, sc.line))));
    const Ctor = TYPED_ARRAY_BY_NAME[jsTypedArray(dtype)];
    if (!Ctor) sc.fail(`no typed array for dtype '${dtype}'`);
    return new Ctor(values) as unknown as AttrValue;
  }
  if (sc.peek() === '[') {
    const body = sc.readGroup('[', ']');
    if (body.trim() === '') return [];
    return splitTopLevel(body, ',').map((part) => parseAttrValue(new Scanner(part, sc.line)));
  }
  if (sc.peek() === '{') {
    const body = sc.readGroup('{', '}');
    const record: Record<string, AttrValue> = {};
    if (body.trim() !== '') {
      for (const entry of splitTopLevel(body, ',')) {
        const idx = entry.indexOf(':');
        if (idx < 0) throw new IRParseError(`malformed attribute record entry '${entry.trim()}'`, sc.line);
        record[entry.slice(0, idx).trim()] = parseAttrValue(new Scanner(entry.slice(idx + 1), sc.line));
      }
    }
    return record as unknown as AttrValue;
  }
  if (sc.text.startsWith('tensor<', sc.pos) || sc.text.startsWith('tuple<', sc.pos) || sc.text.startsWith('token', sc.pos)) {
    return parseType(sc) as unknown as AttrValue;
  }
  const numeric = sc.readWhile((ch) => /[-+0-9.eE]/.test(ch));
  if (numeric === '') sc.fail('expected an attribute value');
  const value = Number(numeric);
  if (Number.isNaN(value)) sc.fail(`invalid number '${numeric}'`);
  return value;
}

function parseAttrs(body: string, line: number): Map<string, AttrValue> {
  const attrs = new Map<string, AttrValue>();
  if (body.trim() === '') return attrs;
  for (const entry of splitTopLevel(body, ',')) {
    const idx = entry.indexOf('=');
    if (idx < 0) throw new IRParseError(`malformed attribute '${entry.trim()}'`, line);
    attrs.set(entry.slice(0, idx).trim(), parseAttrValue(new Scanner(entry.slice(idx + 1), line)));
  }
  return attrs;
}

type Line = { text: string; indent: number; no: number };

type BlockRecord = { argNames: string[]; argTypes: IRType[]; ops: OpRecord[] };
type OpRecord = {
  line: Line;
  opName: string;
  resultNames: string[];
  resultTypes: IRType[];
  attrs: Map<string, AttrValue>;
  operandNames: string[];
  regions: BlockRecord[][];
  deps: Set<string>;
  loc: Location | null;
};

const LOCATION_MARKER = ' loc(';

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
  const inner = sc.readGroup('(', ')');
  const names: string[] = [];
  const types: IRType[] = [];
  if (inner.trim() === '') return { names, types };
  for (const part of splitTopLevel(inner, ',')) {
    const argSc = new Scanner(part, line);
    argSc.expect('%');
    names.push(`%${argSc.identifier()}`);
    argSc.expect(':');
    types.push(parseType(argSc));
  }
  return { names, types };
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

  readBlocks(indent: number): BlockRecord[] {
    const blocks: BlockRecord[] = [];
    for (;;) {
      const line = this.current;
      if (!line || line.indent < indent) break;
      if (line.text.startsWith('^bb')) {
        const sc = new Scanner(line.text.slice('^bb'.length), line.no);
        const { names, types } = readBlockArgs(sc, line.no);
        blocks.push({ argNames: names, argTypes: types, ops: [] });
        this.index++;
        continue;
      }
      if (blocks.length === 0) blocks.push({ argNames: [], argTypes: [], ops: [] });
      blocks[blocks.length - 1].ops.push(this.readOp(indent));
    }
    if (blocks.length === 0) blocks.push({ argNames: [], argTypes: [], ops: [] });
    return blocks;
  }

  readOp(indent: number): OpRecord {
    const source = this.current as Line;
    this.index++;
    const { body, loc: locText } = splitTrailingLocation(source.text);
    const line: Line = body === source.text ? source : { text: body, indent: source.indent, no: source.no };
    const loc = locText === null ? null : parseLocation(locText);
    const sc = new Scanner(line.text, line.no);

    const resultNames: string[] = [];
    const assign = line.text.indexOf(' = ');
    if (assign >= 0 && line.text.startsWith('%')) {
      for (const part of splitTopLevel(line.text.slice(0, assign), ',')) {
        const nameSc = new Scanner(part, line.no);
        nameSc.expect('%');
        resultNames.push(`%${nameSc.identifier()}`);
      }
      sc.pos = assign + ' = '.length;
    }

    const opName = sc.identifier();
    const operandBody = sc.readGroup('(', ')');
    const operandNames = operandBody.trim() === ''
      ? []
      : splitTopLevel(operandBody, ',').map((part) => {
        const opSc = new Scanner(part, line.no);
        opSc.expect('%');
        return `%${opSc.identifier()}`;
      });

    const attrs = sc.peek() === '{' ? parseAttrs(sc.readGroup('{', '}'), line.no) : new Map<string, AttrValue>();
    const resultTypes = sc.eat(':') ? parseTypeList(sc.text.slice(sc.pos), line.no) : [];
    if (resultTypes.length !== resultNames.length) {
      throw new IRParseError(`'${opName}' names ${resultNames.length} results but declares ${resultTypes.length} result types`, line.no);
    }

    const regions: BlockRecord[][] = [];
    while (this.current && this.current.indent === indent && this.current.text === '{') {
      this.index++;
      regions.push(this.readBlocks(indent + 1));
      const closer = this.current;
      if (!closer || closer.indent !== indent || closer.text !== '}') {
        throw new IRParseError(`unterminated region for '${opName}'`, closer ? closer.no : line.no);
      }
      this.index++;
    }

    const deps = new Set<string>(operandNames);
    for (const region of regions) collectRegionDeps(region, deps);
    return { line, opName, resultNames, resultTypes, attrs, operandNames, regions, deps, loc };
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

class Materializer {
  values: Map<string, Value>;

  constructor(values: Map<string, Value>) {
    this.values = values;
  }

  bind(name: string, value: Value, line: number): void {
    if (this.values.has(name)) throw new IRParseError(`value '${name}' is defined twice`, line);
    this.values.set(name, value);
  }

  resolve(name: string, line: number): Value {
    const value = this.values.get(name);
    if (!value) throw new IRParseError(`use of undefined value '${name}'`, line);
    return value;
  }

  fillBlock(block: Block, record: BlockRecord): void {
    const built = new Map<OpRecord, Operation>();
    for (const record0 of dependencyOrder(record.ops)) {
      built.set(record0, this.buildOp(record0));
    }
    for (const op of record.ops) block.pushOp(built.get(op) as Operation);
  }

  buildOp(record: OpRecord): Operation {
    const operands = record.operandNames.map((name) => this.resolve(name, record.line.no));
    const regions = record.regions.map((blocks) => {
      const region = new Region();
      for (const blockRecord of blocks) {
        const block = new Block(blockRecord.argTypes);
        region.addBlock(block);
        for (let i = 0; i < blockRecord.argNames.length; i++) {
          this.bind(blockRecord.argNames[i], block.arguments[i], record.line.no);
        }
        this.fillBlock(block, blockRecord);
      }
      return region;
    });
    const op = new Operation(record.opName, operands, record.resultTypes, record.attrs, regions);
    op.loc = record.loc;
    for (let i = 0; i < record.resultNames.length; i++) {
      this.bind(record.resultNames[i], op.getResult(i), record.line.no);
    }
    return op;
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

function parseSignature(line: Line): { name: string; inputTypes: IRType[]; outputTypes: IRType[]; argNames: string[] } {
  const sc = new Scanner(line.text, line.no);
  sc.expect('func');
  sc.expect('@');
  const name = sc.identifier();
  const argBody = sc.readGroup('(', ')');
  const argNames: string[] = [];
  const inputTypes: IRType[] = [];
  if (argBody.trim() !== '') {
    for (const part of splitTopLevel(argBody, ',')) {
      const argSc = new Scanner(part, line.no);
      argSc.expect('%');
      argNames.push(`%${argSc.identifier()}`);
      argSc.expect(':');
      inputTypes.push(parseType(argSc));
    }
  }
  sc.expect('->');
  const outputTypes = parseTypeList(sc.readGroup('(', ')'), line.no);
  sc.expect('{');
  return { name, inputTypes, outputTypes, argNames };
}

function buildFunctionFrom(lines: Line[], start: number, indentWidth: number): { func: GraphFunction; next: number } {
  const header = lines[start];
  const { name, inputTypes, outputTypes, argNames } = parseSignature(header);
  const func = new GraphFunction(name, inputTypes, outputTypes);

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
    if (!lines[i].text.startsWith('func')) throw new IRParseError(`expected a function, got '${lines[i].text}'`, lines[i].no);
    const { func, next } = buildFunctionFrom(lines, i, indentWidth);
    module.addFunction(func);
    i = next;
  }
  return module;
}

export function parseFunction(text: string, { indentWidth = 2 }: ParseOptions = {}): GraphFunction {
  const lines = toLines(text, indentWidth);
  if (lines.length === 0) throw new IRParseError('empty input', -1);
  if (!lines[0].text.startsWith('func')) throw new IRParseError(`expected a function, got '${lines[0].text}'`, lines[0].no);
  return buildFunctionFrom(lines, 0, indentWidth).func;
}
