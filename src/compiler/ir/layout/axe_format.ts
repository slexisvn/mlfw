import { SymInt } from '../sym_int.js';
import { AxeAxis, AxeLayout, iter } from './axe.js';
import type { SymExpr } from '../sym_int.js';
import type { AxeAxisName, Coord, Iter } from './axe.js';

export type ExprCodec = Readonly<{
  format(expr: SymExpr): string;
  parse(text: string): SymExpr;
}>;

const EXPR_TERMINATORS = new Set([':', '@', ',', ')', ']', '}', '+']);

export class AxeFormatError extends Error {}

export const defaultExprCodec: ExprCodec = Object.freeze({
  format(expr: SymExpr): string {
    return String(expr);
  },
  parse(text: string): SymExpr {
    if (/^-?\d+$/.test(text)) return Number(text);
    if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(text)) return SymInt.var(text);
    throw new AxeFormatError(`cannot read '${text}' as a layout extent or stride; pass an expression codec that can parse it`);
  }
});

class LayoutScanner {
  private _text: string;
  private _pos: number;

  constructor(text: string) {
    this._text = text;
    this._pos = 0;
  }

  skipSpace(): void {
    while (this._pos < this._text.length && /\s/.test(this._text[this._pos])) this._pos++;
  }

  atEnd(): boolean {
    this.skipSpace();
    return this._pos >= this._text.length;
  }

  peek(): string {
    this.skipSpace();
    return this._text[this._pos] ?? '';
  }

  eat(ch: string): boolean {
    if (this.peek() !== ch) return false;
    this._pos++;
    return true;
  }

  expect(ch: string): void {
    if (!this.eat(ch)) throw new AxeFormatError(`expected '${ch}' at position ${this._pos} of '${this._text}'`);
  }

  axis(): AxeAxisName {
    this.skipSpace();
    const start = this._pos;
    while (this._pos < this._text.length && /[A-Za-z0-9_.]/.test(this._text[this._pos])) this._pos++;
    if (this._pos === start) throw new AxeFormatError(`expected an axis name at position ${start} of '${this._text}'`);
    return this._text.slice(start, this._pos);
  }

  expr(): string {
    this.skipSpace();
    const start = this._pos;
    let depth = 0;
    while (this._pos < this._text.length) {
      const ch = this._text[this._pos];
      if (ch === '(') depth++;
      else if (ch === ')' && depth > 0) depth--;
      else if (depth === 0 && EXPR_TERMINATORS.has(ch)) break;
      this._pos++;
    }
    const text = this._text.slice(start, this._pos).trim();
    if (text === '') throw new AxeFormatError(`expected an expression at position ${start} of '${this._text}'`);
    return text;
  }
}

function formatIter(it: Iter, codec: ExprCodec): string {
  return `${codec.format(it.extent)}:${codec.format(it.stride)}@${it.axis}`;
}

export function formatAxeLayout(layout: AxeLayout, codec: ExprCodec = defaultExprCodec): string {
  const parts = [`(${layout.shard.map(it => formatIter(it, codec)).join(', ')})`];
  if (layout.replica.length > 0) {
    parts.push(`[${layout.replica.map(it => formatIter(it, codec)).join(', ')}]`);
  }
  const offset = [...layout.offset.entries()].filter(([, value]) => value !== 0);
  if (offset.length > 0) {
    offset.sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
    parts.push(`{${offset.map(([axis, value]) => `${axis}:${codec.format(value)}`).join(', ')}}`);
  }
  return parts.join(' + ');
}

function parseIters(sc: LayoutScanner, close: string, codec: ExprCodec): Iter[] {
  const out: Iter[] = [];
  if (sc.eat(close)) return out;
  for (;;) {
    const extent = codec.parse(sc.expr());
    sc.expect(':');
    const stride = codec.parse(sc.expr());
    sc.expect('@');
    out.push(iter(extent, stride, sc.axis()));
    if (sc.eat(',')) continue;
    sc.expect(close);
    return out;
  }
}

function parseOffset(sc: LayoutScanner, codec: ExprCodec): Coord {
  const out = new Map<AxeAxisName, SymExpr>();
  if (sc.eat('}')) return out;
  for (;;) {
    const axis = sc.axis();
    sc.expect(':');
    out.set(axis, codec.parse(sc.expr()));
    if (sc.eat(',')) continue;
    sc.expect('}');
    return out;
  }
}

export function parseAxeLayout(text: string, codec: ExprCodec = defaultExprCodec): AxeLayout {
  const sc = new LayoutScanner(text);
  sc.expect('(');
  const shard = parseIters(sc, ')', codec);
  let replica: Iter[] = [];
  let offset: Coord = new Map();

  while (sc.eat('+')) {
    if (sc.eat('[')) {
      if (replica.length > 0) throw new AxeFormatError(`'${text}' declares the replica set twice`);
      replica = parseIters(sc, ']', codec);
      continue;
    }
    if (sc.eat('{')) {
      if (offset.size > 0) throw new AxeFormatError(`'${text}' declares the offset twice`);
      offset = parseOffset(sc, codec);
      continue;
    }
    throw new AxeFormatError(`expected '[' or '{' after '+' in '${text}'`);
  }
  if (!sc.atEnd()) throw new AxeFormatError(`unexpected trailing text in '${text}'`);
  return new AxeLayout(shard, replica, offset);
}

export function formatPermutation(order: readonly number[]): string {
  return `[${order.join(', ')}]`;
}

export function permutationOf(layout: AxeLayout, rank: number): number[] | null {
  if (layout.shard.length !== rank) return null;
  const strides: number[] = [];
  for (const it of layout.shard) {
    if (it.axis !== AxeAxis.MEM) return null;
    if (typeof it.stride !== 'number') return null;
    strides.push(it.stride);
  }
  if (layout.replica.length > 0 || layout.offset.size > 0) return null;
  const order = strides.map((_, dim) => dim).sort((a, b) => strides[b] - strides[a]);
  let expected = 1;
  for (let i = order.length - 1; i >= 0; i--) {
    if (strides[order[i]] !== expected) return null;
    const extent = layout.shard[order[i]].extent;
    if (typeof extent !== 'number') return null;
    expected *= extent;
  }
  return order;
}
