import { tokenize, LangSyntaxError } from '../vendor/tokenizer.js';
import { parse } from '../vendor/parser.js';
import { buildSymbolTable } from './symbol_table.js';

export class DocumentAnalyzer {
  constructor() {
    this._cache = new Map();
  }

  update(uri, text) {
    const prev = this._cache.get(uri);
    const result = analyze(text);
    if (!result.ast && prev?.symbols && prev.symbols.flat.length) {
      result.symbols = prev.symbols;
    }
    this._cache.set(uri, { text, ...result });
    return this._cache.get(uri);
  }

  get(uri) {
    return this._cache.get(uri) ?? null;
  }

  drop(uri) {
    this._cache.delete(uri);
  }
}

function analyze(text) {
  const tokens = safeTokenize(text);
  const ast = safeParse(text);
  let symbols = ast.program ? buildSymbolTable(ast.program, text) : null;
  if (!symbols) {
    const fallback = parseTruncated(text);
    if (fallback) symbols = buildSymbolTable(fallback, text);
  }
  if (!symbols) symbols = emptySymbolTable();
  const errors = [];
  if (tokens.error) errors.push(tokens.error);
  if (ast.error) errors.push(ast.error);
  return { tokens: tokens.value, ast: ast.program, symbols, errors };
}

function parseTruncated(text) {
  const lines = text.split('\n');
  for (let drop = 1; drop <= Math.min(5, lines.length); drop++) {
    const trimmed = lines.slice(0, lines.length - drop).join('\n');
    if (!trimmed.trim()) return null;
    try {
      return parse(trimmed);
    } catch {}
  }
  return null;
}

function emptySymbolTable() {
  const empty = { name: '<empty>', parent: null, children: [], symbols: [], startLine: 1, endLine: 1 };
  return {
    scopes: [empty],
    root: empty,
    flat: [],
    findScopeAt: () => empty,
    resolve: () => null,
  };
}

function safeTokenize(text) {
  try {
    return { value: tokenize(text), error: null };
  } catch (e) {
    return { value: [], error: toDiagnosticError(e, 'Tokenizer') };
  }
}

function safeParse(text) {
  try {
    return { program: parse(text), error: null };
  } catch (e) {
    return { program: null, error: toDiagnosticError(e, 'Parser') };
  }
}

function toDiagnosticError(e, source) {
  if (e instanceof LangSyntaxError) {
    return { message: e.message.replace(/ at \d+:\d+$/, ''), line: e.line ?? 1, column: e.column ?? 1, source };
  }
  return { message: e.message ?? String(e), line: 1, column: 1, source };
}
