import { describe, it, expect } from 'vitest';
import { readFileSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { DocumentAnalyzer } from '../server/analyzer/document_analyzer.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const EXAMPLES = join(HERE, '../../examples');
const LANGUAGE_DATA = JSON.parse(readFileSync(join(HERE, '../language-data.json'), 'utf8'));

describe('DocumentAnalyzer', () => {
  it('parses examples without errors', () => {
    const analyzer = new DocumentAnalyzer();
    const sources = readdirSync(EXAMPLES).filter(f => f.endsWith('.tera'));
    expect(sources.length).toBeGreaterThan(0);
    for (const file of sources) {
      const text = readFileSync(join(EXAMPLES, file), 'utf8');
      const doc = analyzer.update(`file://${file}`, text);
      expect(doc.errors, `errors in ${file}`).toEqual([]);
      expect(doc.ast).toBeTruthy();
      expect(doc.symbols.flat.length).toBeGreaterThan(0);
    }
  });

  it('reports diagnostic for syntax error', () => {
    const analyzer = new DocumentAnalyzer();
    const doc = analyzer.update('file://bad.tera', 'model Foo\n  forward x: x');
    expect(doc.errors.length).toBeGreaterThan(0);
    expect(doc.errors[0]).toHaveProperty('line');
    expect(doc.errors[0]).toHaveProperty('column');
  });

  it('captures model and field symbols', () => {
    const analyzer = new DocumentAnalyzer();
    const text = 'model MLP(hidden):\n  fc1 = Linear(784, hidden)\n  forward x:\n    return fc1(x)\n';
    const doc = analyzer.update('file://m.tera', text);
    const names = doc.symbols.flat.map(s => s.name);
    expect(names).toContain('MLP');
    expect(names).toContain('hidden');
    expect(names).toContain('fc1');
    expect(names).toContain('x');
  });

  it('points forward/train params at their identifier column', () => {
    const analyzer = new DocumentAnalyzer();
    const text = [
      'model Foo():',
      '  forward x:',
      '    return x',
      '  train batch:',
      '    a, b = batch',
      '    return a',
    ].join('\n');
    const doc = analyzer.update('file://p.tera', text);
    const xSym = doc.symbols.resolve('x', { line: 2, character: 12 });
    expect(xSym?.line).toBe(2);
    expect(xSym?.column).toBe(text.split('\n')[1].indexOf('x') + 1);
    const batchSym = doc.symbols.resolve('batch', { line: 4, character: 12 });
    expect(batchSym?.line).toBe(4);
    expect(batchSym?.column).toBe(text.split('\n')[3].indexOf('batch') + 1);
    const aSym = doc.symbols.resolve('a', { line: 5, character: 12 });
    expect(aSym?.line).toBe(5);
    expect(aSym?.column).toBe(text.split('\n')[4].indexOf('a') + 1);
  });

  it('points model and fn symbols at the name, not the keyword', () => {
    const analyzer = new DocumentAnalyzer();
    const text = 'model IrisNet():\n  fc = Linear(4, 3)\n  forward x: return fc(x)\n';
    const doc = analyzer.update('file://n.tera', text);
    const sym = doc.symbols.resolve('IrisNet', { line: 0, character: 8 });
    expect(sym).toBeTruthy();
    expect(sym.line).toBe(1);
    expect(sym.column).toBe(text.indexOf('IrisNet') + 1);
  });

  it('resolves variables to the innermost matching scope', () => {
    const analyzer = new DocumentAnalyzer();
    const text = [
      'x = 1',
      'fn outer():',
      '  x = 2',
      '  y = x',
      'fn other():',
      '  z = x',
      '',
    ].join('\n');
    const doc = analyzer.update('file://r.tera', text);
    const insideOuter = doc.symbols.resolve('x', { line: 3, character: 6 });
    expect(insideOuter?.line).toBe(3);
    const insideOther = doc.symbols.resolve('x', { line: 5, character: 6 });
    expect(insideOther?.line).toBe(1);
  });
});

describe('DocumentAnalyzer type checking', () => {
  const analyzer = () => new DocumentAnalyzer(LANGUAGE_DATA);
  const typeErrors = doc => doc.errors.filter(e => e.source === 'typecheck');

  it('accepts well-typed code', () => {
    const doc = analyzer().update('file://good.tera', 'fn square(n: int) -> int:\n  return n * n');
    expect(typeErrors(doc)).toEqual([]);
  });

  it('flags a return-type violation at its location', () => {
    const doc = analyzer().update('file://bad.tera', 'fn f() -> Tensor:\n  return "hi"');
    const errors = typeErrors(doc);
    expect(errors.length).toBeGreaterThan(0);
    expect(errors[0].line).toBe(2);
  });

  it('flags undefined names', () => {
    const doc = analyzer().update('file://u.tera', 'missing(1)');
    expect(typeErrors(doc).some(e => e.message.includes("undefined name 'missing'"))).toBe(true);
  });

  it('stays parse-only without language data', () => {
    const doc = new DocumentAnalyzer().update('file://x.tera', 'missing(1)');
    expect(doc.errors.filter(e => e.source === 'typecheck')).toEqual([]);
  });

  it('records annotation types on symbols for hover', () => {
    const doc = analyzer().update('file://t.tera', 'fn f(x: int) -> Tensor:\n  return zeros([2])\ny: list[int] = [1]');
    expect(doc.symbols.flat.find(s => s.name === 'x')?.typeName).toBe('int');
    expect(doc.symbols.flat.find(s => s.name === 'y')?.typeName).toBe('list[int]');
    expect(doc.symbols.flat.find(s => s.name === 'f' && s.kind === 'function')?.typeName).toBe('Tensor');
  });

  it('does not flag the type operators as syntax errors', () => {
    const doc = analyzer().update('file://op.tera', 'fn pick(x: int | str) -> int:\n  return 1');
    expect(doc.errors.filter(e => e.source === 'Parser' || e.source === 'Tokenizer')).toEqual([]);
  });

  it('resolves a model field to its builtin type for member hover', () => {
    const text = 'model M():\n  enc = LSTM(2, 3, 1, true)\n  forward (x: Tensor) -> Tensor:\n    return enc(x)';
    const doc = analyzer().update('file://field.tera', text);
    expect(doc.symbols.resolveField('M', 'enc')?.typeName).toBe('LSTM');
    expect(doc.symbols.resolveField('M', 'missing')).toBeNull();
  });
});
