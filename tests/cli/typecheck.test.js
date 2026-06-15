import { describe, expect, it } from 'vitest';
import fs from 'node:fs';
import { checkSource } from '../../src/cli/check.js';
import { parse } from '../../src/cli/parser.js';
import { TeraRuntime } from '../../src/cli/runtime.js';

const diagnose = source => checkSource(source).diagnostics;

describe('Tera type checker', () => {
  describe('accepts well-typed programs', () => {
    const cases = [
      'fn square(n: int) -> int:\n  return n * n',
      'fn f(x: int, y: Tensor) -> Tensor:\n  return y',
      'x: int = 5',
      'x: list[int] = [1, 2, 3]',
      'm: dict[str, int] = {"a": 1}',
      'fn g(s: str) -> str:\n  return s + "!"',
      'fn h(x: int | str) -> int:\n  return 1\nh(1)\nh("a")',
      'x = tensor([1, 2])\ny = x.shape',
      'fn add(a: int, b: int) -> int:\n  return a + b\nz = add(1, 2)',
      'fn make(b: int) -> fn(int) -> int:\n  fn add(x: int) -> int:\n    return b + x\n  return add\ncall = make(1)\nr = call(2)',
    ];
    for (const source of cases) {
      it(JSON.stringify(source), () => expect(diagnose(source)).toEqual([]));
    }
  });

  describe('reports provable mismatches with locations', () => {
    it('flags arithmetic between number and string', () => {
      const [error] = diagnose('z = 1 * "a"');
      expect(error).toMatchObject({ name: 'TypeError', line: 1, column: 7 });
      expect(error.message).toContain("operator '*'");
    });

    it('flags a return value that violates the declared return type', () => {
      const [error] = diagnose('fn f() -> Tensor:\n  return "hi"');
      expect(error).toMatchObject({ name: 'TypeError', line: 2 });
      expect(error.message).toContain('not assignable');
    });

    it('flags a variable assignment that violates its annotation', () => {
      const [error] = diagnose('x: int = "s"');
      expect(error).toMatchObject({ name: 'TypeError', line: 1, column: 1 });
    });

    it('flags wrong argument count against a known function', () => {
      const [error] = diagnose('fn add(a: int, b: int) -> int:\n  return a + b\nadd(1)');
      expect(error.message).toContain('expected 2 argument(s), got 1');
    });

    it('flags undefined names', () => {
      const [error] = diagnose('missing(1)');
      expect(error).toMatchObject({ name: 'TypeError', line: 1, column: 1 });
      expect(error.message).toContain("undefined name 'missing'");
    });

    it('requires param and return annotations (strict)', () => {
      expect(diagnose('fn f(x):\n  return x').length).toBeGreaterThanOrEqual(2);
    });

    it('accumulates multiple diagnostics in one pass', () => {
      expect(diagnose('a = nope\nb = 1 * "x"').length).toBeGreaterThanOrEqual(2);
    });

    it('does not support `any` as a type escape hatch', () => {
      expect(diagnose('v: any = 5').length).toBeGreaterThan(0);
    });
  });

  describe('parser keeps annotations additive', () => {
    it('exposes paramTypes and returnType without changing params', () => {
      const fn = parse('fn f(x: int, y: Tensor) -> Tensor:\n  return y').body[0];
      expect(fn.params).toEqual(['x', 'y']);
      expect(fn.paramTypes).toHaveLength(2);
      expect(fn.returnType).toMatchObject({ kind: 'NameType', name: 'Tensor' });
    });

    it('parses union annotations on variables', () => {
      const assign = parse('x: int | str = 1').body[0];
      expect(assign).toMatchObject({ type: 'Assign', name: 'x' });
      expect(assign.annotation).toMatchObject({ kind: 'UnionType' });
    });

    it('parses function-type annotations', () => {
      const fn = parse('fn make() -> fn(int) -> int:\n  return make').body[0];
      expect(fn.returnType).toMatchObject({ kind: 'FunctionType' });
      expect(fn.returnType.params).toHaveLength(1);
    });

    it('still parses untyped declarations unchanged', () => {
      const fn = parse('fn square(n):\n  return n * n').body[0];
      expect(fn.params).toEqual(['n']);
      expect(fn.paramTypes).toEqual([null]);
      expect(fn.returnType).toBeNull();
    });
  });

  describe('runtime ignores type annotations', () => {
    it('executes annotated assignments unchanged', async () => {
      const result = await new TeraRuntime({ output: () => {} }).execute('x: int = 41\nx + 1');
      expect(result).toBe(42);
    });
  });

  describe('backward compatibility', () => {
    it('still parses every bundled example', () => {
      const dir = new URL('../../examples/', import.meta.url);
      const files = fs.readdirSync(dir).filter(name => name.endsWith('.tera'));
      expect(files.length).toBeGreaterThan(0);
      for (const name of files) {
        expect(() => parse(fs.readFileSync(new URL(name, dir), 'utf8'))).not.toThrow();
      }
    });
  });
});
