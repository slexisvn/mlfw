import { describe, expect, it } from 'vitest';
import { parse } from '../../src/cli/parser.js';

describe('Tensor Lang parser', () => {
  it('preserves source locations on AST nodes', () => {
    const program = parse('\nvalue = missing(1)');
    expect(program.body[0]).toMatchObject({ type: 'Assign', line: 2, column: 1 });
    expect(program.body[0].value.callee).toMatchObject({ type: 'Identifier', line: 2, column: 9 });
  });

  it('parses multidimensional slices', () => {
    const expression = parse('x[:, 1:4:2]').body[0].expression;
    expect(expression.type).toBe('Index');
    expect(expression.items[0]).toMatchObject({ type: 'Slice', start: null, end: null, step: null });
    expect(expression.items[1]).toMatchObject({ type: 'Slice' });
  });

  it('accepts trailing commas in arrays, calls, parameters, and indices', () => {
    expect(() => parse(`
      model MLP(input, hidden,) {
        forward x, {
          return tensor([
            [1, 2],
            [3, 4],
          ],)[0,]
        }
      }
    `)).not.toThrow();
  });
});
