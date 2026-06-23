import { describe, it, expect } from 'vitest';
import {
  IntImmNode, MathOpNode, VariableNode, mathOp
} from '../../../src/compiler/ir/tensor/nodes.js';

function idx(name) {
  return new VariableNode(name, 'index');
}

describe('mathOp constant folding', () => {
  it('folds int + int', () => {
    const result = mathOp('+', new IntImmNode(3), new IntImmNode(5));
    expect(result).toBeInstanceOf(IntImmNode);
    expect(result.value).toBe(8);
  });

  it('folds int * int', () => {
    const result = mathOp('*', new IntImmNode(4), new IntImmNode(7));
    expect(result).toBeInstanceOf(IntImmNode);
    expect(result.value).toBe(28);
  });

  it('folds int % int with Python semantics', () => {
    const result = mathOp('%', new IntImmNode(7), new IntImmNode(3));
    expect(result).toBeInstanceOf(IntImmNode);
    expect(result.value).toBe(1);
  });

  it('folds int // int', () => {
    const result = mathOp('//', new IntImmNode(7), new IntImmNode(3));
    expect(result).toBeInstanceOf(IntImmNode);
    expect(result.value).toBe(2);
  });
});

describe('mathOp identity simplifications — right operand', () => {
  it('x + 0 → x', () => {
    const x = idx('x');
    expect(mathOp('+', x, new IntImmNode(0))).toBe(x);
  });

  it('x - 0 → x', () => {
    const x = idx('x');
    expect(mathOp('-', x, new IntImmNode(0))).toBe(x);
  });

  it('x * 1 → x', () => {
    const x = idx('x');
    expect(mathOp('*', x, new IntImmNode(1))).toBe(x);
  });

  it('x * 0 → 0', () => {
    const result = mathOp('*', idx('x'), new IntImmNode(0));
    expect(result).toBeInstanceOf(IntImmNode);
    expect(result.value).toBe(0);
  });

  it('x // 1 → x', () => {
    const x = idx('x');
    expect(mathOp('//', x, new IntImmNode(1))).toBe(x);
  });

  it('x % 1 → 0', () => {
    const result = mathOp('%', idx('x'), new IntImmNode(1));
    expect(result).toBeInstanceOf(IntImmNode);
    expect(result.value).toBe(0);
  });
});

describe('mathOp identity simplifications — left operand', () => {
  it('0 + x → x', () => {
    const x = idx('x');
    expect(mathOp('+', new IntImmNode(0), x)).toBe(x);
  });

  it('1 * x → x', () => {
    const x = idx('x');
    expect(mathOp('*', new IntImmNode(1), x)).toBe(x);
  });

  it('0 * x → 0', () => {
    const result = mathOp('*', new IntImmNode(0), idx('x'));
    expect(result).toBeInstanceOf(IntImmNode);
    expect(result.value).toBe(0);
  });
});

describe('mathOp preserves non-trivial ops', () => {
  it('x + 5 is a MathOpNode', () => {
    const result = mathOp('+', idx('x'), new IntImmNode(5));
    expect(result).toBeInstanceOf(MathOpNode);
    expect(result.op).toBe('+');
  });

  it('x * 3 is a MathOpNode', () => {
    const result = mathOp('*', idx('x'), new IntImmNode(3));
    expect(result).toBeInstanceOf(MathOpNode);
    expect(result.op).toBe('*');
  });

  it('x % 4 is a MathOpNode', () => {
    const result = mathOp('%', idx('x'), new IntImmNode(4));
    expect(result).toBeInstanceOf(MathOpNode);
    expect(result.op).toBe('%');
  });
});

describe('mathOp integer division/modulo by zero', () => {
  it('does not fold // or % by zero', () => {
    expect(mathOp('//', new IntImmNode(4), new IntImmNode(0)).type).toBe('MathOpNode');
    expect(mathOp('%', new IntImmNode(4), new IntImmNode(0)).type).toBe('MathOpNode');
  });

  it('still folds // and % by a non-zero divisor', () => {
    expect(mathOp('//', new IntImmNode(7), new IntImmNode(2)).type).toBe('IntImmNode');
    expect(mathOp('%', new IntImmNode(7), new IntImmNode(3)).type).toBe('IntImmNode');
  });
});
