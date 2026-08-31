import { describe, it, expect } from 'vitest';
import { Analyzer } from '../../../src/compiler/analysis/analyzer.js';
import {
  irToSymInt, irBound, proveTrue, proveFalse, analyzerForLoops, RewriteSimplify,
} from '../../../src/compiler/analysis/ir_arith.js';
import { SymInt } from '../../../src/compiler/ir/sym_int.js';
import { IntImmNode, VariableNode, MathOpNode, CompareNode, CallExternNode } from '../../../src/compiler/ir/tensor/nodes.js';

const v = (n) => new VariableNode(n, 'index');
const c = (x) => new IntImmNode(x);
const mul = (a, b) => new MathOpNode('*', a, b);
const add = (a, b) => new MathOpNode('+', a, b);
const sub = (a, b) => new MathOpNode('-', a, b);
const fdiv = (a, b) => new MathOpNode('//', a, b);
const fmod = (a, b) => new MathOpNode('%', a, b);

describe('irToSymInt bridge', () => {
  it('maps affine IR expressions to SymInt and yields the same bound', () => {
    const a = analyzerForLoops(new Map([['i', 8]]));
    const expr = add(mul(v('i'), c(2)), c(3));
    const bound = irBound(a, expr);
    expect(bound).toMatchObject({ min: 3, max: 17 });
  });

  it('bridges floordiv/mod by a positive constant', () => {
    const a = analyzerForLoops(new Map([['i', 8]]));
    expect(irBound(a, fdiv(v('i'), c(2)))).toMatchObject({ min: 0, max: 3 });
    expect(irBound(a, fmod(v('i'), c(4)))).toMatchObject({ min: 0, max: 3 });
  });

  it('bridges max/min CallExtern', () => {
    const a = analyzerForLoops(new Map([['i', 8]]));
    expect(irBound(a, new CallExternNode('max', [v('i'), c(2)], 'index'))).toMatchObject({ min: 2, max: 7 });
    expect(irBound(a, new CallExternNode('min', [v('i'), c(2)], 'index'))).toMatchObject({ min: 0, max: 2 });
  });

  it('returns null for non-representable nodes (float div, unknown extern)', () => {
    expect(irToSymInt(new MathOpNode('/', v('i'), c(2)))).toBe(null);
    expect(irToSymInt(new CallExternNode('exp', [v('i')], 'f32'))).toBe(null);
    expect(irToSymInt(fdiv(v('i'), v('j')))).toBe(null);
  });
});

describe('proveTrue / proveFalse over guards', () => {
  it('proves a pad=0 pooling in-bounds guard always holds (oh in [0,2], kh in [0,1], stride 2, inH=6 => ih = oh*2+kh in [0,5])', () => {
    const a = analyzerForLoops(new Map([['oh', 3], ['kh', 2]]));
    const ih = add(mul(v('oh'), c(2)), sub(v('kh'), c(0)));
    const ge = new CompareNode('ge', ih, c(0));
    const lt = new CompareNode('lt', ih, c(6));
    const guard = mul(ge, lt);
    expect(proveTrue(a, guard)).toBe(true);
  });

  it('does NOT prove a padded guard (pad=1 lets the index go negative)', () => {
    const a = analyzerForLoops(new Map([['oh', 3], ['kh', 2]]));
    const ih = add(mul(v('oh'), c(2)), sub(v('kh'), c(1)));
    const ge = new CompareNode('ge', ih, c(0));
    expect(proveTrue(a, ge)).toBe(false);
    expect(proveFalse(a, ge)).toBe(false);
  });

  it('proves a divisible split guard (o*4 + i < 8 with o in [0,1], i in [0,3])', () => {
    const a = analyzerForLoops(new Map([['o', 2], ['i', 4]]));
    const guard = new MathOpNode('<', add(mul(v('o'), c(4)), v('i')), c(8));
    expect(proveTrue(a, guard)).toBe(true);
  });

  it('cannot prove a non-divisible split guard (extent 9, factor 4 => o in [0,2], i in [0,3], max 11 >= 9)', () => {
    const a = analyzerForLoops(new Map([['o', 3], ['i', 4]]));
    const guard = new MathOpNode('<', add(mul(v('o'), c(4)), v('i')), c(9));
    expect(proveTrue(a, guard)).toBe(false);
  });

  it('is sound: unbound variables never prove a guard', () => {
    const a = new Analyzer();
    const guard = new CompareNode('lt', v('n'), c(8));
    expect(proveTrue(a, guard)).toBe(false);
    expect(proveFalse(a, guard)).toBe(false);
  });
});

describe('RewriteSimplify', () => {
  it('folds (a*c)//c -> a and (a*c)%c -> 0 without bounds', () => {
    const s = new RewriteSimplify();
    const a = s.simplify(fdiv(mul(v('a'), c(4)), c(4)));
    expect(a).toMatchObject({ type: 'VariableNode', name: 'a' });
    const m = s.simplify(fmod(mul(v('a'), c(4)), c(4)));
    expect(m).toMatchObject({ type: 'IntImmNode', value: 0 });
  });

  it('uses bounds to fold x%c -> x and x//c -> 0 when 0<=x<c', () => {
    const s = new RewriteSimplify(analyzerForLoops(new Map([['i', 4]])));
    expect(s.simplify(fmod(v('i'), c(4)))).toMatchObject({ type: 'VariableNode', name: 'i' });
    expect(s.simplify(fdiv(v('i'), c(4)))).toMatchObject({ type: 'IntImmNode', value: 0 });
  });

  it('constant-folds and applies identities', () => {
    const s = new RewriteSimplify();
    expect(s.simplify(add(mul(c(2), c(4)), c(1)))).toMatchObject({ type: 'IntImmNode', value: 9 });
    expect(s.simplify(add(v('i'), c(0)))).toMatchObject({ type: 'VariableNode', name: 'i' });
    expect(s.simplify(mul(v('i'), c(1)))).toMatchObject({ type: 'VariableNode', name: 'i' });
  });

  it('folds a provably-true comparison to 1 and a false one to 0', () => {
    const s = new RewriteSimplify(analyzerForLoops(new Map([['i', 8]])));
    expect(s.simplify(new CompareNode('lt', v('i'), c(8)))).toMatchObject({ type: 'IntImmNode', value: 1 });
    expect(s.simplify(new CompareNode('ge', v('i'), c(8)))).toMatchObject({ type: 'IntImmNode', value: 0 });
  });

  it('leaves genuinely symbolic index expressions intact', () => {
    const s = new RewriteSimplify();
    const expr = add(mul(v('i'), c(4)), v('j'));
    const out = s.simplify(expr);
    expect(out.type).toBe('MathOpNode');
    expect(out.op).toBe('+');
  });
});

describe('RewriteSimplify keeps the operand it reasons about', () => {
  const evalAt = (node, name, value) => SymInt.evaluate(irToSymInt(node), new Map([[name, value]]));

  it('leaves a divmod untouched when an identity fold hands the child straight through', () => {
    const s = new RewriteSimplify();
    const quotient = s.simplify(add(c(0), fdiv(v('n'), c(4))));
    const remainder = s.simplify(mul(c(1), fmod(v('n'), c(4))));
    for (const n of [0, 3, 4, 9, 17, 40]) {
      expect(evalAt(quotient, 'n', n)).toBe(Math.floor(n / 4));
      expect(evalAt(remainder, 'n', n)).toBe(n % 4);
    }
  });

  it('still folds the divmod when the dividend really is the small one', () => {
    const s = new RewriteSimplify(analyzerForLoops(new Map([['i', 4]])));
    expect(s.simplify(fdiv(v('i'), c(4)))).toMatchObject({ type: 'IntImmNode', value: 0 });
    expect(s.simplify(add(c(0), fdiv(v('i'), c(4))))).toMatchObject({ type: 'IntImmNode', value: 0 });
  });
});

describe('RewriteSimplify splits an affine index by its divisor', () => {
  const tiled = () => analyzerForLoops(new Map([['io', 4], ['ii', 32]]));

  it('recovers the outer tile index from (io*32 + ii) // 32', () => {
    const s = new RewriteSimplify(tiled());
    const out = s.simplify(fdiv(add(mul(v('io'), c(32)), v('ii')), c(32)));
    expect(out).toMatchObject({ type: 'VariableNode', name: 'io' });
  });

  it('recovers the inner tile index from (io*32 + ii) % 32', () => {
    const s = new RewriteSimplify(tiled());
    const out = s.simplify(fmod(add(mul(v('io'), c(32)), v('ii')), c(32)));
    expect(out).toMatchObject({ type: 'VariableNode', name: 'ii' });
  });

  it('handles the reassociated form the old scaled-var rule could not match', () => {
    const s = new RewriteSimplify(tiled());
    const out = s.simplify(fmod(add(v('ii'), mul(c(32), v('io'))), c(32)));
    expect(out).toMatchObject({ type: 'VariableNode', name: 'ii' });
  });

  it('divides a multiple-of-divisor coefficient down rather than dropping it', () => {
    const s = new RewriteSimplify(analyzerForLoops(new Map([['io', 4], ['ii', 8]])));
    const out = s.simplify(fdiv(add(mul(v('io'), c(64)), v('ii')), c(8)));
    expect(out).toMatchObject({ type: 'MathOpNode', op: '*' });
    expect(out.b).toMatchObject({ type: 'IntImmNode', value: 8 });
  });

  it('refuses to split when the remainder can reach the divisor', () => {
    const s = new RewriteSimplify(analyzerForLoops(new Map([['io', 4], ['ii', 64]])));
    const out = s.simplify(fmod(add(mul(v('io'), c(32)), v('ii')), c(32)));
    expect(out).toMatchObject({ type: 'MathOpNode', op: 'tmod' });
  });

  it('refuses to split when the inner variable is unbounded', () => {
    const s = new RewriteSimplify(analyzerForLoops(new Map([['io', 4]])));
    const out = s.simplify(fmod(add(mul(v('io'), c(32)), v('k')), c(32)));
    expect(out).toMatchObject({ type: 'MathOpNode', op: '%' });
  });
});
