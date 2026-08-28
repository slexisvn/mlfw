import { COMPARE_MATH_OPS, compareDirectionOf } from '../../util/dtype_map.js';
import { DIVMOD_MATH_OPS } from '../../util/divmod.js';
import { Analyzer } from './analyzer.js';
import { SymInt } from './sym_int.js';
import { IntImmNode, MathOpNode, CompareNode, VariableNode, mathOp } from '../ir/tensor/nodes.js';
import type { CallExternNode } from '../ir/tensor/nodes.js';
import { toLinearForm, splitByDivisor, linearFormToNode } from './iter_map.js';
import { symIntToNode } from '../ir/tensor/sym_lower.js';
import type { SymExpr } from './sym_int.js';
import type { IntBound } from './analyzer.js';
import type { LinearForm } from './iter_map.js';
import type { TirNode } from '../ir/tensor/nodes.js';

type SymBinaryName = 'add' | 'sub' | 'mul' | 'div' | 'mod' | 'max' | 'min';
type BoundPredicate = (d: IntBound) => boolean;
type ComparePredicates = Readonly<Record<string, BoundPredicate | undefined>>;

export type DivModSplit = { quotient: TirNode; remainder: TirNode };

const NO_ASSUMPTION = (): void => {};

const MATHOP_TO_SYM: Readonly<Record<string, SymBinaryName | undefined>> = { '+': 'add', '-': 'sub', '*': 'mul', '//': 'div', '%': 'mod', 'tdiv': 'div', 'tmod': 'mod' };

export function irToSymInt(node: TirNode | number | null | undefined): SymExpr | null {
  if (node === null || node === undefined) return null;
  if (typeof node === 'number') return node;
  switch (node.type) {
    case 'IntImmNode':
      return (node as IntImmNode).value;
    case 'VariableNode':
      return SymInt.var((node as VariableNode).name);
    case 'MathOpNode': {
      const math = node as MathOpNode;
      if (math.b === null || math.b === undefined) {
        if (math.op === '-') {
          const a = irToSymInt(math.a);
          return a === null ? null : SymInt.neg(a);
        }
        return null;
      }
      const sym = MATHOP_TO_SYM[math.op];
      if (!sym) return null;
      const a = irToSymInt(math.a);
      if (a === null) return null;
      const b = irToSymInt(math.b);
      if (b === null) return null;
      if (DIVMOD_MATH_OPS.has(math.op)) {
        if (typeof b !== 'number' || b <= 0) return null;
      }
      return SymInt[sym](a, b);
    }
    case 'CallExternNode': {
      const call = node as CallExternNode;
      if (call.args.length !== 2) return null;
      if (call.externName !== 'max' && call.externName !== 'min') return null;
      const a = irToSymInt(call.args[0]);
      if (a === null) return null;
      const b = irToSymInt(call.args[1]);
      if (b === null) return null;
      return call.externName === 'max' ? SymInt.max(a, b) : SymInt.min(a, b);
    }
    default:
      return null;
  }
}

export function irBound(analyzer: Analyzer, node: TirNode | null | undefined): IntBound | null {
  const sym = irToSymInt(node);
  if (sym === null) return null;
  return analyzer.constIntBound(sym);
}

function diffBound(analyzer: Analyzer, a: TirNode | null | undefined, b: TirNode | null | undefined): IntBound | null {
  const sa = irToSymInt(a);
  if (sa === null) return null;
  const sb = irToSymInt(b);
  if (sb === null) return null;
  return analyzer.boundOfDifference(sa, sb);
}

const CMP_TRUE: ComparePredicates = {
  lt: (d) => d.max < 0,
  le: (d) => d.max <= 0,
  gt: (d) => d.min > 0,
  ge: (d) => d.min >= 0,
  eq: (d) => d.min === 0 && d.max === 0,
  ne: (d) => d.min > 0 || d.max < 0,
};

const CMP_FALSE: ComparePredicates = {
  lt: (d) => d.min >= 0,
  le: (d) => d.min > 0,
  gt: (d) => d.max <= 0,
  ge: (d) => d.max < 0,
  eq: (d) => d.min > 0 || d.max < 0,
  ne: (d) => d.min === 0 && d.max === 0,
};

function proveCompare(analyzer: Analyzer, direction: string, a: TirNode, b: TirNode, table: ComparePredicates): boolean {
  const test = table[direction];
  if (!test) return false;
  const d = diffBound(analyzer, a, b);
  if (d === null) return false;
  return test(d);
}

export function proveTrue(analyzer: Analyzer, node: TirNode | null | undefined): boolean {
  if (node === null || node === undefined) return false;
  if (node.type === 'IntImmNode') return (node as IntImmNode).value !== 0;
  if (node.type === 'CompareNode') { const cmp = node as CompareNode; return proveCompare(analyzer, cmp.direction, cmp.a, cmp.b, CMP_TRUE); }
  if (node.type === 'MathOpNode') {
    const math = node as MathOpNode;
    if (math.op === '*' && math.b) return proveTrue(analyzer, math.a) && proveTrue(analyzer, math.b);
    if (COMPARE_MATH_OPS.has(math.op)) return proveCompare(analyzer, compareDirectionOf(math.op), math.a, math.b as TirNode, CMP_TRUE);
  }
  return false;
}

export function proveFalse(analyzer: Analyzer, node: TirNode | null | undefined): boolean {
  if (node === null || node === undefined) return false;
  if (node.type === 'IntImmNode') return (node as IntImmNode).value === 0;
  if (node.type === 'CompareNode') { const cmp = node as CompareNode; return proveCompare(analyzer, cmp.direction, cmp.a, cmp.b, CMP_FALSE); }
  if (node.type === 'MathOpNode') {
    const math = node as MathOpNode;
    if (math.op === '*' && math.b) return proveFalse(analyzer, math.a) || proveFalse(analyzer, math.b);
    if (COMPARE_MATH_OPS.has(math.op)) return proveCompare(analyzer, compareDirectionOf(math.op), math.a, math.b as TirNode, CMP_FALSE);
  }
  return false;
}

export function analyzerForLoops(loopExtents: Iterable<[string, unknown]>): Analyzer {
  const analyzer = new Analyzer();
  for (const [name, extent] of loopExtents) {
    if (typeof extent === 'number' && extent > 0) analyzer.bind(name, 0, extent - 1);
  }
  return analyzer;
}

function formToNode(form: LinearForm): TirNode {
  return linearFormToNode<TirNode>(
    form,
    (name) => new VariableNode(name, 'int32'),
    (value) => new IntImmNode(value),
    (op, a, b) => mathOp(op, a, b),
  );
}

function affineDivMod(analyzer: Analyzer, node: TirNode, divisor: number): DivModSplit | null {
  const form = toLinearForm(node);
  if (!form) return null;
  const parts = splitByDivisor(form, divisor);
  if (!parts) return null;

  const quotient = () => formToNode(parts.divisible.scale(1 / divisor));

  if (parts.remainder.isConstant && parts.remainder.offset === 0) {
    return { quotient: quotient(), remainder: new IntImmNode(0) };
  }

  const remainderNode = formToNode(parts.remainder);
  if (!boundWithin(analyzer, remainderNode, 0, divisor - 1)) return null;
  if (!boundWithin(analyzer, node, 0, Infinity)) return null;

  return { quotient: quotient(), remainder: remainderNode };
}

const NEGATED_DIRECTION: Readonly<Record<string, string | undefined>> = {
  lt: 'ge', le: 'gt', gt: 'le', ge: 'lt', eq: 'ne', ne: 'eq',
};

const DIRECTION_FACTS: Readonly<Record<string, ((a: SymExpr, b: SymExpr) => SymExpr[]) | undefined>> = {
  lt: (a, b) => [SymInt.sub(SymInt.sub(b, a), 1)],
  le: (a, b) => [SymInt.sub(b, a)],
  gt: (a, b) => [SymInt.sub(SymInt.sub(a, b), 1)],
  ge: (a, b) => [SymInt.sub(a, b)],
  eq: (a, b) => [SymInt.sub(a, b), SymInt.sub(b, a)],
  ne: () => [],
};

function comparisonOf(node: TirNode): { direction: string; a: TirNode; b: TirNode } | null {
  if (node.type === 'CompareNode') {
    const cmp = node as CompareNode;
    return { direction: cmp.direction, a: cmp.a, b: cmp.b };
  }
  if (node.type !== 'MathOpNode') return null;
  const math = node as MathOpNode;
  if (!math.b || !COMPARE_MATH_OPS.has(math.op)) return null;
  return { direction: compareDirectionOf(math.op), a: math.a, b: math.b };
}

export function assumeCondition(analyzer: Analyzer, node: TirNode | null | undefined, truth: boolean): () => void {
  if (!node) return NO_ASSUMPTION;
  const comparison = comparisonOf(node);
  if (!comparison) return NO_ASSUMPTION;

  const direction = truth ? comparison.direction : NEGATED_DIRECTION[comparison.direction];
  const facts = direction ? DIRECTION_FACTS[direction] : undefined;
  if (!facts) return NO_ASSUMPTION;

  const a = irToSymInt(comparison.a);
  if (a === null) return NO_ASSUMPTION;
  const b = irToSymInt(comparison.b);
  if (b === null) return NO_ASSUMPTION;

  const releases = facts(a, b).map(fact => analyzer.assumeNonNegative(fact));
  if (releases.length === 0) return NO_ASSUMPTION;
  return () => { for (const release of releases) release(); };
}

export function assumeLoopVar(analyzer: Analyzer, name: string, extent: TirNode | null | undefined): () => void {
  const limit = irToSymInt(extent);
  if (limit === null || typeof limit === 'number') return NO_ASSUMPTION;
  const variable = SymInt.var(name);
  const releases = [analyzer.assumeNonNegative(variable), analyzer.assumeLess(variable, limit)];
  return () => { for (const release of releases) release(); };
}

function exactDivMod(analyzer: Analyzer, op: string, dividend: SymExpr, divisor: number): TirNode | null {
  if (!analyzer.modularSet(dividend).divisibleBy(divisor)) return null;
  if (op === '%') return new IntImmNode(0);
  const quotient = analyzer.exactQuotient(dividend, divisor);
  return quotient === null ? null : symIntToNode(quotient, (name) => new VariableNode(name, 'int32'));
}

function nonNegativeDivMod(analyzer: Analyzer, node: MathOpNode): MathOpNode | null {
  if (node.op !== '//' && node.op !== '%') return null;
  const divisor = node.b as IntImmNode | null;
  if (!divisor || divisor.type !== 'IntImmNode' || divisor.value <= 0) return null;
  if (!boundWithin(analyzer, node.a, 0, Infinity)) return null;
  return new MathOpNode(node.op === '//' ? 'tdiv' : 'tmod', node.a, node.b as TirNode);
}

function boundWithin(analyzer: Analyzer, node: TirNode, lo: number, hi: number): boolean {
  const b = irBound(analyzer, node);
  if (b === null) return false;
  return b.min >= lo && b.max <= hi;
}

export class RewriteSimplify {
  analyzer: Analyzer;

  constructor(analyzer: Analyzer = new Analyzer()) {
    this.analyzer = analyzer;
  }

  simplify(node: TirNode | null | undefined): TirNode | null | undefined {
    if (node === null || node === undefined || typeof node !== 'object') return node;
    switch (node.type) {
      case 'IntImmNode':
      case 'VariableNode':
        return node;
      case 'MathOpNode':
        return this._simplifyMathOp(node as MathOpNode);
      case 'CompareNode':
        return this._simplifyCompare(node as CompareNode);
      default:
        return node;
    }
  }

  _simplifyCompare(node: CompareNode): TirNode {
    const a = this.simplify(node.a) as TirNode;
    const b = this.simplify(node.b) as TirNode;
    if (proveTrue(this.analyzer, new CompareNode(node.direction, a, b))) return new IntImmNode(1);
    if (proveFalse(this.analyzer, new CompareNode(node.direction, a, b))) return new IntImmNode(0);
    if (a === node.a && b === node.b) return node;
    return new CompareNode(node.direction, a, b);
  }

  _simplifyMathOp(node: MathOpNode): TirNode {
    if (node.b === null || node.b === undefined) {
      const a = this.simplify(node.a) as TirNode;
      return a === node.a ? node : new MathOpNode(node.op, a);
    }
    const a = this.simplify(node.a) as TirNode;
    const b = this.simplify(node.b) as TirNode;
    const raw = mathOp(node.op, a, b);
    if (!raw || raw.type !== 'MathOpNode') return raw;
    const folded = raw as MathOpNode;

    if ((folded.op === '//' || folded.op === '%') && (folded.b as TirNode).type === 'IntImmNode' && (folded.b as IntImmNode).value > 0) {
      const c = (folded.b as IntImmNode).value;
      const dividend = irToSymInt(folded.a);
      if (dividend !== null) {
        const bound = this.analyzer.constIntBound(dividend);
        if (bound.min >= 0 && bound.max <= c - 1) {
          return folded.op === '//' ? new IntImmNode(0) : folded.a;
        }
        const exact = exactDivMod(this.analyzer, folded.op, dividend, c);
        if (exact) return exact;
      }
      const split = affineDivMod(this.analyzer, folded.a, c);
      if (split) return folded.op === '//' ? split.quotient : split.remainder;
      const truncated = nonNegativeDivMod(this.analyzer, folded);
      if (truncated) return truncated;
    }
    if (COMPARE_MATH_OPS.has(folded.op)) {
      if (proveTrue(this.analyzer, folded)) return new IntImmNode(1);
      if (proveFalse(this.analyzer, folded)) return new IntImmNode(0);
    }
    return folded;
  }
}
