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

export type SymNode = { node: TirNode; sym: SymExpr | null };
export type DivModSplit = { quotient: SymNode; remainder: SymNode };

const NO_ASSUMPTION = (): void => {};

const MATHOP_TO_SYM: Readonly<Record<string, SymBinaryName | undefined>> = { '+': 'add', '-': 'sub', '*': 'mul', '//': 'div', '%': 'mod', 'tdiv': 'div', 'tmod': 'mod' };
const EXTERN_TO_SYM: Readonly<Record<string, SymBinaryName | undefined>> = { max: 'max', min: 'min' };

const intVar = (name: string): TirNode => new VariableNode(name, 'int32');

function constNode(value: number): SymNode {
  return { node: new IntImmNode(value), sym: value };
}

function symOfMathOp(op: string, a: SymExpr | null, b: SymExpr | null): SymExpr | null {
  const sym = MATHOP_TO_SYM[op];
  if (!sym || a === null || b === null) return null;
  if (DIVMOD_MATH_OPS.has(op) && (typeof b !== 'number' || b <= 0)) return null;
  return SymInt[sym](a, b);
}

export function symOfExtern(name: string, args: readonly (SymExpr | null)[]): SymExpr | null {
  const sym = EXTERN_TO_SYM[name];
  if (!sym || args.length !== 2 || args[0] === null || args[1] === null) return null;
  return SymInt[sym](args[0], args[1]);
}

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
        if (math.op !== '-') return null;
        const a = irToSymInt(math.a);
        return a === null ? null : SymInt.neg(a);
      }
      if (!MATHOP_TO_SYM[math.op]) return null;
      const a = irToSymInt(math.a);
      if (a === null) return null;
      return symOfMathOp(math.op, a, irToSymInt(math.b));
    }
    case 'CallExternNode': {
      const call = node as CallExternNode;
      if (!EXTERN_TO_SYM[call.externName] || call.args.length !== 2) return null;
      return symOfExtern(call.externName, call.args.map(irToSymInt));
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

function decideCompare(bound: IntBound | null, direction: string, table: ComparePredicates): boolean {
  const test = table[direction];
  if (!test || bound === null) return false;
  return test(bound);
}

function proveCompare(analyzer: Analyzer, direction: string, a: TirNode, b: TirNode, table: ComparePredicates): boolean {
  if (!table[direction]) return false;
  return decideCompare(diffBound(analyzer, a, b), direction, table);
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

function formToSymNode(form: LinearForm): SymNode {
  return {
    node: linearFormToNode<TirNode>(form, intVar, (value) => new IntImmNode(value), mathOp),
    sym: linearFormToNode<SymExpr | null>(form, SymInt.var, (value) => value, symOfMathOp),
  };
}

function withinBound(bound: IntBound | null, lo: number, hi: number): boolean {
  return bound !== null && bound.min >= lo && bound.max <= hi;
}

function boundOfSym(analyzer: Analyzer, sym: SymExpr | null): IntBound | null {
  return sym === null ? null : analyzer.constIntBound(sym);
}

function affineDivMod(analyzer: Analyzer, dividend: SymNode, divisor: number): DivModSplit | null {
  const form = toLinearForm(dividend.node);
  if (!form) return null;
  const parts = splitByDivisor(form, divisor);
  if (!parts) return null;

  const quotient = (): SymNode => formToSymNode(parts.divisible.scale(1 / divisor));

  if (parts.remainder.isConstant && parts.remainder.offset === 0) {
    return { quotient: quotient(), remainder: constNode(0) };
  }

  const remainder = formToSymNode(parts.remainder);
  if (!withinBound(boundOfSym(analyzer, remainder.sym), 0, divisor - 1)) return null;
  if (!withinBound(boundOfSym(analyzer, dividend.sym), 0, Infinity)) return null;

  return { quotient: quotient(), remainder };
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

function exactDivMod(analyzer: Analyzer, op: string, dividend: SymExpr, divisor: number): SymNode | null {
  if (!analyzer.modularSet(dividend).divisibleBy(divisor)) return null;
  if (op === '%') return constNode(0);
  const quotient = analyzer.exactQuotient(dividend, divisor);
  if (quotient === null) return null;
  const node = symIntToNode(quotient, intVar);
  return { node, sym: irToSymInt(node) };
}

function truncatedDivMod(op: string, dividend: SymNode, divisor: IntImmNode): SymNode {
  const truncated = op === '//' ? 'tdiv' : 'tmod';
  return { node: new MathOpNode(truncated, dividend.node, divisor), sym: symOfMathOp(truncated, dividend.sym, divisor.value) };
}

export class RewriteSimplify {
  analyzer: Analyzer;

  constructor(analyzer: Analyzer = new Analyzer()) {
    this.analyzer = analyzer;
  }

  simplify(node: TirNode | null | undefined): TirNode | null | undefined {
    if (node === null || node === undefined || typeof node !== 'object') return node;
    return this.rewrite(node).node;
  }

  rewrite(node: TirNode): SymNode {
    switch (node.type) {
      case 'IntImmNode':
        return { node, sym: (node as IntImmNode).value };
      case 'VariableNode':
        return { node, sym: SymInt.var((node as VariableNode).name) };
      case 'MathOpNode': {
        const math = node as MathOpNode;
        const b = math.b === null || math.b === undefined ? null : this.rewrite(math.b);
        return this.mathOp(math.op, this.rewrite(math.a), b);
      }
      case 'CompareNode': {
        const cmp = node as CompareNode;
        return this.compare(cmp.direction, this.rewrite(cmp.a), this.rewrite(cmp.b));
      }
      default:
        return { node, sym: null };
    }
  }

  mathOp(op: string, a: SymNode, b: SymNode | null): SymNode {
    if (b === null) {
      return { node: new MathOpNode(op, a.node), sym: op === '-' && a.sym !== null ? SymInt.neg(a.sym) : null };
    }
    return this._fold(op, a, b) ?? this._rewriteBinary(op, a, b);
  }

  compare(direction: string, a: SymNode, b: SymNode): SymNode {
    return this._decide(direction, a, b) ?? { node: new CompareNode(direction, a.node, b.node), sym: null };
  }

  _fold(op: string, a: SymNode, b: SymNode): SymNode | null {
    const node = mathOp(op, a.node, b.node);
    if (node === a.node) return a;
    if (node === b.node) return b;
    return node.type === 'IntImmNode' ? { node, sym: (node as IntImmNode).value } : null;
  }

  _rewriteBinary(op: string, a: SymNode, b: SymNode): SymNode {
    const divisor = b.node as IntImmNode;
    if ((op === '//' || op === '%') && divisor.type === 'IntImmNode' && divisor.value > 0) {
      const divided = this._divMod(op, a, divisor);
      if (divided) return divided;
    }
    if (COMPARE_MATH_OPS.has(op)) {
      const decided = this._decide(compareDirectionOf(op), a, b);
      if (decided) return decided;
    }
    return { node: new MathOpNode(op, a.node, b.node), sym: symOfMathOp(op, a.sym, b.sym) };
  }

  _decide(direction: string, a: SymNode, b: SymNode): SymNode | null {
    if (a.sym === null || b.sym === null || !CMP_TRUE[direction]) return null;
    const bound = this.analyzer.boundOfDifference(a.sym, b.sym);
    if (decideCompare(bound, direction, CMP_TRUE)) return constNode(1);
    if (decideCompare(bound, direction, CMP_FALSE)) return constNode(0);
    return null;
  }

  _divMod(op: string, dividend: SymNode, divisor: IntImmNode): SymNode | null {
    if (dividend.sym !== null) {
      const bound = this.analyzer.constIntBound(dividend.sym);
      if (bound.min >= 0 && bound.max <= divisor.value - 1) return op === '//' ? constNode(0) : dividend;
      const exact = exactDivMod(this.analyzer, op, dividend.sym, divisor.value);
      if (exact) return exact;
    }
    const split = affineDivMod(this.analyzer, dividend, divisor.value);
    if (split) return op === '//' ? split.quotient : split.remainder;
    return withinBound(boundOfSym(this.analyzer, dividend.sym), 0, Infinity) ? truncatedDivMod(op, dividend, divisor) : null;
  }
}
