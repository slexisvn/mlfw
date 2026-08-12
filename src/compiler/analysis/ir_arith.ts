import { Analyzer } from './analyzer.js';
import { SymInt } from './sym_int.js';
import { IntImmNode, MathOpNode, CompareNode, VariableNode, mathOp } from '../ir/tensor/nodes.js';
import type { CallExternNode } from '../ir/tensor/nodes.js';
import { toLinearForm, splitByDivisor, linearFormToNode } from './iter_map.js';
import type { SymExpr } from './sym_int.js';
import type { IntBound } from './analyzer.js';
import type { LinearForm } from './iter_map.js';
import type { TirNode } from '../ir/tensor/nodes.js';

type SymBinaryName = 'add' | 'sub' | 'mul' | 'div' | 'mod' | 'max' | 'min';
type BoundPredicate = (d: IntBound) => boolean;
type ComparePredicates = Readonly<Record<string, BoundPredicate | undefined>>;

export type DivModSplit = { quotient: TirNode; remainder: TirNode };

const MATHOP_TO_SYM: Readonly<Record<string, SymBinaryName | undefined>> = { '+': 'add', '-': 'sub', '*': 'mul', '//': 'div', '%': 'mod' };
const COMPARE_MATHOPS = new Set(['<', '<=', '>', '>=', '==', '!=']);
const MATHOP_TO_DIRECTION: Readonly<Record<string, string>> = { '<': 'lt', '<=': 'le', '>': 'gt', '>=': 'ge', '==': 'eq', '!=': 'ne' };

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
      if (math.op === '//' || math.op === '%') {
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
  return analyzer.constIntBound(SymInt.sub(sa, sb));
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
    if (COMPARE_MATHOPS.has(math.op)) return proveCompare(analyzer, MATHOP_TO_DIRECTION[math.op], math.a, math.b as TirNode, CMP_TRUE);
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
    if (COMPARE_MATHOPS.has(math.op)) return proveCompare(analyzer, MATHOP_TO_DIRECTION[math.op], math.a, math.b as TirNode, CMP_FALSE);
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
      if (boundWithin(this.analyzer, folded.a, 0, c - 1)) {
        return folded.op === '//' ? new IntImmNode(0) : folded.a;
      }
      const split = affineDivMod(this.analyzer, folded.a, c);
      if (split) return folded.op === '//' ? split.quotient : split.remainder;
    }
    if (COMPARE_MATHOPS.has(folded.op)) {
      if (proveTrue(this.analyzer, folded)) return new IntImmNode(1);
      if (proveFalse(this.analyzer, folded)) return new IntImmNode(0);
    }
    return folded;
  }
}
