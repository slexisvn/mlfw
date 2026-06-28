import { Analyzer } from './analyzer.js';
import { SymInt } from './sym_int.js';
import { IntImmNode, MathOpNode, CompareNode, mathOp } from '../ir/tensor/nodes.js';

const MATHOP_TO_SYM = { '+': 'add', '-': 'sub', '*': 'mul', '//': 'div', '%': 'mod' };
const COMPARE_MATHOPS = new Set(['<', '<=', '>', '>=', '==', '!=']);
const MATHOP_TO_DIRECTION = { '<': 'lt', '<=': 'le', '>': 'gt', '>=': 'ge', '==': 'eq', '!=': 'ne' };

export function irToSymInt(node) {
  if (node === null || node === undefined) return null;
  if (typeof node === 'number') return node;
  switch (node.type) {
    case 'IntImmNode':
      return node.value;
    case 'VariableNode':
      return SymInt.var(node.name);
    case 'MathOpNode': {
      if (node.b === null || node.b === undefined) {
        if (node.op === '-') {
          const a = irToSymInt(node.a);
          return a === null ? null : SymInt.neg(a);
        }
        return null;
      }
      const sym = MATHOP_TO_SYM[node.op];
      if (!sym) return null;
      const a = irToSymInt(node.a);
      if (a === null) return null;
      const b = irToSymInt(node.b);
      if (b === null) return null;
      if (node.op === '//' || node.op === '%') {
        if (typeof b !== 'number' || b <= 0) return null;
      }
      return SymInt[sym](a, b);
    }
    case 'CallExternNode': {
      if (node.args.length !== 2) return null;
      if (node.externName !== 'max' && node.externName !== 'min') return null;
      const a = irToSymInt(node.args[0]);
      if (a === null) return null;
      const b = irToSymInt(node.args[1]);
      if (b === null) return null;
      return node.externName === 'max' ? SymInt.max(a, b) : SymInt.min(a, b);
    }
    default:
      return null;
  }
}

export function irBound(analyzer, node) {
  const sym = irToSymInt(node);
  if (sym === null) return null;
  return analyzer.constIntBound(sym);
}

function diffBound(analyzer, a, b) {
  const sa = irToSymInt(a);
  if (sa === null) return null;
  const sb = irToSymInt(b);
  if (sb === null) return null;
  return analyzer.constIntBound(SymInt.sub(sa, sb));
}

const CMP_TRUE = {
  lt: (d) => d.max < 0,
  le: (d) => d.max <= 0,
  gt: (d) => d.min > 0,
  ge: (d) => d.min >= 0,
  eq: (d) => d.min === 0 && d.max === 0,
  ne: (d) => d.min > 0 || d.max < 0,
};

const CMP_FALSE = {
  lt: (d) => d.min >= 0,
  le: (d) => d.min > 0,
  gt: (d) => d.max <= 0,
  ge: (d) => d.max < 0,
  eq: (d) => d.min > 0 || d.max < 0,
  ne: (d) => d.min === 0 && d.max === 0,
};

function proveCompare(analyzer, direction, a, b, table) {
  const test = table[direction];
  if (!test) return false;
  const d = diffBound(analyzer, a, b);
  if (d === null) return false;
  return test(d);
}

export function proveTrue(analyzer, node) {
  if (node === null || node === undefined) return false;
  if (node.type === 'IntImmNode') return node.value !== 0;
  if (node.type === 'CompareNode') return proveCompare(analyzer, node.direction, node.a, node.b, CMP_TRUE);
  if (node.type === 'MathOpNode') {
    if (node.op === '*' && node.b) return proveTrue(analyzer, node.a) && proveTrue(analyzer, node.b);
    if (COMPARE_MATHOPS.has(node.op)) return proveCompare(analyzer, MATHOP_TO_DIRECTION[node.op], node.a, node.b, CMP_TRUE);
  }
  return false;
}

export function proveFalse(analyzer, node) {
  if (node === null || node === undefined) return false;
  if (node.type === 'IntImmNode') return node.value === 0;
  if (node.type === 'CompareNode') return proveCompare(analyzer, node.direction, node.a, node.b, CMP_FALSE);
  if (node.type === 'MathOpNode') {
    if (node.op === '*' && node.b) return proveFalse(analyzer, node.a) || proveFalse(analyzer, node.b);
    if (COMPARE_MATHOPS.has(node.op)) return proveCompare(analyzer, MATHOP_TO_DIRECTION[node.op], node.a, node.b, CMP_FALSE);
  }
  return false;
}

export function analyzerForLoops(loopExtents) {
  const analyzer = new Analyzer();
  for (const [name, extent] of loopExtents) {
    if (typeof extent === 'number' && extent > 0) analyzer.bind(name, 0, extent - 1);
  }
  return analyzer;
}

function asScaledVar(node) {
  if (!node || node.type !== 'MathOpNode' || node.op !== '*') return null;
  if (node.b && node.b.type === 'IntImmNode' && node.b.value > 0) return { factor: node.a, c: node.b.value };
  if (node.a && node.a.type === 'IntImmNode' && node.a.value > 0) return { factor: node.b, c: node.a.value };
  return null;
}

function boundWithin(analyzer, node, lo, hi) {
  const b = irBound(analyzer, node);
  if (b === null) return false;
  return b.min >= lo && b.max <= hi;
}

export class RewriteSimplify {
  constructor(analyzer = new Analyzer()) {
    this.analyzer = analyzer;
  }

  simplify(node) {
    if (node === null || node === undefined || typeof node !== 'object') return node;
    switch (node.type) {
      case 'IntImmNode':
      case 'VariableNode':
        return node;
      case 'MathOpNode':
        return this._simplifyMathOp(node);
      case 'CompareNode':
        return this._simplifyCompare(node);
      default:
        return node;
    }
  }

  _simplifyCompare(node) {
    const a = this.simplify(node.a);
    const b = this.simplify(node.b);
    if (proveTrue(this.analyzer, new CompareNode(node.direction, a, b))) return new IntImmNode(1);
    if (proveFalse(this.analyzer, new CompareNode(node.direction, a, b))) return new IntImmNode(0);
    if (a === node.a && b === node.b) return node;
    return new CompareNode(node.direction, a, b);
  }

  _simplifyMathOp(node) {
    if (node.b === null || node.b === undefined) {
      const a = this.simplify(node.a);
      return a === node.a ? node : new MathOpNode(node.op, a);
    }
    const a = this.simplify(node.a);
    const b = this.simplify(node.b);
    const folded = mathOp(node.op, a, b);
    if (!folded || folded.type !== 'MathOpNode') return folded;

    if (folded.op === '//' && folded.b.type === 'IntImmNode' && folded.b.value > 0) {
      const c = folded.b.value;
      const scaled = asScaledVar(folded.a);
      if (scaled && scaled.c === c) return scaled.factor;
      if (boundWithin(this.analyzer, folded.a, 0, c - 1)) return new IntImmNode(0);
    }
    if (folded.op === '%' && folded.b.type === 'IntImmNode' && folded.b.value > 0) {
      const c = folded.b.value;
      const scaled = asScaledVar(folded.a);
      if (scaled && scaled.c === c) return new IntImmNode(0);
      if (boundWithin(this.analyzer, folded.a, 0, c - 1)) return folded.a;
    }
    if (COMPARE_MATHOPS.has(folded.op)) {
      if (proveTrue(this.analyzer, folded)) return new IntImmNode(1);
      if (proveFalse(this.analyzer, folded)) return new IntImmNode(0);
    }
    return folded;
  }
}
