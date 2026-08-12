import type { TirNode, IntImmNode, VariableNode, MathOpNode } from '../ir/tensor/nodes.js';

export type VarRange = readonly [min: number, extent: number];
export type RadixFactor = { name: string; coeff: number; extent: number; min: number };
export type RadixDecomposition = { offset: number; extent: number; factors: RadixFactor[] };
export type FormSplit = { divisible: LinearForm; remainder: LinearForm };

export type MakeVarFn<T> = (name: string) => T;
export type MakeConstFn<T> = (value: number) => T;
export type MakeOpFn<T> = (op: string, a: T, b: T) => T;

export class LinearForm {
  offset: number;
  terms: Map<string, number>;

  constructor(offset = 0, terms: Map<string, number> = new Map()) {
    this.offset = offset;
    this.terms = terms;
  }

  static constant(c: number): LinearForm {
    return new LinearForm(c, new Map());
  }

  static variable(name: string): LinearForm {
    return new LinearForm(0, new Map([[name, 1]]));
  }

  add(other: LinearForm): LinearForm {
    const terms = new Map(this.terms);
    for (const [name, coeff] of other.terms) {
      const next = (terms.get(name) || 0) + coeff;
      if (next === 0) terms.delete(name);
      else terms.set(name, next);
    }
    return new LinearForm(this.offset + other.offset, terms);
  }

  negate(): LinearForm {
    const terms = new Map<string, number>();
    for (const [name, coeff] of this.terms) terms.set(name, -coeff);
    return new LinearForm(-this.offset, terms);
  }

  scale(factor: number): LinearForm {
    if (factor === 0) return LinearForm.constant(0);
    const terms = new Map<string, number>();
    for (const [name, coeff] of this.terms) terms.set(name, coeff * factor);
    return new LinearForm(this.offset * factor, terms);
  }

  get isConstant(): boolean {
    return this.terms.size === 0;
  }
}

export function toLinearForm(expr: TirNode | null | undefined): LinearForm | null {
  if (!expr) return null;
  switch (expr.type) {
    case 'IntImmNode':
      return Number.isInteger((expr as IntImmNode).value) ? LinearForm.constant((expr as IntImmNode).value) : null;
    case 'VariableNode':
      return LinearForm.variable((expr as VariableNode).name);
    case 'MathOpNode': {
      const math = expr as MathOpNode;
      const a = toLinearForm(math.a);
      if (!a) return null;
      if (math.b === null || math.b === undefined) return null;
      const b = toLinearForm(math.b);
      if (!b) return null;
      switch (math.op) {
        case '+': return a.add(b);
        case '-': return a.add(b.negate());
        case '*':
          if (a.isConstant) return b.scale(a.offset);
          if (b.isConstant) return a.scale(b.offset);
          return null;
        default: return null;
      }
    }
    default:
      return null;
  }
}

export function composeForm(form: LinearForm | null | undefined, varForms: ReadonlyMap<string, LinearForm>): LinearForm | null {
  if (!form) return null;
  let result = LinearForm.constant(form.offset);
  for (const [name, coeff] of form.terms) {
    const bound = varForms.get(name);
    if (!bound) return null;
    result = result.add(bound.scale(coeff));
  }
  return result;
}

export function splitByDivisor(form: LinearForm, divisor: number): FormSplit | null {
  if (!Number.isInteger(divisor) || divisor <= 0) return null;
  const divisible = new Map<string, number>();
  const remainder = new Map<string, number>();
  for (const [name, coeff] of form.terms) {
    if (coeff % divisor === 0) divisible.set(name, coeff);
    else remainder.set(name, coeff);
  }
  return {
    divisible: new LinearForm(0, divisible),
    remainder: new LinearForm(form.offset, remainder),
  };
}

export function linearFormToNode<T>(form: LinearForm, makeVar: MakeVarFn<T>, makeConst: MakeConstFn<T>, makeOp: MakeOpFn<T>): T {
  let node: T | null = null;
  for (const [name, coeff] of form.terms) {
    const term = coeff === 1 ? makeVar(name) : makeOp('*', makeVar(name), makeConst(coeff));
    node = node === null ? term : makeOp('+', node, term);
  }
  if (node === null) return makeConst(form.offset);
  if (form.offset === 0) return node;
  return makeOp('+', node, makeConst(form.offset));
}

export function mixedRadixDecomposition(form: LinearForm | null | undefined, varRanges: ReadonlyMap<string, VarRange>): RadixDecomposition | null {
  if (!form) return null;

  const factors: RadixFactor[] = [];
  let offset = form.offset;
  for (const [name, coeff] of form.terms) {
    const range = varRanges.get(name);
    if (!range) return null;
    const [min, extent] = range;
    if (extent <= 0 || coeff <= 0) return null;
    offset += coeff * min;
    factors.push({ name, coeff, extent, min });
  }

  factors.sort((x, y) => x.coeff - y.coeff);
  let stride = 1;
  for (const factor of factors) {
    if (factor.coeff !== stride) return null;
    stride *= factor.extent;
  }

  return { offset, extent: stride, factors };
}

export function coverRangeOfForm(form: LinearForm | null | undefined, varRanges: ReadonlyMap<string, VarRange>): [number, number] | null {
  const decomposition = mixedRadixDecomposition(form, varRanges);
  return decomposition ? [decomposition.offset, decomposition.extent] : null;
}

export function exactCoverRange(expr: TirNode | null | undefined, varRanges: ReadonlyMap<string, VarRange>): [number, number] | null {
  return coverRangeOfForm(toLinearForm(expr), varRanges);
}
