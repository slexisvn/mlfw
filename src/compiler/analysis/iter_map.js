export class LinearForm {
  constructor(offset = 0, terms = new Map()) {
    this.offset = offset;
    this.terms = terms;
  }

  static constant(c) {
    return new LinearForm(c, new Map());
  }

  static variable(name) {
    return new LinearForm(0, new Map([[name, 1]]));
  }

  add(other) {
    const terms = new Map(this.terms);
    for (const [name, coeff] of other.terms) {
      const next = (terms.get(name) || 0) + coeff;
      if (next === 0) terms.delete(name);
      else terms.set(name, next);
    }
    return new LinearForm(this.offset + other.offset, terms);
  }

  negate() {
    const terms = new Map();
    for (const [name, coeff] of this.terms) terms.set(name, -coeff);
    return new LinearForm(-this.offset, terms);
  }

  scale(factor) {
    if (factor === 0) return LinearForm.constant(0);
    const terms = new Map();
    for (const [name, coeff] of this.terms) terms.set(name, coeff * factor);
    return new LinearForm(this.offset * factor, terms);
  }

  get isConstant() {
    return this.terms.size === 0;
  }
}

export function toLinearForm(expr) {
  if (!expr) return null;
  switch (expr.type) {
    case 'IntImmNode':
      return Number.isInteger(expr.value) ? LinearForm.constant(expr.value) : null;
    case 'VariableNode':
      return LinearForm.variable(expr.name);
    case 'MathOpNode': {
      const a = toLinearForm(expr.a);
      if (!a) return null;
      if (expr.b === null || expr.b === undefined) return null;
      const b = toLinearForm(expr.b);
      if (!b) return null;
      switch (expr.op) {
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

export function exactCoverRange(expr, varRanges) {
  const form = toLinearForm(expr);
  if (!form) return null;

  const factors = [];
  for (const [name, coeff] of form.terms) {
    const range = varRanges.get(name);
    if (!range) return null;
    const [min, extent] = range;
    if (min !== 0 || extent <= 0 || coeff <= 0) return null;
    factors.push({ coeff, extent });
  }

  if (factors.length === 0) return [form.offset, 1];

  factors.sort((x, y) => x.coeff - y.coeff);
  let stride = 1;
  for (const factor of factors) {
    if (factor.coeff !== stride) return null;
    stride *= factor.extent;
  }

  return [form.offset, stride];
}
