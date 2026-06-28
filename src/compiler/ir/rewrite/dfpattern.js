class DFPattern {
  match(op, bindings) { return false; }
}

class AnyPattern extends DFPattern {
  match() { return true; }
}

class OpPattern extends DFPattern {
  constructor(name, operandPatterns) {
    super();
    this.name = name;
    this.operandPatterns = operandPatterns;
  }
  match(op, bindings) {
    if (!op || op.opName !== this.name) return false;
    if (this.operandPatterns.length === 0) return true;
    if (op.numOperands < this.operandPatterns.length) return false;
    for (let i = 0; i < this.operandPatterns.length; i++) {
      const producer = op.getOperand(i).definingOp;
      if (!this.operandPatterns[i].match(producer, bindings)) return false;
    }
    return true;
  }
}

class AttrPattern extends DFPattern {
  constructor(inner, key, value) {
    super();
    this.inner = inner;
    this.key = key;
    this.value = value;
  }
  match(op, bindings) {
    if (!this.inner.match(op, bindings)) return false;
    if (!op || typeof op.getAttr !== 'function') return false;
    const v = op.getAttr(this.key);
    return this.value === undefined ? v !== undefined && v !== null : v === this.value;
  }
}

class AltPattern extends DFPattern {
  constructor(patterns) {
    super();
    this.patterns = patterns;
  }
  match(op, bindings) {
    for (const p of this.patterns) {
      const trial = { ...bindings };
      if (p.match(op, trial)) {
        Object.assign(bindings, trial);
        return true;
      }
    }
    return false;
  }
}

class CapturePattern extends DFPattern {
  constructor(name, inner) {
    super();
    this.name = name;
    this.inner = inner;
  }
  match(op, bindings) {
    if (!this.inner.match(op, bindings)) return false;
    bindings[this.name] = op;
    return true;
  }
}

export { DFPattern };
export const wildcard = () => new AnyPattern();
export const isOp = (name, ...operandPatterns) => new OpPattern(name, operandPatterns);
export const hasAttr = (inner, key, value) => new AttrPattern(inner, key, value);
export const alt = (...patterns) => new AltPattern(patterns);
export const capture = (name, inner = new AnyPattern()) => new CapturePattern(name, inner);

export function matchPattern(pattern, op) {
  const bindings = {};
  return pattern.match(op, bindings) ? bindings : null;
}
