import type { Operation } from '../graph/operation.js';
import type { AttrValue } from '../graph/types.js';

export type PatternBindings = Record<string, Operation>;

class DFPattern {
  match(op: Operation | null, bindings: PatternBindings): boolean { return false; }
}

class AnyPattern extends DFPattern {
  override match(): boolean { return true; }
}

class OpPattern extends DFPattern {
  name: string;
  operandPatterns: readonly DFPattern[];

  constructor(name: string, operandPatterns: readonly DFPattern[]) {
    super();
    this.name = name;
    this.operandPatterns = operandPatterns;
  }
  override match(op: Operation | null, bindings: PatternBindings): boolean {
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
  inner: DFPattern;
  key: string;
  value: AttrValue | undefined;

  constructor(inner: DFPattern, key: string, value: AttrValue | undefined) {
    super();
    this.inner = inner;
    this.key = key;
    this.value = value;
  }
  override match(op: Operation | null, bindings: PatternBindings): boolean {
    if (!this.inner.match(op, bindings)) return false;
    if (!op || typeof op.getAttr !== 'function') return false;
    const v = op.getAttr(this.key);
    return this.value === undefined ? v !== undefined && v !== null : v === this.value;
  }
}

class AltPattern extends DFPattern {
  patterns: readonly DFPattern[];

  constructor(patterns: readonly DFPattern[]) {
    super();
    this.patterns = patterns;
  }
  override match(op: Operation | null, bindings: PatternBindings): boolean {
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
  name: string;
  inner: DFPattern;

  constructor(name: string, inner: DFPattern) {
    super();
    this.name = name;
    this.inner = inner;
  }
  override match(op: Operation | null, bindings: PatternBindings): boolean {
    if (!this.inner.match(op, bindings)) return false;
    bindings[this.name] = op as Operation;
    return true;
  }
}

export { DFPattern };
export const wildcard = (): DFPattern => new AnyPattern();
export const isOp = (name: string, ...operandPatterns: DFPattern[]): DFPattern => new OpPattern(name, operandPatterns);
export const hasAttr = (inner: DFPattern, key: string, value?: AttrValue): DFPattern => new AttrPattern(inner, key, value);
export const alt = (...patterns: DFPattern[]): DFPattern => new AltPattern(patterns);
export const capture = (name: string, inner: DFPattern = new AnyPattern()): DFPattern => new CapturePattern(name, inner);

export function matchPattern(pattern: DFPattern, op: Operation | null): PatternBindings | null {
  const bindings: PatternBindings = {};
  return pattern.match(op, bindings) ? bindings : null;
}
