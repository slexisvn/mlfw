import { describe, it, expect } from 'vitest';
import { wildcard, isOp, hasAttr, alt, capture, matchPattern } from '../../../src/compiler/passes/rewrite/dfpattern.js';

function mkOp(opName, operands = [], attrs = {}) {
  return {
    opName,
    numOperands: operands.length,
    getOperand: (i) => ({ definingOp: operands[i] }),
    getAttr: (k) => (k in attrs ? attrs[k] : null),
  };
}

describe('DFPattern matcher', () => {
  const x = mkOp('arg');
  const inner = mkOp('transpose', [x]);
  const outer = mkOp('transpose', [inner]);

  it('matches a nested subgraph structurally', () => {
    expect(matchPattern(isOp('transpose', isOp('transpose', wildcard())), outer)).toBeTruthy();
    expect(matchPattern(isOp('transpose', isOp('transpose', wildcard())), inner)).toBe(null);
  });

  it('binds captured nodes', () => {
    const pat = capture('outer', isOp('transpose', capture('inner', isOp('transpose'))));
    const b = matchPattern(pat, outer);
    expect(b.outer).toBe(outer);
    expect(b.inner).toBe(inner);
  });

  it('matches by attribute', () => {
    const op = mkOp('reduce', [], { reduce_type: 'sum' });
    expect(matchPattern(hasAttr(isOp('reduce'), 'reduce_type', 'sum'), op)).toBeTruthy();
    expect(matchPattern(hasAttr(isOp('reduce'), 'reduce_type', 'max'), op)).toBe(null);
  });

  it('alternation', () => {
    const pat = alt(isOp('add'), isOp('transpose'));
    expect(matchPattern(pat, outer)).toBeTruthy();
    expect(matchPattern(pat, mkOp('mul'))).toBe(null);
  });

  it('wildcard matches anything', () => {
    expect(matchPattern(wildcard(), x)).toBeTruthy();
  });
});
