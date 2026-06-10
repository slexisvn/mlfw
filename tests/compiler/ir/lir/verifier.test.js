import { describe, it, expect } from 'vitest';
import { verifyLIR } from '../../../../src/compiler/ir/lir/verifier.js';
import { LIRFunc, LIRMetadata } from '../../../../src/compiler/ir/lir/nodes.js';
import {
  ForNode, SeqNode, LetStmtNode, EvaluateNode,
  MathOpNode, VariableNode, IntImmNode, ForKind,
} from '../../../../src/compiler/ir/tensor/nodes.js';

function idx(name) { return new VariableNode(name, 'index'); }

function lirFunc(body) {
  return new LIRFunc('f', [], body, new Map(), [], new Map(), new LIRMetadata());
}

function evalUse(varName) {
  return new EvaluateNode(new MathOpNode('+', new VariableNode(varName, 'index'), new IntImmNode(1)));
}

describe('verifyLIR unbound variables', () => {
  it('accepts a loop variable used inside its body', () => {
    const body = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, evalUse('i'));
    const errors = verifyLIR(lirFunc(body));
    expect(errors.length).toBe(0);
  });

  it('accepts a let-bound variable used in its body', () => {
    const v = new VariableNode('t', 'index');
    const body = new LetStmtNode(v, new IntImmNode(3), evalUse('t'));
    const errors = verifyLIR(lirFunc(body));
    expect(errors.length).toBe(0);
  });

  it('flags use of an unbound variable', () => {
    const body = evalUse('ghost');
    const errors = verifyLIR(lirFunc(body));
    expect(errors.some(e => /unbound variable 'ghost'/.test(e.message))).toBe(true);
  });
});

describe('verifyLIR scope restoration', () => {
  it('does not leak a loop variable to a sibling statement', () => {
    const loop = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, evalUse('i'));
    const sibling = evalUse('i');
    const body = new SeqNode([loop, sibling]);
    const errors = verifyLIR(lirFunc(body));
    expect(errors.some(e => /unbound variable 'i'/.test(e.message))).toBe(true);
  });

  it('does not leak a let variable to a sibling statement', () => {
    const v = new VariableNode('t', 'index');
    const letStmt = new LetStmtNode(v, new IntImmNode(1), evalUse('t'));
    const sibling = evalUse('t');
    const body = new SeqNode([letStmt, sibling]);
    const errors = verifyLIR(lirFunc(body));
    expect(errors.some(e => /unbound variable 't'/.test(e.message))).toBe(true);
  });
});
