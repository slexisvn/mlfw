import { describe, it, expect } from 'vitest';
import { verifyLIR } from '../../../../src/compiler/ir/lir/verifier.js';
import { LIRFunc, LIRMetadata } from '../../../../src/compiler/ir/lir/nodes.js';
import {
  ForNode, SeqNode, LetStmtNode, EvaluateNode,
  MathOpNode, VariableNode, IntImmNode, ForKind,
} from '../../../../src/compiler/ir/tensor/nodes.js';
import { buildFunction, IRBuilder } from '../../../../src/compiler/ir/graph/builder.js';
import { GraphFunction } from '../../../../src/compiler/ir/graph/function.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { lowerGraphToPrimFunc } from '../../../../src/compiler/passes/lowering/graph_to_tensor.js';
import { lowerToLIR } from '../../../../src/compiler/passes/lowering/tensor_to_lir.js';
import { WasmTarget } from '../../../../src/backend/target.js';

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

function buildAuto(name, inTypes, build) {
  const probe = new GraphFunction(name, inTypes, []);
  const r = build(new IRBuilder(probe), probe.args);
  return buildFunction(name, inTypes, [r.getResult(0).type], (b, a) => { b.returnOp([build(b, a).getResult(0)]); });
}

describe('verifyLIR on real lowered graphs across lowering-rule categories (pipeline never runs verifyLIR)', () => {
  const F = ScalarType.F32;
  const T = (sh) => new TensorType(sh, F);
  const CASES = [
    ['elementwise_chain', [T([4, 5]), T([4, 5])], (b, a) => b.relu(b.add(b.mul(a[0], a[1]).getResult(0), a[0]).getResult(0))],
    ['reduce_sum_mid', [T([2, 3, 4])], (b, a) => b.reduce(a[0], b.scalarConstant(0, F).getResult(0), [1], 'sum')],
    ['reduce_max_keepdim', [T([2, 3, 4])], (b, a) => b.reduce(a[0], b.scalarConstant(-1e30, F).getResult(0), [2], 'max')],
    ['argmax', [T([2, 3, 4])], (b, a) => b.argmax(a[0], 1, false)],
    ['matmul', [T([4, 5]), T([5, 6])], (b, a) => b.matmul(a[0], a[1])],
    ['transpose', [T([2, 3, 4])], (b, a) => b.transpose(a[0], [2, 0, 1])],
    ['slice_step', [T([6, 8])], (b, a) => b.slice(a[0], [0, 1], [6, 8], [2, 3])],
    ['pad', [T([3, 4])], (b, a) => b.pad(a[0], b.scalarConstant(0, F).getResult(0), [1, 0], [0, 2], [0, 0])],
    ['concat', [T([2, 3]), T([2, 5])], (b, a) => b.concat([a[0], a[1]], 1)],
    ['broadcast', [T([4])], (b, a) => b.broadcast(a[0], [3, 4], [1])],
    ['pool2d_max', [T([1, 2, 6, 6])], (b, a) => b.pool2d(a[0], 'max', [2, 2], [2, 2], [[0, 0], [0, 0]])],
    ['pool2d_avg_pad', [T([1, 2, 6, 6])], (b, a) => b.pool2d(a[0], 'avg', [3, 3], [1, 1], [[1, 1], [1, 1]])],
    ['conv_dilation_groups', [T([1, 4, 7, 7]), T([4, 2, 3, 3])], (b, a) => b.conv(a[0], a[1], [1, 1], [[1, 1], [1, 1]], { dilation: [2, 2], groups: 2 })],
    ['resize_bilinear', [T([1, 2, 4, 4])], (b, a) => b.resize(a[0], [8, 8], 'bilinear')],
    ['resize_nearest_down', [T([1, 2, 4, 4])], (b, a) => b.resize(a[0], [3, 3], 'nearest')],
  ];
  for (const [name, inTypes, build] of CASES) {
    it(`${name} lowers to LIR with no verifier errors`, () => {
      const func = buildAuto(name, inTypes, build);
      const lir = lowerToLIR(lowerGraphToPrimFunc(func), WasmTarget());
      const errors = verifyLIR(lir);
      expect(errors.map((e) => e.message), `${name}: ${errors.map((e) => e.message).join('; ')}`).toEqual([]);
    });
  }
});
