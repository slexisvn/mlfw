import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType } from '../../../../src/compiler/ir/graph/types.js';
import { MultiOutputFusionPass } from '../../../../src/compiler/passes/fusion/multi_output_fusion.js';
import { PassResult } from '../../../../src/compiler/passes/pass.js';
import { F32 } from '../../../_utils/ir_fixture.js';


function findOps(func, opName) {
  const result = [];
  for (const op of func.ops()) {
    if (op.opName === opName) result.push(op);
  }
  return result;
}

function assertNoUseBeforeDef(func) {
  const defined = new Set(func.entryBlock.arguments);
  for (const op of func.entryBlock.ops()) {
    for (let i = 0; i < op.numOperands; i++) {
      const operand = op.getOperand(i);
      if (operand.isBlockArgument()) continue;
      expect(defined.has(operand)).toBe(true);
    }
    for (let i = 0; i < op.numResults; i++) defined.add(op.getResult(i));
  }
}

describe('MultiOutputFusionPass — insertion point dominates operands', () => {
  it('does not insert merged fusion before a late-defined shared operand', () => {
    const t = new TensorType([8, 8], F32);
    const func = buildFunction('f', [t, t], [t, t], (b, args) => {
      const left = b.fusion([args[0]], [t], 'elementwise', (ib, bargs) => {
        ib.yieldOp([ib.neg(bargs[0]).getResult(0)]);
      });
      const late = b.exp(args[1]);
      const right = b.fusion([args[0], late.getResult(0)], [t], 'elementwise', (ib, bargs) => {
        ib.yieldOp([ib.add(bargs[0], bargs[1]).getResult(0)]);
      });
      b.returnOp([left.getResult(0), right.getResult(0)]);
    });

    expect(new MultiOutputFusionPass().run(func)).toBe(PassResult.CHANGED);
    expect(findOps(func, 'fusion').length).toBe(1);
    assertNoUseBeforeDef(func);
  });
});
