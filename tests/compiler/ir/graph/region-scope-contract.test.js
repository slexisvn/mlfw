import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { verifyFunction } from '../../../../src/compiler/ir/graph/verifier.js';
import { compileGraph } from '../../../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../../../src/compiler/support/target.js';
import { TensorType } from '../../../../src/compiler/ir/graph/types.js';
import { F32 as F } from '../../../_utils/ir_fixture.js';


function buildIfProgram(predTrue) {
  const xT = new TensorType([4], F);
  return buildFunction('iff', [xT], [xT], (b, a) => {
    const thr = b.scalarConstant(predTrue ? -100 : 100, F).getResult(0);
    const s = b.reduce(a[0], b.scalarConstant(0, F).getResult(0), [0], 'sum').getResult(0);
    const pred = b.compare(s, thr, 'gt').getResult(0);
    const two = b.broadcast(b.scalarConstant(2, F).getResult(0), [4], []).getResult(0);
    const one = b.broadcast(b.scalarConstant(1, F).getResult(0), [4], []).getResult(0);
    const ifo = b.ifOp(pred, [xT],
      (tb) => { tb.yieldOp([tb.mul(a[0], two).getResult(0)]); },
      (eb) => { eb.yieldOp([eb.add(a[0], one).getResult(0)]); });
    b.returnOp([ifo.getResult(0)]);
  });
}

function runIf(predTrue) {
  const r = compileGraph(buildIfProgram(predTrue), CPUTarget());
  const out = new Float32Array(4);
  r.run('iff', new Float32Array([1, 2, 3, 4]), out);
  return [...out];
}

describe('region scoping contract', () => {
  it('compiles and runs a region (if) program: then-branch computes x*2', () => {
    expect(runIf(true)).toEqual([2, 4, 6, 8]);
  });

  it('compiles and runs a region (if) program: else-branch computes x+1', () => {
    expect(runIf(false)).toEqual([2, 3, 4, 5]);
  });

  it('ops() iterates the top-level scope only; region bodies are reached via opsRecursive()', () => {
    const func = buildIfProgram(true);
    const top = func.opsArray().map((o) => o.opName);
    const recursive = [...func.opsRecursive()].map((o) => o.opName);
    expect(top).toContain('if');
    expect(top).not.toContain('mul');
    expect(recursive).toContain('mul');
  });

  it('the verifier rejects a top-level use of a value defined inside a region (scope escape)', () => {
    const func = buildIfProgram(true);
    const innerValue = [...func.opsRecursive()].find((o) => o.opName === 'mul').getResult(0);
    func.getReturnOp().replaceOperand(0, innerValue);
    expect(verifyFunction(func).some((e) => /used before definition/.test(e.message))).toBe(true);
  });
});
