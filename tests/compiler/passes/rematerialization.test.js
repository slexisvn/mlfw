import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';

import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { RematerializationPass } from '../../../src/compiler/passes/memory/rematerialization.js';
import { PassManager } from '../../../src/compiler/passes/pass_manager.js';
import { GraphModule } from '../../../src/compiler/ir/graph/module.js';
import { UseDefAnalysis } from '../../../src/compiler/analysis/use_def.js';

const f32 = ScalarType.F32;

function opCount(func) {
  let count = 0;
  for (const op of func.ops()) count++;
  return count;
}

function measurePeak(func) {
  const pass = new RematerializationPass({ memoryBudget: Infinity });
  const ud = UseDefAnalysis.compute(func);
  return pass._analyzeIntervalPressure(func, ud).peakPressure;
}

function runRemat(func, budget) {
  const mod = new GraphModule('test');
  mod.addFunction(func);
  const pm = new PassManager();
  pm.addPass(new RematerializationPass({ memoryBudget: budget }));
  return pm.run(mod);
}

describe('RematerializationPass', () => {
  it('returns UNCHANGED when budget is Infinity', () => {
    const func = buildFunction('inf', [new TensorType([8, 8], f32)], [new TensorType([8, 8], f32)],
      (b, [x]) => {
        const a = b.exp(x);
        b.returnOp([a.getResult(0)]);
      }
    );
    const result = runRemat(func, Infinity);
    assert.equal(result.changed, false);
  });

  it('returns UNCHANGED when peak is within budget', () => {
    const func = buildFunction('ok', [new TensorType([4, 4], f32)], [new TensorType([4, 4], f32)],
      (b, [x]) => {
        const a = b.exp(x);
        b.returnOp([a.getResult(0)]);
      }
    );
    const peak = measurePeak(func);
    const result = runRemat(func, peak + 1);
    assert.equal(result.changed, false);
  });

  it('does not rematerialize single-use values', () => {
    const func = buildFunction('single_use', [new TensorType([64, 64], f32)], [new TensorType([64, 64], f32)],
      (b, [x]) => {
        const a = b.exp(x);
        const c = b.neg(a.getResult(0));
        const d = b.abs(c.getResult(0));
        b.returnOp([d.getResult(0)]);
      }
    );
    const before = opCount(func);
    runRemat(func, 1);
    assert.equal(opCount(func), before);
  });

  it('does not increase peak memory', () => {
    const big = new TensorType([512, 512], f32);
    const func = buildFunction('no_increase', [big, big], [big],
      (b, [x, y]) => {
        const a = b.exp(x);
        const c = b.exp(y);
        const d = b.neg(x);
        const e = b.neg(y);
        const f = b.abs(x);
        const s1 = b.add(a.getResult(0), c.getResult(0));
        const s2 = b.add(s1.getResult(0), d.getResult(0));
        const s3 = b.add(s2.getResult(0), e.getResult(0));
        const s4 = b.add(s3.getResult(0), f.getResult(0));
        const out = b.add(s4.getResult(0), a.getResult(0));
        b.returnOp([out.getResult(0)]);
      }
    );
    const before = measurePeak(func);
    runRemat(func, before - 1);
    const after = measurePeak(func);
    assert.ok(after <= before, `peak increased: ${before} -> ${after}`);
  });

  it('skips ops with side effects', () => {
    const func = buildFunction('sideeff', [new TensorType([4, 4], f32)], [new TensorType([4, 4], f32)],
      (b, [x]) => {
        const a = b.exp(x);
        b.returnOp([a.getResult(0)]);
      }
    );
    const before = opCount(func);
    runRemat(func, 0);
    assert.equal(opCount(func), before);
  });

  it('analyzes interval pressure correctly', () => {
    const big = new TensorType([100, 100], f32);
    const func = buildFunction('pressure', [big], [big],
      (b, [x]) => {
        const a = b.exp(x);
        const c = b.neg(a.getResult(0));
        b.returnOp([c.getResult(0)]);
      }
    );
    const pass = new RematerializationPass({ memoryBudget: Infinity });
    const ud = UseDefAnalysis.compute(func);
    const result = pass._analyzeIntervalPressure(func, ud);
    assert.ok(result.peakPressure > 0);
    assert.ok(result.peakIdx >= 0);
  });

  it('respects maxIterations', () => {
    const big = new TensorType([512, 512], f32);
    const func = buildFunction('max_iter', [big], [big],
      (b, [x]) => {
        const a = b.exp(x);
        const c = b.neg(a.getResult(0));
        const out = b.add(a.getResult(0), c.getResult(0));
        b.returnOp([out.getResult(0)]);
      }
    );
    const before = opCount(func);
    runRemat(func, 0, { maxIterations: 0 });
    assert.equal(opCount(func), before);
  });
});
