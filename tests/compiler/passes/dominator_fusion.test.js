import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';

import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { DominatorFusionPass } from '../../../src/compiler/passes/fusion/dominator_fusion.js';
import { PassResult } from '../../../src/compiler/passes/pass.js';
import { PassManager } from '../../../src/compiler/passes/pass_manager.js';
import { GraphModule } from '../../../src/compiler/ir/graph/module.js';
import { CPUTarget } from '../../../src/compiler/backend/target.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';

const f32 = ScalarType.F32;
const t44 = new TensorType([4, 4], f32);

function countFusions(func) {
  let count = 0;
  for (const op of func.ops()) if (op.opName === 'fusion') count++;
  return count;
}

function innerOpNames(fusionOp) {
  const names = [];
  for (const op of fusionOp.regions[0].entryBlock.ops()) {
    if (op.opName !== 'yield') names.push(op.opName);
  }
  return names;
}

function runDomFusion(func, config = {}) {
  const mod = new GraphModule('test');
  mod.addFunction(func);
  const pm = new PassManager();
  pm.addPass(new DominatorFusionPass(config));
  return pm.run(mod);
}

describe('DominatorFusionPass', () => {
  it('fuses elementwise chain', () => {
    const func = buildFunction('chain', [t44, t44], [t44],
      (b, [x, y]) => {
        const a = b.add(x, y);
        const e = b.exp(a.getResult(0));
        const n = b.neg(e.getResult(0));
        b.returnOp([n.getResult(0)]);
      }
    );
    const result = runDomFusion(func);
    assert.equal(result.changed, true);
    assert.equal(countFusions(func), 1);
    const fusion = [...func.ops()].find(op => op.opName === 'fusion');
    const inner = innerOpNames(fusion);
    assert.ok(inner.includes('add'));
    assert.ok(inner.includes('exp'));
    assert.ok(inner.includes('neg'));
  });

  it('fuses diamond graph in single pass', () => {
    const func = buildFunction('diamond', [t44, t44], [t44],
      (b, [x, y]) => {
        const a = b.add(x, y);
        const e1 = b.exp(a.getResult(0));
        const e2 = b.neg(a.getResult(0));
        const m = b.mul(e1.getResult(0), e2.getResult(0));
        b.returnOp([m.getResult(0)]);
      }
    );
    const result = runDomFusion(func);
    assert.equal(result.changed, true);
    assert.ok(countFusions(func) >= 1);
  });

  it('stops at opaque ops', () => {
    const func = buildFunction('opaque_barrier', [t44, t44], [t44],
      (b, [x, y]) => {
        const a = b.add(x, y);
        const mm = b.matmul(a.getResult(0), y);
        const e = b.exp(mm.getResult(0));
        b.returnOp([e.getResult(0)]);
      }
    );
    runDomFusion(func);
    let hasDot = false;
    for (const op of func.ops()) {
      if (op.opName === 'dot') hasDot = true;
      if (op.opName === 'fusion') {
        const inner = innerOpNames(op);
        assert.ok(!inner.includes('dot'));
      }
    }
  });

  it('returns UNCHANGED for single-op graph', () => {
    const func = buildFunction('single', [t44], [t44],
      (b, [x]) => {
        const e = b.exp(x);
        b.returnOp([e.getResult(0)]);
      }
    );
    const result = runDomFusion(func);
    assert.equal(result.changed, false);
  });

  it('compiles end-to-end with dominator strategy', () => {
    const func = buildFunction('e2e_dom', [t44, t44], [t44],
      (b, [x, y]) => {
        const a = b.add(x, y);
        const e = b.exp(a.getResult(0));
        const n = b.neg(e.getResult(0));
        const m = b.mul(n.getResult(0), x);
        b.returnOp([m.getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget(), { fusionStrategy: 'dominator' });
    assert.ok(compiled.listKernels().includes('e2e_dom'));
  });

  it('respects maxFusionSize', () => {
    const func = buildFunction('size', [t44, t44], [t44],
      (b, [x, y]) => {
        const a = b.add(x, y);
        const e = b.exp(a.getResult(0));
        const n = b.neg(e.getResult(0));
        b.returnOp([n.getResult(0)]);
      }
    );
    const result = runDomFusion(func, { maxFusionSize: 2 });
    if (result.changed) {
      for (const op of func.ops()) {
        if (op.opName === 'fusion') {
          assert.ok(innerOpNames(op).length <= 2);
        }
      }
    }
  });
});
