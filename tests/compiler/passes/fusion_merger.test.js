import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';

import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { Operation } from '../../../src/compiler/ir/graph/operation.js';
import { Block, Region } from '../../../src/compiler/ir/graph/block.js';
import { FusionMergerPass } from '../../../src/compiler/passes/fusion/fusion_merger.js';
import { PassResult } from '../../../src/compiler/passes/pass.js';
import { PassManager } from '../../../src/compiler/passes/pass_manager.js';
import { GraphModule } from '../../../src/compiler/ir/graph/module.js';

const f32 = ScalarType.F32;
const t44 = new TensorType([4, 4], f32);

function makeFusion(b, operands, innerBuilder) {
  const r = new Region();
  const blk = new Block(operands.map(o => o.type));
  r.addBlock(blk);
  const innerResults = innerBuilder(blk);
  blk.pushOp(new Operation('yield', innerResults, []));
  const outputTypes = innerResults.map(v => v.type);
  return b._buildOp('fusion', operands, outputTypes, { fusion_kind: 'kElementwise' }, [r]);
}

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

function runMerger(func, config = {}) {
  const mod = new GraphModule('test');
  mod.addFunction(func);
  const pm = new PassManager();
  pm.addPass(new FusionMergerPass(config));
  return pm.run(mod);
}

describe('FusionMergerPass', () => {
  it('merges adjacent producer-consumer fusions', () => {
    const func = buildFunction('merge_pc', [t44, t44, t44], [t44],
      (b, [a, w, c]) => {
        const f1 = makeFusion(b, [a, w], blk => {
          const add = new Operation('add', [blk.arguments[0], blk.arguments[1]], [t44]);
          blk.pushOp(add);
          return [add.getResult(0)];
        });
        const f2 = makeFusion(b, [f1.getResult(0), c], blk => {
          const mul = new Operation('mul', [blk.arguments[0], blk.arguments[1]], [t44]);
          blk.pushOp(mul);
          return [mul.getResult(0)];
        });
        b.returnOp([f2.getResult(0)]);
      }
    );
    assert.equal(countFusions(func), 2);
    const result = runMerger(func);
    assert.equal(result.changed, true);
    assert.equal(countFusions(func), 1);
    const fusion = [...func.ops()].find(op => op.opName === 'fusion');
    assert.deepStrictEqual(innerOpNames(fusion), ['add', 'mul']);
    assert.equal(fusion.numOperands, 3);
  });

  it('skips independent fusions with no data edge', () => {
    const func = buildFunction('no_edge', [t44, t44, t44, t44], [t44],
      (b, [a, w, c, d]) => {
        const f1 = makeFusion(b, [a, w], blk => {
          const add = new Operation('add', [blk.arguments[0], blk.arguments[1]], [t44]);
          blk.pushOp(add);
          return [add.getResult(0)];
        });
        const f2 = makeFusion(b, [c, d], blk => {
          const mul = new Operation('mul', [blk.arguments[0], blk.arguments[1]], [t44]);
          blk.pushOp(mul);
          return [mul.getResult(0)];
        });
        const out = b.add(f1.getResult(0), f2.getResult(0));
        b.returnOp([out.getResult(0)]);
      }
    );
    assert.equal(countFusions(func), 2);
    const result = runMerger(func);
    assert.equal(result.changed, false);
    assert.equal(countFusions(func), 2);
  });

  it('respects maxFusionSize limit', () => {
    const func = buildFunction('size_limit', [t44, t44, t44], [t44],
      (b, [a, w, c]) => {
        const f1 = makeFusion(b, [a, w], blk => {
          const add = new Operation('add', [blk.arguments[0], blk.arguments[1]], [t44]);
          blk.pushOp(add);
          return [add.getResult(0)];
        });
        const f2 = makeFusion(b, [f1.getResult(0), c], blk => {
          const mul = new Operation('mul', [blk.arguments[0], blk.arguments[1]], [t44]);
          blk.pushOp(mul);
          return [mul.getResult(0)];
        });
        b.returnOp([f2.getResult(0)]);
      }
    );
    const result = runMerger(func, { maxFusionSize: 1 });
    assert.equal(result.changed, false);
  });

  it('deduplicates shared operands during merge', () => {
    const func = buildFunction('shared_op', [t44, t44], [t44],
      (b, [x, y]) => {
        const f1 = makeFusion(b, [x, y], blk => {
          const add = new Operation('add', [blk.arguments[0], blk.arguments[1]], [t44]);
          blk.pushOp(add);
          return [add.getResult(0)];
        });
        const f2 = makeFusion(b, [f1.getResult(0), x], blk => {
          const mul = new Operation('mul', [blk.arguments[0], blk.arguments[1]], [t44]);
          blk.pushOp(mul);
          return [mul.getResult(0)];
        });
        b.returnOp([f2.getResult(0)]);
      }
    );
    const result = runMerger(func);
    assert.equal(result.changed, true);
    const fusion = [...func.ops()].find(op => op.opName === 'fusion');
    assert.equal(fusion.numOperands, 2);
  });
});
