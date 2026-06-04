import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';

import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { Operation } from '../../../src/compiler/ir/graph/operation.js';
import { Block, Region } from '../../../src/compiler/ir/graph/block.js';
import { MultiOutputFusionPass } from '../../../src/compiler/passes/fusion/multi_output_fusion.js';
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

function runMOF(func, config = {}) {
  const mod = new GraphModule('test');
  mod.addFunction(func);
  const pm = new PassManager();
  pm.addPass(new MultiOutputFusionPass(config));
  return pm.run(mod);
}

describe('MultiOutputFusionPass', () => {
  it('merges sibling fusions sharing common input', () => {
    const func = buildFunction('shared', [t44, t44, t44], [t44, t44],
      (b, [x, a, c]) => {
        const f1 = makeFusion(b, [x, a], blk => {
          const op = new Operation('add', [blk.arguments[0], blk.arguments[1]], [t44]);
          blk.pushOp(op);
          return [op.getResult(0)];
        });
        const f2 = makeFusion(b, [x, c], blk => {
          const op = new Operation('mul', [blk.arguments[0], blk.arguments[1]], [t44]);
          blk.pushOp(op);
          return [op.getResult(0)];
        });
        b.returnOp([f1.getResult(0), f2.getResult(0)]);
      }
    );
    assert.equal(countFusions(func), 2);
    const result = runMOF(func);
    assert.equal(result.changed, true);
    assert.equal(countFusions(func), 1);
    const fusion = [...func.ops()].find(op => op.opName === 'fusion');
    assert.equal(fusion.numResults, 2);
  });

  it('rejects producer-consumer fusions', () => {
    const func = buildFunction('pc', [t44, t44], [t44],
      (b, [x, y]) => {
        const f1 = makeFusion(b, [x, y], blk => {
          const op = new Operation('add', [blk.arguments[0], blk.arguments[1]], [t44]);
          blk.pushOp(op);
          return [op.getResult(0)];
        });
        const f2 = makeFusion(b, [f1.getResult(0), x], blk => {
          const op = new Operation('mul', [blk.arguments[0], blk.arguments[1]], [t44]);
          blk.pushOp(op);
          return [op.getResult(0)];
        });
        b.returnOp([f2.getResult(0)]);
      }
    );
    const result = runMOF(func);
    assert.equal(result.changed, false);
    assert.equal(countFusions(func), 2);
  });

  it('deduplicates shared operands', () => {
    const func = buildFunction('dedup', [t44, t44, t44], [t44, t44],
      (b, [x, a, c]) => {
        const f1 = makeFusion(b, [x, a], blk => {
          const op = new Operation('add', [blk.arguments[0], blk.arguments[1]], [t44]);
          blk.pushOp(op);
          return [op.getResult(0)];
        });
        const f2 = makeFusion(b, [x, c], blk => {
          const op = new Operation('mul', [blk.arguments[0], blk.arguments[1]], [t44]);
          blk.pushOp(op);
          return [op.getResult(0)];
        });
        b.returnOp([f1.getResult(0), f2.getResult(0)]);
      }
    );
    runMOF(func);
    const fusion = [...func.ops()].find(op => op.opName === 'fusion');
    assert.equal(fusion.numOperands, 3);
  });

  it('respects maxOutputs limit', () => {
    const func = buildFunction('max_out', [t44, t44, t44], [t44, t44],
      (b, [x, a, c]) => {
        const f1 = makeFusion(b, [x, a], blk => {
          const op = new Operation('add', [blk.arguments[0], blk.arguments[1]], [t44]);
          blk.pushOp(op);
          return [op.getResult(0)];
        });
        const f2 = makeFusion(b, [x, c], blk => {
          const op = new Operation('mul', [blk.arguments[0], blk.arguments[1]], [t44]);
          blk.pushOp(op);
          return [op.getResult(0)];
        });
        b.returnOp([f1.getResult(0), f2.getResult(0)]);
      }
    );
    const result = runMOF(func, { maxOutputs: 1 });
    assert.equal(result.changed, false);
  });
});
