import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';

import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { printFunction } from '../../../src/compiler/ir/printer/printer.js';

import { FusionPass } from '../../../src/compiler/passes/fusion/fusion_pass.js';
import { PassResult } from '../../../src/compiler/passes/pass.js';
import { FusionGroupBuilder, FusionGroup } from '../../../src/compiler/passes/fusion/fusion_groups.js';
import { FusionLegality, FusionKind } from '../../../src/compiler/passes/fusion/fusion_analysis.js';
import { FusionCostModel } from '../../../src/compiler/passes/fusion/fusion_cost.js';

import { CPUTarget, GPUTarget } from '../../../src/backend/target.js';
import { RuntimeTensor } from '../../../src/compiler/runtime/runtime.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { canInlineFuse } from '../../../src/compiler/passes/lowering/graph_to_tensor.js';
import { TraceLog, TraceLevel } from '../../../src/compiler/pipeline/trace.js';

const f32 = ScalarType.F32;
const f32_4x4 = new TensorType([4, 4], f32);

describe('FusionPass', () => {
  it('fuses elementwise operations', () => {
    const func = buildFunction('test_fusion', [f32_4x4, f32_4x4], [f32_4x4], (b, [x, y]) => {
      const add = b.add(x, y);
      const mul = b.mul(add.getResult(0), x);
      const sub = b.sub(mul.getResult(0), y);
      b.returnOp([sub.getResult(0)]);
    });

    const pass = new FusionPass({ cost: { minBenefitRatio: 1.0, launchOverheadUs: 100 } });
    const result = pass.run(func, null);

    assert.equal(result, PassResult.CHANGED);
    const text = printFunction(func);

    assert.ok(text.includes('fusion'));
    assert.equal(func.findOps(op => op.opName === 'add').length, 0);
    const topLevelOps = [...func.entryBlock.ops()];
    const fusionOp = topLevelOps.find(op => op.opName === 'fusion');
    assert.ok(fusionOp);
    assert.equal(fusionOp.numRegions, 1);
  });

  it('fuses broadcast into elementwise', () => {
    const f32_scalar = new TensorType([], f32);
    const func = buildFunction('test_bcast', [f32_4x4, f32_scalar], [f32_4x4], (b, [x, s]) => {
      const bcast = b.broadcast(s, [4, 4], []);
      const add = b.add(x, bcast.getResult(0));
      b.returnOp([add.getResult(0)]);
    });

    const builder = new FusionGroupBuilder(new FusionLegality({}));
    const groups = builder.buildAllGroups(func);
    assert.equal(groups.length, 1);
    assert.equal(groups[0].size, 2);
    assert.equal(groups[0].kind, FusionKind.BROADCAST);
  });

  it('fuses elementwise into reduction (prologue)', () => {
    const func = buildFunction('test_prologue', [f32_4x4, f32_4x4], [f32_4x4], (b, [x, y]) => {
      const add = b.add(x, y);
      const reduce = b.reduce(add.getResult(0), b.scalarConstant(0, f32).getResult(0), [1], 'sum');
      b.returnOp([reduce.getResult(0)]);
    });

    const builder = new FusionGroupBuilder(new FusionLegality({}));
    const groups = builder.buildAllGroups(func);
    assert.equal(groups.length, 1);
    assert.ok(groups[0].ops.some(o => o.opName === 'reduce'));
    assert.ok(groups[0].ops.some(o => o.opName === 'add'));
    assert.equal(groups[0].kind, FusionKind.REDUCTION);
  });

  it('fuses reduction into elementwise (epilogue)', () => {
    const func = buildFunction('test_epilogue', [f32_4x4, f32_4x4], [f32_4x4], (b, [x, bias]) => {
      const reduce = b.reduce(x, b.scalarConstant(0, f32).getResult(0), [1], 'sum');
      const add = b.add(reduce.getResult(0), bias);
      b.returnOp([add.getResult(0)]);
    });

    const builder = new FusionGroupBuilder(new FusionLegality({}));
    const groups = builder.buildAllGroups(func);
    assert.equal(groups.length, 1);
    assert.ok(groups[0].ops.some(o => o.opName === 'reduce'));
    assert.ok(groups[0].ops.some(o => o.opName === 'add'));
    assert.equal(groups[0].kind, FusionKind.REDUCTION);
  });

  it('does not fuse opaque ops in generic fusion', () => {
    const func = buildFunction('test_opaque', [f32_4x4, f32_4x4], [f32_4x4], (b, [x, y]) => {
      const dot = b.dot(x, y, [1], [0]);
      const add = b.add(dot.getResult(0), x);
      b.returnOp([add.getResult(0)]);
    });

    const builder = new FusionGroupBuilder(new FusionLegality({}));
    const groups = builder.buildAllGroups(func);
    assert.equal(groups.length, 0);
  });

  it('EpilogueFusionPass handles opaque+elementwise patterns', async () => {
    const { EpilogueFusionPass } = await import('../../../src/compiler/passes/fusion/epilogue_fusion.js');
    const func = buildFunction('test_epilogue_dot', [f32_4x4, f32_4x4], [f32_4x4], (b, [x, y]) => {
      const dot = b.dot(x, y, [1], [0]);
      const add = b.add(dot.getResult(0), x);
      b.returnOp([add.getResult(0)]);
    });

    const pass = new EpilogueFusionPass();
    const result = pass.run(func);
    assert.equal(result, PassResult.CHANGED);
    assert.ok(func.findOp(op => op.opName === 'fused_dot_epilogue'));
  });

  it('does not fuse multiple reductions together', () => {
    const func = buildFunction('test_multi_reduce', [f32_4x4], [f32_4x4], (b, [x]) => {
      const reduce1 = b.reduce(x, b.scalarConstant(0, f32).getResult(0), [1], 'sum');
      const reduce2 = b.reduce(reduce1.getResult(0), b.scalarConstant(0, f32).getResult(0), [0], 'sum');
      b.returnOp([reduce2.getResult(0)]);
    });

    const builder = new FusionGroupBuilder(new FusionLegality({}));
    const groups = builder.buildAllGroups(func);
    assert.equal(groups.length, 0);
  });

  it('fuses long chain of elementwise operations (vertical)', () => {
    const func = buildFunction('test_long_chain', [f32_4x4], [f32_4x4], (b, [x]) => {
      let curr = x;
      for (let i = 0; i < 10; i++) {
        const op = b.add(curr, x);
        curr = op.getResult(0);
      }
      b.returnOp([curr]);
    });

    const builder = new FusionGroupBuilder(new FusionLegality({}));
    const groups = builder.buildAllGroups(func);
    assert.equal(groups.length, 1);
    assert.equal(groups[0].size, 10);
    assert.equal(groups[0].kind, FusionKind.ELEMENTWISE);
  });

  it('fuses producer into multiple consumers', () => {
    const func = buildFunction('test_multi_consumer', [f32_4x4], [f32_4x4, f32_4x4], (b, [x]) => {
      const p = b.exp(x);
      const c1 = b.add(p.getResult(0), x);
      const c2 = b.sub(p.getResult(0), x);
      b.returnOp([c1.getResult(0), c2.getResult(0)]);
    });

    const builder = new FusionGroupBuilder(new FusionLegality({}));
    const groups = builder.buildAllGroups(func);
    assert.equal(groups.length, 1);
    assert.equal(groups[0].size, 3);
  });

  it('does not fuse if it creates a cycle', () => {
    const func2 = buildFunction('test_cycle_opaque', [f32_4x4], [f32_4x4], (b, [x]) => {
      const o1 = b.exp(x);
      const o2 = b.dot(o1.getResult(0), o1.getResult(0), [1], [0]);
      const o3 = b.add(o1.getResult(0), o2.getResult(0));
      b.returnOp([o3.getResult(0)]);
    });

    const pass2 = new FusionPass({ target: { allowEpilogueFusion: false } });
    pass2.run(func2, null);

    let fusionCount = 0;
    for (const op of func2.ops()) {
      if (op.opName === 'fusion') fusionCount++;
    }
    assert.equal(fusionCount, 0);
  });

  it('extends FunctionPass and integrates with PassManager', () => {
    const pass = new FusionPass({});
    assert.ok(pass instanceof FusionPass);
    assert.equal(pass.name, 'FusionPass');
    assert.ok(typeof pass.run === 'function');
  });

  it('emits fusion decisions via trace at DEBUG level', () => {
    const events = [];
    const trace = new TraceLog({ level: TraceLevel.DEBUG, sink: e => events.push(e) });

    const func = buildFunction('test', [f32_4x4, f32_4x4], [f32_4x4], (b, [x, y]) => {
      const add = b.add(x, y);
      const mul = b.mul(add.getResult(0), x);
      b.returnOp([mul.getResult(0)]);
    });

    const pass = new FusionPass({ cost: { launchOverheadUs: 100 } });
    pass.trace = trace;
    pass.run(func, null);

    const decisions = events.filter(e => e.type === 'fusion_decision');
    assert.ok(decisions.length > 0);
    for (const d of decisions) {
      assert.ok(typeof d.fuse === 'boolean');
      assert.ok(typeof d.groupSize === 'number');
    }
  });
});

describe('FusionCostModel', () => {
  it('rejects fusion when library call would be lost', () => {
    const func = buildFunction('test', [f32_4x4, f32_4x4], [f32_4x4], (b, [x, y]) => {
      const dot = b.dot(x, y, [1], [0]);
      const add = b.add(dot.getResult(0), x);
      b.returnOp([add.getResult(0)]);
    });

    const group = new FusionGroup(0);
    const dotOp = func.findOp(op => op.opName === 'dot');
    const addOp = func.findOp(op => op.opName === 'add');
    group.addOp(dotOp);
    group.addOp(addOp);

    const costModel = new FusionCostModel({ libraryOps: new Set(['dot']) });
    const decision = costModel.shouldFuse(group);
    assert.equal(decision.fuse, false);
    assert.ok(decision.reason.includes('library call'));
  });

  it('rejects fusion when register pressure exceeds limit', () => {
    const func = buildFunction('test', [f32_4x4], [f32_4x4], (b, [x]) => {
      let curr = x;
      for (let i = 0; i < 10; i++) {
        curr = b.add(curr, x).getResult(0);
      }
      b.returnOp([curr]);
    });

    const builder = new FusionGroupBuilder(new FusionLegality({}));
    const groups = builder.buildAllGroups(func);

    const costModel = new FusionCostModel({ maxRegistersPerThread: 1 });
    const decision = costModel.shouldFuse(groups[0]);
    assert.equal(decision.fuse, false);
    assert.ok(decision.reason.includes('register'));
  });
});

describe('FusionLegality shape checking', () => {
  it('rejects elementwise fusion with mismatched shapes', () => {
    const f32_2x3 = new TensorType([2, 3], f32);
    const func = buildFunction('test', [f32_4x4, f32_2x3], [f32_4x4], (b, [x, y]) => {
      const exp_x = b.exp(x);
      const exp_y = b.exp(y);
      b.returnOp([exp_x.getResult(0)]);
    });

    const legality = new FusionLegality({});
    const expX = func.findOp(op => op.opName === 'exp' && op.getOperand(0) === func.args[0]);
    const expY = func.findOp(op => op.opName === 'exp' && op.getOperand(0) === func.args[1]);
    const result = legality.canFuse(expX, expY);
    assert.equal(result.legal, false);
    assert.ok(result.reason.includes('shape mismatch'));
  });

  it('accepts elementwise fusion with matching shapes', () => {
    const func = buildFunction('test', [f32_4x4], [f32_4x4], (b, [x]) => {
      const a = b.exp(x);
      const m = b.log(a.getResult(0));
      b.returnOp([m.getResult(0)]);
    });

    const legality = new FusionLegality({});
    const expOp = func.findOp(op => op.opName === 'exp');
    const logOp = func.findOp(op => op.opName === 'log');
    const result = legality.canFuse(expOp, logOp);
    assert.equal(result.legal, true);
  });
});

describe('Merge re-legality (Fix #4)', () => {
  it('rejects merge that would create multiple reductions', () => {
    const func = buildFunction('test', [f32_4x4, f32_4x4], [f32_4x4], (b, [x, y]) => {
      const add1 = b.add(x, y);
      const reduce1 = b.reduce(add1.getResult(0), b.scalarConstant(0, f32).getResult(0), [1], 'sum');
      const add2 = b.add(x, x);
      const reduce2 = b.reduce(add2.getResult(0), b.scalarConstant(0, f32).getResult(0), [1], 'sum');
      const final = b.add(reduce1.getResult(0), reduce2.getResult(0));
      b.returnOp([final.getResult(0)]);
    });

    const groupA = new FusionGroup(0);
    const groupB = new FusionGroup(1);
    for (const op of func.ops()) {
      if (op.opName === 'reduce') {
        if (groupA.size === 0) groupA.addOp(op);
        else groupB.addOp(op);
      }
    }

    const legality = new FusionLegality({});
    const result = legality.canMergeGroups(groupA, groupB);
    assert.equal(result.legal, false);
    assert.ok(result.reason.includes('multiple reductions'));
  });
});

describe('Fusion:Inline fusion builder registry', () => {
  it('reports inline-fusable for all elementwise ops', () => {
    for (const op of ['add', 'sub', 'mul', 'div', 'exp', 'log', 'sqrt', 'tanh', 'sin', 'cos', 'pow', 'rem', 'abs', 'neg', 'ceil', 'floor', 'round', 'sign']) {
      assert.ok(canInlineFuse(op), `${op} should be inline-fusable`);
    }
  });

  it('reports inline-fusable for special ops', () => {
    for (const op of ['compare', 'select', 'clamp', 'convert', 'broadcast_in_dim', 'broadcast']) {
      assert.ok(canInlineFuse(op), `${op} should be inline-fusable`);
    }
  });

  it('reports non-fusable for reduce/dot/conv', () => {
    for (const op of ['reduce', 'dot', 'conv', 'slice', 'gather']) {
      assert.ok(!canInlineFuse(op), `${op} should NOT be inline-fusable`);
    }
  });
});

describe('Fusion:FusionLegality lowerable check', () => {
  it('rejects fusion when op has no lowering rule', () => {
    const legality = new FusionLegality({});
    const f32_4 = new TensorType([4], f32);
    const func = buildFunction('test', [f32_4], [f32_4, f32_4], (b, [x]) => {
      const c = b.customCall('unknown_fn', [x], [f32_4]);
      const e = b.exp(x);
      b.returnOp([c.getResult(0), e.getResult(0)]);
    });
    const customOp = func.findOp(op => op.opName === 'custom_call');
    const expOp = func.findOp(op => op.opName === 'exp');
    const result = legality.canFuse(customOp, expOp);
    assert.equal(result.legal, false);
    assert.ok(result.reason.includes('no lowering rule'));
  });
});

describe('Fusion:Reduction fusion gating', () => {
  it('rejects elementwise->reduction fusion when disabled', () => {
    const f32_4x4 = new TensorType([4, 4], f32);
    const func = buildFunction('test', [f32_4x4, f32_4x4], [f32_4x4], (b, [x, y]) => {
      const add = b.add(x, y);
      const reduce = b.reduce(add.getResult(0), b.scalarConstant(0, f32).getResult(0), [1], 'sum');
      b.returnOp([reduce.getResult(0)]);
    });
    const legality = new FusionLegality({ allowReductionFusion: false });
    const builder = new FusionGroupBuilder(legality);
    const groups = builder.buildAllGroups(func);
    assert.equal(groups.length, 0);
  });

  it('allows elementwise->reduction fusion when enabled', () => {
    const f32_4x4 = new TensorType([4, 4], f32);
    const func = buildFunction('test', [f32_4x4, f32_4x4], [f32_4x4], (b, [x, y]) => {
      const add = b.add(x, y);
      const reduce = b.reduce(add.getResult(0), b.scalarConstant(0, f32).getResult(0), [1], 'sum');
      b.returnOp([reduce.getResult(0)]);
    });
    const legality = new FusionLegality({ allowReductionFusion: true });
    const builder = new FusionGroupBuilder(legality);
    const groups = builder.buildAllGroups(func);
    assert.ok(groups.length > 0);
  });
});

describe('Fusion:Cost model recompute cost', () => {
  it('accounts for recompute cost in DAG fusion', () => {
    const f32_4x4 = new TensorType([4, 4], f32);
    const func = buildFunction('test', [f32_4x4], [f32_4x4], (b, [x]) => {
      const e = b.exp(x);
      const a = b.add(e.getResult(0), x);
      const m = b.mul(e.getResult(0), a.getResult(0));
      b.returnOp([m.getResult(0)]);
    });
    const legality = new FusionLegality({});
    const builder = new FusionGroupBuilder(legality);
    const groups = builder.buildAllGroups(func);
    assert.ok(groups.length > 0);
    const cost = new FusionCostModel({ launchOverheadUs: 100 });
    const groupCost = cost.estimateGroupCost(groups[0]);
    assert.ok(groupCost.recomputeCost > 0, 'should have recompute cost for exp used twice');
    assert.ok(groupCost.fusedFLOPs > groupCost.unfusedFLOPs, 'fused FLOPs should exceed unfused due to recompute');
  });
});

describe('Fusion:CSE in fused expressions', () => {
  it('fused code with DAG reuse generates let-bindings', () => {
    const f32_4 = new TensorType([4], f32);
    const func = buildFunction('cse_test', [f32_4], [f32_4], (b, [x]) => {
      const e = b.exp(x);
      const a = b.add(e.getResult(0), x);
      const m = b.mul(e.getResult(0), a.getResult(0));
      b.returnOp([m.getResult(0)]);
    });
    const compiled = compileGraph(func, CPUTarget(), { enableFusion: true });
    const source = compiled.getSource('cse_test');
    assert.ok(source.includes('cse'), `generated code should contain CSE variable, got:\n${source}`);
  });

  it('CSE fused code produces correct results', () => {
    const f32_4 = new TensorType([4], f32);
    const func = buildFunction('cse_correct', [f32_4], [f32_4], (b, [x]) => {
      const e = b.exp(x);
      const a = b.add(e.getResult(0), x);
      const m = b.mul(e.getResult(0), a.getResult(0));
      b.returnOp([m.getResult(0)]);
    });
    const compiled = compileGraph(func, CPUTarget(), { enableFusion: true });
    const X = RuntimeTensor.fromArray([0, 1, -1, 2], [4]);
    const out = RuntimeTensor.zeros([4]);
    compiled.run('cse_correct', X, out);
    for (let i = 0; i < 4; i++) {
      const x = X.data[i];
      const expected = Math.exp(x) * (Math.exp(x) + x);
      assert.ok(Math.abs(out.data[i] - expected) < 1e-4, `at ${i}: expected ${expected}, got ${out.data[i]}`);
    }
  });
});

describe('Fusion:Multi-output fusion', () => {
  it('fuses producer consumed by two outputs', () => {
    const f32_4 = new TensorType([4], f32);
    const func = buildFunction('multi_out', [f32_4], [f32_4, f32_4], (b, [x]) => {
      const e = b.exp(x);
      const a = b.add(e.getResult(0), x);
      const s = b.sub(e.getResult(0), x);
      b.returnOp([a.getResult(0), s.getResult(0)]);
    });
    const compiled = compileGraph(func, CPUTarget(), { enableFusion: true });
    const X = RuntimeTensor.fromArray([0, 1, -1, 2], [4]);
    const out1 = RuntimeTensor.zeros([4]);
    const out2 = RuntimeTensor.zeros([4]);
    compiled.run('multi_out', X, out1, out2);
    for (let i = 0; i < 4; i++) {
      const x = X.data[i];
      assert.ok(Math.abs(out1.data[i] - (Math.exp(x) + x)) < 1e-4);
      assert.ok(Math.abs(out2.data[i] - (Math.exp(x) - x)) < 1e-4);
    }
  });
});

describe('Fusion:Broadcast index mapping in fusion', () => {
  it('fuses broadcast with size-1 dimension correctly', () => {
    const func = buildFunction('fused_bcast',
      [new TensorType([3, 4], f32), new TensorType([1, 4], f32)],
      [new TensorType([3, 4], f32)],
      (b, [x, bias]) => {
        const bcast = b.broadcast(bias, [3, 4], [0, 1]);
        const result = b.add(x, bcast.getResult(0));
        b.returnOp([result.getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget(), { enableFusion: true });
    const X = RuntimeTensor.fromArray([1,2,3,4, 5,6,7,8, 9,10,11,12], [3, 4]);
    const bias = RuntimeTensor.fromArray([10, 20, 30, 40], [1, 4]);
    const out = RuntimeTensor.zeros([3, 4]);
    compiled.run('fused_bcast', X, bias, out);
    assert.deepEqual([...out.data], [11,22,33,44, 15,26,37,48, 19,30,41,52]);
  });
});

describe('Fusion:Target capability gating', () => {
  it('generic fusion rejects opaque ops (dot), EpilogueFusionPass handles them', () => {
    const f32_4x4 = new TensorType([4, 4], f32);
    const func = buildFunction('test', [f32_4x4, f32_4x4], [f32_4x4], (b, [x, y]) => {
      const dot = b.dot(x, y, [1], [0]);
      const add = b.add(dot.getResult(0), x);
      b.returnOp([add.getResult(0)]);
    });
    const legality = new FusionLegality({});
    const builder = new FusionGroupBuilder(legality);
    const groups = builder.buildAllGroups(func);
    assert.equal(groups.length, 0, 'generic fusion should reject opaque dot');
  });

  it('EpilogueFusionPass respects target.enableEpilogueFusion=false', async () => {
    const { EpilogueFusionPass } = await import('../../../src/compiler/passes/fusion/epilogue_fusion.js');
    const { PassResult } = await import('../../../src/compiler/passes/pass.js');
    const f32_4x4 = new TensorType([4, 4], f32);
    const func = buildFunction('test', [f32_4x4, f32_4x4], [f32_4x4], (b, [x, y]) => {
      const dot = b.dot(x, y, [1], [0]);
      const add = b.add(dot.getResult(0), x);
      b.returnOp([add.getResult(0)]);
    });
    const pass = new EpilogueFusionPass({ target: { enableEpilogueFusion: false } });
    assert.equal(pass.run(func), PassResult.UNCHANGED);
  });
});

describe('Fusion:compare+select+clamp fused chain', () => {
  it('fuses compare->select->clamp correctly', () => {
    const f32_4 = new TensorType([4], f32);
    const func = buildFunction('csc_chain', [f32_4, f32_4], [f32_4], (b, [x, y]) => {
      const cmp = b.compare(x, y, 'gt');
      const sel = b.select(cmp.getResult(0), x, y);
      const lo = b.broadcast(b.scalarConstant(0, f32).getResult(0), [4], []);
      const hi = b.broadcast(b.scalarConstant(10, f32).getResult(0), [4], []);
      const result = b.clamp(lo.getResult(0), sel.getResult(0), hi.getResult(0));
      b.returnOp([result.getResult(0)]);
    });
    const compiled = compileGraph(func, CPUTarget(), { enableFusion: true });
    const X = RuntimeTensor.fromArray([-5, 15, 3, 8], [4]);
    const Y = RuntimeTensor.fromArray([2, 2, 4, 4], [4]);
    const out = RuntimeTensor.zeros([4]);
    compiled.run('csc_chain', X, Y, out);
    assert.deepEqual([...out.data], [2, 10, 4, 8]);
  });
});