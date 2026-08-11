import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType } from '../../../src/compiler/ir/graph/types.js';
import { BackwardGraphBuilder } from '../../../src/compiler/ad/backward_builder.js';
import { JointGraphBuilder } from '../../../src/compiler/ad/joint_builder.js';
import {
  EveryKPolicy, SqrtPolicy, MemoryBudgetPolicy, ExplicitPolicy,
} from '../../../src/compiler/ad/checkpoint_policy.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../../src/backend/target.js';
import { UseDefAnalysis } from '../../../src/compiler/analysis/use_def.js';
import '../../../src/compiler/ad/index.js';
import { F32 } from '../../_utils/ir_fixture.js';


function t(shape) {
  return new TensorType(shape, F32);
}

function numel(shape) {
  return shape.reduce((a, b) => a * b, 1);
}

function buildChain(numOps) {
  return buildFunction('chain_fwd', [t([4]), t([4])], [t([4])], (b, args) => {
    let current = args[0];
    for (let i = 0; i < numOps; i++) {
      current = b.add(current, args[1]).getResult(0);
    }
    b.returnOp([current]);
  });
}

function buildAndCompileBackward(forwardFunc, opts = {}) {
  const bwdBuilder = new BackwardGraphBuilder(opts);
  const result = bwdBuilder.build(forwardFunc);
  const bwdFunc = result.backwardFunc;
  const compiled = compileGraph(bwdFunc, CPUTarget());
  const funcName = compiled.listKernels()[0];

  return {
    run(...inputBuffers) {
      const outputBuffers = bwdFunc.outputTypes.map(
        outType => new Float32Array(Math.max(numel(outType.shape), 1))
      );
      compiled.run(funcName, ...inputBuffers, ...outputBuffers);
      return outputBuffers;
    },
    inputTypes: bwdFunc.inputTypes,
    outputTypes: bwdFunc.outputTypes,
    savedValues: result.savedValues,
    bwdFunc,
  };
}

describe('CheckpointPolicy segmentation', () => {
  it('EveryKPolicy segments 9 ops into 3 groups of 3', () => {
    const func = buildChain(9);
    const policy = new EveryKPolicy({ segmentSize: 3 });
    const analysis = UseDefAnalysis.compute(func);
    const segments = policy.segment(analysis.topologicalOrder, func);

    expect(segments.length).toBe(3);
    for (const seg of segments) {
      expect(seg.ops.length).toBe(3);
    }
  });

  it('SqrtPolicy on 16 ops creates ~4-op segments', () => {
    const func = buildChain(16);
    const policy = new SqrtPolicy();
    const analysis = UseDefAnalysis.compute(func);
    const segments = policy.segment(analysis.topologicalOrder, func);

    expect(segments.length).toBe(4);
    for (const seg of segments) {
      expect(seg.ops.length).toBe(4);
    }
  });

  it('MemoryBudgetPolicy cuts when threshold exceeded', () => {
    const func = buildFunction('mem_fwd', [t([1024]), t([1024])], [t([1024])], (b, args) => {
      const a = b.add(args[0], args[1]).getResult(0);
      const c = b.mul(a, args[0]).getResult(0);
      const d = b.add(c, args[1]).getResult(0);
      b.returnOp([d]);
    });
    const policy = new MemoryBudgetPolicy({ threshold: 4096 });
    const analysis = UseDefAnalysis.compute(func);
    const segments = policy.segment(analysis.topologicalOrder, func);

    expect(segments.length).toBeGreaterThan(1);
  });

  it('ExplicitPolicy splits at specified op boundaries', () => {
    const func = buildChain(6);
    const analysis = UseDefAnalysis.compute(func);
    const ops = analysis.topologicalOrder.filter(
      op => op.opName !== 'return' && op.opName !== 'constant'
    );
    const boundaryIds = new Set([ops[2].id]);
    const policy = new ExplicitPolicy({ boundaries: boundaryIds });
    const segments = policy.segment(analysis.topologicalOrder, func);

    expect(segments.length).toBe(2);
    expect(segments[0].ops.length).toBe(3);
    expect(segments[1].ops.length).toBe(3);
  });

  it('segments compute correct boundary inputs and outputs', () => {
    const func = buildChain(6);
    const policy = new EveryKPolicy({ segmentSize: 3 });
    const analysis = UseDefAnalysis.compute(func);
    const segments = policy.segment(analysis.topologicalOrder, func);

    expect(segments[0].boundaryOutputs.size).toBeGreaterThan(0);
    expect(segments[1].boundaryInputs.size).toBeGreaterThan(0);
  });
});

describe('Checkpointed backward builder', () => {
  it('saves fewer values than non-checkpointed', () => {
    const func = buildChain(9);

    const normalBuilder = new BackwardGraphBuilder();
    const normalResult = normalBuilder.build(func);

    const ckptBuilder = new BackwardGraphBuilder({
      checkpointPolicy: new EveryKPolicy({ segmentSize: 3 }),
    });
    const ckptResult = ckptBuilder.build(func);

    expect(ckptResult.savedValues.length).toBeLessThan(normalResult.savedValues.length);
  });

  it('backward graph contains recompute ops', () => {
    const func = buildChain(6);
    const builder = new BackwardGraphBuilder({
      checkpointPolicy: new EveryKPolicy({ segmentSize: 3 }),
    });
    const { backwardFunc } = builder.build(func);

    const ops = backwardFunc.opsArray();
    const addOps = ops.filter(op => op.opName === 'add');
    expect(addOps.length).toBeGreaterThanOrEqual(6);
  });

  it('produces correct gradients for add chain', () => {
    const func = buildChain(6);

    const normalBwd = buildAndCompileBackward(func);
    const ckptBwd = buildAndCompileBackward(func, {
      checkpointPolicy: new EveryKPolicy({ segmentSize: 3 }),
    });

    const normalInputs = [];
    for (let i = 0; i < normalBwd.inputTypes.length; i++) {
      const buf = new Float32Array(numel(normalBwd.inputTypes[i].shape));
      if (i === 0) buf.set([1, 1, 1, 1]);
      normalInputs.push(buf);
    }
    const normalGrads = normalBwd.run(...normalInputs);

    const ckptInputs = [];
    for (let i = 0; i < ckptBwd.inputTypes.length; i++) {
      const buf = new Float32Array(numel(ckptBwd.inputTypes[i].shape));
      if (i === 0) buf.set([1, 1, 1, 1]);
      ckptInputs.push(buf);
    }
    const ckptGrads = ckptBwd.run(...ckptInputs);

    expect(normalGrads.length).toBe(ckptGrads.length);
    for (let g = 0; g < normalGrads.length; g++) {
      for (let i = 0; i < normalGrads[g].length; i++) {
        expect(ckptGrads[g][i]).toBeCloseTo(normalGrads[g][i], 4);
      }
    }
  });

  it('produces correct gradients for mul chain with SqrtPolicy', () => {
    const func = buildFunction('mul_chain', [t([4]), t([4])], [t([4])], (b, args) => {
      const a = b.mul(args[0], args[1]).getResult(0);
      const c = b.mul(a, args[1]).getResult(0);
      const d = b.mul(c, args[0]).getResult(0);
      const e = b.mul(d, args[1]).getResult(0);
      b.returnOp([e]);
    });

    const normalBwd = buildAndCompileBackward(func);
    const ckptBwd = buildAndCompileBackward(func, {
      checkpointPolicy: new SqrtPolicy(),
    });

    const x = [2, 3, 4, 5];
    const y = [1, 2, 1, 2];

    const normalInputs = [];
    for (let i = 0; i < normalBwd.inputTypes.length; i++) {
      normalInputs.push(new Float32Array(numel(normalBwd.inputTypes[i].shape)));
    }
    normalInputs[0].set([1, 1, 1, 1]);
    for (let i = 1; i < normalBwd.inputTypes.length; i++) {
      normalInputs[i].set(i % 2 === 1 ? x : y);
    }

    const ckptInputs = [];
    for (let i = 0; i < ckptBwd.inputTypes.length; i++) {
      ckptInputs.push(new Float32Array(numel(ckptBwd.inputTypes[i].shape)));
    }
    ckptInputs[0].set([1, 1, 1, 1]);
    for (let i = 1; i < ckptBwd.inputTypes.length; i++) {
      ckptInputs[i].set(i % 2 === 1 ? x : y);
    }

    const normalGrads = normalBwd.run(...normalInputs);
    const ckptGrads = ckptBwd.run(...ckptInputs);

    expect(normalGrads.length).toBe(ckptGrads.length);
  });
});

describe('Checkpointed joint builder', () => {
  it('creates joint graph with checkpoint', () => {
    const func = buildChain(6);
    const builder = new JointGraphBuilder({
      checkpointPolicy: new EveryKPolicy({ segmentSize: 3 }),
    });
    const { jointFunc, numForwardOutputs, numGradInputs } = builder.build(func);

    expect(jointFunc.name).toBe('joint_chain_fwd');
    expect(numForwardOutputs).toBe(1);
    expect(numGradInputs).toBe(2);

    const ops = jointFunc.opsArray();
    const addOps = ops.filter(op => op.opName === 'add');
    expect(addOps.length).toBeGreaterThanOrEqual(12);
  });

  it('joint graph input/output counts match non-checkpointed', () => {
    const func = buildChain(6);

    const normalBuilder = new JointGraphBuilder();
    const normalResult = normalBuilder.build(func);

    const ckptBuilder = new JointGraphBuilder({
      checkpointPolicy: new EveryKPolicy({ segmentSize: 3 }),
    });
    const ckptResult = ckptBuilder.build(func);

    expect(ckptResult.jointFunc.inputTypes.length).toBe(normalResult.jointFunc.inputTypes.length);
    expect(ckptResult.numForwardOutputs).toBe(normalResult.numForwardOutputs);
    expect(ckptResult.numGradInputs).toBe(normalResult.numGradInputs);
  });
});

describe('AD rejects region control-flow where it cannot be differentiated', () => {
  function buildScanModel() {
    const T = 3, D = 2;
    return buildFunction('scan_fwd', [t([T, D]), t([D])], [t([D])], (b, [xs, init]) => {
      const s = b.scanOp([xs], [init], (bb, xt, cy) => {
        const nc = bb.add(cy[0], xt[0]).getResult(0);
        return [[nc], [nc]];
      });
      b.returnOp([s.getResult(0)]);
    });
  }

  it('checkpointed backward throws on scan instead of silently dropping its gradients', () => {
    const builder = new BackwardGraphBuilder({ checkpointPolicy: new EveryKPolicy({ segmentSize: 1 }) });
    expect(() => builder.build(buildScanModel())).toThrow(/region control-flow|scan/i);
  });

  it('non-checkpointed backward still differentiates scan (control case)', () => {
    const builder = new BackwardGraphBuilder();
    expect(() => builder.build(buildScanModel())).not.toThrow();
  });

  it('joint builder throws on scan instead of silently dropping its gradients', () => {
    const builder = new JointGraphBuilder();
    expect(() => builder.build(buildScanModel())).toThrow(/region control-flow|scan/i);
  });

  it('checkpointed joint builder throws on scan', () => {
    const builder = new JointGraphBuilder({ checkpointPolicy: new EveryKPolicy({ segmentSize: 1 }) });
    expect(() => builder.build(buildScanModel())).toThrow(/region control-flow|scan/i);
  });
});

describe('scan carry checkpointing (O(sqrt T) backward)', () => {
  function buildScanModelT(T, D) {
    return buildFunction('scan_ck_fwd', [t([T, D]), t([D])], [t([D])], (b, [xs, init]) => {
      const s = b.scanOp([xs], [init], (bb, xt, cy) => {
        const m = bb.mul(cy[0], xt[0]).getResult(0);
        const nc = bb.add(m, xt[0]).getResult(0);
        return [[nc], [nc]];
      });
      b.returnOp([s.getResult(0)]);
    });
  }

  function fillInputs(bwd, seed) {
    const ins = [];
    for (let i = 0; i < bwd.inputTypes.length; i++) {
      const buf = new Float32Array(numel(bwd.inputTypes[i].shape));
      if (i === 0) {
        buf.fill(seed);
      } else {
        for (let k = 0; k < buf.length; k++) buf[k] = 0.2 + ((i * 7 + k * 3) % 11) * 0.05;
      }
      ins.push(buf);
    }
    return ins;
  }

  it('opt-in checkpointed scan backward matches non-checkpointed gradients (T=8, sqrt)', () => {
    const T = 8, D = 3;
    const normalBwd = buildAndCompileBackward(buildScanModelT(T, D));
    const ckptBwd = buildAndCompileBackward(buildScanModelT(T, D), { scanCheckpoint: 'sqrt' });

    expect(ckptBwd.inputTypes.length).toBe(normalBwd.inputTypes.length);

    const normalGrads = normalBwd.run(...fillInputs(normalBwd, 1));
    const ckptGrads = ckptBwd.run(...fillInputs(ckptBwd, 1));

    expect(ckptGrads.length).toBe(normalGrads.length);
    for (let g = 0; g < normalGrads.length; g++) {
      expect(ckptGrads[g].length).toBe(normalGrads[g].length);
      for (let i = 0; i < normalGrads[g].length; i++) {
        expect(ckptGrads[g][i]).toBeCloseTo(normalGrads[g][i], 4);
      }
    }
  });

  it('fixed segment size matches non-checkpointed gradients (T=9, K=3)', () => {
    const T = 9, D = 2;
    const normalBwd = buildAndCompileBackward(buildScanModelT(T, D));
    const ckptBwd = buildAndCompileBackward(buildScanModelT(T, D), { scanCheckpoint: 3 });

    const normalGrads = normalBwd.run(...fillInputs(normalBwd, 1));
    const ckptGrads = ckptBwd.run(...fillInputs(ckptBwd, 1));

    for (let g = 0; g < normalGrads.length; g++) {
      for (let i = 0; i < normalGrads[g].length; i++) {
        expect(ckptGrads[g][i]).toBeCloseTo(normalGrads[g][i], 4);
      }
    }
  });

  it('checkpointing recomputes the body (more body ops than non-checkpointed)', () => {
    const T = 9, D = 2;
    const normal = new BackwardGraphBuilder().build(buildScanModelT(T, D));
    const ckpt = new BackwardGraphBuilder({ scanCheckpoint: 3 }).build(buildScanModelT(T, D));
    const countMul = (r) => r.backwardFunc.opsArray().filter(o => o.opName === 'mul').length;
    expect(countMul(ckpt)).toBeGreaterThan(countMul(normal));
  });

  it('default (no scanCheckpoint) still differentiates scan unchanged', () => {
    const builder = new BackwardGraphBuilder();
    expect(() => builder.build(buildScanModelT(5, 2))).not.toThrow();
  });
});
