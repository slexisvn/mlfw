import { describe, it, expect, beforeEach } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { GraphModule } from '../../../src/compiler/ir/graph/module.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { AnalysisManager } from '../../../src/compiler/analysis/analysis_manager.js';
import { UseDefAnalysis } from '../../../src/compiler/analysis/use_def.js';
import { PostDominanceAnalysis } from '../../../src/compiler/analysis/dominance.js';
import { MemoryEffectAnalysis } from '../../../src/compiler/analysis/memory_effect.js';
import { LivenessAnalysis } from '../../../src/compiler/analysis/liveness.js';
import { PassManager } from '../../../src/compiler/passes/pass_manager.js';
import { FunctionPass, PassResult } from '../../../src/compiler/passes/pass.js';
import { DCEPass } from '../../../src/compiler/passes/simplify/dce.js';
import { PriorityFusionPass } from '../../../src/compiler/passes/fusion/priority_fusion.js';
import { CPUTarget } from '../../../src/compiler/support/target.js';

const t = new TensorType([4], ScalarType.F32);

function graph() {
  return buildFunction('f', [t, t], [t], (b, args) => {
    const s = b.add(args[0], args[1]);
    const m = b.mul(s.getResult(0), args[0]);
    b.returnOp([b.tanh(m.getResult(0)).getResult(0)]);
  });
}

function moduleOf(func) {
  const module = new GraphModule('m');
  module.addFunction(func);
  return module;
}

let computeCount = 0;

class CountedAnalysis {
  static get name() { return 'counted'; }
  static get depKey() { return 'counted'; }
  static get dependencies() { return [UseDefAnalysis]; }
  static compute(func, deps) {
    computeCount++;
    return { ops: deps.useDef.topologicalOrder.length };
  }
}

class ReadingPass extends FunctionPass {
  constructor(name, result = PassResult.UNCHANGED) {
    super(name);
    this.requiredAnalyses = [CountedAnalysis];
    this._result = result;
    this.seen = [];
  }
  run(func, analysisManager) {
    this.seen.push(this.getAnalysis(CountedAnalysis, func, analysisManager));
    return this._result;
  }
}

beforeEach(() => { computeCount = 0; });

describe('analyses are computed once and shared across passes', () => {
  it('two passes that read the same analysis compute it once', () => {
    const pm = new PassManager();
    const a = new ReadingPass('a');
    const b = new ReadingPass('b');
    pm.addPass(a);
    pm.addPass(b);
    pm.run(moduleOf(graph()));

    expect(computeCount).toBe(1);
    expect(a.seen[0]).toBe(b.seen[0]);
  });

  it('a pass that reports CHANGED forces the next pass to recompute', () => {
    const pm = new PassManager();
    pm.addPass(new ReadingPass('mutator', PassResult.CHANGED));
    pm.addPass(new ReadingPass('reader'));
    pm.run(moduleOf(graph()));

    expect(computeCount).toBe(2);
  });

  it('a dependency shared by two analyses is computed once', () => {
    const func = graph();
    const manager = new AnalysisManager();
    const useDef = manager.getAnalysis(UseDefAnalysis, func);
    const pdom = manager.getAnalysis(PostDominanceAnalysis, func);
    const counted = manager.getAnalysis(CountedAnalysis, func);

    expect(counted.ops).toBe(useDef.topologicalOrder.length);
    expect(pdom.idom.size).toBeGreaterThan(0);
    expect(manager.getAnalysis(UseDefAnalysis, func)).toBe(useDef);
  });

  it('a pass reached without a manager still resolves dependencies and caches locally', () => {
    const pass = new ReadingPass('standalone');
    const func = graph();
    const first = pass.getAnalysis(CountedAnalysis, func);
    const second = pass.getAnalysis(CountedAnalysis, func);

    expect(first).toBe(second);
    expect(computeCount).toBe(1);
  });
});

describe('the passes that use analyses declare them', () => {
  it('DCE requires and preserves the memory-effect analysis', () => {
    const dce = new DCEPass();
    expect(dce.requiredAnalyses).toContain(MemoryEffectAnalysis);
    expect(dce.preservedAnalyses.has(MemoryEffectAnalysis)).toBe(true);
  });

  it('preserving an analysis keeps it across a CHANGED pass', () => {
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      b.neg(args[0]);
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const manager = new AnalysisManager();
    const before = manager.getAnalysis(MemoryEffectAnalysis, func);

    const pm = new PassManager();
    pm.analysisManager = manager;
    pm.addPass(new DCEPass());
    const result = pm.run(moduleOf(func));

    expect(result.changed).toBe(true);
    expect(manager.getAnalysis(MemoryEffectAnalysis, func)).toBe(before);
  });

  it('an analysis that was not preserved is dropped by the same CHANGED pass', () => {
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      b.neg(args[0]);
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const manager = new AnalysisManager();
    const before = manager.getAnalysis(UseDefAnalysis, func);

    const pm = new PassManager();
    pm.analysisManager = manager;
    pm.addPass(new DCEPass());
    pm.run(moduleOf(func));

    expect(manager.getAnalysis(UseDefAnalysis, func)).not.toBe(before);
  });

  it('the fusion and remat passes name the analyses they read', () => {
    expect(new PriorityFusionPass({ target: CPUTarget() }).requiredAnalyses).toContain(UseDefAnalysis);
  });

  it('remat reads liveness through the manager rather than recomputing it', () => {
    const manager = new AnalysisManager();
    const func = graph();
    const liveness = manager.getAnalysis(LivenessAnalysis, func);
    expect(manager.getAnalysis(LivenessAnalysis, func)).toBe(liveness);
    expect(liveness.peakPressure).toBeGreaterThan(0);
    expect(liveness.liveAtPeak.size).toBeGreaterThan(0);
  });
});
