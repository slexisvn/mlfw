import { describe, it, expect } from 'vitest';
import { PassContext, FunctionPass } from '../../../src/compiler/passes/pass.js';
import { registerTirPass, clearTirPasses, snapshotTirPasses, tirPassesForPhase } from '../../../src/compiler/pipeline/tir_pass_registry.js';
import { buildTirPipeline } from '../../../src/compiler/pipeline/tir_pipeline.js';

class NamedPass extends FunctionPass {
  constructor(name, optLevel) { super(name); this.optLevel = optLevel; }
  run() {}
}

describe('PassContext require-set', () => {
  it('runs a required pass even when its optLevel exceeds the context optLevel', () => {
    const ctx = new PassContext({ optLevel: 0, requiredPasses: ['Critical'] });
    expect(ctx.shouldRun(new NamedPass('Critical', 5))).toBe(true);
    expect(ctx.shouldRun(new NamedPass('Optional', 5))).toBe(false);
  });

  it('disabled still wins over required', () => {
    const ctx = new PassContext({ optLevel: 9, disabledPasses: ['X'], requiredPasses: ['X'] });
    expect(ctx.shouldRun(new NamedPass('X', 0))).toBe(false);
  });
});

describe('TIR pass registry', () => {
  it('appends registered TIR passes by phase/priority into the pipeline', () => {
    const before = snapshotTirPasses();
    try {
      clearTirPasses();
      let tag = null;
      class Marker { constructor(t) { this.name = 'Marker'; this.t = t; } run() {} }
      registerTirPass(() => new Marker('post'), { phase: 'post' });
      registerTirPass(() => new Marker('pre'), { phase: 'pre' });

      const config = { optimization: { loopPartition: false, detectAccumulators: false }, scheduling: { enabled: false, autotune: false }, target: { isWebGPU: () => false }, memory: {} };
      const passes = buildTirPipeline(config);
      const markers = passes.filter(p => p.name === 'Marker');
      expect(markers.length).toBe(2);
      expect(passes[0].name).toBe('Marker');
      expect(passes[0].t).toBe('pre');
      expect(passes[passes.length - 1].t).toBe('post');
      expect(tirPassesForPhase('pre', config).length).toBe(1);
    } finally {
      clearTirPasses();
      for (const e of before) registerTirPass(e.factory, { phase: e.phase, priority: e.priority });
    }
  });
});
