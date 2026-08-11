import { describe, it, expect } from 'vitest';
import { TirPassManager } from '../../../src/compiler/passes/tir_pass_manager.js';
import { PrimFuncPass, TirModulePass } from '../../../src/compiler/passes/tir_pass.js';
import { TirModule } from '../../../src/compiler/ir/tensor/module.js';
import { TraceLog, TraceLevel } from '../../../src/compiler/pipeline/trace.js';
import { PrimFunc, SeqNode, EvaluateNode, VariableNode } from '../../../src/compiler/ir/tensor/nodes.js';

function pf(name) {
  return new PrimFunc(name, [], new SeqNode([]));
}

function mod(...names) {
  const m = new TirModule('test');
  for (const n of names) m.addFunction(pf(n));
  return m;
}

class CorruptingPass extends PrimFuncPass {
  constructor(name, only = null) {
    super(name, name);
    this.only = only;
  }
  run(primFunc) {
    if (this.only === null || primFunc.name === this.only) {
      primFunc.body = new SeqNode([new EvaluateNode(new VariableNode('escaped', 'int32'))]);
    }
    return primFunc;
  }
}

function mkCtx({ resilient = false, sink = null } = {}) {
  return {
    trace: new TraceLog({ level: sink ? TraceLevel.VERBOSE : TraceLevel.SILENT, sink }),
    errors: [],
    failed: new Set(),
    resilient,
  };
}

class RecordPass extends PrimFuncPass {
  constructor(name, log, failOn = null) {
    super(name, name);
    this.log = log;
    this.failOn = failOn;
    this.begins = 0;
    this.ends = 0;
  }
  begin() { this.begins++; }
  end() { this.ends++; }
  run(pf) {
    this.log.push(`${this.name}:${pf.name}`);
    if (pf.name === this.failOn) throw new Error(`boom on ${pf.name}`);
    return pf;
  }
}

describe('TirPassManager', () => {
  it('runs each pass over each func, passes outer / funcs inner', () => {
    const log = [];
    const pm = new TirPassManager();
    pm.addPass(new RecordPass('P1', log));
    pm.addPass(new RecordPass('P2', log));
    pm.run(mod('f1', 'f2'), mkCtx());
    expect(log).toEqual(['P1:f1', 'P1:f2', 'P2:f1', 'P2:f2']);
  });

  it('calls begin/end exactly once per pass', () => {
    const pm = new TirPassManager();
    const p = new RecordPass('P', []);
    pm.addPass(p);
    pm.run(mod('a', 'b'), mkCtx());
    expect(p.begins).toBe(1);
    expect(p.ends).toBe(1);
  });

  it('replaces a func in place when a pass returns a new PrimFunc', () => {
    const replacement = pf('f1');
    class Swap extends PrimFuncPass {
      constructor() { super('Swap', 'Swap'); }
      run() { return replacement; }
    }
    const pm = new TirPassManager();
    pm.addPass(new Swap());
    const module = mod('f1');
    pm.run(module, mkCtx());
    expect(module.getFunction('f1')).toBe(replacement);
  });

  it('resilient: a throwing func is failed and skipped by later passes; siblings continue', () => {
    const log = [];
    const pm = new TirPassManager();
    pm.addPass(new RecordPass('P1', log, 'f1'));
    pm.addPass(new RecordPass('P2', log));
    const ctx = mkCtx({ resilient: true });
    pm.run(mod('f1', 'f2'), ctx);

    expect(ctx.failed.has('f1')).toBe(true);
    expect(ctx.errors).toHaveLength(1);
    expect(ctx.errors[0].funcName).toBe('f1');
    expect(ctx.errors[0].phase).toBe('P1');
    expect(log).toEqual(['P1:f1', 'P1:f2', 'P2:f2']);
  });

  it('strict: a throwing func breaks the current pass loop but later passes still run on survivors', () => {
    const log = [];
    const pm = new TirPassManager();
    pm.addPass(new RecordPass('P1', log, 'f1'));
    pm.addPass(new RecordPass('P2', log));
    const ctx = mkCtx({ resilient: false });
    pm.run(mod('f1', 'f2'), ctx);

    expect(ctx.failed.has('f1')).toBe(true);
    expect(log).toEqual(['P1:f1', 'P2:f2']);
  });

  it('checkEachPass names the pass that produced invalid IR (strict throws)', () => {
    const pm = new TirPassManager();
    pm.addPass(new CorruptingPass('Breaker'));
    pm.setCheckEachPass(true);
    expect(() => pm.run(mod('bad'), mkCtx())).toThrow(/pass 'Breaker' produced invalid IR.*Unbound variable used: escaped/);
  });

  it('checkEachPass records the error and fails only that func (resilient)', () => {
    const pm = new TirPassManager();
    pm.addPass(new CorruptingPass('Breaker', 'bad'));
    pm.setCheckEachPass(true);
    const ctx = mkCtx({ resilient: true });
    pm.run(mod('bad', 'ok'), ctx);
    expect(ctx.failed.has('bad')).toBe(true);
    expect(ctx.failed.has('ok')).toBe(false);
    expect(ctx.errors[0].phase).toBe('verification');
    expect(ctx.errors[0].passName).toBe('Breaker');
  });

  it('checkEachPass off lets invalid TIR through', () => {
    const pm = new TirPassManager();
    pm.addPass(new CorruptingPass('Breaker'));
    const ctx = mkCtx();
    pm.run(mod('bad'), ctx);
    expect(ctx.errors).toEqual([]);
  });

  it('runs a module-level pass once over the whole module, not once per function', () => {
    const seen = [];
    class Rename extends TirModulePass {
      constructor() { super('Rename', 'Rename'); }
      runModule(module) {
        seen.push(module.functionNames().join(','));
        module.addFunction(pf('generated'));
      }
    }
    const pm = new TirPassManager();
    pm.addPass(new Rename());
    const module = mod('f1', 'f2');
    pm.run(module, mkCtx());
    expect(seen).toEqual(['f1,f2']);
    expect(module.functionNames()).toEqual(['f1', 'f2', 'generated']);
  });

  it('emits phase start/end events per pass phase', () => {
    const events = [];
    const pm = new TirPassManager();
    pm.addPass(new RecordPass('Alpha', []));
    pm.addPass(new RecordPass('Beta', []));
    pm.run(mod('f'), mkCtx({ sink: (e) => events.push(e) }));
    const phases = events.filter((e) => e.type === 'phase');
    expect(phases.filter((e) => e.phase === 'Alpha' && e.action === 'start')).toHaveLength(1);
    expect(phases.filter((e) => e.phase === 'Alpha' && e.action === 'end')).toHaveLength(1);
    expect(phases.filter((e) => e.phase === 'Beta' && e.action === 'start')).toHaveLength(1);
  });
});
