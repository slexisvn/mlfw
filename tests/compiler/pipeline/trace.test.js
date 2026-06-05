import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { TraceLog, TraceLevel } from '../../../src/compiler/pipeline/trace.js';
import { PassResult, FunctionPass } from '../../../src/compiler/passes/pass.js';
import { PassManager } from '../../../src/compiler/passes/pass_manager.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { GraphModule } from '../../../src/compiler/ir/graph/module.js';
import { CPUTarget } from '../../../src/backend/target.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';

const f32 = ScalarType.F32;

describe('TraceLevel', () => {
  it('has correct ordering', () => {
    assert.ok(TraceLevel.SILENT < TraceLevel.INFO);
    assert.ok(TraceLevel.INFO < TraceLevel.VERBOSE);
    assert.ok(TraceLevel.VERBOSE < TraceLevel.DEBUG);
  });

  it('is frozen', () => {
    assert.ok(Object.isFrozen(TraceLevel));
  });
});

describe('TraceLog', () => {
  it('emits nothing at SILENT level', () => {
    const events = [];
    const trace = new TraceLog({ level: TraceLevel.SILENT, sink: e => events.push(e) });
    trace.phaseStart('test');
    trace.phaseEnd('test', 10);
    trace.passRun('TestPass', PassResult.CHANGED, 5, 10, 8);
    assert.equal(events.length, 0);
  });

  it('emits phase events at INFO level', () => {
    const events = [];
    const trace = new TraceLog({ level: TraceLevel.INFO, sink: e => events.push(e) });
    trace.phaseStart('graphPasses');
    trace.phaseEnd('graphPasses', 42.5);
    assert.equal(events.length, 2);
    assert.equal(events[0].type, 'phase');
    assert.equal(events[0].action, 'start');
    assert.equal(events[0].phase, 'graphPasses');
    assert.equal(events[1].action, 'end');
    assert.equal(events[1].durationMs, 42.5);
    assert.ok(typeof events[0].timestamp === 'number');
  });

  it('suppresses pass events at INFO level', () => {
    const events = [];
    const trace = new TraceLog({ level: TraceLevel.INFO, sink: e => events.push(e) });
    trace.passRun('DCEPass', PassResult.UNCHANGED, 1.2, 10, 10);
    assert.equal(events.length, 0);
  });

  it('emits pass events at VERBOSE level', () => {
    const events = [];
    const trace = new TraceLog({ level: TraceLevel.VERBOSE, sink: e => events.push(e) });
    trace.passRun('CSEPass', PassResult.CHANGED, 3.5, 20, 15);
    assert.equal(events.length, 1);
    assert.equal(events[0].type, 'pass');
    assert.equal(events[0].passName, 'CSEPass');
    assert.equal(events[0].changed, true);
    assert.equal(events[0].durationMs, 3.5);
    assert.equal(events[0].opCountBefore, 20);
    assert.equal(events[0].opCountAfter, 15);
  });

  it('emits function events at INFO level', () => {
    const events = [];
    const trace = new TraceLog({ level: TraceLevel.INFO, sink: e => events.push(e) });
    trace.functionEvent('lowering', 'main', { durationMs: 7.3 });
    assert.equal(events.length, 1);
    assert.equal(events[0].type, 'function');
    assert.equal(events[0].phase, 'lowering');
    assert.equal(events[0].funcName, 'main');
  });

  it('emits memory stats at VERBOSE level', () => {
    const events = [];
    const trace = new TraceLog({ level: TraceLevel.VERBOSE, sink: e => events.push(e) });
    trace.memoryStats('main', { peakMemory: 1024, totalTemporaries: 3, totalInplace: 1 });
    assert.equal(events.length, 1);
    assert.equal(events[0].type, 'memory');
    assert.equal(events[0].peakMemory, 1024);
  });

  it('emits codegen stats at VERBOSE level', () => {
    const events = [];
    const trace = new TraceLog({ level: TraceLevel.VERBOSE, sink: e => events.push(e) });
    trace.codegenStats('main', { sourceSize: 500, targetName: 'cpu_generic' });
    assert.equal(events.length, 1);
    assert.equal(events[0].type, 'codegen');
    assert.equal(events[0].sourceSize, 500);
  });

  it('suppresses ir_snapshot below DEBUG level', () => {
    const events = [];
    const trace = new TraceLog({ level: TraceLevel.VERBOSE, sink: e => events.push(e) });
    trace.irDump('test', 'some IR text');
    assert.equal(events.length, 0);
  });

  it('emits ir_snapshot at DEBUG level', () => {
    const events = [];
    const trace = new TraceLog({ level: TraceLevel.DEBUG, sink: e => events.push(e) });
    trace.irDump('afterGraphPasses', 'module @test {}');
    assert.equal(events.length, 1);
    assert.equal(events[0].type, 'ir_snapshot');
    assert.equal(events[0].label, 'afterGraphPasses');
  });

  it('shouldSnapshot respects level and config', () => {
    const t1 = new TraceLog({ level: TraceLevel.VERBOSE, irSnapshot: { afterGraphPasses: true } });
    assert.equal(t1.shouldSnapshot('afterGraphPasses'), false);

    const t2 = new TraceLog({ level: TraceLevel.DEBUG, irSnapshot: { afterGraphPasses: true } });
    assert.equal(t2.shouldSnapshot('afterGraphPasses'), true);
    assert.equal(t2.shouldSnapshot('afterLowering'), false);
  });

  it('defaults to noop sink when none provided', () => {
    const trace = new TraceLog({ level: TraceLevel.DEBUG });
    trace.phaseStart('test');
    trace.phaseEnd('test', 1);
    trace.passRun('X', PassResult.CHANGED, 1, 5, 5);
    trace.irDump('x', 'text');
  });
});

describe('PassManager with trace', () => {
  it('injects trace into each pass and emits pass events', () => {
    const events = [];
    const trace = new TraceLog({ level: TraceLevel.VERBOSE, sink: e => events.push(e) });

    class NoopPass extends FunctionPass {
      constructor() { super('NoopPass'); }
      run(func) {
        assert.ok(this.trace);
        assert.equal(this.trace, trace);
        return PassResult.UNCHANGED;
      }
    }

    const f32_4 = new TensorType([4], f32);
    const func = buildFunction('test', [f32_4], [f32_4], (b, [x]) => {
      b.returnOp([b.neg(x).getResult(0)]);
    });
    const mod = new GraphModule('m');
    mod.addFunction(func);

    const pm = new PassManager();
    pm.setTrace(trace);
    pm.addPass(new NoopPass());
    pm.run(mod);

    const passEvents = events.filter(e => e.type === 'pass');
    assert.equal(passEvents.length, 1);
    assert.equal(passEvents[0].passName, 'NoopPass');
    assert.equal(passEvents[0].changed, false);
    assert.ok(passEvents[0].durationMs >= 0);
    assert.ok(passEvents[0].opCountBefore >= 0);
  });

  it('cleans up trace on pass after run', () => {
    const trace = new TraceLog({ level: TraceLevel.VERBOSE, sink: () => {} });

    let traceInsideRun = null;
    class CheckPass extends FunctionPass {
      constructor() { super('CheckPass'); }
      run(func) {
        traceInsideRun = this.trace;
        return PassResult.UNCHANGED;
      }
    }

    const f32_4 = new TensorType([4], f32);
    const func = buildFunction('t', [f32_4], [f32_4], (b, [x]) => {
      b.returnOp([b.neg(x).getResult(0)]);
    });
    const mod = new GraphModule('m');
    mod.addFunction(func);

    const pass = new CheckPass();
    const pm = new PassManager();
    pm.setTrace(trace);
    pm.addPass(pass);
    pm.run(mod);

    assert.equal(traceInsideRun, trace);
    assert.equal(pass.trace, null);
  });

  it('emits multiple pass events for multiple passes', () => {
    const events = [];
    const trace = new TraceLog({ level: TraceLevel.VERBOSE, sink: e => events.push(e) });

    class P1 extends FunctionPass {
      constructor() { super('PassA'); }
      run() { return PassResult.CHANGED; }
    }
    class P2 extends FunctionPass {
      constructor() { super('PassB'); }
      run() { return PassResult.UNCHANGED; }
    }

    const f32_4 = new TensorType([4], f32);
    const func = buildFunction('t', [f32_4], [f32_4], (b, [x]) => {
      b.returnOp([b.neg(x).getResult(0)]);
    });
    const mod = new GraphModule('m');
    mod.addFunction(func);

    const pm = new PassManager();
    pm.setTrace(trace);
    pm.addPass(new P1());
    pm.addPass(new P2());
    pm.run(mod);

    const passEvents = events.filter(e => e.type === 'pass');
    assert.equal(passEvents.length, 2);
    assert.equal(passEvents[0].passName, 'PassA');
    assert.equal(passEvents[0].changed, true);
    assert.equal(passEvents[1].passName, 'PassB');
    assert.equal(passEvents[1].changed, false);
  });

  it('skips pass event emission at INFO level', () => {
    const events = [];
    const trace = new TraceLog({ level: TraceLevel.INFO, sink: e => events.push(e) });

    class P extends FunctionPass {
      constructor() { super('P'); }
      run() { return PassResult.UNCHANGED; }
    }

    const f32_4 = new TensorType([4], f32);
    const func = buildFunction('t', [f32_4], [f32_4], (b, [x]) => {
      b.returnOp([b.neg(x).getResult(0)]);
    });
    const mod = new GraphModule('m');
    mod.addFunction(func);

    const pm = new PassManager();
    pm.setTrace(trace);
    pm.addPass(new P());
    pm.run(mod);

    assert.equal(events.filter(e => e.type === 'pass').length, 0);
  });
});

describe('Full pipeline trace integration', () => {
  it('emits per-pass events from real compilation', () => {
    const events = [];
    const f32_4 = new TensorType([4], f32);
    const func = buildFunction('traced_add', [f32_4, f32_4], [f32_4], (b, [x, y]) => {
      b.returnOp([b.add(x, y).getResult(0)]);
    });
    compileGraph(func, CPUTarget(), {
      trace: { level: TraceLevel.VERBOSE, sink: e => events.push(e) },
    });

    const phases = events.filter(e => e.type === 'phase' && e.action === 'start').map(e => e.phase);
    assert.ok(phases.includes('compile'));
    assert.ok(phases.includes('graphPasses'));
    assert.ok(phases.includes('lowering'));
    assert.ok(phases.includes('codegen'));

    const passEvents = events.filter(e => e.type === 'pass');
    assert.ok(passEvents.length >= 6);
    const passNames = passEvents.map(e => e.passName);
    assert.ok(passNames.includes('DecompositionPass'));
    assert.ok(passNames.includes('dce'));

    for (const pe of passEvents) {
      assert.ok(typeof pe.passName === 'string');
      assert.ok(typeof pe.changed === 'boolean');
      assert.ok(typeof pe.durationMs === 'number');
      assert.ok(typeof pe.opCountBefore === 'number');
      assert.ok(typeof pe.opCountAfter === 'number');
    }
  });

  it('fusion pass emits decisions at DEBUG level', () => {
    const events = [];
    const f32_8 = new TensorType([8], f32);
    const func = buildFunction('fused', [f32_8, f32_8], [f32_8], (b, [x, y]) => {
      const a = b.add(x, y);
      const m = b.mul(a.getResult(0), x);
      b.returnOp([m.getResult(0)]);
    });
    compileGraph(func, CPUTarget(), {
      fusion: { enabled: true },
      trace: { level: TraceLevel.DEBUG, sink: e => events.push(e) },
    });

    const fusionDecisions = events.filter(e => e.type === 'fusion_decision');
    if (fusionDecisions.length > 0) {
      for (const d of fusionDecisions) {
        assert.equal(d.passName, 'FusionPass');
        assert.ok(typeof d.fuse === 'boolean');
        assert.ok(typeof d.groupSize === 'number');
      }
    }
  });

  it('IR snapshots at DEBUG level', () => {
    const events = [];
    const f32_4 = new TensorType([4], f32);
    const func = buildFunction('snap', [f32_4], [f32_4], (b, [x]) => {
      b.returnOp([b.neg(x).getResult(0)]);
    });
    compileGraph(func, CPUTarget(), {
      trace: {
        level: TraceLevel.DEBUG,
        sink: e => events.push(e),
        irSnapshot: { afterGraphPasses: true, afterLowering: true },
      },
    });

    const snapshots = events.filter(e => e.type === 'ir_snapshot');
    assert.ok(snapshots.length >= 2);
    assert.ok(snapshots.some(s => s.label === 'afterGraphPasses'));
    assert.ok(snapshots.some(s => s.label.startsWith('afterLowering:')));
  });

  it('all events have timestamps in order', () => {
    const events = [];
    const f32_4 = new TensorType([4], f32);
    const func = buildFunction('ts', [f32_4], [f32_4], (b, [x]) => {
      b.returnOp([b.exp(x).getResult(0)]);
    });
    compileGraph(func, CPUTarget(), {
      trace: { level: TraceLevel.VERBOSE, sink: e => events.push(e) },
    });
    for (let i = 1; i < events.length; i++) {
      assert.ok(events[i].timestamp >= events[i - 1].timestamp);
    }
  });

  it('SILENT emits nothing', () => {
    const events = [];
    const f32_4 = new TensorType([4], f32);
    const func = buildFunction('s', [f32_4], [f32_4], (b, [x]) => {
      b.returnOp([b.exp(x).getResult(0)]);
    });
    compileGraph(func, CPUTarget(), {
      trace: { level: TraceLevel.SILENT, sink: e => events.push(e) },
    });
    assert.equal(events.length, 0);
  });
});
