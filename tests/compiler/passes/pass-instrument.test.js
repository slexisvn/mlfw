import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { GraphModule } from '../../../src/compiler/ir/graph/module.js';
import { Compiler } from '../../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../../src/backend/target.js';
import { PassResult } from '../../../src/compiler/passes/pass.js';
import { IRLevel } from '../../../src/compiler/ir/verify.js';

function moduleWithDeadOp() {
  const t = new TensorType([8], ScalarType.F32);
  const func = buildFunction('instrumented', [t, t], [t], (b, args) => {
    const [a, c] = args;
    b.sub(a, c);
    b.mul(a, a);
    const scaled = b.mul(a, c).getResult(0);
    const shifted = b.add(scaled, a).getResult(0);
    b.returnOp([b.tanh(shifted).getResult(0)]);
  });
  const mod = new GraphModule('instrument_test');
  mod.addFunction(func);
  return mod;
}

function countGraphOps(module) {
  let total = 0;
  for (const func of module) total += func.numOps();
  return total;
}

class Recorder {
  constructor() {
    this.events = [];
    this.open = [];
  }

  runBeforePass(pass, target, level) {
    this.open.push(pass.name);
    this.events.push({
      kind: 'before',
      pass: pass.name,
      level,
      ops: level === IRLevel.GRAPH_MODULE ? countGraphOps(target) : -1,
    });
  }

  runAfterPass(pass, target, level, result) {
    this.events.push({
      kind: 'after',
      pass: pass.name,
      level,
      result,
      ops: level === IRLevel.GRAPH_MODULE ? countGraphOps(target) : -1,
      depth: this.open.length,
    });
    this.open.pop();
  }

  at(level) {
    return this.events.filter(e => e.level === level);
  }

  pairFor(passName) {
    const before = this.events.find(e => e.kind === 'before' && e.pass === passName);
    const after = this.events.find(e => e.kind === 'after' && e.pass === passName);
    return { before, after };
  }
}

function compileInstrumented(opts = {}) {
  const recorder = new Recorder();
  const result = new Compiler({ target: CPUTarget(), instruments: [recorder], ...opts })
    .compile(moduleWithDeadOp());
  return { recorder, result };
}

describe('pass instrumentation', () => {
  it('fires for every pass at all three IR levels', () => {
    const { recorder, result } = compileInstrumented();

    expect(result.succeeded).toBe(true);
    for (const level of [IRLevel.GRAPH_MODULE, IRLevel.TIR, IRLevel.LIR]) {
      expect(recorder.at(level).length).toBeGreaterThan(0);
    }

    const graphPasses = recorder.at(IRLevel.GRAPH_MODULE).map(e => e.pass);
    expect(graphPasses).toContain('dce');
    expect(graphPasses).toContain('canonicalize');

    const tirPasses = new Set(recorder.at(IRLevel.TIR).map(e => e.pass));
    expect(tirPasses.size).toBeGreaterThan(1);
  });

  it('pairs every before with an after, in order', () => {
    const { recorder } = compileInstrumented();

    let depth = 0;
    for (const event of recorder.events) {
      if (event.kind === 'before') {
        depth++;
      } else {
        expect(event.depth).toBe(depth);
        depth--;
      }
      expect(depth).toBeGreaterThanOrEqual(0);
    }
    expect(depth).toBe(0);
    expect(recorder.open).toHaveLength(0);
  });

  it('replays the repeats of a fixed-point group', () => {
    const { recorder } = compileInstrumented();

    const dceRuns = recorder.events.filter(e => e.kind === 'before' && e.pass === 'dce');
    expect(dceRuns.length).toBeGreaterThan(1);
  });

  it('hands over IR that already reflects what the pass did', () => {
    const { recorder } = compileInstrumented();

    const { before, after } = recorder.pairFor('dce');
    expect(after.ops).toBeLessThan(before.ops);
    expect(after.result).toBe(PassResult.CHANGED);

    const unchanged = recorder.events.find(e =>
      e.kind === 'after' && e.level === IRLevel.GRAPH_MODULE && e.result === PassResult.UNCHANGED);
    expect(unchanged).toBeDefined();
  });

  it('reports the level a pass ran at', () => {
    const { recorder } = compileInstrumented();

    const levels = new Set(recorder.events.map(e => e.level));
    expect([...levels].sort()).toEqual([IRLevel.GRAPH_MODULE, IRLevel.LIR, IRLevel.TIR].sort());

    const firstLevels = recorder.events.filter(e => e.kind === 'before').map(e => e.level);
    expect(firstLevels.indexOf(IRLevel.GRAPH_MODULE)).toBeLessThan(firstLevels.indexOf(IRLevel.TIR));
    expect(firstLevels.indexOf(IRLevel.TIR)).toBeLessThan(firstLevels.indexOf(IRLevel.LIR));
  });

  it('accepts an instrument that only implements one hook', () => {
    const seen = [];
    const result = new Compiler({
      target: CPUTarget(),
      instruments: [{ runAfterPass: (pass) => seen.push(pass.name) }],
    }).compile(moduleWithDeadOp());

    expect(result.succeeded).toBe(true);
    expect(seen.length).toBeGreaterThan(0);
  });

  it('compiles identically with no instruments attached', () => {
    const plain = new Compiler({ target: CPUTarget() }).compile(moduleWithDeadOp());
    const { result } = compileInstrumented();

    expect(plain.succeeded).toBe(true);
    expect(result.listKernels()).toEqual(plain.listKernels());
    expect(result.getSource('instrumented')).toBe(plain.getSource('instrumented'));
  });
});
