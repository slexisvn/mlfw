import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { GraphModule } from '../../../src/compiler/ir/graph/module.js';
import { Compiler } from '../../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../../src/backend/target.js';
import { PassContext } from '../../../src/compiler/passes/pass.js';
import { IRLevel } from '../../../src/compiler/ir/verify.js';
import { TraceLevel } from '../../../src/compiler/pipeline/trace.js';

function moduleUnderTest() {
  const t = new TensorType([8], ScalarType.F32);
  const func = buildFunction('skippable', [t, t], [t], (b, args) => {
    const [a, c] = args;
    const scaled = b.mul(a, c).getResult(0);
    const shifted = b.add(scaled, a).getResult(0);
    b.returnOp([b.tanh(shifted).getResult(0)]);
  });
  const mod = new GraphModule('pass_disable_test');
  mod.addFunction(func);
  return mod;
}

class PassLog {
  constructor() {
    this.ran = [];
  }

  runBeforePass(pass, target, level) {
    this.ran.push({ pass: pass.name, level });
  }

  namesAt(level) {
    return this.ran.filter(e => e.level === level).map(e => e.pass);
  }
}

function compile(disabledPasses = []) {
  const log = new PassLog();
  const skipped = [];
  const result = new Compiler({
    target: CPUTarget(),
    instruments: [log],
    passContext: disabledPasses.length > 0 ? new PassContext({ disabledPasses }) : null,
    trace: {
      level: TraceLevel.INFO,
      sink: event => { if (event.type === 'pass_skipped') skipped.push(event); },
    },
  }).compile(moduleUnderTest());
  return { log, result, skipped };
}

const GRAPH_PASS = 'dce';
const TIR_PASS = 'SimplifyPass';
const LIR_PASS = 'FlatIndexSimplifyPass';

describe('disabling a pass', () => {
  it('runs every level of the pipeline when nothing is disabled', () => {
    const { log, result } = compile();

    expect(result.succeeded).toBe(true);
    expect(log.namesAt(IRLevel.GRAPH_MODULE)).toContain(GRAPH_PASS);
    expect(log.namesAt(IRLevel.TIR)).toContain(TIR_PASS);
    expect(log.namesAt(IRLevel.LIR)).toContain(LIR_PASS);
  });

  it.each([
    [GRAPH_PASS, IRLevel.GRAPH_MODULE],
    [TIR_PASS, IRLevel.TIR],
    [LIR_PASS, IRLevel.LIR],
  ])('skips %s and reports it as skipped at its own level', (name, level) => {
    const { log, result, skipped } = compile([name]);

    expect(result.succeeded).toBe(true);
    expect(log.namesAt(level)).not.toContain(name);
    expect(skipped.map(e => e.passName)).toContain(name);
    expect(skipped.find(e => e.passName === name).irLevel).toBe(level);
  });

  it('leaves the rest of the level running', () => {
    const before = compile().log.namesAt(IRLevel.TIR);
    const after = compile([TIR_PASS]).log.namesAt(IRLevel.TIR);

    expect(before).toContain(TIR_PASS);
    expect(after).toEqual(before.filter(name => name !== TIR_PASS));
    expect(after.length).toBeGreaterThan(0);
  });

  it('reports nothing skipped when nothing is disabled', () => {
    expect(compile().skipped).toHaveLength(0);
  });
});
