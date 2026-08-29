import { describe, it, expect, afterEach } from 'vitest';
import { Compiler, CompilerConfig } from '../../../src/compiler/pipeline/compiler.js';
import { VerifyLevel, normalizeVerifyLevel, checkIRInvariants } from '../../../src/compiler/pipeline/invariant_check.js';
import { IRLevel, verifyIR, irLevels, registerIRVerifier, getIRVerifier, unregisterIRVerifier } from '../../../src/compiler/ir/verify.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { GraphModule } from '../../../src/compiler/ir/graph/module.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { CPUTarget } from '../../../src/backend/target.js';
import { FunctionPass, PassResult } from '../../../src/compiler/passes/pass.js';
import { PrimFuncPass } from '../../../src/compiler/passes/tir_pass.js';
import { buildLirPipeline } from '../../../src/compiler/pipeline/lir_pipeline.js';
import { registerGraphPass, clearGraphPasses } from '../../../src/compiler/pipeline/graph_pass_registry.js';
import { registerTirPass, clearTirPasses, snapshotTirPasses } from '../../../src/compiler/pipeline/tir_pass_registry.js';
import { PrimFunc, SeqNode, EvaluateNode, VariableNode } from '../../../src/compiler/ir/tensor/nodes.js';

const UNREGISTERED_OP = '__unregistered_invariant_op__';

function addModule(name = 'f') {
  const t = new TensorType([4], ScalarType.F32);
  const mod = new GraphModule('m');
  mod.addFunction(buildFunction(name, [t, t], [t], (b, args) => {
    b.returnOp([b.add(args[0], args[1]).getResult(0)]);
  }));
  return mod;
}

class GraphCorruptingPass extends FunctionPass {
  constructor() { super('GraphBreaker'); }
  run(func) {
    const op = func.opsArray().find((o) => o.opName === 'add');
    if (!op) return PassResult.UNCHANGED;
    op.opName = UNREGISTERED_OP;
    return PassResult.CHANGED;
  }
}

class TirCorruptingPass extends PrimFuncPass {
  constructor() { super('TirBreaker', 'TirBreaker'); }
  run(primFunc) {
    primFunc.body = new SeqNode([new EvaluateNode(new VariableNode('escaped_tir_var', 'int32'))]);
    return primFunc;
  }
}

describe('IR verifier registry', () => {
  it('covers every IR level the pipeline produces', () => {
    expect(irLevels().sort()).toEqual([IRLevel.GRAPH_FUNC, IRLevel.GRAPH_MODULE, IRLevel.LIR, IRLevel.TIR].sort());
  });

  it('normalizes each verifier to plain message strings', () => {
    const mod = addModule();
    mod.getFunction('f').opsArray().find((o) => o.opName === 'add').opName = UNREGISTERED_OP;
    const messages = verifyIR(IRLevel.GRAPH_MODULE, mod);
    expect(messages.length).toBeGreaterThan(0);
    for (const m of messages) expect(typeof m).toBe('string');
    expect(messages.join('; ')).toContain(UNREGISTERED_OP);
  });

  it('reports no errors for valid IR at every level', () => {
    expect(verifyIR(IRLevel.GRAPH_MODULE, addModule())).toEqual([]);
    expect(verifyIR(IRLevel.TIR, new PrimFunc('p', [], new SeqNode([])))).toEqual([]);
  });

  it('throws on an unregistered IR level rather than silently passing', () => {
    expect(() => verifyIR('not-a-level', {})).toThrow(/No IR verifier registered/);
  });

  it('is extensible with a new level', () => {
    const before = irLevels().length;
    registerIRVerifier('test-level', () => ['synthetic']);
    try {
      expect(verifyIR('test-level', null)).toEqual(['synthetic']);
      expect(irLevels().length).toBe(before + 1);
    } finally {
      unregisterIRVerifier('test-level');
    }
    expect(irLevels().length).toBe(before);
  });
});

describe('checkIRInvariants', () => {
  it('returns null for valid IR', () => {
    expect(checkIRInvariants(IRLevel.GRAPH_MODULE, addModule(), 'm')).toBeNull();
  });

  it('names the producing pass when given one', () => {
    const mod = addModule();
    mod.getFunction('f').opsArray().find((o) => o.opName === 'add').opName = UNREGISTERED_OP;
    const err = checkIRInvariants(IRLevel.GRAPH_MODULE, mod, 'm', 'SomePass');
    expect(err.phase).toBe('verification');
    expect(err.passName).toBe('SomePass');
    expect(err.message).toContain("pass 'SomePass' produced invalid IR");
  });

  it('omits the pass prefix at IR boundaries', () => {
    const mod = addModule();
    mod.getFunction('f').opsArray().find((o) => o.opName === 'add').opName = UNREGISTERED_OP;
    const err = checkIRInvariants(IRLevel.GRAPH_MODULE, mod, 'm');
    expect(err.passName).toBeNull();
    expect(err.message).not.toContain('produced invalid IR');
  });
});

describe('VerifyLevel config', () => {
  it('defaults to each-pass so compiles are checked without opting in', () => {
    const cfg = new CompilerConfig({ target: CPUTarget() });
    expect(cfg.verify).toBe(VerifyLevel.EACH_PASS);
    expect(cfg.verifyEnabled).toBe(true);
    expect(cfg.verifyEachPass).toBe(true);
  });

  it('exposes boundaries as verification without per-pass checking', () => {
    const cfg = new CompilerConfig({ target: CPUTarget(), verify: VerifyLevel.BOUNDARIES });
    expect(cfg.verifyEnabled).toBe(true);
    expect(cfg.verifyEachPass).toBe(false);
  });

  it('disables everything at off', () => {
    const cfg = new CompilerConfig({ target: CPUTarget(), verify: VerifyLevel.OFF });
    expect(cfg.verifyEnabled).toBe(false);
    expect(cfg.verifyEachPass).toBe(false);
  });

  it('rejects an invalid level loudly instead of silently defaulting', () => {
    expect(() => new CompilerConfig({ target: CPUTarget(), verify: false })).toThrow(/Invalid verify level/);
    expect(() => new CompilerConfig({ target: CPUTarget(), verify: 'full' })).toThrow(/Invalid verify level/);
  });
});

describe('Compiler attributes corruption to the pass that caused it', () => {
  afterEach(() => clearGraphPasses());

  it('names the graph pass at each-pass, and only that pass', () => {
    registerGraphPass(() => new GraphCorruptingPass(), { phase: 'pre' });
    const compiler = new Compiler({ target: CPUTarget(), errorMode: 'resilient' });
    const result = compiler.compile(addModule());

    expect(result.errors.length).toBeGreaterThan(0);
    const err = result.errors[0];
    expect(err.passName).toBe('GraphBreaker');
    expect(err.message).toContain("pass 'GraphBreaker' produced invalid IR");
  });

  it('names the graph pass in the thrown error under the default strict mode', () => {
    registerGraphPass(() => new GraphCorruptingPass(), { phase: 'pre' });
    const compiler = new Compiler({ target: CPUTarget() });

    let thrown = null;
    try {
      compiler.compile(addModule());
    } catch (e) {
      thrown = e;
    }

    expect(thrown).toBeTruthy();
    expect(thrown.message).toContain('GraphBreaker');
    expect(thrown.message).toContain("pass 'GraphBreaker' produced invalid IR");
    expect(thrown.message).toContain(UNREGISTERED_OP);
  });

  it('surfaces the same corruption without a pass name at boundaries level', () => {
    registerGraphPass(() => new GraphCorruptingPass(), { phase: 'pre' });
    const compiler = new Compiler({
      target: CPUTarget(), errorMode: 'resilient', verify: VerifyLevel.BOUNDARIES,
    });
    const result = compiler.compile(addModule());

    expect(result.errors.length).toBeGreaterThan(0);
    expect(result.errors[0].passName).toBeNull();
    expect(result.errors[0].message).toContain(UNREGISTERED_OP);
  });

  it('lets the corruption reach lowering when verification is off', () => {
    registerGraphPass(() => new GraphCorruptingPass(), { phase: 'pre' });
    const compiler = new Compiler({
      target: CPUTarget(), errorMode: 'resilient', verify: VerifyLevel.OFF,
    });
    const result = compiler.compile(addModule());

    expect(result.errors.length).toBeGreaterThan(0);
    expect(result.errors.every((e) => e.phase !== 'verification')).toBe(true);
  });
});

describe('TIR corruption is attributed to its pass', () => {
  afterEach(() => clearTirPasses());

  it('names the TIR pass that produced invalid TensorIR', () => {
    const before = snapshotTirPasses();
    clearTirPasses();
    for (const e of before) registerTirPass(e.factory, { phase: e.phase, priority: e.priority });
    registerTirPass(() => new TirCorruptingPass(), { phase: 'post' });

    const compiler = new Compiler({ target: CPUTarget(), errorMode: 'resilient' });
    const result = compiler.compile(addModule());

    const err = result.errors.find((e) => e.passName === 'TirBreaker');
    expect(err).toBeTruthy();
    expect(err.message).toContain('escaped_tir_var');

    clearTirPasses();
    for (const e of before) registerTirPass(e.factory, { phase: e.phase, priority: e.priority });
  });
});

describe('LIR is verified at the boundary, not only under a debug flag', () => {
  const originalLirVerifier = getIRVerifier(IRLevel.LIR);
  afterEach(() => registerIRVerifier(IRLevel.LIR, originalLirVerifier));

  function countingLirVerifier() {
    const calls = { n: 0 };
    registerIRVerifier(IRLevel.LIR, (lirFunc) => { calls.n++; return originalLirVerifier(lirFunc); });
    return calls;
  }

  it('runs the LIR verifier on a plain compile at boundaries level', () => {
    const calls = countingLirVerifier();
    new Compiler({ target: CPUTarget(), verify: VerifyLevel.BOUNDARIES }).compile(addModule());
    expect(calls.n).toBe(1);
  });

  it('runs after every LIR pass as well as at the boundary under each-pass level', () => {
    const calls = countingLirVerifier();
    const lirPassCount = buildLirPipeline(new CompilerConfig({ target: CPUTarget() })).length;
    new Compiler({ target: CPUTarget(), verify: VerifyLevel.EACH_PASS }).compile(addModule());
    expect(calls.n).toBe(lirPassCount + 1);
    expect(lirPassCount).toBeGreaterThan(0);
  });

  it('skips it entirely when verification is off', () => {
    const calls = countingLirVerifier();
    new Compiler({ target: CPUTarget(), verify: VerifyLevel.OFF }).compile(addModule());
    expect(calls.n).toBe(0);
  });

  it('surfaces a real LIR verification failure through the compile result', () => {
    registerIRVerifier(IRLevel.LIR, () => ['synthetic LIR breakage']);
    const compiler = new Compiler({ target: CPUTarget(), errorMode: 'resilient' });
    const result = compiler.compile(addModule());
    expect(result.errors.some((e) => e.message.includes('synthetic LIR breakage'))).toBe(true);
  });
});
