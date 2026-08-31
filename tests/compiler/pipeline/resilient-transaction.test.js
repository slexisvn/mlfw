import { describe, it, expect, afterEach } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { GraphModule } from '../../../src/compiler/ir/graph/module.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { Compiler } from '../../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../../src/compiler/support/target.js';
import { registerGraphPass, clearGraphPasses } from '../../../src/compiler/pipeline/graph_pass_registry.js';
import { FunctionPass, ModulePass, PassResult } from '../../../src/compiler/passes/pass.js';
import { verifyFunction } from '../../../src/compiler/ir/graph/verifier.js';

const EVIL_ATTR = '__evil_partial_mutation__';

class EvilPass extends FunctionPass {
  constructor() { super('EvilPass'); }
  run(func) {
    if (func.name !== 'evil') return PassResult.UNCHANGED;
    const op = func.opsArray().find((o) => o.opName !== 'return');
    if (op) op.setAttr(EVIL_ATTR, true);
    throw new Error('boom mid-rewrite');
  }
}

class EvilModulePass extends ModulePass {
  constructor() { super('EvilModulePass'); }
  run(module) {
    module.removeFunction('good');
    const evil = module.getFunction('evil');
    if (evil) {
      const op = evil.opsArray().find((o) => o.opName !== 'return');
      if (op) op.setAttr(EVIL_ATTR, true);
    }
    throw new Error('boom mid-module-rewrite');
  }
}

function f32(shape) { return new TensorType(shape, ScalarType.F32); }

function makeModule() {
  const t = f32([4]);
  const evil = buildFunction('evil', [t, t], [t], (b, args) => {
    b.returnOp([b.add(args[0], args[1]).getResult(0)]);
  });
  const good = buildFunction('good', [t, t], [t], (b, args) => {
    b.returnOp([b.mul(args[0], args[1]).getResult(0)]);
  });
  const mod = new GraphModule('m');
  mod.addFunction(evil);
  mod.addFunction(good);
  return { mod, evil, good };
}

function hasEvilAttr(func) {
  return func.opsArray().some((o) => o.hasAttr(EVIL_ATTR));
}

describe('resilient mode is transactional', () => {
  afterEach(() => clearGraphPasses());

  it('does not corrupt the caller IR when a pass throws mid-rewrite (resilient)', () => {
    registerGraphPass(() => new EvilPass(), { phase: 'pre' });
    const { mod, evil, good } = makeModule();

    const compiler = new Compiler({ target: CPUTarget(), errorMode: 'resilient' });
    const result = compiler.compile(mod);

    expect(hasEvilAttr(evil)).toBe(false);
    expect(verifyFunction(evil)).toHaveLength(0);
    expect(verifyFunction(good)).toHaveLength(0);

    expect(result.errors.some((e) => e.funcName === 'evil')).toBe(true);
    expect(result.errors.some((e) => e.funcName === 'good')).toBe(false);

    const out = new Float32Array(4);
    result.run('good', new Float32Array([1, 2, 3, 4]), new Float32Array([2, 2, 2, 2]), out);
    expect([...out]).toEqual([2, 4, 6, 8]);
  });

  it('rolls the working module back when a module pass mutates and then throws', () => {
    registerGraphPass(() => new EvilModulePass(), { phase: 'pre' });
    const { mod, evil, good } = makeModule();

    const compiler = new Compiler({ target: CPUTarget(), errorMode: 'resilient' });
    const result = compiler.compile(mod);

    expect(hasEvilAttr(evil)).toBe(false);
    expect(mod.hasFunction('good')).toBe(true);
    expect(result.errors.length).toBeGreaterThan(0);

    const out = new Float32Array(4);
    result.run('good', new Float32Array([1, 2, 3, 4]), new Float32Array([2, 2, 2, 2]), out);
    expect([...out]).toEqual([2, 4, 6, 8]);
  });

  it('still throws in strict mode when a pass throws', () => {
    registerGraphPass(() => new EvilPass(), { phase: 'pre' });
    const { mod } = makeModule();

    const compiler = new Compiler({ target: CPUTarget(), errorMode: 'strict' });
    expect(() => compiler.compile(mod)).toThrow();
  });

  it('a resilient compile with no failures produces working kernels', () => {
    const { mod } = makeModule();
    const result = new Compiler({ target: CPUTarget(), errorMode: 'resilient' }).compile(mod);
    expect(result.errors.length).toBe(0);

    const evilOut = new Float32Array(4);
    result.run('evil', new Float32Array([1, 2, 3, 4]), new Float32Array([1, 1, 1, 1]), evilOut);
    expect([...evilOut]).toEqual([2, 3, 4, 5]);

    const goodOut = new Float32Array(4);
    result.run('good', new Float32Array([1, 2, 3, 4]), new Float32Array([2, 2, 2, 2]), goodOut);
    expect([...goodOut]).toEqual([2, 4, 6, 8]);
  });
});
