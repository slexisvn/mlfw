import { describe, it, expect } from 'vitest';
import { PassManager } from '../../../src/compiler/passes/pass_manager.js';
import { FunctionPass, ModulePass, PassResult } from '../../../src/compiler/passes/pass.js';

class FakeFunc {
  constructor(name) {
    this.name = name;
    this.version = 0;
  }
  numOps() { return 1; }
  bumpVersion() { this.version++; }
}

class FakeModule {
  constructor(funcs) {
    this.name = 'm';
    this._funcs = funcs;
  }
  [Symbol.iterator]() { return this._funcs[Symbol.iterator](); }
}

class ScriptedFunctionPass extends FunctionPass {
  constructor(name, resultFor) {
    super(name);
    this.resultFor = resultFor;
    this.seen = [];
  }
  run(func) {
    this.seen.push(func.name);
    return this.resultFor(func);
  }
}

class ScriptedModulePass extends ModulePass {
  constructor(name, result) {
    super(name);
    this.result = result;
  }
  run() { return this.result; }
}

describe('PassManager FAILED handling', () => {
  it('non-resilient FunctionPass FAILED invalidates analyses and surfaces error', () => {
    const fA = new FakeFunc('a');
    const mod = new FakeModule([fA]);
    const pm = new PassManager();
    let invalidated = null;
    pm.analysisManager.invalidate = (func) => { invalidated = func; };

    pm.addPass(new ScriptedFunctionPass('p1', () => PassResult.FAILED));
    const out = pm.run(mod);

    expect(invalidated).toBe(fA);
    expect(out.errors).not.toBeNull();
    expect(out.errors[0].passName).toBe('p1');
    expect(out.failedFunctions.has('a')).toBe(true);
  });

  it('resilient FunctionPass FAILED flags the function and skips it downstream', () => {
    const fA = new FakeFunc('a');
    const fB = new FakeFunc('b');
    const mod = new FakeModule([fA, fB]);
    const pm = new PassManager();
    pm.analysisManager.invalidate = () => {};

    const failer = new ScriptedFunctionPass('p1', (f) => f.name === 'a' ? PassResult.FAILED : PassResult.UNCHANGED);
    const downstream = new ScriptedFunctionPass('p2', () => PassResult.UNCHANGED);
    pm.addPass(failer);
    pm.addPass(downstream);

    const out = pm.run(mod, { errorMode: 'resilient' });

    expect(out.failedFunctions.has('a')).toBe(true);
    expect(downstream.seen).toEqual(['b']);
    expect(out.errors[0].funcName).toBe('a');
  });

  it('ModulePass FAILED invalidates all analyses and records an error', () => {
    const mod = new FakeModule([new FakeFunc('a')]);
    const pm = new PassManager();
    let invalidatedAll = false;
    pm.analysisManager.invalidateAll = () => { invalidatedAll = true; };

    pm.addPass(new ScriptedModulePass('mp', PassResult.FAILED));
    const out = pm.run(mod);

    expect(invalidatedAll).toBe(true);
    expect(out.errors).not.toBeNull();
    expect(out.errors[0].passName).toBe('mp');
  });
});
