import { describe, it, expect } from 'vitest';
import { PassManager, FixedPointGroup } from '../../../src/compiler/passes/pass_manager.js';
import { FunctionPass, ModulePass, PassResult } from '../../../src/compiler/passes/pass.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { GraphModule } from '../../../src/compiler/ir/graph/module.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';

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

const UNREGISTERED_OP = '__unregistered_test_op__';

function f32(shape) { return new TensorType(shape, ScalarType.F32); }

function realModule(names) {
  const t = f32([4]);
  const mod = new GraphModule('m');
  for (const name of names) {
    mod.addFunction(buildFunction(name, [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    }));
  }
  return mod;
}

class CorruptingPass extends FunctionPass {
  constructor(name, shouldCorrupt = () => true) {
    super(name);
    this.shouldCorrupt = shouldCorrupt;
    this.seen = [];
  }
  run(func) {
    this.seen.push(func.name);
    if (!this.shouldCorrupt(func)) return PassResult.UNCHANGED;
    func.opsArray().find((o) => o.opName === 'add').opName = UNREGISTERED_OP;
    return PassResult.CHANGED;
  }
}

describe('PassManager attributes invalid IR to the producing pass', () => {
  it('FunctionPass that CHANGED into invalid IR is flagged and pipeline stops (strict)', () => {
    const mod = realModule(['a']);
    const pm = new PassManager();
    pm.setCheckEachPass(true);

    const downstream = new CorruptingPass('downstream', () => false);
    pm.addPass(new CorruptingPass('breaker'));
    pm.addPass(downstream);

    const out = pm.run(mod);

    expect(out.errors[0].passName).toBe('breaker');
    expect(out.errors[0].message).toContain('produced invalid IR');
    expect(out.errors[0].message).toContain(UNREGISTERED_OP);
    expect(out.failedFunctions.has('a')).toBe(true);
    expect(downstream.seen).toEqual([]);
  });

  it('a CHANGED pass that keeps IR valid passes through untouched', () => {
    const mod = realModule(['a']);
    const pm = new PassManager();
    pm.setCheckEachPass(true);

    const downstream = new CorruptingPass('downstream', () => false);
    pm.addPass(new ScriptedFunctionPass('p1', () => PassResult.CHANGED));
    pm.addPass(downstream);

    const out = pm.run(mod);

    expect(out.errors).toBeNull();
    expect(downstream.seen).toEqual(['a']);
  });

  it('checkEachPass off lets invalid IR through to the next pass', () => {
    const mod = realModule(['a']);
    const pm = new PassManager();

    const downstream = new CorruptingPass('downstream', () => false);
    pm.addPass(new CorruptingPass('breaker'));
    pm.addPass(downstream);

    const out = pm.run(mod);

    expect(out.errors).toBeNull();
    expect(downstream.seen).toEqual(['a']);
  });

  it('resilient mode flags the function but continues other functions', () => {
    const mod = realModule(['a', 'b']);
    const pm = new PassManager();
    pm.setCheckEachPass(true);

    const downstream = new CorruptingPass('downstream', () => false);
    pm.addPass(new CorruptingPass('changer', (f) => f.name === 'a'));
    pm.addPass(downstream);

    const out = pm.run(mod, { errorMode: 'resilient' });

    expect(out.failedFunctions.has('a')).toBe(true);
    expect(out.failedFunctions.has('b')).toBe(false);
    expect(downstream.seen).toEqual(['b']);
  });

  it('ModulePass that CHANGED into invalid IR is attributed to the pass', () => {
    const mod = realModule(['a']);
    const pm = new PassManager();
    pm.setCheckEachPass(true);

    class CorruptingModulePass extends ModulePass {
      constructor() { super('mp'); }
      run(module) {
        module.getFunction('a').opsArray().find((o) => o.opName === 'add').opName = UNREGISTERED_OP;
        return PassResult.CHANGED;
      }
    }

    pm.addPass(new CorruptingModulePass());
    const out = pm.run(mod);

    expect(out.errors[0].passName).toBe('mp');
    expect(out.errors[0].message).toContain(UNREGISTERED_OP);
  });
});

describe('FixedPointGroup iterates passes to convergence', () => {
  it('re-runs a group until no pass reports CHANGED', () => {
    const mod = new FakeModule([new FakeFunc('a')]);
    const pm = new PassManager();
    pm.analysisManager.invalidate = () => {};

    let n = 0;
    const settling = new ScriptedFunctionPass('settle', () => (n++ < 2 ? PassResult.CHANGED : PassResult.UNCHANGED));
    pm.addPass(new FixedPointGroup('canon', [settling], 8));
    const out = pm.run(mod);

    expect(settling.seen.length).toBe(3);
    expect(out.changed).toBe(true);
  });

  it('stops at maxIterations when a pass never settles', () => {
    const mod = new FakeModule([new FakeFunc('a')]);
    const pm = new PassManager();
    pm.analysisManager.invalidate = () => {};

    const churner = new ScriptedFunctionPass('churn', () => PassResult.CHANGED);
    pm.addPass(new FixedPointGroup('canon', [churner], 3));
    pm.run(mod);

    expect(churner.seen.length).toBe(3);
  });

  it('converges regardless of having a one-shot change among stable passes', () => {
    const mod = new FakeModule([new FakeFunc('a')]);
    const pm = new PassManager();
    pm.analysisManager.invalidate = () => {};

    let once = false;
    const a = new ScriptedFunctionPass('a', () => (once ? PassResult.UNCHANGED : (once = true, PassResult.CHANGED)));
    const b = new ScriptedFunctionPass('b', () => PassResult.UNCHANGED);
    pm.addPass(new FixedPointGroup('canon', [a, b], 8));
    pm.run(mod);

    expect(a.seen.length).toBe(2);
    expect(b.seen.length).toBe(2);
  });
});
