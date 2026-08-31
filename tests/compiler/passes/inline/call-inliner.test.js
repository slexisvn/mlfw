import { describe, it, expect } from 'vitest';
import { buildFunction, IRBuilder } from '../../../../src/compiler/ir/graph/builder.js';
import { GraphModule } from '../../../../src/compiler/ir/graph/module.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { Operation } from '../../../../src/compiler/ir/graph/operation.js';
import { CallInlinerPass } from '../../../../src/compiler/passes/inline/call_inliner.js';
import { PassResult } from '../../../../src/compiler/passes/pass.js';
import { verifyFunction } from '../../../../src/compiler/ir/graph/verifier.js';
import { Compiler } from '../../../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../../../src/compiler/support/target.js';

const t = (shape) => new TensorType(shape, ScalarType.F32);

function callOp(callee, operands, resultTypes) {
  return new Operation('call', operands, resultTypes, { callee });
}

function moduleWithCall({ callee = 'scaled', calleeResults = [t([4])] } = {}) {
  const module = new GraphModule('m');
  module.addFunction(buildFunction('scaled', [t([4])], [t([4])], (b, a) => {
    b.returnOp([b.mul(a[0], a[0]).getResult(0)]);
  }));
  const main = buildFunction('main', [t([4])], [t([4])], (b, a) => {
    b.returnOp([a[0]]);
  });
  module.addFunction(main);

  const block = main.entryBlock;
  const ret = block.lastOp;
  const call = callOp(callee, [main.args[0]], calleeResults);
  block.insertBefore(call, ret);
  ret.operands[0].removeUse(ret._operandLinks[0]);
  ret.operands[0] = call.getResult(0);
  call.getResult(0).addUse(ret._operandLinks[0]);
  return { module, main, call };
}

describe('call op verification', () => {
  it('accepts a well-formed call to a sibling function', () => {
    const { main } = moduleWithCall();
    expect(verifyFunction(main).map(String)).toEqual([]);
  });

  it('reports an unknown-arity call against the callee signature', () => {
    const { module, main } = moduleWithCall();
    module.addFunction(buildFunction('scaled', [t([4]), t([4])], [t([4])], (b, a) => {
      b.returnOp([b.mul(a[0], a[1]).getResult(0)]);
    }));
    const errors = verifyFunction(main).map(String);
    expect(errors.some((e) => /passes 1 operands but it takes 2/.test(e))).toBe(true);
  });

  it('reports a result-type mismatch against the callee signature', () => {
    const { main } = moduleWithCall({ calleeResults: [t([8])] });
    const errors = verifyFunction(main).map(String);
    expect(errors.some((e) => /result 0 has type/.test(e))).toBe(true);
  });

  it('rejects a directly recursive call', () => {
    const module = new GraphModule('m');
    const self = buildFunction('loopy', [t([4])], [t([4])], (b, a) => { b.returnOp([a[0]]); });
    module.addFunction(self);
    const block = self.entryBlock;
    const ret = block.lastOp;
    block.insertBefore(callOp('loopy', [self.args[0]], [t([4])]), ret);
    expect(verifyFunction(self).map(String).some((e) => /directly recursive/.test(e))).toBe(true);
  });
});

describe('CallInlinerPass splices the callee body into the caller', () => {
  it('replaces the call with the callee body and rewires its results', () => {
    const { module, main, call } = moduleWithCall();
    expect(new CallInlinerPass().run(module)).toBe(PassResult.CHANGED);

    expect(call.parentBlock).toBeNull();
    const names = [...main.entryBlock].map((op) => op.opName);
    expect(names).toEqual(['mul', 'return']);
    expect(verifyFunction(main).map(String)).toEqual([]);
  });

  it('is a no-op on a module without calls', () => {
    const module = new GraphModule('m');
    module.addFunction(buildFunction('f', [t([4])], [t([4])], (b, a) => { b.returnOp([a[0]]); }));
    expect(new CallInlinerPass().run(module)).toBe(PassResult.UNCHANGED);
  });

  it('inlines a chain bottom-up so nested calls disappear too', () => {
    const module = new GraphModule('m');
    module.addFunction(buildFunction('leaf', [t([4])], [t([4])], (b, a) => {
      b.returnOp([b.mul(a[0], a[0]).getResult(0)]);
    }));
    const mid = buildFunction('mid', [t([4])], [t([4])], (b, a) => { b.returnOp([a[0]]); });
    module.addFunction(mid);
    const midRet = mid.entryBlock.lastOp;
    const midCall = callOp('leaf', [mid.args[0]], [t([4])]);
    mid.entryBlock.insertBefore(midCall, midRet);
    midRet.operands[0].removeUse(midRet._operandLinks[0]);
    midRet.operands[0] = midCall.getResult(0);
    midCall.getResult(0).addUse(midRet._operandLinks[0]);

    const top = buildFunction('top', [t([4])], [t([4])], (b, a) => { b.returnOp([a[0]]); });
    module.addFunction(top);
    const topRet = top.entryBlock.lastOp;
    const topCall = callOp('mid', [top.args[0]], [t([4])]);
    top.entryBlock.insertBefore(topCall, topRet);
    topRet.operands[0].removeUse(topRet._operandLinks[0]);
    topRet.operands[0] = topCall.getResult(0);
    topCall.getResult(0).addUse(topRet._operandLinks[0]);

    new CallInlinerPass().run(module);
    expect([...top.entryBlock].map((op) => op.opName)).toEqual(['mul', 'return']);
    expect([...mid.entryBlock].map((op) => op.opName)).toEqual(['mul', 'return']);
  });

  it('refuses a call cycle instead of looping forever', () => {
    const module = new GraphModule('m');
    for (const [name, target] of [['a', 'b'], ['b', 'a']]) {
      const fn = buildFunction(name, [t([4])], [t([4])], (b, args) => { b.returnOp([args[0]]); });
      module.addFunction(fn);
      fn.entryBlock.insertBefore(callOp(target, [fn.args[0]], [t([4])]), fn.entryBlock.lastOp);
    }
    expect(() => new CallInlinerPass().run(module)).toThrow(/recursive call cycle/);
  });

  it('a module whose entry calls a helper compiles and runs like the inlined program', () => {
    const { module } = moduleWithCall();
    const result = new Compiler({ target: CPUTarget() }).compile(module);
    const input = Float32Array.from([1, 2, 3, 4]);
    const out = new Float32Array(4);
    result.run('main', input, out);
    expect(Array.from(out)).toEqual([1, 4, 9, 16]);
  });
});
