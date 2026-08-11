import { describe, it, expect } from 'vitest';
import { TirModule } from '../../../../src/compiler/ir/tensor/module.js';
import {
  PrimFunc, SeqNode, EvaluateNode, VariableNode, ForNode, IntImmNode, ForKind,
  BufferStoreNode,
} from '../../../../src/compiler/ir/tensor/nodes.js';
import { Buffer } from '../../../../src/compiler/ir/tensor/buffer.js';

function wellFormed(name) {
  const out = new Buffer('out', [4], 'float32', 'global');
  const i = new VariableNode('i', 'int32');
  const body = new ForNode(i, new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL,
    new BufferStoreNode(out, [i], new IntImmNode(0)));
  return new PrimFunc(name, [], body, new Map([['out', out]]));
}

function malformed(name) {
  return new PrimFunc(name, [], new SeqNode([new EvaluateNode(new VariableNode('escaped', 'int32'))]));
}

describe('TirModule keyed function table', () => {
  it('indexes functions by name and back-references the module', () => {
    const module = new TirModule('m');
    const f = module.addFunction(wellFormed('a'));
    expect(module.getFunction('a')).toBe(f);
    expect(f._module).toBe(module);
    expect(module.hasFunction('a')).toBe(true);
    expect(module.functionNames()).toEqual(['a']);
    expect(module.functionCount).toBe(1);
  });

  it('bumps the version on every structural change', () => {
    const module = new TirModule('m');
    const start = module.version;
    module.addFunction(wellFormed('a'));
    const afterAdd = module.version;
    module.replaceFunction('a', wellFormed('a'));
    const afterReplace = module.version;
    module.removeFunction('a');
    expect(afterAdd).toBeGreaterThan(start);
    expect(afterReplace).toBeGreaterThan(afterAdd);
    expect(module.version).toBeGreaterThan(afterReplace);
    expect(module.removeFunction('a')).toBe(false);
  });

  it('replaceFunction rekeys the table when the replacement is renamed', () => {
    const module = new TirModule('m');
    module.addFunction(wellFormed('a'));
    module.replaceFunction('a', wellFormed('b'));
    expect(module.functionNames()).toEqual(['b']);
    expect(module.getFunction('a')).toBeNull();
  });

  it('refuses to replace a function that is not in the module', () => {
    const module = new TirModule('m');
    expect(() => module.replaceFunction('missing', wellFormed('missing'))).toThrow(/no function 'missing'/);
  });

  it('iterates in insertion order over both the iterator and the generator', () => {
    const module = new TirModule('m');
    for (const n of ['x', 'y', 'z']) module.addFunction(wellFormed(n));
    expect([...module].map((f) => f.name)).toEqual(['x', 'y', 'z']);
    expect([...module.functions()].map((f) => f.name)).toEqual(['x', 'y', 'z']);
  });

  it('verify routes each function through the TIR verifier and prefixes the name', () => {
    const module = new TirModule('m');
    module.addFunction(wellFormed('good'));
    module.addFunction(malformed('bad'));
    const errors = module.verify();
    expect(errors.some((e) => e.startsWith('bad: ') && /escaped/.test(e))).toBe(true);
    expect(errors.some((e) => e.startsWith('good: '))).toBe(false);
  });

  it('verify flags an empty module', () => {
    expect(new TirModule('m').verify()).toEqual(['Module has no functions']);
  });
});
