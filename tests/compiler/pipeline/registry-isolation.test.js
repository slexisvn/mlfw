import { describe, it, expect, afterEach } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { GraphModule } from '../../../src/compiler/ir/graph/module.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { Compiler } from '../../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../../src/backend/target.js';
import { lowerPointwise } from '../../../src/compiler/passes/lowering/lowering_registry.js';
import { MathOpNode, FloatImmNode } from '../../../src/compiler/ir/tensor/nodes.js';
import { registerGraphPass, clearGraphPasses } from '../../../src/compiler/pipeline/graph_pass_registry.js';
import { FunctionPass, PassResult } from '../../../src/compiler/passes/pass.js';

function f32(shape) { return new TensorType(shape, ScalarType.F32); }

function negModule() {
  const t = f32([3]);
  const func = buildFunction('f', [t], [t], (b, [x]) => {
    b.returnOp([b.neg(x).getResult(0)]);
  });
  const mod = new GraphModule('m');
  mod.addFunction(func);
  return mod;
}

const doubleRule = (ctx, op, inputs, outputs) =>
  lowerPointwise(ctx, op, inputs, outputs, (o, loads) => new MathOpNode('*', loads[0], new FloatImmNode(2)));

describe('per-Compiler lowering-rule isolation', () => {
  it('an instance can override a lowering rule without leaking to other instances or the global registry', () => {
    const overridden = new Compiler({ target: CPUTarget(), loweringRules: { neg: doubleRule } });
    const r1 = overridden.compile(negModule());
    const out1 = new Float32Array(3);
    r1.run(r1.listKernels()[0], new Float32Array([1, 2, 3]), out1);
    expect([...out1]).toEqual([2, 4, 6]);

    const def = new Compiler({ target: CPUTarget() });
    const r2 = def.compile(negModule());
    const out2 = new Float32Array(3);
    r2.run(r2.listKernels()[0], new Float32Array([1, 2, 3]), out2);
    expect([...out2]).toEqual([-1, -2, -3]);
  });

  it('the default compiler is unchanged when no overrides are given', () => {
    const def = new Compiler({ target: CPUTarget() });
    const r = def.compile(negModule());
    const out = new Float32Array(3);
    r.run(r.listKernels()[0], new Float32Array([1, 2, 3]), out);
    expect([...out]).toEqual([-1, -2, -3]);
  });
});

describe('per-Compiler graph-pass hook isolation', () => {
  afterEach(() => clearGraphPasses());

  it('each instance snapshots its hooks at construction and is unaffected by later global mutation', () => {
    let aRan = 0;
    let bRan = 0;
    class PassA extends FunctionPass { constructor() { super('PassA'); } run() { aRan++; return PassResult.UNCHANGED; } }
    class PassB extends FunctionPass { constructor() { super('PassB'); } run() { bRan++; return PassResult.UNCHANGED; } }

    registerGraphPass(() => new PassA(), { phase: 'post' });
    const compilerA = new Compiler({ target: CPUTarget() });
    clearGraphPasses();
    registerGraphPass(() => new PassB(), { phase: 'post' });
    const compilerB = new Compiler({ target: CPUTarget() });

    compilerA.compile(negModule());
    expect(aRan).toBeGreaterThan(0);
    expect(bRan).toBe(0);

    compilerB.compile(negModule());
    expect(bRan).toBeGreaterThan(0);
  });
});
