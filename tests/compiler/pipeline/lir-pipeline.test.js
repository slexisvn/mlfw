import { describe, it, expect, afterEach } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { GraphModule } from '../../../src/compiler/ir/graph/module.js';
import { CPUTarget } from '../../../src/compiler/support/target.js';
import { Compiler, CompilerConfig } from '../../../src/compiler/pipeline/compiler.js';
import { lowerGraphToPrimFunc } from '../../../src/compiler/passes/lowering/graph_to_tensor.js';
import { lowerToLIR } from '../../../src/compiler/passes/lowering/tensor_to_lir.js';
import { buildLirPipeline } from '../../../src/compiler/pipeline/lir_pipeline.js';
import { registerLirPass, clearLirPasses, snapshotLirPasses } from '../../../src/compiler/pipeline/lir_pass_registry.js';
import { LirFuncPass } from '../../../src/compiler/passes/lir_pass.js';
import { LirPassManager } from '../../../src/compiler/passes/lir_pass_manager.js';
import { FlatIndexSimplifyPass } from '../../../src/compiler/passes/simplify/flat_index_simplify.js';
import { TraceLog } from '../../../src/compiler/support/trace.js';
import { collect } from '../../../src/compiler/ir/ir_visitor.js';
import {
  LIRFunc, LIRFlatStoreNode, LIRMetadata,
} from '../../../src/compiler/ir/lir/nodes.js';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import { IntImmNode, FloatImmNode, MathOpNode, VariableNode } from '../../../src/compiler/ir/tensor/nodes.js';

const f32 = (shape) => new TensorType(shape, ScalarType.F32);
const runCtx = () => ({ trace: new TraceLog(), errors: [], failed: new Set(), resilient: false });

function lirOf(build, argTypes, retTypes) {
  const target = CPUTarget();
  const func = buildFunction('f', argTypes, retTypes, build);
  return lowerToLIR(lowerGraphToPrimFunc(func, target, null), target);
}

function foldableOffsetFunc() {
  const b = new Buffer('x', [16], 'float32', 'global');
  const offset = new MathOpNode('+', new MathOpNode('*', new IntImmNode(2), new IntImmNode(3)), new IntImmNode(1));
  const body = new LIRFlatStoreNode(b, offset, new FloatImmNode(1), 'float32');
  return new LIRFunc('k', [], body, new Map([[new VariableNode('x', 'handle'), b]]), [], new Map(), new LIRMetadata());
}

afterEach(() => clearLirPasses());

describe('LIR pass pipeline', () => {
  it('exists and carries at least one pass', () => {
    const passes = buildLirPipeline(new CompilerConfig({ target: CPUTarget() }));
    expect(passes.length).toBeGreaterThan(0);
    expect(passes.some(p => p instanceof FlatIndexSimplifyPass)).toBe(true);
  });

  it('lets a pass register into a phase without touching the pipeline builder', () => {
    const before = buildLirPipeline(new CompilerConfig({ target: CPUTarget() })).length;
    class Marker extends LirFuncPass {
      constructor() { super('Marker'); }
      run(func) { return func; }
    }
    registerLirPass(() => new Marker(), { phase: 'pre' });
    const passes = buildLirPipeline(new CompilerConfig({ target: CPUTarget() }));
    expect(passes.length).toBe(before + 1);
    expect(passes[0]).toBeInstanceOf(Marker);
    expect(snapshotLirPasses().length).toBe(1);
  });

  it('drops a factory that opts out for this config', () => {
    const before = buildLirPipeline(new CompilerConfig({ target: CPUTarget() })).length;
    registerLirPass((config) => (config.optimization.fastMath ? new FlatIndexSimplifyPass() : null));
    expect(buildLirPipeline(new CompilerConfig({ target: CPUTarget() })).length).toBe(before);
    expect(buildLirPipeline(new CompilerConfig({ target: CPUTarget(), optimization: { fastMath: true } })).length).toBe(before + 1);
  });
});

describe('LirPassManager', () => {
  it('runs each pass over every function and adopts a replacement', () => {
    const seen = [];
    class Rename extends LirFuncPass {
      constructor() { super('Rename'); }
      run(func) { seen.push(func.name); func.name = func.name + '_done'; return func; }
    }
    const funcs = [foldableOffsetFunc(), foldableOffsetFunc()];
    funcs[1].name = 'k2';
    const pm = new LirPassManager();
    pm.addPass(new Rename());
    pm.run(funcs, runCtx());
    expect(seen).toEqual(['k', 'k2']);
    expect(funcs.map(f => f.name)).toEqual(['k_done', 'k2_done']);
  });

  it('records a failing function and keeps going in resilient mode', () => {
    class Boom extends LirFuncPass {
      constructor() { super('Boom'); }
      run(func) { if (func.name === 'k') throw new Error('kaboom'); return func; }
    }
    const funcs = [foldableOffsetFunc(), foldableOffsetFunc()];
    funcs[1].name = 'k2';
    const ctx = { ...runCtx(), resilient: true };
    const pm = new LirPassManager();
    pm.addPass(new Boom());
    pm.run(funcs, ctx);
    expect(ctx.errors.map(e => e.message)).toEqual(['kaboom']);
    expect([...ctx.failed]).toEqual(['k']);
  });

  it('skips functions already marked failed', () => {
    const seen = [];
    class Note extends LirFuncPass {
      constructor() { super('Note'); }
      run(func) { seen.push(func.name); return func; }
    }
    const funcs = [foldableOffsetFunc()];
    const ctx = { ...runCtx(), failed: new Set(['k']) };
    const pm = new LirPassManager();
    pm.addPass(new Note());
    pm.run(funcs, ctx);
    expect(seen).toEqual([]);
  });
});

describe('FlatIndexSimplifyPass', () => {
  it('folds arithmetic in a flattened offset that no earlier phase could reach', () => {
    const func = foldableOffsetFunc();
    expect(collect(func.body, (n) => n.type === 'MathOpNode').length).toBe(2);

    new FlatIndexSimplifyPass().run(func, { trace: new TraceLog() });

    expect(collect(func.body, (n) => n.type === 'MathOpNode').length).toBe(0);
    expect(func.body.offsetExpr.type).toBe('IntImmNode');
    expect(func.body.offsetExpr.value).toBe(7);
  });

  it('leaves a genuinely dynamic offset alone', () => {
    const b = new Buffer('x', [16], 'float32', 'global');
    const offset = new MathOpNode('+', new VariableNode('i', 'int32'), new IntImmNode(1));
    const func = new LIRFunc('k', [], new LIRFlatStoreNode(b, offset, new FloatImmNode(1), 'float32'),
      new Map([[new VariableNode('x', 'handle'), b]]), [], new Map(), new LIRMetadata());

    new FlatIndexSimplifyPass().run(func, { trace: new TraceLog() });
    expect(func.body.offsetExpr.type).toBe('MathOpNode');
  });

  it('keeps a real compile correct end to end', () => {
    const t = f32([4]);
    const mod = new GraphModule('m');
    mod.addFunction(buildFunction('f', [t, t], [t], (b, args) => {
      b.returnOp([b.mul(b.add(args[0], args[1]).getResult(0), args[1]).getResult(0)]);
    }));
    const result = new Compiler({ target: CPUTarget() }).compile(mod);
    const out = new Float32Array(4);
    result.run('f', new Float32Array([1, 2, 3, 4]), new Float32Array([10, 20, 30, 40]), out);
    expect([...out]).toEqual([110, 440, 990, 1760]);
  });
});

describe('lowered LIR still verifies after the pass pipeline', () => {
  it('produces a well-formed LIR function for a fused elementwise graph', () => {
    const t = f32([8]);
    const lir = lirOf((b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    }, [t, t], [t]);
    new FlatIndexSimplifyPass().run(lir, { trace: new TraceLog() });
    expect(lir.type).toBe('LIRFunc');
    expect(collect(lir.body, (n) => n.type === 'LIRFlatStoreNode').length).toBeGreaterThan(0);
  });
});
