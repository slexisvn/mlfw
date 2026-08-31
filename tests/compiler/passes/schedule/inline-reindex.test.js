import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { lowerGraphToPrimFunc } from '../../../../src/compiler/passes/lowering/graph_to_tensor.js';
import { InlineReindexPass } from '../../../../src/compiler/passes/schedule/inline_reindex_pass.js';
import { CompilerConfig } from '../../../../src/compiler/pipeline/compiler.js';
import { TraceLog, TraceLevel } from '../../../../src/compiler/support/trace.js';
import { FuncAttr } from '../../../../src/compiler/ir/func_attrs.js';
import { CPUTarget, CUDATarget } from '../../../../src/compiler/support/target.js';
import { walk } from '../../../../src/compiler/ir/ir_visitor.js';

const t = (shape) => new TensorType(shape, ScalarType.F32);

function transposeThenAdd() {
  const src = t([4, 8]);
  const dst = t([8, 4]);
  return lowerGraphToPrimFunc(buildFunction('reindex', [src, dst], [dst], (b, args) => {
    const moved = b.transpose(args[0], [1, 0]).getResult(0);
    b.returnOp([b.add(moved, args[1]).getResult(0)]);
  }));
}

function blockNames(primFunc) {
  const names = [];
  walk(primFunc, (node) => { if (node.type === 'BlockNode') names.push(node.name); });
  return names;
}

function runPass(primFunc, { target = CUDATarget(), scheduling = { enabled: true }, level = TraceLevel.DEBUG } = {}) {
  const events = [];
  const trace = new TraceLog({ level, sink: (event) => events.push(event) });
  new InlineReindexPass(new CompilerConfig({ target, scheduling })).run(primFunc, { trace });
  const explains = events.filter((e) => e.type === 'explain');
  return { primFunc, explains, decisions: explains.map((e) => e.decision) };
}

describe('InlineReindexPass folds a block that only renumbers indices', () => {
  it('inlines the transpose into the consumer that reads it once', () => {
    const before = blockNames(transposeThenAdd());
    const { primFunc, explains } = runPass(transposeThenAdd());
    const after = blockNames(primFunc);

    expect(before.some((name) => name.startsWith('transpose'))).toBe(true);
    expect(after.some((name) => name.startsWith('transpose'))).toBe(false);
    expect(explains.map((e) => e.subject)).toEqual(before.filter((name) => name.startsWith('transpose')));
    expect(explains.every((e) => e.decision === 'inlined')).toBe(true);
  });

  it('says nothing at a trace level that does not ask for explanations', () => {
    const { explains } = runPass(transposeThenAdd(), { level: TraceLevel.VERBOSE });

    expect(explains).toEqual([]);
  });
});

describe('InlineReindexPass says why it left a function alone', () => {
  it('leaves a CPU function alone, where no kernel launch is saved', () => {
    const { primFunc, decisions } = runPass(transposeThenAdd(), { target: CPUTarget() });

    expect(decisions).toEqual(['left-alone']);
    expect(blockNames(primFunc).some((name) => name.startsWith('transpose'))).toBe(true);
  });

  it('leaves the loop nests as lowered when scheduling is switched off', () => {
    const off = { enabled: false, gpuTiling: false, autotune: false };
    const { primFunc, decisions } = runPass(transposeThenAdd(), { scheduling: off });

    expect(decisions).toEqual(['left-alone']);
    expect(blockNames(primFunc).some((name) => name.startsWith('transpose'))).toBe(true);
  });

  it('leaves a function whose body belongs to an external kernel alone', () => {
    const primFunc = transposeThenAdd();
    primFunc.setAttr(FuncAttr.EXTERNAL_CODEGEN, true);

    const { decisions } = runPass(primFunc);

    expect(decisions).toEqual(['left-alone']);
    expect(blockNames(primFunc).some((name) => name.startsWith('transpose'))).toBe(true);
  });

  it('leaves a function with no index-only block alone', () => {
    const shape = t([4, 8]);
    const primFunc = lowerGraphToPrimFunc(buildFunction('mul', [shape, shape], [shape], (b, args) => {
      b.returnOp([b.mul(args[0], args[1]).getResult(0)]);
    }));

    const { decisions } = runPass(primFunc);

    expect(decisions).toEqual(['left-alone']);
  });

  it('gives a reason with every decision it reports', () => {
    const reported = [
      ...runPass(transposeThenAdd()).explains,
      ...runPass(transposeThenAdd(), { target: CPUTarget() }).explains,
    ];

    expect(reported.length).toBeGreaterThan(1);
    expect(reported.every((e) => e.category === 'inline-reindex' && typeof e.reason === 'string' && e.reason !== '')).toBe(true);
  });
});
