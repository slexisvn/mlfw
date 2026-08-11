import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { compileGraph } from '../../../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../../../src/backend/target.js';
import { F32 as F, T as t } from '../../../_utils/ir_fixture.js';


const PROGRAMS = [
  {
    name: 'matmul-chain',
    inShapes: [[4, 4], [4, 4]],
    outShape: [4, 4],
    inputs: [Array.from({ length: 16 }, (_, i) => (i % 5) - 2), Array.from({ length: 16 }, (_, i) => ((i * 3) % 7) - 3)],
    build: (b, [x, w]) => {
      const a = b.matmul(x, w).getResult(0);
      const r = b.relu(a).getResult(0);
      return b.matmul(r, w).getResult(0);
    },
  },
  {
    name: 'transpose-relu',
    inShapes: [[3, 4]],
    outShape: [4, 3],
    inputs: [Array.from({ length: 12 }, (_, i) => i - 6)],
    build: (b, [x]) => {
      const a = b.transpose(x, [1, 0]).getResult(0);
      return b.relu(a).getResult(0);
    },
  },
  {
    name: 'reduce-add',
    inShapes: [[2, 4]],
    outShape: [2],
    inputs: [[1, -2, 3, -4, 5, -6, 7, -8]],
    build: (b, [x]) => {
      const s = b.reduce(x, b.scalarConstant(0, F).getResult(0), [1], 'sum').getResult(0);
      return b.relu(s).getResult(0);
    },
  },
  {
    name: 'elementwise-chain',
    inShapes: [[8], [8]],
    outShape: [8],
    inputs: [[1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2, 2, 2, 2, 2, 2]],
    build: (b, [x, y]) => {
      const a = b.add(x, y).getResult(0);
      const m = b.mul(a, x).getResult(0);
      return b.neg(m).getResult(0);
    },
  },
];

function compileRun(prog, opts) {
  const func = buildFunction('f', prog.inShapes.map(t), [t(prog.outShape)], (b, args) => {
    b.returnOp([prog.build(b, args)]);
  });
  const r = compileGraph(func, CPUTarget(), opts);
  const out = new Float32Array(prog.outShape.reduce((a, c) => a * c, 1));
  r.run(r.listKernels()[0], ...prog.inputs.map((a) => new Float32Array(a)), out);
  return { out: [...out], source: r.getSource(r.listKernels()[0]) };
}

describe('CPU pool allocation', () => {
  for (const prog of PROGRAMS) {
    for (const fusion of [true, false]) {
      it(`${prog.name} (fusion=${fusion}): pooled output is identical to per-buffer output`, () => {
        const off = compileRun(prog, { fusion: { enabled: fusion } });
        const on = compileRun(prog, { fusion: { enabled: fusion }, memory: { poolAllocation: true } });
        expect(on.out).toEqual(off.out);
      });
    }
  }

  it('materializes a shared backing pool when there are reusable buffers', () => {
    const prog = PROGRAMS.find((p) => p.name === 'elementwise-chain');
    const on = compileRun(prog, { fusion: { enabled: false }, memory: { poolAllocation: true } });
    expect(on.source).toContain('_mem_pool');
    expect(on.source).toContain('new Float32Array(_mem_pool');
  });
});
