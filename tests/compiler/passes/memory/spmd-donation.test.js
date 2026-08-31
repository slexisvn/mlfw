import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { compileGraph } from '../../../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../../../src/compiler/support/target.js';
import { Buffer } from '../../../../src/compiler/ir/tensor/buffer.js';
import {
  PrimFunc, BlockNode, BufferStoreNode, BufferLoadNode, ForNode, VariableNode, IntImmNode, MathOpNode, ForKind
} from '../../../../src/compiler/ir/tensor/nodes.js';
import { BufferLiveness } from '../../../../src/compiler/passes/memory/buffer_liveness.js';
import { InplaceAnalysis } from '../../../../src/compiler/passes/memory/inplace_analysis.js';
import { T as t } from '../../../_utils/ir_fixture.js';


describe('SPMD collectives (mesh-axis model; multi-device runtime maps the axis to GPUs)', () => {
  it('all_reduce sums across the device axis', () => {
    const f = buildFunction('ar', [t([2, 3])], [t([2, 3])], (b, [x]) => {
      b.returnOp([b.allReduce(x).getResult(0)]);
    });
    const c = compileGraph(f, CPUTarget());
    const o = new Float32Array(6);
    c.run(c.listKernels()[0], new Float32Array([0, 1, 2, 10, 11, 12]), o);
    expect([...o]).toEqual([10, 12, 14, 10, 12, 14]);
  });

  it('all_gather concatenates shards onto every device', () => {
    const f = buildFunction('ag', [t([2, 2])], [t([2, 4])], (b, [x]) => {
      b.returnOp([b.allGather(x).getResult(0)]);
    });
    const c = compileGraph(f, CPUTarget());
    const o = new Float32Array(8);
    c.run(c.listKernels()[0], new Float32Array([0, 1, 10, 11]), o);
    expect([...o]).toEqual([0, 1, 10, 11, 0, 1, 10, 11]);
  });

  it('all_gather honors non-default mesh_axis/gather_dim attrs', () => {
    const f = buildFunction('ag_nd', [t([2, 3])], [t([6, 3])], (b, [x]) => {
      b.returnOp([b.allGather(x, { meshAxis: 1, gatherDim: 0 }).getResult(0)]);
    });
    const c = compileGraph(f, CPUTarget());
    const o = new Float32Array(18);
    c.run(c.listKernels()[0], new Float32Array([0, 1, 2, 10, 11, 12]), o);
    expect([...o]).toEqual([0, 0, 0, 10, 10, 10, 1, 1, 1, 11, 11, 11, 2, 2, 2, 12, 12, 12]);
  });
});

describe('buffer donation (in/out aliasing)', () => {
  it('lets an output safely reuse a donated input buffer; refuses param aliasing otherwise', () => {
    const N = 8;
    const IN = new Buffer('IN', [N], 'f32', 'global');
    const OUT = new Buffer('OUT', [N], 'f32', 'global');
    const i = new VariableNode('i', 'int32');
    const blk = new BlockNode('e', [{ iterVar: i, binding: i }], [{ buffer: IN }], [{ buffer: OUT }],
      new BufferStoreNode(OUT, [i], new MathOpNode('*', new BufferLoadNode(IN, [i]), new IntImmNode(2))));
    const func = new PrimFunc('f', [], new ForNode(i, new IntImmNode(0), new IntImmNode(N), ForKind.SERIAL, blk),
      new Map([['IN', IN], ['OUT', OUT]]));
    const liveness = BufferLiveness.analyze(func);

    expect(InplaceAnalysis.analyze(func, liveness)).toHaveLength(0);

    const donated = InplaceAnalysis.analyze(func, liveness, new Set([IN, OUT]));
    expect(donated).toHaveLength(1);
    expect(donated[0].srcBuffer).toBe(IN);
    expect(donated[0].dstBuffer).toBe(OUT);
  });
});
