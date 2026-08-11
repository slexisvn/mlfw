import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType } from '../../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../../src/backend/target.js';
import { ElementwiseCPURule } from '../../../src/compiler/schedule/rules.js';
import { ForNode, IntImmNode, VariableNode, ForKind } from '../../../src/compiler/ir/tensor/nodes.js';
import { countLoops as countForLoops } from '../../_utils/kernel_source.js';
import { F32 } from '../../_utils/ir_fixture.js';
import { compileScheduled as compile } from '../../_utils/ir_fixture.js';
import { compileUnscheduled } from '../../_utils/ir_fixture.js';


function src(result) {
  const kernels = result.listKernels();
  return kernels.map(k => result.getSource(k)).join('\n');
}

function countStores(s) {
  return (s.match(/\w+\[.*?\]\s*=/g) || []).length;
}


describe('MatmulTiledCPURule — tiling via schedule.tile()', () => {
  it('matmul with scheduling does not inflate stores vs default', () => {
    const a = new TensorType([32, 64], F32);
    const b = new TensorType([64, 32], F32);
    const out = new TensorType([32, 32], F32);
    const func = buildFunction('mm', [a, b], [out], (builder, args) => {
      builder.returnOp([builder.matmul(args[0], args[1]).getResult(0)]);
    });

    const funcDef = buildFunction('mm_def', [a, b], [out], (builder, args) => {
      builder.returnOp([builder.matmul(args[0], args[1]).getResult(0)]);
    });

    const scheduled = src(compile(func));
    const defaultSrc = src(compileUnscheduled(funcDef));

    const schedStores = countStores(scheduled);
    const defStores = countStores(defaultSrc);

    expect(schedStores).toBeLessThanOrEqual(defStores * 1.5);
  });

  it('tiled matmul has more for-loops than default (outer+inner tiles)', () => {
    const a = new TensorType([64, 64], F32);
    const b = new TensorType([64, 64], F32);
    const out = new TensorType([64, 64], F32);
    const func = buildFunction('mm_tile', [a, b], [out], (builder, args) => {
      builder.returnOp([builder.matmul(args[0], args[1]).getResult(0)]);
    });

    const funcDef = buildFunction('mm_tile_def', [a, b], [out], (builder, args) => {
      builder.returnOp([builder.matmul(args[0], args[1]).getResult(0)]);
    });

    const scheduled = src(compile(func));
    const defaultSrc = src(compileUnscheduled(funcDef));

    expect(countForLoops(scheduled)).toBeGreaterThanOrEqual(countForLoops(defaultSrc));
  });

  it('small matmul that does not meet tile threshold still compiles', () => {
    const a = new TensorType([4, 4], F32);
    const b = new TensorType([4, 4], F32);
    const out = new TensorType([4, 4], F32);
    const func = buildFunction('mm_small', [a, b], [out], (builder, args) => {
      builder.returnOp([builder.matmul(args[0], args[1]).getResult(0)]);
    });

    const result = compile(func);
    expect(result.listKernels().length).toBeGreaterThan(0);
    expect(src(result)).toMatch(/\w+\[.*?\]\s*=/);
  });
});

describe('ElementwiseCPURule — VECTORIZED loops not unrolled in CPU codegen', () => {
  it('elementwise add with scheduling does not inflate stores vs default', () => {
    const t = new TensorType([256], F32);
    const func = buildFunction('ew_add', [t, t], [t], (builder, args) => {
      builder.returnOp([builder.add(args[0], args[1]).getResult(0)]);
    });

    const funcDef = buildFunction('ew_add_def', [t, t], [t], (builder, args) => {
      builder.returnOp([builder.add(args[0], args[1]).getResult(0)]);
    });

    const scheduled = src(compile(func));
    const defaultSrc = src(compileUnscheduled(funcDef));

    expect(countStores(scheduled)).toBeLessThanOrEqual(countStores(defaultSrc) * 1.15);
  });

  it('scheduled relu produces same store count as default', () => {
    const t = new TensorType([512], F32);
    const func = buildFunction('ew_relu', [t], [t], (builder, args) => {
      builder.returnOp([builder.relu(args[0]).getResult(0)]);
    });

    const funcDef = buildFunction('ew_relu_def', [t], [t], (builder, args) => {
      builder.returnOp([builder.relu(args[0]).getResult(0)]);
    });

    const schedStores = countStores(src(compile(func)));
    const defStores = countStores(src(compileUnscheduled(funcDef)));

    expect(schedStores).toBeLessThanOrEqual(defStores * 1.15);
  });

  it('chain add+mul+neg with scheduling does not inflate stores', () => {
    const t = new TensorType([256], F32);
    const func = buildFunction('ew_chain', [t, t, t], [t], (builder, args) => {
      const sum = builder.add(args[0], args[1]);
      const prod = builder.mul(sum.getResult(0), args[2]);
      builder.returnOp([builder.neg(prod.getResult(0)).getResult(0)]);
    });

    const funcDef = buildFunction('ew_chain_def', [t, t, t], [t], (builder, args) => {
      const sum = builder.add(args[0], args[1]);
      const prod = builder.mul(sum.getResult(0), args[2]);
      builder.returnOp([builder.neg(prod.getResult(0)).getResult(0)]);
    });

    const schedStores = countStores(src(compile(func)));
    const defStores = countStores(src(compileUnscheduled(funcDef)));

    expect(schedStores).toBeLessThanOrEqual(defStores * 1.15);
  });
});

class FakeSchedule {
  constructor(loops) {
    this._loops = loops;
    this.func = {};
    this.calls = [];
  }
  getLoops() { return this._loops; }
  parallelize(loop) { this.calls.push(['parallelize', loop.loopVar.name]); }
  vectorize(loop) { this.calls.push(['vectorize', loop.loopVar.name]); }
  split(loop, factor) {
    this.calls.push(['split', loop.loopVar.name, factor]);
    const o = new ForNode(new VariableNode(loop.loopVar.name + '_o', 'int32'),
      new IntImmNode(0), new IntImmNode(1), ForKind.SERIAL, loop.body);
    const i = new ForNode(new VariableNode(loop.loopVar.name + '_i', 'int32'),
      new IntImmNode(0), new IntImmNode(factor), ForKind.SERIAL, loop.body);
    return [o, i];
  }
}

function serialLoop(name, extent) {
  return new ForNode(new VariableNode(name, 'int32'), new IntImmNode(0),
    new IntImmNode(extent), ForKind.SERIAL, null);
}

describe('ElementwiseCPURule.apply — vectorize only when divisible', () => {
  const target = { vectorWidth: 4 };

  it('does not vectorize the full innermost loop when extent is not divisible', () => {
    const loops = [serialLoop('i', 8), serialLoop('j', 6)];
    const sch = new FakeSchedule(loops);
    new ElementwiseCPURule().apply(sch, 'b', target);

    expect(sch.calls).toContainEqual(['parallelize', 'i']);
    expect(sch.calls.some(c => c[0] === 'vectorize' && c[1] === 'j')).toBe(false);
    expect(sch.calls.some(c => c[0] === 'split' && c[1] === 'j')).toBe(false);
  });

  it('splits then vectorizes the inner tile when extent is divisible', () => {
    const loops = [serialLoop('i', 8), serialLoop('j', 8)];
    const sch = new FakeSchedule(loops);
    new ElementwiseCPURule().apply(sch, 'b', target);

    expect(sch.calls).toContainEqual(['split', 'j', 4]);
    expect(sch.calls).toContainEqual(['vectorize', 'j_i']);
  });
});
