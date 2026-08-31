import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { UseDefAnalysis } from '../../../src/compiler/analysis/use_def.js';
import { topoSortOpSet } from '../../../src/compiler/ir/graph/graph_algorithms.js';
import { opHasSideEffects, effectPredecessors } from '../../../src/compiler/ir/graph/op_traits.js';
import { DCEPass } from '../../../src/compiler/passes/simplify/dce.js';
import { PassResult } from '../../../src/compiler/passes/pass.js';

const t = () => new TensorType([4], ScalarType.F32);

function effectfulOrder(ops) {
  const names = [];
  for (const op of ops) {
    if (opHasSideEffects(op)) names.push(op.getAttr('call_target_name') || op.opName);
  }
  return names;
}

function twoIndependentCustomCalls() {
  return buildFunction('f', [t()], [t()], (b, args) => {
    b.customCall('sink_a', [args[0]], [t()]);
    b.customCall('sink_b', [args[0]], [t()]);
    b.returnOp([b.neg(args[0]).getResult(0)]);
  });
}

describe('side-effecting ops keep their program order through every linearization', () => {
  it('chains effectful ops so the topological order cannot float them past each other', () => {
    const func = twoIndependentCustomCalls();
    const chain = effectPredecessors([...func.ops()]);

    const calls = [...func.ops()].filter(op => op.opName === 'custom_call');
    expect(calls.length).toBe(2);
    expect(chain.get(calls[1])).toBe(calls[0]);
    expect(chain.has(calls[0])).toBe(false);
  });

  it('UseDefAnalysis emits the effectful ops in program order', () => {
    const func = twoIndependentCustomCalls();
    const topo = UseDefAnalysis.compute(func).topologicalOrder;
    expect(effectfulOrder(topo)).toEqual(['sink_a', 'sink_b']);
  });

  it('topoSortOpSet emits the effectful ops in program order', () => {
    const func = twoIndependentCustomCalls();
    const ordered = topoSortOpSet([...func.ops()]);
    expect(effectfulOrder(ordered)).toEqual(['sink_a', 'sink_b']);
  });

  it('orders a region op by the effects nested inside it', () => {
    const func = buildFunction('f', [t()], [t()], (b, args) => {
      b.customCall('sink_a', [args[0]], [t()]);
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });
    const call = [...func.ops()].find(op => op.opName === 'custom_call');
    expect(opHasSideEffects(call)).toBe(true);

    const pure = [...func.ops()].find(op => op.opName === 'neg');
    expect(opHasSideEffects(pure)).toBe(false);
  });

  it('leaves pure ops free to reorder', () => {
    const func = buildFunction('f', [t()], [t()], (b, args) => {
      b.neg(args[0]);
      b.returnOp([b.add(args[0], args[0]).getResult(0)]);
    });
    expect(effectPredecessors([...func.ops()]).size).toBe(0);
  });
});

describe('scatter is a value-producing op, not an effect', () => {
  const scatterOpts = {
    updateWindowDims: [1],
    insertedWindowDims: [0],
    scatterDimsToOperandDims: [0],
    indexVectorDim: 1,
  };

  function scatterFunc() {
    const operandT = new TensorType([5, 3], ScalarType.F32);
    const indicesT = new TensorType([2, 1], ScalarType.I32);
    const updatesT = new TensorType([2, 3], ScalarType.F32);
    return buildFunction('f', [operandT, indicesT, updatesT], [operandT], (b, args) => {
      b.scatter(args[0], args[1], args[2], scatterOpts);
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });
  }

  it('does not declare a side effect', () => {
    const func = scatterFunc();
    const scatter = [...func.ops()].find(op => op.opName === 'scatter');
    expect(scatter).toBeDefined();
    expect(opHasSideEffects(scatter)).toBe(false);
  });

  it('lets DCE erase a scatter nobody reads', () => {
    const func = scatterFunc();
    expect(new DCEPass().run(func)).toBe(PassResult.CHANGED);
    expect([...func.ops()].some(op => op.opName === 'scatter')).toBe(false);
  });

  it('keeps a scatter whose result is returned', () => {
    const operandT = new TensorType([5, 3], ScalarType.F32);
    const indicesT = new TensorType([2, 1], ScalarType.I32);
    const updatesT = new TensorType([2, 3], ScalarType.F32);
    const func = buildFunction('f', [operandT, indicesT, updatesT], [operandT], (b, args) => {
      b.returnOp([b.scatter(args[0], args[1], args[2], scatterOpts).getResult(0)]);
    });
    expect(new DCEPass().run(func)).toBe(PassResult.UNCHANGED);
    expect([...func.ops()].some(op => op.opName === 'scatter')).toBe(true);
  });
});
