import { ScalarType, TensorType } from '../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../src/compiler/support/target.js';
import { BlockRealizeNode, IterVarKind } from '../../src/compiler/ir/tensor/nodes.js';

export const F32 = ScalarType.F32;
export const F16 = ScalarType.F16;
export const I32 = ScalarType.I32;

export const T = (shape) => new TensorType(shape, F32);

export const typedT = (shape, dtype) => new TensorType(shape, dtype);

export const compileCPU = (func, opts = {}) => compileGraph(func, CPUTarget(), opts);

export const compileFor = (func, target, opts = {}) => compileGraph(func, target, opts);

export const compileScheduled = (func, target = CPUTarget(), opts = {}) =>
  compileGraph(func, target, { scheduling: { enabled: true }, ...opts });

export const compileUnscheduled = (func, target = CPUTarget(), opts = {}) =>
  compileGraph(func, target, { scheduling: { enabled: false }, ...opts });

export function runKernel(result, name, inputs, outputShapes) {
  const inArrays = inputs.map((i) => (i instanceof Float32Array ? i : new Float32Array(i)));
  const outArrays = outputShapes.map((s) => new Float32Array(s.reduce((a, b) => a * b, 1)));
  result.run(name, ...inArrays, ...outArrays);
  return outArrays;
}

export const spatialIter = (v, binding = v) => new BlockRealizeNode(v, binding, IterVarKind.DATA_PAR);

export const reduceIter = (v, binding = v) => new BlockRealizeNode(v, binding, IterVarKind.COMM_REDUCE);
