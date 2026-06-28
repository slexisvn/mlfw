import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { SymInt } from '../../../src/compiler/analysis/sym_int.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { CPUTarget, CUDATarget } from '../../../src/backend/target.js';
import { RuntimeTensor } from '../../../src/runtime/runtime.js';
import { cudaDeps } from './cuda-setup.js';

const F32 = ScalarType.F32;
const n = SymInt.var('n');
const m = SymInt.var('m');
const p = SymInt.var('p');

function addFn() {
  const t = new TensorType([n, 4], F32);
  return buildFunction('add_sym', [t, t], [t], (b, [x, y]) =>
    b.returnOp([b.add(x, y).getResult(0)]));
}

function dotFn(K) {
  const lhs = new TensorType([m, K], F32);
  const rhs = new TensorType([K, p], F32);
  const outT = new TensorType([m, p], F32);
  return buildFunction('dot_sym', [lhs, rhs], [outT], (b, [a, c]) =>
    b.returnOp([b.dot(a, c, [1], [0]).getResult(0)]));
}

describe('CUDA symbolic-shape source generation', () => {
  it('emits int shape params used in loop bounds and indices', () => {
    const src = compileGraph(dotFn(8), CUDATarget()).getSource('dot_sym');
    expect(src).toMatch(/__global__ void dot_sym\([^)]*int _sym_m[^)]*int _sym_p/);
    expect(src).toMatch(/< _sym_m/);
    expect(src).toMatch(/< _sym_p/);
    expect(src).toMatch(/\* _sym_p\)/);
  });
});

describe.skipIf(!cudaDeps)('CUDA symbolic-shape execution on real GPU', () => {
  it('elementwise add binds the symbolic dim from input shapes', async () => {
    const res = await compileGraph(addFn(), CUDATarget());
    for (const N of [3, 9]) {
      const xa = Float32Array.from({ length: N * 4 }, (_, i) => Math.sin(i * 0.7 + 1));
      const ya = Float32Array.from({ length: N * 4 }, (_, i) => Math.cos(i * 0.4 + 2));
      const gpu = new Float32Array(N * 4);
      await res.runAsync('add_sym',
        RuntimeTensor.fromArray(xa, [N, 4], 'f32'),
        RuntimeTensor.fromArray(ya, [N, 4], 'f32'),
        new RuntimeTensor(gpu, [N, 4], 'f32'));
      for (let i = 0; i < gpu.length; i++) expect(gpu[i]).toBeCloseTo(xa[i] + ya[i], 4);
    }
  });

  it('matmul with symbolic M and N matches the CPU reference', async () => {
    const K = 8;
    const gpuRes = await compileGraph(dotFn(K), CUDATarget());
    const cpuRes = compileGraph(dotFn(K), CPUTarget());
    for (const [M, P] of [[2, 3], [5, 4]]) {
      const la = Float32Array.from({ length: M * K }, (_, i) => Math.sin(i * 0.31 + 1));
      const ra = Float32Array.from({ length: K * P }, (_, i) => Math.cos(i * 0.27 + 2));

      const cpu = new Float32Array(M * P);
      cpuRes.run('dot_sym',
        RuntimeTensor.fromArray(la, [M, K], 'f32'),
        RuntimeTensor.fromArray(ra, [K, P], 'f32'),
        new RuntimeTensor(cpu, [M, P], 'f32'));

      const gpu = new Float32Array(M * P);
      await gpuRes.runAsync('dot_sym',
        RuntimeTensor.fromArray(la, [M, K], 'f32'),
        RuntimeTensor.fromArray(ra, [K, P], 'f32'),
        new RuntimeTensor(gpu, [M, P], 'f32'));

      for (let i = 0; i < gpu.length; i++) expect(gpu[i]).toBeCloseTo(cpu[i], 3);
    }
  });
});
