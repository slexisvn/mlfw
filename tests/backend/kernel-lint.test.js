import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../src/compiler/ir/graph/builder.js';
import { compileGraph } from '../../src/compiler/pipeline/compiler.js';
import { CPUTarget, WasmTarget, CUDATarget, WebGPUTarget } from '../../src/backend/target.js';
import { lintKernel, lintKernelStrict } from '../_utils/kernel_lint.js';
import { F32, T as t } from '../_utils/ir_fixture.js';


const MODELS = {
  matmul_relu: () => buildFunction('mm_relu', [t([8, 16]), t([16, 16])], [t([8, 16])], (b, a) => {
    const m = b.matmul(a[0], a[1]).getResult(0);
    b.returnOp([b.relu(m).getResult(0)]);
  }),
  softmax: () => buildFunction('sm', [t([8, 16])], [t([8, 16])], (b, a) => {
    b.returnOp([b.softmax(a[0], 1).getResult(0)]);
  }),
  layernorm: () => buildFunction('ln', [t([8, 16]), t([16]), t([16])], [t([8, 16])], (b, a) => {
    b.returnOp([b._inferAndBuild('layer_norm', [a[0], a[1], a[2]], { axis: 1, epsilon: 1e-5 }).getResult(0)]);
  }),
  reduce_sum: () => buildFunction('rs', [t([8, 16])], [t([8])], (b, a) => {
    const z = b.scalarConstant(0, F32).getResult(0);
    b.returnOp([b.reduce(a[0], z, [1], 'sum').getResult(0)]);
  }),
  reduce_max: () => buildFunction('rmax', [t([8, 16])], [t([8])], (b, a) => {
    const z = b.scalarConstant(-Infinity, F32).getResult(0);
    b.returnOp([b.reduce(a[0], z, [1], 'max').getResult(0)]);
  }),
  conv: () => buildFunction('cv', [t([1, 1, 8, 8]), t([2, 1, 3, 3])], [t([1, 2, 6, 6])], (b, a) => {
    b.returnOp([b.conv(a[0], a[1], [1, 1], [[0, 0], [0, 0]]).getResult(0)]);
  }),
  pool: () => buildFunction('pl', [t([1, 1, 8, 8])], [t([1, 1, 4, 4])], (b, a) => {
    b.returnOp([b.pool2d(a[0], 'max', [2, 2], [2, 2], [[0, 0], [0, 0]]).getResult(0)]);
  }),
  elementwise_chain: () => buildFunction('ew', [t([8, 16]), t([8, 16])], [t([8, 16])], (b, a) => {
    const s = b.add(a[0], a[1]).getResult(0);
    const m = b.mul(s, a[0]).getResult(0);
    b.returnOp([b.tanh(m).getResult(0)]);
  }),
  matmul_odd: () => buildFunction('mmodd', [t([50, 50]), t([50, 50])], [t([50, 50])], (b, a) => {
    b.returnOp([b.matmul(a[0], a[1]).getResult(0)]);
  }),
};

const TARGETS = { cpu: CPUTarget, wasm: WasmTarget, gpu: CUDATarget, webgpu: WebGPUTarget };

describe('static kernel quality lint across backends', () => {
  for (const [mname, build] of Object.entries(MODELS)) {
    for (const [tname, makeTarget] of Object.entries(TARGETS)) {
      it(`${mname} on ${tname} emits no malformed kernel source`, () => {
        const result = compileGraph(build(), makeTarget(), { scheduling: { enabled: true } });
        const kernels = result.listKernels();
        expect(kernels.length).toBeGreaterThan(0);
        for (const k of kernels) {
          const src = result.getSource(k);
          expect(src && src.length).toBeGreaterThan(0);
          const issues = lintKernelStrict(src);
          expect(issues, `${mname}/${tname}/${k}: ${JSON.stringify(issues)}`).toEqual([]);
        }
      });
    }
  }
});

describe('kernel lint self-check', () => {
  it('detects unbalanced braces', () => {
    expect(lintKernelStrict('void k() { a[0] = 1;').some(i => i.kind === 'unbalanced')).toBe(true);
  });
  it('detects a self-assign store', () => {
    expect(lintKernelStrict('__global__ void k(float* x) { x[i] = x[i]; }').some(i => i.kind === 'self_assign')).toBe(true);
  });
  it('detects JS artifacts in a native kernel', () => {
    expect(lintKernel('__global__ void k() { float a = undefined; }').issues.some(i => i.kind === 'js_artifact')).toBe(true);
  });
  it('passes a clean cuda kernel', () => {
    const src = '__global__ void k(float* x, float* y) { const int i = threadIdx.x; if (i < 16) { y[i] = x[i] + 1.0f; } }';
    expect(lintKernelStrict(src)).toEqual([]);
  });
});
