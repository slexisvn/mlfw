import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { CPUTarget, CUDATarget } from '../../../src/compiler/support/target.js';
import { f16ToF32, bf16ToF32, f32ToF16, f32ToBf16 } from '../../../src/tensor/utils/half.js';
import { tensor, matmul, relu, tanh } from '../../../src/index.js';
import * as nn from '../../../src/nn/index.js';
import { compile } from '../../../src/tracing/compile.js';
import { buildGpt2 } from '../../_utils/gpt2_model.js';
import { cudaDeps } from '../../_utils/cuda.js';

let cudaRuntime = null;
if (cudaDeps) cudaRuntime = await import('#io/cuda_runtime');

const flat = (v) => Array.from(v && typeof v.contiguous === 'function' ? v.contiguous().data : v.data);

describe.skipIf(!cudaDeps)('CUDA execution, graphs & models', () => {
  describe('execution matches CPU on real GPU', () => {
    const F32 = ScalarType.F32;
    const prod = (s) => s.reduce((a, b) => a * b, 1);
    const TA = { i32: Int32Array, i64: BigInt64Array, f16: Uint16Array, bf16: Uint16Array };

    async function expectDtypeMatchesCpu(dtype) {
      const t = new TensorType([64], ScalarType[dtype.toUpperCase()]);
      const fn = () => buildFunction('op', [t, t], [t], (b, a) => b.returnOp([b.mul(b.add(a[0], a[1]).getResult(0), a[0]).getResult(0)]));
      const C = TA[dtype];
      const a = new C(64), b = new C(64);
      for (let i = 0; i < 64; i++) {
        if (dtype === 'i64') { a[i] = BigInt((i % 7) - 3); b[i] = BigInt((i % 5) - 2); }
        else if (dtype === 'i32') { a[i] = (i % 7) - 3; b[i] = (i % 5) - 2; }
        else if (dtype === 'f16') { a[i] = f32ToF16(0.1 + (i % 8) * 0.1); b[i] = f32ToF16(0.2 + (i % 6) * 0.1); }
        else { a[i] = f32ToBf16(0.1 + (i % 8) * 0.1); b[i] = f32ToBf16(0.2 + (i % 6) * 0.1); }
      }
      const cpu = new C(64), gpu = new C(64);
      compileGraph(fn(), CPUTarget()).run('op', a, b, cpu);
      await compileGraph(fn(), CUDATarget()).runAsync('op', a, b, gpu);
      if (dtype === 'i32' || dtype === 'i64') {
        for (let i = 0; i < 64; i++) expect(gpu[i]).toBe(cpu[i]);
      } else {
        const dec = dtype === 'f16' ? f16ToF32 : bf16ToF32;
        for (let i = 0; i < 64; i++) expect(dec(gpu[i])).toBeCloseTo(dec(cpu[i]), 2);
      }
    }

    function rnd(n, seed) {
      const a = new Float32Array(n);
      for (let i = 0; i < n; i++) a[i] = Math.sin(i * 0.3 + seed) * 0.5;
      return a;
    }

    async function expectCudaMatchesCpu(name, mk, inShapes, outShape, opts, tol = 2e-3) {
      const ins = inShapes.map((s, k) => rnd(Math.max(1, prod(s)), k + 1));
      const cpu = new Float32Array(prod(outShape));
      const gpu = new Float32Array(prod(outShape));
      compileGraph(mk(), CPUTarget(), opts).run(name, ...ins, cpu);
      await compileGraph(mk(), CUDATarget(), opts).runAsync(name, ...ins, gpu);
      let maxErr = 0;
      for (let i = 0; i < cpu.length; i++) maxErr = Math.max(maxErr, Math.abs(cpu[i] - gpu[i]) / (1 + Math.abs(cpu[i])));
      expect(maxErr, `${name} maxRelErr`).toBeLessThan(tol);
    }

    const elementwise = () => buildFunction('ew', [new TensorType([256], F32), new TensorType([256], F32)], [new TensorType([256], F32)], (b, a) => {
      let v = b.mul(a[0], a[1]).getResult(0);
      v = b.tanh(v).getResult(0);
      v = b.add(v, a[0]).getResult(0);
      v = b.sigmoid(v).getResult(0);
      b.returnOp([v]);
    });
    const matmul = () => buildFunction('mm', [new TensorType([16, 24], F32), new TensorType([24, 16], F32)], [new TensorType([16, 16], F32)], (b, a) => b.returnOp([b.matmul(a[0], a[1]).getResult(0)]));
    const reduction = () => buildFunction('red', [new TensorType([16, 24], F32), new TensorType([], F32)], [new TensorType([16], F32)], (b, a) => b.returnOp([b.reduce(a[0], a[1], [1], 'sum').getResult(0)]));
    const softmax = () => buildFunction('sm', [new TensorType([16, 24], F32)], [new TensorType([16, 24], F32)], (b, a) => b.returnOp([b.softmax(a[0]).getResult(0)]));
    const layernorm = () => buildFunction('ln', [new TensorType([16, 24], F32), new TensorType([24], F32), new TensorType([24], F32)], [new TensorType([16, 24], F32)], (b, a) => b.returnOp([b.layernorm(a[0], a[1], a[2]).getResult(0)]));
    const conv = () => buildFunction('cv', [new TensorType([1, 3, 12, 12], F32), new TensorType([6, 3, 3, 3], F32)], [new TensorType([1, 6, 10, 10], F32)], (b, a) => b.returnOp([b.conv(a[0], a[1], [1, 1], [[0, 0], [0, 0]]).getResult(0)]));
    const gelu = () => buildFunction('gl', [new TensorType([256], F32)], [new TensorType([256], F32)], (b, a) => b.returnOp([b.gelu(a[0]).getResult(0)]));

    it('elementwise chain', () => expectCudaMatchesCpu('ew', elementwise, [[256], [256]], [256], {}));
    it('matmul', () => expectCudaMatchesCpu('mm', matmul, [[16, 24], [24, 16]], [16, 16], {}));
    it('reduction sum', () => expectCudaMatchesCpu('red', reduction, [[16, 24], []], [16], {}));
    it('softmax', () => expectCudaMatchesCpu('sm', softmax, [[16, 24]], [16, 24], {}));
    it('layernorm', () => expectCudaMatchesCpu('ln', layernorm, [[16, 24], [24], [24]], [16, 24], {}));
    it('conv', () => expectCudaMatchesCpu('cv', conv, [[1, 3, 12, 12], [6, 3, 3, 3]], [1, 6, 10, 10], {}));
    it('gelu', () => expectCudaMatchesCpu('gl', gelu, [[256]], [256], {}));

    for (const [cfgName, opts] of [['scheduled', { scheduling: { enabled: true } }], ['autotune', { scheduling: { enabled: true, autotune: true, seed: 7 } }]]) {
      it(`elementwise chain (${cfgName})`, () => expectCudaMatchesCpu('ew', elementwise, [[256], [256]], [256], opts));
      it(`matmul (${cfgName})`, () => expectCudaMatchesCpu('mm', matmul, [[16, 24], [24, 16]], [16, 16], opts));
      it(`reduction sum (${cfgName})`, () => expectCudaMatchesCpu('red', reduction, [[16, 24], []], [16], opts));
      it(`softmax (${cfgName})`, () => expectCudaMatchesCpu('sm', softmax, [[16, 24]], [16, 24], opts));
      it(`layernorm (${cfgName})`, () => expectCudaMatchesCpu('ln', layernorm, [[16, 24], [24], [24]], [16, 24], opts));
      it(`conv (${cfgName})`, () => expectCudaMatchesCpu('cv', conv, [[1, 3, 12, 12], [6, 3, 3, 3]], [1, 6, 10, 10], opts));
      it(`gelu (${cfgName})`, () => expectCudaMatchesCpu('gl', gelu, [[256]], [256], opts));
    }

    for (const N of [128, 256]) {
      const big = () => buildFunction('mmN', [new TensorType([N, N], F32), new TensorType([N, N], F32)], [new TensorType([N, N], F32)], (b, a) => b.returnOp([b.matmul(a[0], a[1]).getResult(0)]));
      it(`scheduled matmul ${N}x${N}`, () => expectCudaMatchesCpu('mmN', big, [[N, N], [N, N]], [N, N], { scheduling: { enabled: true } }));
    }

    for (const dtype of ['i32', 'i64', 'f16', 'bf16']) {
      it(`dtype ${dtype}`, () => expectDtypeMatchesCpu(dtype));
    }
  });

  describe('full models matching CPU on real GPU', () => {
    async function maxRelErr(fwd, inputs) {
      const cpu = flat(await compile({ forward: fwd }, inputs, { target: CPUTarget() })(...inputs));
      const gpu = flat(await compile({ forward: fwd }, inputs, { target: CUDATarget() })(...inputs));
      expect(gpu.length).toBe(cpu.length);
      let e = 0;
      for (let i = 0; i < cpu.length; i++) e = Math.max(e, Math.abs(cpu[i] - gpu[i]) / (1 + Math.abs(cpu[i])));
      return e;
    }

    it('GPT-2 decoder compiled on CUDA matches CPU', async () => {
      const { fwd, ids } = buildGpt2(nn, tensor, { V: 64, D: 48, H: 3, FF: 96, L: 2, SEQ: 8 });
      const e = await maxRelErr(fwd, [ids]);
      expect(e, `gpt2 cuda-vs-cpu maxRelErr (${cudaDeps.arch})`).toBeLessThan(2e-3);
    }, 180000);
  });

  describe('graph capture/replay on the compiled plan', () => {
    const maxRelErr = (a, b) => { let e = 0; for (let i = 0; i < a.length; i++) e = Math.max(e, Math.abs(a[i] - b[i]) / (1 + Math.abs(a[i]))); return e; };

    function mk(n, seed) { return Array.from({ length: n }, (_, i) => Math.sin(seed + i * 0.21) * 0.4); }

    it('graphed replay matches non-graphed and CPU on a native matmul chain', async () => {
      const x = tensor(mk(6 * 8, 1), { shape: [6, 8] });
      const w1 = tensor(mk(8 * 12, 2), { shape: [8, 12] });
      const w2 = tensor(mk(12 * 5, 3), { shape: [12, 5] });
      const fwd = (a, p, q) => tanh(matmul(relu(matmul(a, p)), q));

      const cpu = flat(await compile({ forward: fwd }, [x, w1, w2], { target: CPUTarget() })(x, w1, w2));

      const cf = compile({ forward: fwd }, [x, w1, w2], { target: CUDATarget() });
      expect(cf.result().module.executionPlan, 'native matmul chain produced a multi-kernel plan').toBeTruthy();

      cudaRuntime.setCudaGraphEnabled(false);
      const noGraph = flat(await cf(x, w1, w2));
      cudaRuntime.setCudaGraphEnabled(true);
      let graphed;
      try { graphed = flat(await cf(x, w1, w2)); } finally { cudaRuntime.setCudaGraphEnabled(false); }

      expect(maxRelErr(cpu, noGraph), 'plan vs CPU').toBeLessThan(2e-3);
      expect(maxRelErr(noGraph, graphed), 'graphed vs non-graphed').toBeLessThan(1e-5);
    }, 120000);

    it('repeated graphed replays are stable', async () => {
      const x = tensor(mk(4 * 6, 7), { shape: [4, 6] });
      const w1 = tensor(mk(6 * 6, 8), { shape: [6, 6] });
      const w2 = tensor(mk(6 * 3, 9), { shape: [6, 3] });
      const fwd = (a, p, q) => matmul(relu(matmul(a, p)), q);
      const cf = compile({ forward: fwd }, [x, w1, w2], { target: CUDATarget() });
      cudaRuntime.setCudaGraphEnabled(true);
      try {
        const first = flat(await cf(x, w1, w2));
        for (let i = 0; i < 5; i++) {
          expect(maxRelErr(first, flat(await cf(x, w1, w2)))).toBeLessThan(1e-6);
        }
      } finally { cudaRuntime.setCudaGraphEnabled(false); }
    }, 120000);
  });
});
