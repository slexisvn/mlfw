import { describe, it, expect } from 'vitest';
import { tensor, matmul, relu, tanh, sigmoid, sum } from '../../../src/index.js';
import { ones } from '../../../src/tensor/factory/creation_ops.js';
import { compileWithBackward } from '../../../src/tracing/compile_backward.js';
import { CUDATarget, CPUTarget } from '../../../src/backend/target.js';
import { cudaDeps } from './cuda-setup.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { QuantizationScheme, QuantizationParams } from '../../../src/compiler/ir/graph/quantization_types.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';

describe.skipIf(!cudaDeps)('CUDA training & quantization', () => {
  describe('compiled gradients match eager autograd on real GPU', () => {
    const flat = (v) => Array.from(v && v.contiguous ? v.contiguous().data : v.data);

    function mulberry32(seed) {
      let a = seed >>> 0;
      return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    }
    function data(rng, s, lo, hi) {
      const n = s.reduce((a, b) => a * b, 1);
      const flatArr = [];
      for (let i = 0; i < n; i++) flatArr.push(lo + (hi - lo) * rng());
      const nest = (fl, sh) => sh.length === 1 ? fl.slice(0, sh[0]) : Array.from({ length: sh[0] }, (_, i) => nest(fl.slice(i * fl.length / sh[0], (i + 1) * fl.length / sh[0]), sh.slice(1)));
      return nest(flatArr, s);
    }

    const PROGRAMS = [
      { name: 'matmul', shapes: [[4, 6], [6, 5]], fwd: (x, y) => matmul(x, y) },
      { name: 'matmul_relu', shapes: [[4, 6], [6, 5]], fwd: (x, y) => relu(matmul(x, y)) },
      { name: 'mlp_chain', shapes: [[3, 4], [4, 5], [5, 2]], fwd: (x, a, b) => matmul(relu(matmul(x, a)), b) },
      { name: 'deep_chain', shapes: [[2, 4], [4, 4], [4, 3]], fwd: (x, a, b) => tanh(matmul(relu(matmul(x, a)), b)) },
      { name: 'sigmoid_mlp', shapes: [[5, 8], [8, 4]], fwd: (x, a) => sigmoid(matmul(x, a)) },
    ];

    async function checkBackwardCuda(prog) {
      const rng = mulberry32(7000 + prog.name.length * 31);
      const datas = prog.shapes.map((s) => data(rng, s, -1, 1));
      const eagerInputs = datas.map((d) => tensor(d, { requiresGrad: true }));
      sum(prog.fwd(...eagerInputs)).backward();
      const eagerGrads = eagerInputs.map((x) => flat(x.grad));

      const inputs = datas.map((d) => tensor(d));
      const cf = compileWithBackward({ forward: (...a) => prog.fwd(...a) }, inputs, { target: CUDATarget() });
      let out = cf(...inputs); if (out && out.then) out = await out;
      const g = ones(Array.isArray(out) ? out[0].shape : out.shape);
      let cg = cf.backward(g); if (cg && cg.then) cg = await cg;
      const compiledGrads = cg.map((t) => flat(t));

      expect(compiledGrads.length).toBe(eagerGrads.length);
      let maxErr = 0;
      for (let i = 0; i < eagerGrads.length; i++) {
        expect(compiledGrads[i].length).toBe(eagerGrads[i].length);
        for (let k = 0; k < eagerGrads[i].length; k++) {
          maxErr = Math.max(maxErr, Math.abs(eagerGrads[i][k] - compiledGrads[i][k]) / (1 + Math.abs(eagerGrads[i][k])));
        }
      }
      return maxErr;
    }

    for (const prog of PROGRAMS) {
      it(`${prog.name} backward on CUDA`, async () => {
        expect(await checkBackwardCuda(prog), `${prog.name} grad maxRelErr`).toBeLessThan(3e-3);
      }, 60000);
    }

    it('big MLP (batch 64, hidden 512, 3 layers) backward on CUDA', async () => {
      const prog = { name: 'big', shapes: [[64, 512], [512, 512], [512, 128]], fwd: (x, a, b) => relu(matmul(relu(matmul(x, a)), b)) };
      expect(await checkBackwardCuda(prog), 'big mlp grad maxRelErr').toBeLessThan(3e-3);
    }, 120000);
  });

  describe('int8 quantization matches CPU on real GPU', () => {
    const F = ScalarType.F32, I8 = ScalarType.I8, I32 = ScalarType.I32;
    const T = (s, d = F) => new TensorType(s, d);
    const SYM = QuantizationScheme.PER_TENSOR_SYMMETRIC;

    async function diffCuda(mk, A, outN, tol = 2e-3) {
      const cpu = new Float32Array(outN), gpu = new Float32Array(outN);
      compileGraph(mk(), CPUTarget()).run('q', A, cpu);
      await compileGraph(mk(), CUDATarget()).runAsync('q', A, gpu);
      let e = 0;
      for (let i = 0; i < outN; i++) e = Math.max(e, Math.abs(cpu[i] - gpu[i]) / (1 + Math.abs(cpu[i])));
      return e;
    }

    function fakeQuant(n, scale, zp) {
      return () => buildFunction('q', [T([n])], [T([n])], (b, a) => {
        const q = b._buildOp('quantize', [a[0]], [T([n], I8)], { scale, zero_point: zp, scheme: SYM, target_dtype: I8 });
        const dq = b._buildOp('dequantize', [q.getResult(0)], [T([n])], { scale, zero_point: zp, scheme: SYM, target_dtype: F });
        b.returnOp([dq.getResult(0)]);
      });
    }

    function int8Matmul(M, K, N, A, aScale) {
      const W = Float32Array.from({ length: K * N }, (_, i) => Math.sin(i * 1.1) * 3);
      const wp = QuantizationParams.fromConstantArray([...W], SYM, I8);
      const wInt8 = wp.quantizeArray([...W]);
      const sv = [];
      for (let n = 0; n < N; n++) sv.push(aScale * wp.getScalarScale());
      return () => buildFunction('q', [T([M, K])], [T([M, N])], (b, a) => {
        const aq = b._buildOp('quantize', [a[0]], [T([M, K], I8)], { scale: aScale, zero_point: 0, scheme: SYM, target_dtype: I8 });
        const wc = b.constant(wInt8, T([K, N], I8));
        const qd = b._buildOp('quantized_dot', [aq.getResult(0), wc.getResult(0)], [T([M, N], I32)], { lhs_contracting: [1], rhs_contracting: [0], lhs_zero_point: 0, rhs_zero_point: 0, lhs_scale: aScale, rhs_scale: 1, output_scale: 1, output_zero_point: 0 });
        const cf = b._buildOp('convert', [qd.getResult(0)], [T([M, N], F)], { target_dtype: F });
        const svc = b.constant(sv, T([N], F));
        const bc = b._buildOp('broadcast_in_dim', [svc.getResult(0)], [T([M, N], F)], { broadcast_dimensions: [1], result_shape: [M, N] });
        b.returnOp([b._buildOp('mul', [cf.getResult(0), bc.getResult(0)], [T([M, N], F)], {}).getResult(0)]);
      });
    }

    it('fake-quant (quantize/dequantize) round-trip', async () => {
      const qp = QuantizationParams.fromRange(-6, 6, SYM, I8);
      const e = await diffCuda(fakeQuant(16, qp.getScalarScale(), qp.getScalarZeroPoint()), Float32Array.from({ length: 16 }, (_, i) => Math.sin(i) * 3), 16);
      expect(e).toBeLessThan(2e-3);
    }, 60000);

    it('int8 quantized_dot small', async () => {
      const e = await diffCuda(int8Matmul(4, 8, 6, Float32Array.from({ length: 32 }, (_, i) => Math.sin(i * 0.7)), 0.05), Float32Array.from({ length: 32 }, (_, i) => Math.sin(i * 0.7)), 24);
      expect(e).toBeLessThan(2e-3);
    }, 60000);

    it('int8 quantized_dot 64x128x32', async () => {
      const A = Float32Array.from({ length: 64 * 128 }, (_, i) => Math.sin(i * 0.3));
      const e = await diffCuda(int8Matmul(64, 128, 32, A, 0.02), A, 64 * 32);
      expect(e).toBeLessThan(2e-3);
    }, 60000);
  });
});
