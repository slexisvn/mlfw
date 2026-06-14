import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { QuantizationScheme, QuantizationParams } from '../../../src/compiler/ir/graph/quantization_types.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { CPUTarget, CUDATarget } from '../../../src/backend/target.js';
import { cudaDeps } from './cuda-setup.js';

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

describe.skipIf(!cudaDeps)('CUDA int8 quantization matches CPU on real GPU', () => {
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
