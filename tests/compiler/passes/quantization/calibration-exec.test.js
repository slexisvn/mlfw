import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { compileGraph, Compiler } from '../../../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../../../src/backend/target.js';
import { QuantizationScheme } from '../../../../src/compiler/ir/graph/quantization_types.js';
import { collectCalibration } from '../../../../src/compiler/analysis/calibrate_exec.js';
import { tensor, Linear, ReLU, Sequential, manual_seed } from '../../../../src/index.js';
import { compile } from '../../../../src/tracing/compile.js';
import { TraceLevel } from '../../../../src/compiler/pipeline/trace.js';
import { F32 as F, T } from '../../../_utils/ir_fixture.js';


function rnd(seed) { let s = seed; return () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; }; }

const M = 8, K = 64, N = 32;
const rw = rnd(2);
const W = new Float32Array(K * N);
for (let k = 0; k < K; k++) for (let n = 0; n < N; n++) W[k * N + n] = (rw() - 0.5) * 2 * (0.05 + (n / N) * 2);

const buildLinear = () => buildFunction('q', [T([M, K])], [T([M, N])], (b, a) => {
  const wc = b.constant([...W], T([K, N]));
  b.returnOp([b.matmul(a[0], wc.getResult(0)).getResult(0)]);
});

function makeSmallActivationInput(seed) {
  const r = rnd(seed); const x = new Float32Array(M * K);
  for (let i = 0; i < x.length; i++) x[i] = (r() - 0.5) * 0.6;
  return x;
}
function reference(X) {
  const ref = new Float32Array(M * N);
  for (let m = 0; m < M; m++) for (let n = 0; n < N; n++) {
    let acc = 0; for (let k = 0; k < K; k++) acc += X[m * K + k] * W[k * N + n];
    ref[m * N + n] = acc;
  }
  return ref;
}
function relErr(out, ref) {
  let e = 0, den = 0;
  for (let i = 0; i < out.length; i++) { e += Math.abs(out[i] - ref[i]); den += Math.abs(ref[i]); }
  return e / den;
}

const compileFn = (mod, tgt) => new Compiler({ target: tgt }).compile(mod);

describe('activation calibration end-to-end', () => {
  const X = makeSmallActivationInput(1);
  const ref = reference(X);
  const batches = [];
  for (let t = 0; t < 4; t++) batches.push([makeSmallActivationInput(100 + t)]);

  function run(q) {
    const r = compileGraph(buildLinear(), CPUTarget(), {
      quantization: { enabled: true, quantizableOps: new Set(['dot']), ...q },
    });
    const out = new Float32Array(M * N);
    r.run('q', X, out);
    return relErr(out, ref);
  }

  it('default [-6,6] activation range is lossy on small activations (the bug)', () => {
    const err = run({ scheme: QuantizationScheme.PER_TENSOR_SYMMETRIC });
    expect(err).toBeGreaterThan(0.05);
  });

  it('calibration collapses activation error to int8-grade (~1%)', () => {
    const err = run({ scheme: QuantizationScheme.PER_TENSOR_SYMMETRIC, calibrationData: batches });
    expect(err).toBeLessThan(0.02);
  });

  it('calibration is a >=4x improvement over the uncalibrated default', () => {
    const uncal = run({ scheme: QuantizationScheme.PER_TENSOR_SYMMETRIC });
    const cal = run({ scheme: QuantizationScheme.PER_TENSOR_SYMMETRIC, calibrationData: batches });
    expect(uncal / cal).toBeGreaterThan(4);
  });

  it('per-channel + calibration also reaches int8-grade accuracy', () => {
    const err = run({ scheme: QuantizationScheme.PER_CHANNEL, calibrationData: batches });
    expect(err).toBeLessThan(0.02);
  });
});

describe('calibration through an intermediate activation (2-layer + relu)', () => {
  const H = 64, N2 = 24, K2 = 48, MM = 8;
  const r2 = rnd(7);
  const W1 = new Float32Array(K2 * H); for (let i = 0; i < W1.length; i++) W1[i] = (r2() - 0.5) * 0.5;
  const W2 = new Float32Array(H * N2); for (let i = 0; i < W2.length; i++) W2[i] = (r2() - 0.5) * 0.5;

  const build = () => buildFunction('q', [T([MM, K2])], [T([MM, N2])], (b, a) => {
    const w1 = b.constant([...W1], T([K2, H])).getResult(0);
    const w2 = b.constant([...W2], T([H, N2])).getResult(0);
    const h = b.matmul(a[0], w1).getResult(0);
    const hr = b.relu(h).getResult(0);
    b.returnOp([b.matmul(hr, w2).getResult(0)]);
  });
  function inp(seed) { const r = rnd(seed); const x = new Float32Array(MM * K2); for (let i = 0; i < x.length; i++) x[i] = (r() - 0.5) * 0.6; return x; }
  function fref(X) {
    const h = new Float32Array(MM * H);
    for (let m = 0; m < MM; m++) for (let j = 0; j < H; j++) { let a = 0; for (let k = 0; k < K2; k++) a += X[m * K2 + k] * W1[k * H + j]; h[m * H + j] = Math.max(0, a); }
    const o = new Float32Array(MM * N2);
    for (let m = 0; m < MM; m++) for (let n = 0; n < N2; n++) { let a = 0; for (let j = 0; j < H; j++) a += h[m * H + j] * W2[j * N2 + n]; o[m * N2 + n] = a; }
    return o;
  }
  it('calibrating the intermediate (relu) activation reaches ~1%', () => {
    const X = inp(1); const ref = fref(X);
    const batches = []; for (let t = 0; t < 6; t++) batches.push([inp(200 + t)]);
    const r = compileGraph(build(), CPUTarget(), { quantization: { enabled: true, quantizableOps: new Set(['dot']), scheme: QuantizationScheme.PER_TENSOR_SYMMETRIC, calibrationData: batches } });
    const out = new Float32Array(MM * N2); r.run('q', X, out);
    expect(relErr(out, ref)).toBeLessThan(0.03);
  });
});

describe('calibration survives the rewrites that run before quantization', () => {
  const MT = 8, KT = 48, NT = 24;
  const rt = rnd(11);
  const WT = new Float32Array(NT * KT);
  for (let i = 0; i < WT.length; i++) WT[i] = (rt() - 0.5) * 0.4;

  // dot(x, transpose(wT)) — canonicalization folds the transpose into the dot's
  // contracting dimensions, so the operand quantization sees is wT itself.
  const build = () => buildFunction('q', [T([MT, KT]), T([NT, KT])], [T([MT, NT])], (b, a) => {
    const w = b.transpose(a[1], [1, 0]).getResult(0);
    b.returnOp([b.matmul(a[0], w).getResult(0)]);
  });
  function inp(seed) { const r = rnd(seed); const x = new Float32Array(MT * KT); for (let i = 0; i < x.length; i++) x[i] = (r() - 0.5) * 0.6; return x; }
  function fref(X) {
    const o = new Float32Array(MT * NT);
    for (let m = 0; m < MT; m++) for (let n = 0; n < NT; n++) { let acc = 0; for (let k = 0; k < KT; k++) acc += X[m * KT + k] * WT[n * KT + k]; o[m * NT + n] = acc; }
    return o;
  }
  function run(q) {
    let ir = null;
    const r = compileGraph(build(), CPUTarget(), {
      quantization: { enabled: true, quantizableOps: new Set(['dot']), scheme: QuantizationScheme.PER_TENSOR_SYMMETRIC, ...q },
      trace: { level: TraceLevel.DEBUG, irSnapshot: { afterGraphPasses: true }, sink: (e) => { if (e.type === 'ir_snapshot') ir = e.text; } },
    });
    const X = inp(3);
    const out = new Float32Array(MT * NT);
    r.run('q', X, WT, out);
    return { err: relErr(out, fref(X)), ir };
  }

  it('folds the transpose away, so the observed value no longer exists at quantization time', () => {
    const { ir } = run({});
    expect(ir).not.toMatch(/transpose/);
    expect(ir).toMatch(/quantized_dot/);
  });

  it('calibrating the transposed weight operand reaches int8-grade accuracy', () => {
    const batches = []; for (let t = 0; t < 4; t++) batches.push([inp(300 + t), WT]);
    expect(run({ calibrationData: batches }).err).toBeLessThan(0.02);
  });

  it('without calibration the same graph stays at the lossy default range', () => {
    expect(run({}).err).toBeGreaterThan(0.05);
  });
});

describe('calibration through the public compile() path (traced model, captured parameters)', () => {
  const raw = Float32Array.from({ length: M * K }, (_, i) => ((i % 31) / 31) - 0.5);
  const X = tensor([...raw]).reshape([M, K]);
  const build = () => { manual_seed(0); return new Sequential(new Linear(K, 4 * M), new ReLU(), new Linear(4 * M, N)); };
  const quant = { enabled: true, quantizableOps: new Set(['dot']) };

  async function outputs(opts) {
    const compiled = compile(build(), [X], { target: CPUTarget(), ...opts });
    await compiled._ready;
    return (await compiled(X)).toArray().flat();
  }

  it('compiles when a batch holds only the user inputs, not the captured parameters', async () => {
    const out = await outputs({ quantization: { ...quant, calibrationData: [[X]] } });
    expect(out.length).toBe(M * N);
    expect(out.every(Number.isFinite)).toBe(true);
  });

  it('accepts a batch given as tensors or as raw arrays', async () => {
    const fromTensors = await outputs({ quantization: { ...quant, calibrationData: [[X]] } });
    const fromArrays = await outputs({ quantization: { ...quant, calibrationData: [[raw]] } });
    expect(fromArrays).toEqual(fromTensors);
  });

  it('calibrated activations over folded weights reach int8-grade accuracy', async () => {
    const ref = await outputs({});
    const cal = await outputs({ foldWeights: true, quantization: { ...quant, calibrationData: [[X]] } });
    expect(relErr(cal, ref)).toBeLessThan(0.02);
  });

  it('calibration alone reaches int8-grade accuracy, without folding the weights', async () => {
    const ref = await outputs({});
    const cal = await outputs({ quantization: { ...quant, calibrationData: [[X]] } });
    expect(relErr(cal, ref)).toBeLessThan(0.02);
  });

  it('folded weights alone leave the activation range at the lossy default', async () => {
    const ref = await outputs({});
    const uncal = await outputs({ foldWeights: true, quantization: quant });
    expect(relErr(uncal, ref)).toBeGreaterThan(0.04);
  });
});

describe('collectCalibration standalone', () => {
  it('returns a CalibrationResult with observed ranges for activations (arg feeding dot)', () => {
    const func = buildLinear();
    const batches = [[makeSmallActivationInput(1)], [makeSmallActivationInput(2)]];
    const result = collectCalibration(func, CPUTarget(), batches, { quantizableOps: new Set(['dot']), compileFn });
    const arg = func.args[0];
    expect(result.hasData(arg)).toBe(true);
    const range = result.getRange(arg);
    expect(range.max).toBeLessThan(0.5);
    expect(range.min).toBeGreaterThan(-0.5);
  });

  it('does not observe constant weights (those calibrate from data directly)', () => {
    const func = buildLinear();
    const result = collectCalibration(func, CPUTarget(), [[makeSmallActivationInput(1)]], { quantizableOps: new Set(['dot']), compileFn });
    const wConst = func.findOp(o => o.opName === 'constant');
    expect(result.hasData(wConst.getResult(0))).toBe(false);
  });

  it('throws on async (GPU) targets', () => {
    const func = buildLinear();
    const fakeGpu = { isGPU: () => true, supportsInt8: true };
    expect(() => collectCalibration(func, fakeGpu, [[makeSmallActivationInput(1)]], { compileFn })).toThrow(/async/i);
  });

  it('throws when no batches are supplied', () => {
    const func = buildLinear();
    expect(() => collectCalibration(func, CPUTarget(), [], { compileFn })).toThrow(/batch/i);
  });
});
