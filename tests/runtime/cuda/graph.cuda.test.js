import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, Adam, Linear, LSTM, Embedding, CrossEntropyLoss, relu, mean, mul, sub, add, div, cat, MSELoss, TensorDataset, DataLoader } from '../../../src/index.js';
import { LightningModule, Trainer } from '../../../src/lightning/index.js';
import { cudnnAvailable } from '../../../src/runtime/node/cuda/cudnn.js';
import { GPU_DEVICE } from '../../../src/tensor/types/device.js';
import { preloadCudaRuntime } from '../../../src/runtime/backend_registry.js';
import { cudaDeps } from '../../_utils/cuda.js';
import { getDevice } from '../../../src/runtime/node/cuda/device.js';
import { getProgram } from '../../../src/runtime/node/cuda/program.js';
import { launch, f32, i32 } from '../../../src/runtime/node/cuda/launcher.js';
import { acquire, release, copyHostToDevice, copyDeviceToHost, copyHostToDeviceAsync } from '../../../src/runtime/node/cuda/memory.js';
import { cu } from '../../../src/runtime/node/cuda/ffi.js';
import { cublasMatmulDevice } from '../../../src/runtime/node/cuda/cublas.js';
import { beginEagerCapture, endEagerCapture, replay, syncStream, destroyEagerGraph } from '../../../src/runtime/node/cuda/eager_graph.js';
import { stepIncKernelSource, adamGraphKernelSource, deviceClipGradNorm } from '../../../src/runtime/node/cuda_runtime.js';

const maxAbsErr = (a, b) => {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
};
const grid1d = (n) => [Math.max(Math.ceil(n / 256), 1), 1, 1];
const BLOCK = [256, 1, 1];

const SCALE_SRC = `extern "C" __global__ void scale2(const float* in, float* out, int n) {
  int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) out[i] = in[i] * 2.0f;
}`;
const RELU_SRC = `extern "C" __global__ void relu_ip(float* x, int n) {
  int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) { float v = x[i]; x[i] = v > 0.0f ? v : 0.0f; }
}`;

function cpuChain(A0, B, M, K, N) {
  const out = new Float32Array(M * N);
  for (let r = 0; r < M; r++) {
    for (let c = 0; c < N; c++) {
      let acc = 0;
      for (let k = 0; k < K; k++) acc += (A0[r * K + k] * 2) * B[k * N + c];
      out[r * N + c] = Math.max(acc, 0);
    }
  }
  return out;
}

describe.skipIf(!cudaDeps)('CUDA eager graph capture/replay primitive', () => {
  beforeAll(async () => { await preloadCudaRuntime(); getDevice(); });

  it('captures a scale -> cuBLAS sgemm -> relu chain and replays it bit-exactly', () => {
    const M = 48, K = 32, N = 24;
    const mk = M * K, kn = K * N, mn = M * N;
    const A0 = new Float32Array(mk).map((_, i) => Math.sin(i * 0.13) - 0.3);
    const B = new Float32Array(kn).map((_, i) => Math.cos(i * 0.07));

    const dA0 = acquire(mk * 4), dA = acquire(mk * 4), dB = acquire(kn * 4), dC = acquire(mn * 4);
    copyHostToDevice(dA0, A0);
    copyHostToDevice(dB, B);
    const scale = getProgram(SCALE_SRC, 'scale2').func;
    const reluf = getProgram(RELU_SRC, 'relu_ip').func;

    const runChain = (sync) => {
      launch(scale, grid1d(mk), BLOCK, 0, [dA0, dA], [mk], sync);
      cublasMatmulDevice(M, N, K, dA, dB, dC);
      launch(reluf, grid1d(mn), BLOCK, 0, [dC], [mn], sync);
    };

    runChain(true);
    syncStream();
    const eagerOut = new Float32Array(mn);
    copyDeviceToHost(eagerOut, dC);

    const cpuOut = cpuChain(A0, B, M, K, N);
    expect(maxAbsErr(eagerOut, cpuOut), 'eager device vs cpu').toBeLessThan(1e-3);
    expect(eagerOut.some((x) => x > 0.1), 'output is non-trivial').toBe(true);

    beginEagerCapture();
    let captured;
    try { runChain(false); captured = endEagerCapture(); }
    catch (e) { try { endEagerCapture(); } catch (_) {} throw e; }

    const origLaunch = cu.launchKernel;
    let kernelLaunches = 0;
    cu.launchKernel = (...args) => { kernelLaunches++; return origLaunch(...args); };
    try {
      for (let rep = 0; rep < 4; rep++) {
        cu.memsetD8(dA, 0, mk * 4);
        cu.memsetD8(dC, 0, mn * 4);
        replay(captured.exec);
        syncStream();
        const graphOut = new Float32Array(mn);
        copyDeviceToHost(graphOut, dC);
        expect(maxAbsErr(graphOut, eagerOut), `graph replay ${rep} vs eager`).toBe(0);
      }
    } finally {
      cu.launchKernel = origLaunch;
    }
    expect(kernelLaunches, 'no per-op cuLaunchKernel during replay').toBe(0);

    destroyEagerGraph(captured);
    release(dA0, mk * 4); release(dA, mk * 4); release(dB, kn * 4); release(dC, mn * 4);
  }, 60000);

  it('device-step Adam (step_inc + adam_graph) tracks bias-correction across replays', () => {
    const n = 257;
    const beta1 = 0.9, beta2 = 0.999, eps = 1e-8, lr = 0.01, wd = 0.0;
    const ob1 = 1 - beta1, ob2 = 1 - beta2;
    const fb1 = Math.fround(beta1), fb2 = Math.fround(beta2);
    const w0 = new Float32Array(n).map((_, i) => Math.sin(i * 0.21));
    const g = new Float32Array(n).map((_, i) => 0.5 * Math.cos(i * 0.11));

    const dW = acquire(n * 4), dG = acquire(n * 4), dM = acquire(n * 4), dV = acquire(n * 4), dT = acquire(4);
    copyHostToDevice(dW, w0);
    copyHostToDevice(dG, g);
    cu.memsetD8(dM, 0, n * 4);
    cu.memsetD8(dV, 0, n * 4);
    cu.memsetD8(dT, 0, 4);

    const si = stepIncKernelSource();
    const ag = adamGraphKernelSource('f32');
    const stepInc = getProgram(si.source, si.name).func;
    const adamG = getProgram(ag.source, ag.name).func;

    const STEPS = 8;
    beginEagerCapture();
    let captured;
    try {
      launch(stepInc, [1, 1, 1], [1, 1, 1], 0, [dT], [], false);
      launch(adamG, grid1d(n), BLOCK, 0, [dW, dG, dM, dV, dT], [n, f32(beta1), f32(beta2), f32(ob1), f32(ob2), f32(eps), f32(lr), f32(wd)], false);
      captured = endEagerCapture();
    } catch (e) { try { endEagerCapture(); } catch (_) {} throw e; }

    for (let s = 0; s < STEPS; s++) { replay(captured.exec); }
    syncStream();
    const graphW = new Float32Array(n);
    copyDeviceToHost(graphW, dW);

    const w = Float64Array.from(w0), m = new Float64Array(n), v = new Float64Array(n);
    for (let t = 1; t <= STEPS; t++) {
      const bc1 = 1 - Math.pow(fb1, t), bc2 = 1 - Math.pow(fb2, t);
      const ss = lr / bc1, bc2s = Math.sqrt(bc2);
      for (let i = 0; i < n; i++) {
        const gi = g[i] + wd * w[i];
        m[i] = beta1 * m[i] + ob1 * gi;
        v[i] = beta2 * v[i] + ob2 * gi * gi;
        w[i] = w[i] - ss * m[i] / (Math.sqrt(v[i]) / bc2s + eps);
      }
    }

    expect(maxAbsErr(graphW, w), 'graph Adam vs host Adam over 8 steps').toBeLessThan(1e-3);
    destroyEagerGraph(captured);
    release(dW, n * 4); release(dG, n * 4); release(dM, n * 4); release(dV, n * 4); release(dT, 4);
  }, 60000);
});

describe.skipIf(!cudaDeps)('CUDA launcher scalar param typing', () => {
  beforeAll(async () => { await preloadCudaRuntime(); getDevice(); });

  const PROBE_SRC = `extern "C" __global__ void scalar_probe(float* fout, int* iout, int n, int ival, float fval) {
  int i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= n) return;
  fout[i] = fval; iout[i] = ival;
}`;

  it('writes an integer-valued float scalar as float32, not an int32 denormal', () => {
    const n = 64;
    const probe = getProgram(PROBE_SRC, 'scalar_probe').func;
    const fDev = acquire(n * 4), iDev = acquire(n * 4);
    try {
      launch(probe, grid1d(n), BLOCK, 0, [fDev, iDev], [n, i32(7), f32(1.0)], true);
      syncStream();
      const fOut = new Float32Array(n), iOut = new Int32Array(n);
      copyDeviceToHost(fOut, fDev);
      copyDeviceToHost(iOut, iDev);
      for (let i = 0; i < n; i++) {
        expect(fOut[i], `f32(1.0) read as float at ${i}`).toBe(1.0);
        expect(iOut[i], `i32(7) read as int at ${i}`).toBe(7);
      }

      launch(probe, grid1d(n), BLOCK, 0, [fDev, iDev], [n, i32(7), f32(1000.0)], true);
      syncStream();
      copyDeviceToHost(fOut, fDev);
      expect(fOut[0], 'f32(1000.0) read as float').toBe(1000.0);
    } finally {
      release(fDev, n * 4); release(iDev, n * 4);
    }
  }, 60000);

  it('documents the prevented bug: an untagged integer-valued float is mis-encoded as a denormal', () => {
    const n = 8;
    const probe = getProgram(PROBE_SRC, 'scalar_probe').func;
    const fDev = acquire(n * 4), iDev = acquire(n * 4);
    try {
      launch(probe, grid1d(n), BLOCK, 0, [fDev, iDev], [n, 7, 1.0], true);
      syncStream();
      const fOut = new Float32Array(n);
      copyDeviceToHost(fOut, fDev);
      expect(fOut[0], 'untagged 1.0 is corrupted (proves the f32 test is sensitive)').not.toBe(1.0);
      expect(Math.abs(fOut[0]), 'untagged 1.0 collapses to a near-zero denormal').toBeLessThan(1e-40);
    } finally {
      release(fDev, n * 4); release(iDev, n * 4);
    }
  }, 60000);
});

describe.skipIf(!cudaDeps)('CUDA eager whole-step graph (forward + backward + Adam)', () => {
  let resident;
  beforeAll(async () => {
    await preloadCudaRuntime();
    getDevice();
    resident = await import('../../../src/runtime/node/cuda/resident.js');
  });

  const flat = (v) => Array.from(v && typeof v.contiguous === 'function' ? v.contiguous().data : v.data);

  it('captures fwd+bwd+Adam once and replays a multi-step trajectory matching non-graphed eager', () => {
    const IN = 16, HID = 24, OUT = 8, B = 8, STEPS = 7, WARMUP = 3, lr = 0.01;
    const seq = Array.from({ length: STEPS }, (_, s) =>
      Float32Array.from({ length: B * IN }, (_, i) => Math.sin(i * 0.21 + s * 1.7) * 0.5));
    const targetData = Float32Array.from({ length: B * OUT }, (_, i) => Math.cos(i * 0.13));

    const build = () => {
      const l1 = new Linear(IN, HID), l2 = new Linear(HID, OUT);
      l1.to(GPU_DEVICE); l2.to(GPU_DEVICE);
      return { l1, l2, params: [...l1.parameters(), ...l2.parameters()] };
    };
    const trainStep = (m, x, tgt, opt) => {
      for (const p of m.params) p.grad = null;
      const pred = m.l2.forward(relu(m.l1.forward(x)));
      const diff = sub(pred, tgt);
      const loss = mean(mul(diff, diff));
      loss.backward();
      opt.step();
      return loss;
    };
    const gpuInput = (data) => tensor(Array.from(data), { shape: [B, IN] }).to(GPU_DEVICE);

    resident.setEagerDeferred(true);
    resident.setCudaGraphArmed(true);

    const A = build();
    const initParams = A.params.map((p) => Float32Array.from(flat(p)));
    const tgtA = tensor(Array.from(targetData), { shape: [B, OUT] }).to(GPU_DEVICE);
    const optA = new Adam(A.params, { lr });
    const lossA = [];
    for (let s = 0; s < STEPS; s++) {
      const loss = trainStep(A, gpuInput(seq[s]), tgtA, optA);
      lossA.push(loss.item());
      resident.flushDeferred();
    }
    const finalA = A.params.map((p) => Float32Array.from(flat(p)));

    const Bm = build();
    Bm.params.forEach((p, i) => p._impl.storage.rawData.set(initParams[i]));
    const tgtB = tensor(Array.from(targetData), { shape: [B, OUT] }).to(GPU_DEVICE);
    const optB = new Adam(Bm.params, { lr });
    const lossB = [];

    for (let s = 0; s < WARMUP; s++) {
      const loss = trainStep(Bm, gpuInput(seq[s]), tgtB, optB);
      lossB.push(loss.item());
      resident.flushDeferred();
    }

    const xCap = gpuInput(seq[WARMUP]);
    const xRaw = xCap._impl.storage.rawData;
    resident.deviceBufferForInput(xRaw);
    const inputDptr = resident.deviceBufferDptr(xRaw);

    let captured, lossTensor;
    beginEagerCapture();
    try {
      lossTensor = trainStep(Bm, xCap, tgtB, optB);
      captured = endEagerCapture();
    } catch (e) { try { endEagerCapture(); } catch (_) {} throw e; }

    const lossDptr = resident.deviceBufferDptr(lossTensor._impl.storage.rawData);
    const lossScratch = new Float32Array(1);

    const origLaunch = cu.launchKernel;
    let replayLaunches = 0;
    for (let s = WARMUP; s < STEPS; s++) {
      copyHostToDeviceAsync(inputDptr, seq[s]);
      if (s === STEPS - 1) cu.launchKernel = (...a) => { replayLaunches++; return origLaunch(...a); };
      replay(captured.exec);
      syncStream();
      if (s === STEPS - 1) cu.launchKernel = origLaunch;
      copyDeviceToHost(lossScratch, lossDptr);
      lossB.push(lossScratch[0]);
    }

    const finalB = Bm.params.map((p) => Float32Array.from(flat(p)));

    destroyEagerGraph(captured);
    resident.clearCapturePins();
    resident.flushDeferred();
    resident.setCudaGraphArmed(false);
    resident.setEagerDeferred(true);

    for (let s = 0; s < STEPS; s++) {
      const rel = Math.abs(lossA[s] - lossB[s]) / (1 + Math.abs(lossA[s]));
      expect(rel, `loss step ${s}: eager=${lossA[s]} graph=${lossB[s]}`).toBeLessThan(2e-3);
    }
    for (let i = 0; i < finalA.length; i++) {
      expect(maxAbsErr(finalA[i], finalB[i]), `final param ${i} eager vs graph`).toBeLessThan(2e-4);
    }
    expect(replayLaunches, 'graph replay issues no per-op cuLaunchKernel').toBe(0);
  }, 120000);
});

describe.skipIf(!cudaDeps || !cudnnAvailable())('CUDA eager whole-step graph with cuDNN LSTM', () => {
  let resident;
  beforeAll(async () => {
    await preloadCudaRuntime();
    getDevice();
    resident = await import('../../../src/runtime/node/cuda/resident.js');
  });

  const flat = (v) => Array.from(v && typeof v.contiguous === 'function' ? v.contiguous().data : v.data);

  it('captures LSTM fwd+bwd+Adam and replays a trajectory matching non-graphed eager', () => {
    const IN = 8, H = 8, LAYERS = 2, OUT = 4, B = 4, T = 5, STEPS = 6, WARMUP = 3, lr = 0.01;
    const seq = Array.from({ length: STEPS }, (_, s) =>
      Float32Array.from({ length: B * T * IN }, (_, i) => Math.sin(i * 0.19 + s * 1.3) * 0.4));
    const targetData = Float32Array.from({ length: B * OUT }, (_, i) => Math.cos(i * 0.17));

    const build = () => {
      const lstm = new LSTM(IN, H, LAYERS);
      const lin = new Linear(H, OUT);
      lstm.to(GPU_DEVICE); lin.to(GPU_DEVICE);
      return { lstm, lin, params: [...lstm.parameters(), ...lin.parameters()] };
    };
    const trainStep = (m, x, tgt, opt) => {
      for (const p of m.params) p.grad = null;
      const out = m.lstm.forward(x);
      const y = Array.isArray(out) ? out[0] : out;
      const lastH = y.select(1, T - 1);
      const pred = m.lin.forward(lastH);
      const diff = sub(pred, tgt);
      const loss = mean(mul(diff, diff));
      loss.backward();
      opt.step();
      return loss;
    };
    const gpuInput = (data) => tensor(Array.from(data), { shape: [B, T, IN] }).to(GPU_DEVICE);

    resident.setEagerDeferred(true);
    resident.setCudaGraphArmed(true);

    const A = build();
    const initParams = A.params.map((p) => Float32Array.from(flat(p)));
    const tgtA = tensor(Array.from(targetData), { shape: [B, OUT] }).to(GPU_DEVICE);
    const optA = new Adam(A.params, { lr });
    const lossA = [];
    for (let s = 0; s < STEPS; s++) {
      lossA.push(trainStep(A, gpuInput(seq[s]), tgtA, optA).item());
      resident.flushDeferred();
    }
    const finalA = A.params.map((p) => Float32Array.from(flat(p)));

    const Bm = build();
    Bm.params.forEach((p, i) => p._impl.storage.rawData.set(initParams[i]));
    const tgtB = tensor(Array.from(targetData), { shape: [B, OUT] }).to(GPU_DEVICE);
    const optB = new Adam(Bm.params, { lr });
    const lossB = [];
    for (let s = 0; s < WARMUP; s++) {
      lossB.push(trainStep(Bm, gpuInput(seq[s]), tgtB, optB).item());
      resident.flushDeferred();
    }

    const xCap = gpuInput(seq[WARMUP]);
    const xRaw = xCap._impl.storage.rawData;
    resident.deviceBufferForInput(xRaw);
    resident.pinResident(xRaw);
    const inputDptr = resident.deviceBufferDptr(xRaw);

    let captured, lossTensor;
    beginEagerCapture();
    try { lossTensor = trainStep(Bm, xCap, tgtB, optB); captured = endEagerCapture(); }
    catch (e) { try { endEagerCapture(); } catch (_) {} resident.clearCapturePins(); throw e; }

    const lossDptr = resident.deviceBufferDptr(lossTensor._impl.storage.rawData);
    const lossScratch = new Float32Array(1);
    for (let s = WARMUP; s < STEPS; s++) {
      copyHostToDeviceAsync(inputDptr, seq[s]);
      replay(captured.exec);
      syncStream();
      copyDeviceToHost(lossScratch, lossDptr);
      lossB.push(lossScratch[0]);
    }
    const finalB = Bm.params.map((p) => Float32Array.from(flat(p)));

    destroyEagerGraph(captured);
    resident.clearCapturePins();
    resident.flushDeferred();
    resident.setCudaGraphArmed(false);
    resident.setEagerDeferred(true);

    for (let s = 0; s < STEPS; s++) {
      const rel = Math.abs(lossA[s] - lossB[s]) / (1 + Math.abs(lossA[s]));
      expect(rel, `loss step ${s}: eager=${lossA[s]} graph=${lossB[s]}`).toBeLessThan(3e-3);
    }
    for (let i = 0; i < finalA.length; i++) {
      expect(maxAbsErr(finalA[i], finalB[i]), `final param ${i} eager vs graph`).toBeLessThan(1e-3);
    }
  }, 120000);
});

class MiniSeq2Seq extends LightningModule {
  constructor(vocab, embed, hidden, layers) {
    super();
    this.embed = new Embedding(vocab, embed);
    this.encoder = new LSTM(embed, hidden, layers, true);
    this.decoder = new LSTM(embed, hidden, layers, true);
    this.head = new Linear(2 * hidden, vocab);
    this.lossFn = new CrossEntropyLoss('sum', 0);
  }
  trainingStep(batch) {
    const [q, din, dout] = batch;
    const [enc, encState] = this.encoder.forward(this.embed.forward(q));
    const [dec] = this.decoder.forward(this.embed.forward(din), encState);
    const scores = dec.matmul(enc.transpose(1, 2));
    const context = scores.softmax(2).matmul(enc);
    const combined = cat([dec, context], 2);
    const bsz = q.shape[0], outLen = din.shape[1];
    const flat = combined.reshape([bsz * outLen, combined.shape[2]]);
    const logits = this.head.forward(flat);
    const logits3d = logits.reshape([bsz, outLen, logits.shape[1]]);
    let total = null;
    for (let t = 0; t < outLen; t++) {
      const l = this.lossFn.forward(logits3d.select(1, t), dout.select(1, t));
      total = total === null ? l : add(total, l);
    }
    const loss = div(total, bsz);
    this.log('train_loss', loss);
    return loss;
  }
  configureOptimizers() { return new Adam(this.parameters(), { lr: 0.004 }); }
}

describe.skipIf(!cudaDeps || !cudnnAvailable())('Trainer cudaGraph end-to-end: seq2seq (LSTM + attention + embedding + CE)', () => {
  beforeAll(async () => { await preloadCudaRuntime(); getDevice(); });
  const flat = (v) => Array.from(v && typeof v.contiguous === 'function' ? v.contiguous().data : v.data);
  const quiet = { logger: false, enableCheckpointing: false, enableProgress: false };

  function makeLoader(vocab, N, B, qlen, anslen) {
    const rnd = (n) => Array.from({ length: n }, (_, i) => 1 + ((i * 7 + 3) % (vocab - 1)));
    const q = tensor(rnd(N * qlen), { shape: [N, qlen], dtype: 'i32' });
    const din = tensor(rnd(N * anslen), { shape: [N, anslen], dtype: 'i32' });
    const dout = tensor(rnd(N * anslen), { shape: [N, anslen], dtype: 'i32' });
    return new DataLoader(new TensorDataset(q, din, dout), { batchSize: B });
  }

  it('captures the full seq2seq step and trains to match non-graphed eager', async () => {
    const VOCAB = 40, EMBED = 24, HIDDEN = 32, LAYERS = 2, B = 12, N = 48, QLEN = 5, ANSLEN = 6;
    const mA = new MiniSeq2Seq(VOCAB, EMBED, HIDDEN, LAYERS);
    const mB = new MiniSeq2Seq(VOCAB, EMBED, HIDDEN, LAYERS);
    [...mB.parameters()].forEach((p, i) => p._impl.storage.rawData.set([...mA.parameters()][i]._impl.storage.rawData));

    const tA = new Trainer({ accelerator: 'gpu', maxEpochs: 2, ...quiet, cudaGraph: false });
    await tA.fit(mA, makeLoader(VOCAB, N, B, QLEN, ANSLEN));
    const tB = new Trainer({ accelerator: 'gpu', maxEpochs: 2, ...quiet, cudaGraph: true, cudaGraphWarmupSteps: 3 });
    await tB.fit(mB, makeLoader(VOCAB, N, B, QLEN, ANSLEN));

    expect(mB._cudaGraphPhase, 'seq2seq capture succeeded').toBe('replay');
    const pa = [...mA.parameters()].map((p) => Float32Array.from(flat(p)));
    const pb = [...mB.parameters()].map((p) => Float32Array.from(flat(p)));
    for (let i = 0; i < pa.length; i++) {
      expect(maxAbsErr(pa[i], pb[i]), `seq2seq param ${i} eager vs graph`).toBeLessThan(2e-3);
    }
  }, 180000);
});

class RegMLP extends LightningModule {
  constructor(inF, hidF) {
    super();
    this.l1 = new Linear(inF, hidF);
    this.l2 = new Linear(hidF, 1);
    this.lossFn = new MSELoss();
  }
  forward(x) { return this.l2.forward(relu(this.l1.forward(x))); }
  trainingStep(batch) {
    const [x, y] = batch;
    const loss = this.lossFn.forward(this.forward(x), y);
    this.log('train_loss', loss);
    return loss;
  }
  configureOptimizers() { return new Adam(this.parameters(), { lr: 0.01 }); }
}

describe.skipIf(!cudaDeps)('Trainer(accelerator="gpu", cudaGraph=true) end-to-end', () => {
  beforeAll(async () => { await preloadCudaRuntime(); getDevice(); });

  const flat = (v) => Array.from(v && typeof v.contiguous === 'function' ? v.contiguous().data : v.data);
  const quiet = { logger: false, enableCheckpointing: false, enableProgress: false };

  function makeLoader(IN, N, B) {
    const xData = Float32Array.from({ length: N * IN }, (_, i) => Math.sin(i * 0.17) * 0.5);
    const yData = Float32Array.from({ length: N }, (_, i) => Math.cos(i * 0.23));
    const x = tensor(Array.from(xData), { shape: [N, IN] });
    const y = tensor(Array.from(yData), { shape: [N, 1] });
    return new DataLoader(new TensorDataset(x, y), { batchSize: B });
  }

  it('matches non-graphed eager training (final params) and exercises the graph', async () => {
    const IN = 12, HID = 16, B = 10, N = 40;

    const mA = new RegMLP(IN, HID);
    const mB = new RegMLP(IN, HID);
    const pa0 = [...mA.parameters()], pb0 = [...mB.parameters()];
    pb0.forEach((p, i) => p._impl.storage.rawData.set(pa0[i]._impl.storage.rawData));

    const tA = new Trainer({ accelerator: 'gpu', maxEpochs: 2, ...quiet, cudaGraph: false });
    await tA.fit(mA, makeLoader(IN, N, B));

    const tB = new Trainer({ accelerator: 'gpu', maxEpochs: 2, ...quiet, cudaGraph: true, cudaGraphWarmupSteps: 3 });
    await tB.fit(mB, makeLoader(IN, N, B));

    expect(mB._cudaGraphPhase, 'graph capture succeeded').toBe('replay');

    const pa = [...mA.parameters()].map((p) => Float32Array.from(flat(p)));
    const pb = [...mB.parameters()].map((p) => Float32Array.from(flat(p)));
    for (let i = 0; i < pa.length; i++) {
      expect(maxAbsErr(pa[i], pb[i]), `final param ${i} eager vs graph`).toBeLessThan(1e-3);
    }
  }, 120000);

  it('rejects incompatible configs', async () => {
    const m = new RegMLP(8, 8);
    const bad = new Trainer({ accelerator: 'cpu', maxEpochs: 1, ...quiet, cudaGraph: true });
    await expect(bad.fit(m, makeLoader(8, 20, 10))).rejects.toThrow(/accelerator="gpu"/);

    const m2 = new RegMLP(8, 8);
    const bad2 = new Trainer({ accelerator: 'gpu', maxEpochs: 1, ...quiet, cudaGraph: true, gradientClipVal: 1.0, gradientClipAlgorithm: 'value' });
    await expect(bad2.fit(m2, makeLoader(8, 20, 10))).rejects.toThrow(/gradient_clip_algorithm/);
  }, 120000);

  it('device-side grad clipping applies the correct global-norm coefficient', async () => {
    await preloadCudaRuntime();
    const resident = await import('../../../src/runtime/node/cuda/resident.js');
    resident.setEagerDeferred(true);
    const IN = 8, H = 16, OUT = 4, B = 6;
    const l1 = new Linear(IN, H), l2 = new Linear(H, OUT);
    l1.to(GPU_DEVICE); l2.to(GPU_DEVICE);
    const params = [...l1.parameters(), ...l2.parameters()];
    const x = tensor(Array.from({ length: B * IN }, (_, i) => Math.sin(i * 0.3)), { shape: [B, IN] }).to(GPU_DEVICE);
    const tgt = tensor(Array.from({ length: B * OUT }, (_, i) => Math.cos(i * 0.2)), { shape: [B, OUT] }).to(GPU_DEVICE);

    for (const p of params) p.grad = null;
    const pred = l2.forward(relu(l1.forward(x)));
    const d = sub(pred, tgt);
    mean(mul(d, d)).backward();

    const before = params.map((p) => flat(p.grad));
    let sumsq = 0;
    for (const g of before) for (const v of g) sumsq += v * v;
    const norm = Math.sqrt(sumsq), maxNorm = norm * 0.5;
    const expected = Math.min(1, maxNorm / (norm + 1e-6));

    deviceClipGradNorm(params, maxNorm);
    const after = params.map((p) => flat(p.grad));
    let maxErr = 0;
    for (let i = 0; i < before.length; i++)
      for (let j = 0; j < before[i].length; j++)
        if (Math.abs(before[i][j]) > 1e-6) maxErr = Math.max(maxErr, Math.abs(after[i][j] / before[i][j] - expected));

    resident.flushDeferred();
    resident.setEagerDeferred(true);
    expect(expected, 'clip is active (coef < 1)').toBeLessThan(1);
    expect(maxErr, 'device coef vs host expected coef').toBeLessThan(1e-4);
  }, 120000);

  it('graphs norm-clipping in the captured step and trains to match eager (integer clip val)', async () => {
    const IN = 12, HID = 16, B = 10, N = 40;
    const mA = new RegMLP(IN, HID);
    const mB = new RegMLP(IN, HID);
    [...mB.parameters()].forEach((p, i) => p._impl.storage.rawData.set([...mA.parameters()][i]._impl.storage.rawData));

    const tA = new Trainer({ accelerator: 'gpu', maxEpochs: 2, ...quiet, cudaGraph: false, gradientClipVal: 1000 });
    await tA.fit(mA, makeLoader(IN, N, B));
    const tB = new Trainer({ accelerator: 'gpu', maxEpochs: 2, ...quiet, cudaGraph: true, cudaGraphWarmupSteps: 3, gradientClipVal: 1000 });
    await tB.fit(mB, makeLoader(IN, N, B));

    expect(mB._cudaGraphPhase, 'capture succeeded with clipping in graph').toBe('replay');
    const pa = [...mA.parameters()].map((p) => Float32Array.from(flat(p)));
    const pb = [...mB.parameters()].map((p) => Float32Array.from(flat(p)));
    for (let i = 0; i < pa.length; i++) {
      expect(maxAbsErr(pa[i], pb[i]), `clipped param ${i} eager vs graph`).toBeLessThan(2e-3);
    }
  }, 120000);
});
