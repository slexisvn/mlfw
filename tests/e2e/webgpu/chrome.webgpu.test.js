import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { mulberry32 } from '../../_utils/rng.js';
import { createServer } from 'http';
import { existsSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';

const PROJECT_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');

function findChrome() {
  if (process.env.CHROME_PATH && existsSync(process.env.CHROME_PATH)) return process.env.CHROME_PATH;
  const candidates = [
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/usr/bin/google-chrome',
    '/usr/bin/chromium-browser',
    '/usr/bin/chromium',
  ];
  for (const c of candidates) if (existsSync(c)) return c;
  return null;
}

async function tryLoad() {
  const chromePath = findChrome();
  if (!chromePath) return null;
  let puppeteer, esbuild;
  try {
    puppeteer = (await import('puppeteer-core')).default;
    esbuild = await import('esbuild');
  } catch (_) {
    return null;
  }
  return { chromePath, puppeteer, esbuild };
}

const deps = await tryLoad();

const PAGE_RUNNER = `
import * as M from './bundle.js';
const { tensor, add, sub, mul, div, matmul, relu, sum, mean, max, min, softmax,
  sigmoid, tanh, exp, log, sqrt, abs, neg, where, gt, clamp, compile, CPUTarget, WebGPUTarget, layer_norm } = M;
function flat(t){ return Array.from(t.contiguous ? t.contiguous().data : t.data); }
const UNARY={relu,sigmoid,tanh,exp,abs,neg,
  sqrt:(x)=>sqrt(add(abs(x),tensor(0.5))),log:(x)=>log(add(abs(x),tensor(1))),clamp:(x)=>clamp(x,-0.5,0.5)};
const BINARY={add,sub,mul,div:(a,b)=>div(a,add(abs(b),tensor(1))),max:(a,b)=>where(gt(a,b),a,b)};
const REDUCE={sum,mean,max,min};
function runProg(ops,xs){let cur=xs[0];let bi=1;for(const o of ops){
  if(o.k==='u')cur=UNARY[o.op](cur);
  else if(o.k==='b')cur=BINARY[o.op](cur,xs[bi++%xs.length]);
  else if(o.k==='r'){const d=Math.min(o.dim,cur.shape.length-1);cur=REDUCE[o.op](cur,d,o.keep);}
  else if(o.k==='sm')cur=softmax(cur,cur.shape.length-1);
  else if(o.k==='t'){if(cur.shape.length>=2)cur=cur.transpose(0,1);}
  else if(o.k==='mm')cur=matmul(cur,xs[bi++%xs.length]);
  else if(o.k==='ln'){const n=cur.shape[cur.shape.length-1];cur=layer_norm(cur,[n],null,null,1e-5);}
}return cur;}
window.runSpec = async (spec) => {
  const xs = spec.inputs.map(a=>tensor(a));
  const out = { name: spec.name };
  try { const cc = compile({forward:(...a)=>runProg(spec.ops,a)}, xs, {target:CPUTarget()}); out.cpu = flat(await cc(...xs)); }
  catch(e){ out.cpuErr = String(e && e.message || e); }
  try { const gc = compile({forward:(...a)=>runProg(spec.ops,a)}, xs, {target:WebGPUTarget()}); out.gpu = flat(await gc(...xs)); }
  catch(e){ out.gpuErr = String(e && e.message || e); }
  return out;
};
window.runCase = async (src, inputs, config) => {
  const fn = eval('('+src+')');
  const xs = inputs.map(s => tensor(s.data, s.dtype ? { dtype: s.dtype } : undefined));
  const arr = (t)=>Array.from((t.contiguous ? t.contiguous() : t).data, v => typeof v === 'bigint' ? Number(v) : v);
  const out = {};
  try { const c = compile({forward:(...a)=>fn(M,...a)}, xs, {target:CPUTarget()}); out.cpu = arr(await c(...xs)); } catch(e){ out.cpuErr = String(e && e.message || e); }
  try { const g = compile({forward:(...a)=>fn(M,...a)}, xs, {target:WebGPUTarget(), ...(config || {})}); out.gpu = arr(await g(...xs)); } catch(e){ out.gpuErr = String(e && e.message || e); }
  return out;
};
window.runSequential = async (layers, inputs) => {
  const mk = (spec) => {
    if (spec.t === 'linear') return new M.Linear(spec.i, spec.o);
    if (spec.t === 'relu') return new M.ReLU();
    if (spec.t === 'sigmoid') return new M.Sigmoid();
    return new M.Tanh();
  };
  const model = new M.Sequential(...layers.map(mk));
  const xs = inputs.map(d => tensor(d));
  const arr = (t) => Array.from((t.contiguous ? t.contiguous() : t).data);
  const out = {};
  try { out.eager = arr(model.forward(xs[0])); } catch(e) { out.eagerErr = String(e && e.message || e); }
  try {
    const g = compile(model, [xs[0]], { target: WebGPUTarget() });
    out.runs = [];
    for (const x of xs) { const r = await g(x); out.runs.push(arr(r)); out.shape = r.shape; }
  } catch(e) { out.gpuErr = String(e && e.message || e); }
  return out;
};
window.runRNN = async (kind, E, H, T, B, seed) => {
  let a = seed >>> 0;
  const r = () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const numel = s => s.reduce((x, y) => x * y, 1);
  const nest = (f, s) => { if (s.length === 1) return f.slice(0, s[0]); const sub = numel(s.slice(1)); const o = []; for (let i = 0; i < s[0]; i++) o.push(nest(f.slice(i * sub, (i + 1) * sub), s.slice(1))); return o; };
  const Mod = kind === 'gru' ? M.GRU : M.LSTM;
  const m = new Mod(E, H, 1, true);
  const f = []; for (let i = 0; i < B * T * E; i++) f.push(r() * 2 - 1);
  const inp = tensor(nest(f, [B, T, E]));
  const wrap = { forward: (...a) => m.forward(...a)[0] };
  const out = {};
  try { const c = compile(wrap, [inp], { target: CPUTarget() }); out.cpu = flat(await c(inp)); } catch (e) { out.cpuErr = String(e && e.message || e); }
  try { const g = compile(wrap, [inp], { target: WebGPUTarget() }); out.gpu = flat(await g(inp)); } catch (e) { out.gpuErr = String(e && e.message || e); }
  return out;
};
window.runRNNLinear = async (kind, E, H, V, T, B, seed) => {
  let a = seed >>> 0;
  const r = () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const numel = s => s.reduce((x, y) => x * y, 1);
  const nest = (f, s) => { if (s.length === 1) return f.slice(0, s[0]); const sub = numel(s.slice(1)); const o = []; for (let i = 0; i < s[0]; i++) o.push(nest(f.slice(i * sub, (i + 1) * sub), s.slice(1))); return o; };
  const m = new (kind === 'gru' ? M.GRU : M.LSTM)(E, H, 1, true);
  const lin = new M.Linear(H, V);
  const f = []; for (let i = 0; i < B * T * E; i++) f.push(r() * 2 - 1);
  const inp = tensor(nest(f, [B, T, E]));
  const wrap = { forward: (xx) => lin.forward(m.forward(xx)[0]) };
  const out = {};
  try { const c = compile(wrap, [inp], { target: CPUTarget() }); out.cpu = flat(await c(inp)); } catch (e) { out.cpuErr = String(e && e.message || e); }
  try { const g = compile(wrap, [inp], { target: WebGPUTarget() }); out.gpu = flat(await g(inp)); } catch (e) { out.gpuErr = String(e && e.message || e); }
  return out;
};
const seededRng = (seed) => { let a = seed >>> 0; return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; };
const nestArr = (f, s) => { if (s.length === 1) return f.slice(0, s[0]); const sub = s.slice(1).reduce((x, y) => x * y, 1); const o = []; for (let i = 0; i < s[0]; i++) o.push(nestArr(f.slice(i * sub, (i + 1) * sub), s.slice(1))); return o; };
const readDevice = (t) => Array.from(t._impl.storage.rawData.subarray(0, t.numel));
window.runEagerLinear = async (inF, outF, B, seed) => {
  await M.preloadWebGPU();
  const r = seededRng(seed);
  const xf = []; for (let i = 0; i < B * inF; i++) xf.push(r() * 2 - 1);
  const x = tensor(nestArr(xf, [B, inF]));
  const lin = new M.Linear(inF, outF);
  const cpu = flat(lin.forward(x));
  lin.to(M.WEBGPU_DEVICE);
  const og = lin.forward(x.to(M.WEBGPU_DEVICE));
  await M.flushWebGPUEager();
  return { cpu, gpu: readDevice(og) };
};
window.runEagerLSTM = async (kind, E, H, T, B, seed) => {
  await M.preloadWebGPU();
  const r = seededRng(seed);
  const f = []; for (let i = 0; i < B * T * E; i++) f.push(r() * 2 - 1);
  const inp = tensor(nestArr(f, [B, T, E]));
  const m = new (kind === 'gru' ? M.GRU : M.LSTM)(E, H, 1, true);
  const cpu = flat(sum(m.forward(inp)[0], 2));
  m.to(M.WEBGPU_DEVICE);
  const red = sum(m.forward(inp.to(M.WEBGPU_DEVICE))[0], 2);
  await M.flushWebGPUEager();
  return { cpu, gpu: readDevice(red) };
};
window.runFusedLSTM = async (E, H, T, seed) => {
  await M.preloadWebGPU();
  const r = seededRng(seed);
  const f = []; for (let i = 0; i < T * E; i++) f.push(r() * 2 - 1);
  const inp = tensor(nestArr(f, [1, T, E]));
  const m = new M.LSTM(E, H, 1, true);
  const cpu = flat(m.forward(inp)[0]);
  m.to(M.WEBGPU_DEVICE);
  const og = m.forward(inp.to(M.WEBGPU_DEVICE))[0];
  await M.flushWebGPUEager();
  return { cpu, gpu: readDevice(og) };
};
window.runFusedGRU = async (E, H, T, L, seed) => {
  await M.preloadWebGPU();
  const r = seededRng(seed);
  const f = []; for (let i = 0; i < T * E; i++) f.push(r() * 2 - 1);
  const inp = tensor(nestArr(f, [1, T, E]));
  const m = new M.GRU(E, H, L, true);
  const cpuOut = m.forward(inp);
  const cpu = flat(cpuOut[0]);
  const cpuH = flat(cpuOut[1]);
  m.to(M.WEBGPU_DEVICE);
  const og = m.forward(inp.to(M.WEBGPU_DEVICE));
  await M.flushWebGPUEager();
  return { cpu, gpu: readDevice(og[0]), cpuH, gpuH: readDevice(og[1]) };
};
window.runTrainerPredict = async (arch, inF, outF, B, nb, seed) => {
  await M.preloadWebGPU();
  const r = seededRng(seed);
  class Net extends M.lightning.LightningModule {
    constructor() {
      super();
      if (arch === 'lstm') { this.rnn = new M.LSTM(inF, 16, 1, true); this.head = new M.Linear(16, outF); }
      else { this.body = new M.Linear(inF, 16); this.head = new M.Linear(16, outF); }
    }
    forward(x) {
      if (arch === 'lstm') return this.head.forward(sum(this.rnn.forward(x)[0], 1));
      return this.head.forward(M.relu(this.body.forward(x)));
    }
    configureOptimizers() { return null; }
  }
  const net = new Net();
  const shape = arch === 'lstm' ? [B, 4, inF] : [B, inF];
  const per = shape.reduce((a, b) => a * b, 1);
  const batches = [];
  for (let bi = 0; bi < nb; bi++) { const xf = []; for (let i = 0; i < per; i++) xf.push(r() * 2 - 1); batches.push(tensor(nestArr(xf, shape))); }
  const cpu = batches.map(x => flat(net.forward(x)));
  const loader = { length: batches.length, [Symbol.iterator]() { let i = 0; return { next: () => i < batches.length ? { value: batches[i++], done: false } : { value: undefined, done: true } }; } };
  const trainer = new M.lightning.Trainer({ accelerator: 'webgpu', logger: false, enableProgress: false });
  const preds = await trainer.predict(net, loader);
  return { cpu, gpu: preds.map(readDevice) };
};
window.runSeq2Seq = async (V, E, H, Lay, T, A, seed) => {
  const r = seededRng(seed);
  const emb = new M.Embedding(V, E), enc = new M.LSTM(E, H, Lay, true), dec = new M.LSTM(E, H, Lay, true), head = new M.Linear(2 * H, V);
  const qd = []; for (let i = 0; i < T; i++) qd.push(Math.floor(r() * V));
  const dd = []; for (let i = 0; i < A; i++) dd.push(Math.floor(r() * V));
  const q = tensor([qd], { dtype: 'i32' }), din = tensor([dd], { dtype: 'i32' });
  const fwd = (qq, d2) => {
    const e = enc.forward(emb.forward(qq))[0];
    const d = dec.forward(emb.forward(d2))[0];
    const w = softmax(matmul(d, e.transpose(1, 2)), 2);
    const cm = M.cat([d, matmul(w, e)], 2);
    return head.forward(cm.reshape([A, 2 * H]));
  };
  const wrap = { forward: (qq, d2) => fwd(qq, d2) };
  const out = {};
  try { const c = compile(wrap, [q, din], { target: CPUTarget() }); out.cpu = flat(await c(q, din)); } catch (e) { out.cpuErr = String(e && e.message || e); }
  try { const g = compile(wrap, [q, din], { target: WebGPUTarget() }); out.gpu = flat(await g(q, din)); } catch (e) { out.gpuErr = String(e && e.message || e); }
  return out;
};
window.__gpu = !!navigator.gpu;
window.__ready = true;
`;

const PAGE_HTML = `<!doctype html><html><head><meta charset="utf-8"></head><body>
<script type="module" src="./runner.js"></script></body></html>`;

function numel(s) { return s.reduce((a, b) => a * b, 1); }
function nest(f, s) { if (s.length === 1) return f.slice(0, s[0]); const sub = numel(s.slice(1)); const o = []; for (let i = 0; i < s[0]; i++) o.push(nest(f.slice(i * sub, (i + 1) * sub), s.slice(1))); return o; }
function mkArr(rng, shape, lo, hi) { const n = numel(shape); const f = []; for (let i = 0; i < n; i++) f.push(lo + (hi - lo) * rng()); return nest(f, shape); }

describe.skipIf(!deps)('webgpu via Chrome (differential vs CPU)', () => {
  let browser, server, page, port, bundleCode, hasGpu;

  beforeAll(async () => {
    const built = await deps.esbuild.build({
      stdin: {
        contents: "export * from './src/index.js';\n"
          + "export { compile } from './src/tracing/compile.js';\n"
          + "export { CPUTarget, WebGPUTarget } from './src/backend/target.js';\n"
          + "export { layer_norm } from './src/nn/functional/normalization.js';\n"
          + "export { conv2d } from './src/nn/functional/conv.js';\n"
          + "export { max_pool2d, avg_pool2d } from './src/nn/functional/pooling.js';\n"
          + "export { split } from './src/tensor/ops/ops.js';\n"
          + "export { WEBGPU_DEVICE } from './src/tensor/types/device.js';\n"
          + "export { preloadWebGPU } from './src/runtime/backend_registry.js';\n"
          + "export { flushWebGPUEager } from './src/runtime/webgpu.js';\n",
        resolveDir: PROJECT_ROOT,
        loader: 'js',
      },
      bundle: true, format: 'esm', platform: 'browser', conditions: ['browser'],
      external: ['webgpu'], write: false,
    });
    bundleCode = built.outputFiles[0].text;

    server = createServer((req, res) => {
      const p = req.url.split('?')[0];
      if (p === '/' || p === '/page.html') { res.writeHead(200, { 'content-type': 'text/html' }); res.end(PAGE_HTML); }
      else if (p === '/runner.js') { res.writeHead(200, { 'content-type': 'text/javascript' }); res.end(PAGE_RUNNER); }
      else if (p === '/bundle.js') { res.writeHead(200, { 'content-type': 'text/javascript' }); res.end(bundleCode); }
      else { res.writeHead(404); res.end('nf'); }
    });
    await new Promise(r => server.listen(0, r));
    port = server.address().port;

    browser = await deps.puppeteer.launch({
      executablePath: deps.chromePath, headless: 'new',
      args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--no-sandbox', '--use-angle=default'],
    });
    page = await browser.newPage();
    await page.goto(`http://localhost:${port}/page.html`, { waitUntil: 'networkidle0' });
    await page.waitForFunction('window.__ready===true', { timeout: 20000 });
    hasGpu = await page.evaluate('window.__gpu');
  }, 60000);

  afterAll(async () => {
    if (browser) await browser.close();
    if (server) await new Promise(r => server.close(r));
  });

  async function diff(spec, tol = 3e-3) {
    const res = await page.evaluate((sp) => window.runSpec(sp), spec);
    expect(res.cpuErr, `cpu error: ${res.cpuErr}`).toBeUndefined();
    expect(res.gpuErr, `gpu error: ${res.gpuErr}`).toBeUndefined();
    expect(res.gpu.length).toBe(res.cpu.length);
    let maxErr = 0, bad = -1;
    for (let i = 0; i < res.cpu.length; i++) {
      const e = Math.abs(res.cpu[i] - res.gpu[i]) / (1 + Math.abs(res.cpu[i]));
      if (e > maxErr) { maxErr = e; bad = i; }
    }
    expect(maxErr, `${spec.name} idx ${bad}: cpu=${res.cpu[bad]} gpu=${res.gpu[bad]}`).toBeLessThan(tol);
    return res;
  }

  it('Chrome exposes WebGPU', () => {
    expect(hasGpu).toBe(true);
  });

  it('forward ignoring extra inputs (unused-binding pruning)', async () => {
    const res = await diff({
      name: 'unused',
      ops: [{ k: 'u', op: 'relu' }, { k: 'r', op: 'sum', dim: 1, keep: false }],
      inputs: [[[1, -2, 3], [-4, 5, -6]], [[9, 9, 9], [9, 9, 9]], [[7, 7, 7], [7, 7, 7]]],
    });
    expect(res.gpu).toEqual([4, 5]);
  });

  it('deep fused kernel: layernorm + softmax chain', async () => {
    await diff({
      name: 'deep_ln',
      ops: [{ k: 'ln' }, { k: 'sm' }, { k: 'ln' }, { k: 'sm' }, { k: 'ln' }, { k: 'sm' }, { k: 'ln' }, { k: 'sm' }],
      inputs: [mkArr(mulberry32(1), [4, 4], -1, 1)],
    });
  });

  it('matmul + norm + softmax chain', async () => {
    const ins = [mkArr(mulberry32(2), [3, 3], -1, 1), mkArr(mulberry32(3), [3, 3], -1, 1), mkArr(mulberry32(4), [3, 3], -1, 1)];
    await diff({ name: 'mm_chain', ops: [{ k: 'mm' }, { k: 'mm' }, { k: 'ln' }, { k: 'mm' }, { k: 'sm' }, { k: 'mm' }], inputs: ins });
  });

  async function caseEq(src, inputs, config) {
    const res = await page.evaluate((s, ins, c) => window.runCase(s, ins, c), src, inputs, config || null);
    expect(res.cpuErr, `cpu error: ${res.cpuErr}`).toBeUndefined();
    expect(res.gpuErr, `gpu error: ${res.gpuErr}`).toBeUndefined();
    expect(res.gpu).toEqual(res.cpu);
    return res;
  }

  async function caseClose(src, inputs, config, tol = 3e-3) {
    const res = await page.evaluate((s, ins, c) => window.runCase(s, ins, c), src, inputs, config || null);
    expect(res.cpuErr, `cpu error: ${res.cpuErr}`).toBeUndefined();
    expect(res.gpuErr, `gpu error: ${res.gpuErr}`).toBeUndefined();
    let maxErr = 0;
    for (let i = 0; i < res.cpu.length; i++) maxErr = Math.max(maxErr, Math.abs(res.cpu[i] - res.gpu[i]) / (1 + Math.abs(res.cpu[i])));
    expect(maxErr, `maxErr=${maxErr}`).toBeLessThan(tol);
    return res;
  }

  async function seq(layers, inputs) {
    const res = await page.evaluate((l, i) => window.runSequential(l, i), layers, inputs);
    expect(res.eagerErr, `eager error: ${res.eagerErr}`).toBeUndefined();
    expect(res.gpuErr, `gpu error: ${res.gpuErr}`).toBeUndefined();
    return res;
  }

  const closeTo = (a, b) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);

  it('Linear forward [1,4]->[1,2] matches eager', async () => {
    const res = await seq([{ t: 'linear', i: 4, o: 2 }], [[[1, 2, 3, 4]]]);
    expect(res.shape).toEqual([1, 2]);
    expect(closeTo(res.runs[0], res.eager)).toBeLessThan(1e-3);
  });

  it('Linear + ReLU clamps negatives', async () => {
    const res = await seq([{ t: 'linear', i: 4, o: 4 }, { t: 'relu' }], [[[1, -1, 2, -2]]]);
    expect(Math.min(...res.runs[0])).toBeGreaterThanOrEqual(0);
    expect(closeTo(res.runs[0], res.eager)).toBeLessThan(1e-3);
  });

  it('Linear + Sigmoid bounds [0,1]', async () => {
    const res = await seq([{ t: 'linear', i: 3, o: 3 }, { t: 'sigmoid' }], [[[10, -10, 0]]]);
    expect(Math.min(...res.runs[0])).toBeGreaterThanOrEqual(0);
    expect(Math.max(...res.runs[0])).toBeLessThanOrEqual(1);
    expect(closeTo(res.runs[0], res.eager)).toBeLessThan(1e-3);
  });

  it('Linear + Tanh bounds [-1,1]', async () => {
    const res = await seq([{ t: 'linear', i: 3, o: 3 }, { t: 'tanh' }], [[[5, -5, 0]]]);
    expect(Math.min(...res.runs[0])).toBeGreaterThanOrEqual(-1);
    expect(Math.max(...res.runs[0])).toBeLessThanOrEqual(1);
    expect(closeTo(res.runs[0], res.eager)).toBeLessThan(1e-3);
  });

  it('repeated runs on the same input are bit-identical', async () => {
    const res = await seq([{ t: 'linear', i: 4, o: 2 }], [[[1, 2, 3, 4]], [[1, 2, 3, 4]]]);
    expect(res.runs[1]).toEqual(res.runs[0]);
  });

  it('different inputs produce different outputs', async () => {
    const res = await seq([{ t: 'linear', i: 4, o: 2 }], [[[1, 0, 0, 0]], [[0, 0, 0, 1]]]);
    expect(res.runs[1]).not.toEqual(res.runs[0]);
  });

  it('zero input produces finite output', async () => {
    const res = await seq([{ t: 'linear', i: 4, o: 2 }], [[[0, 0, 0, 0]]]);
    expect(res.runs[0].every(Number.isFinite), `non-finite in ${JSON.stringify(res.runs[0])}`).toBe(true);
  });

  it('3-layer MLP executes correctly', async () => {
    const res = await seq(
      [{ t: 'linear', i: 8, o: 16 }, { t: 'relu' }, { t: 'linear', i: 16, o: 8 }, { t: 'relu' }, { t: 'linear', i: 8, o: 4 }],
      [[[1, 2, 3, 4, 5, 6, 7, 8]]],
    );
    expect(res.shape).toEqual([1, 4]);
    expect(res.runs[0].every(Number.isFinite), `non-finite in ${JSON.stringify(res.runs[0])}`).toBe(true);
    expect(closeTo(res.runs[0], res.eager)).toBeLessThan(1e-3);
  });

  it('chained reduces with unused inputs match CPU (multi-kernel buffers)', async () => {
    await caseClose(
      '(M, a, b, c) => M.neg(M.mean(M.neg(M.mean(a, 1, false)), 1, false))',
      [
        { data: [[0.5, -0.3], [0.2, 0.9]] },
        { data: [[0.1, 0.4], [-0.2, 0.7]] },
        { data: [[0.3, 0.1], [0.6, -0.5]] },
      ],
      null,
      1e-4,
    );
  });

  it('scheduled+autotuned conv2d + relu + maxpool matches CPU (multi-workgroup serialized)', async () => {
    const mk = (seed, shape) => {
      const n = numel(shape);
      const d = [];
      for (let i = 0; i < n; i++) d.push(Math.abs(Math.sin(i * 1.3 + seed)) * 2 - 0.5);
      return nest(d, shape);
    };
    await caseClose(
      '(M, x, w) => M.max_pool2d(M.relu(M.conv2d(x, w, null, [1, 1], [1, 1], [1, 1], 1)), [2, 2], [2, 2], [0, 0])',
      [{ data: mk(1, [1, 2, 8, 8]) }, { data: mk(2, [3, 2, 3, 3]) }],
      { scheduling: { enabled: true, autotune: true } },
      3e-3,
    );
  });

  it('large RNN per-step dispatch matches CPU (carry ping-pong + scalar-replaced fused kernels)', async () => {
    for (const [kind, E, H, T, B] of [['lstm', 32, 64, 8, 32], ['gru', 32, 64, 6, 32]]) {
      const res = await page.evaluate((k, e, h, t, b) => window.runRNN(k, e, h, t, b, 123), kind, E, H, T, B);
      expect(res.cpuErr, `${kind} cpu error: ${res.cpuErr}`).toBeUndefined();
      expect(res.gpuErr, `${kind} gpu error: ${res.gpuErr}`).toBeUndefined();
      expect(res.gpu.length).toBe(res.cpu.length);
      let maxErr = 0;
      for (let i = 0; i < res.cpu.length; i++) maxErr = Math.max(maxErr, Math.abs(res.cpu[i] - res.gpu[i]) / (1 + Math.abs(res.cpu[i])));
      expect(maxErr, `${kind} maxErr=${maxErr}`).toBeLessThan(3e-3);
    }
  }, 60000);

  it('compiled RNN -> Linear matches CPU at B=1 (scan->matmul split preserves recurrence)', async () => {
    for (const kind of ['lstm', 'gru']) {
      const res = await page.evaluate((k) => window.runRNNLinear(k, 64, 128, 500, 20, 1, 99), kind);
      expect(res.cpuErr, `${kind} cpu error: ${res.cpuErr}`).toBeUndefined();
      expect(res.gpuErr, `${kind} gpu error: ${res.gpuErr}`).toBeUndefined();
      expect(res.gpu.length).toBe(res.cpu.length);
      let maxErr = 0;
      for (let i = 0; i < res.cpu.length; i++) maxErr = Math.max(maxErr, Math.abs(res.cpu[i] - res.gpu[i]) / (1 + Math.abs(res.cpu[i])));
      expect(maxErr, `${kind} maxErr=${maxErr}`).toBeLessThan(3e-3);
    }
  }, 60000);

  it('compiled RNN -> Linear matches CPU across batch (scan->matmul split, recurrence + no private-mem overflow)', async () => {
    for (const [kind, B] of [['lstm', 1], ['lstm', 16], ['lstm', 64], ['gru', 1], ['gru', 16], ['gru', 64]]) {
      const res = await page.evaluate((k, b) => window.runRNNLinear(k, 64, 128, 500, 20, b, 99), kind, B);
      expect(res.cpuErr, `${kind} B${B} cpu error: ${res.cpuErr}`).toBeUndefined();
      expect(res.gpuErr, `${kind} B${B} gpu error: ${res.gpuErr}`).toBeUndefined();
      expect(res.gpu.length).toBe(res.cpu.length);
      let maxErr = 0;
      for (let i = 0; i < res.cpu.length; i++) maxErr = Math.max(maxErr, Math.abs(res.cpu[i] - res.gpu[i]) / (1 + Math.abs(res.cpu[i])));
      expect(maxErr, `${kind} B${B} maxErr=${maxErr}`).toBeLessThan(3e-3);
    }
  }, 120000);

  it('multi-scan seq2seq (2L encoder + decoder + attention) compile matches CPU (N-scan per-step plan, the chatbot bug)', async () => {
    if (!hasGpu) return;
    const res = await page.evaluate(() => window.runSeq2Seq(40, 16, 300, 2, 4, 3, 7));
    expect(res.cpuErr, `cpu error: ${res.cpuErr}`).toBeUndefined();
    expect(res.gpuErr, `gpu error: ${res.gpuErr}`).toBeUndefined();
    expect(res.gpu.length).toBe(res.cpu.length);
    let maxErr = 0, bad = -1;
    for (let i = 0; i < res.cpu.length; i++) {
      const e = Math.abs(res.cpu[i] - res.gpu[i]) / (1 + Math.abs(res.cpu[i]));
      if (e > maxErr) { maxErr = e; bad = i; }
    }
    expect(maxErr, `seq2seq idx ${bad}: cpu=${res.cpu[bad]} gpu=${res.gpu[bad]}`).toBeLessThan(3e-3);
  }, 120000);

  it('i32 reduce and elementwise', async () => {
    const r = await caseEq("(M,x)=>M.sum(x,1)", [{ data: [[1, 2, 3], [4, 5, 6]], dtype: 'i32' }]);
    expect(r.gpu).toEqual([6, 15]);
    await caseEq("(M,a,b)=>M.mul(a,b)", [{ data: [[1, 2], [3, 4]], dtype: 'i32' }, { data: [[5, 6], [7, 8]], dtype: 'i32' }]);
  });

  it('narrow integer dtypes (i16/i8/ui8) with wraparound', async () => {
    await caseEq("(M,a,b)=>M.add(a,b)", [{ data: [[10, 20]], dtype: 'i16' }, { data: [[1, 2]], dtype: 'i16' }]);
    await caseEq("(M,a,b)=>M.add(a,b)", [{ data: [[1, 2]], dtype: 'i8' }, { data: [[3, 4]], dtype: 'i8' }]);
    await caseEq("(M,a,b)=>M.add(a,b)", [{ data: [[100, 150]], dtype: 'ui8' }, { data: [[50, 60]], dtype: 'ui8' }]);
  });

  it('bool select via where', async () => {
    const r = await caseEq("(M,c,a,b)=>M.where(c,a,b)", [{ data: [[1, 0]], dtype: 'bool' }, { data: [[5, 5]] }, { data: [[9, 9]] }]);
    expect(r.gpu).toEqual([5, 9]);
  });

  it('integer index tensors: index_select / gather pick the right rows', async () => {
    const r = await caseEq("(M,x)=>M.index_select(x,0,M.tensor([0,2],{dtype:'i32'}))", [{ data: [[1, 2], [3, 4], [5, 6]] }]);
    expect(r.gpu).toEqual([1, 2, 5, 6]);
    const g = await caseEq("(M,x)=>M.index_select(x,1,M.tensor([2,0],{dtype:'i32'}))", [{ data: [[1, 2, 3], [4, 5, 6]] }]);
    expect(g.gpu).toEqual([3, 1, 6, 4]);
  });

  it('f16 / bf16 half precision match CPU bit-for-bit', async () => {
    await caseEq("(M,a,b)=>M.add(a,b)", [{ data: [[1.5, 2.5]], dtype: 'f16' }, { data: [[0.5, 0.5]], dtype: 'f16' }]);
    await caseEq("(M,a,b)=>M.add(a,b)", [{ data: [[1.5, 2.5]], dtype: 'bf16' }, { data: [[0.5, 0.5]], dtype: 'bf16' }]);
  });

  it('i64 (>32-bit-safe small values) and f64', async () => {
    const r = await caseEq("(M,a,b)=>M.add(a,b)", [{ data: [[100, 200]], dtype: 'i64' }, { data: [[1, 2]], dtype: 'i64' }]);
    expect(r.gpu).toEqual([101, 202]);
    await caseEq("(M,x)=>M.sum(x,1)", [{ data: [[1.5, 2.5], [3.5, 4.5]], dtype: 'f64' }]);
  });

  const SCHED = { scheduling: { enabled: true } };
  const AUTOTUNE = { scheduling: { enabled: true, autotune: true } };
  const grid = (rows, cols, seed) => Array.from({ length: rows }, (_, i) => Array.from({ length: cols }, (_, j) => Math.sin((i * cols + j + seed) * 0.7)));

  it('erf / erfc / lgamma / gamma match CPU (WGSL helper-function lowering)', async () => {
    const pos = { data: Array.from({ length: 24 }, (_, i) => 0.15 + i * 0.19) };
    const signed = { data: Array.from({ length: 24 }, (_, i) => -2.2 + i * 0.19) };

    await caseClose("(M,x)=>M.erf(x)", [signed]);
    await caseClose("(M,x)=>M.erfc(x)", [signed]);
    await caseClose("(M,x)=>M.lgamma(x)", [pos]);
    await caseClose("(M,x)=>M.gamma(x)", [pos]);

    const nested = await caseClose("(M,x)=>M.erf(M.erf(M.lgamma(x)))", [pos]);
    expect(nested.cpu.every(Number.isFinite)).toBe(true);
  });

  it('negative-argument lgamma takes the reflection branch on GPU too', async () => {
    const neg = { data: [-0.3, -1.4, -2.6, -3.7, -4.2, -0.75] };
    const res = await caseClose("(M,x)=>M.lgamma(x)", [neg]);
    expect(res.cpu.every(Number.isFinite)).toBe(true);
  });

  it('max_pool2d / avg_pool2d match CPU (bundled node-type tag regression)', async () => {
    const x = { data: [[grid(8, 8, 1), grid(8, 8, 2)]] };
    const mp = await caseClose("(M,x)=>M.max_pool2d(x,[2,2],[2,2])", [x]);
    expect(mp.cpu.every(Number.isFinite)).toBe(true);
    await caseClose("(M,x)=>M.avg_pool2d(x,[2,2],[2,2])", [x]);
  });

  it('scheduled reduction matches CPU (no thread-race)', async () => {
    await caseClose("(M,x)=>M.sum(M.relu(x),1)", [{ data: grid(16, 16, 1) }], SCHED);
    await caseClose("(M,x)=>M.sum(M.relu(x),1)", [{ data: grid(16, 16, 1) }], AUTOTUNE);
    await caseClose("(M,x)=>M.sum(x,0)", [{ data: grid(12, 20, 2) }], AUTOTUNE);
  });

  it('scheduled multi-axis reduction: conv2d matches CPU', async () => {
    const x = { data: [Array.from({ length: 2 }, (_, c) => grid(10, 10, c * 5))] };
    const w = { data: Array.from({ length: 3 }, (_, o) => Array.from({ length: 2 }, (_, c) => grid(3, 3, o * 7 + c))) };
    await caseClose("(M,x,w)=>M.conv2d(x,w,null,[1,1],[[0,0],[0,0]])", [x, w], SCHED);
    await caseClose("(M,x,w)=>M.conv2d(x,w,null,[1,1],[[0,0],[0,0]])", [x, w], AUTOTUNE);
  });

  it('scheduled softmax / layernorm match CPU', async () => {
    await caseClose("(M,x)=>M.softmax(x,1)", [{ data: grid(8, 16, 3) }], AUTOTUNE);
    await caseClose("(M,x)=>M.layer_norm(x,[32],null,null,1e-5)", [{ data: grid(8, 32, 4) }], AUTOTUNE);
  });

  it('scheduled matmul with elementwise prefix matches CPU (cross-thread intermediate)', async () => {
    const a = { data: grid(7, 7, 1) };
    const b = { data: grid(7, 7, 2) };
    await caseClose("(M,a,b)=>M.matmul(M.tanh(a),b)", [a, b], SCHED);
    await caseClose("(M,a,b)=>M.matmul(M.tanh(a),b)", [a, b], AUTOTUNE);
    await caseClose("(M,a,b)=>M.relu(M.matmul(M.relu(a),b))", [a, b], AUTOTUNE);
  });

  it('scheduled wide bottleneck matches CPU (cross-workgroup intermediate serialized)', async () => {
    const x = { data: grid(2, 4, 1) };
    const w1 = { data: grid(4, 64, 2) };
    const w2 = { data: grid(64, 3, 3) };
    await caseClose("(M,x,w1,w2)=>M.matmul(M.relu(M.matmul(x,w1)),w2)", [x, w1, w2], SCHED);
    await caseClose("(M,x,w1,w2)=>M.matmul(M.relu(M.matmul(x,w1)),w2)", [x, w1, w2], AUTOTUNE);
  });

  it('matmul wider than one workgroup feeding offset-split matches CPU (cross-extent serialized)', async () => {
    const x = { data: grid(1, 128, 1) };
    const w = { data: grid(128, 1024, 2) };
    const src = "(M,x,w)=>{const g=M.matmul(x,w);const p=M.split(g,256,-1);return M.add(M.add(M.sigmoid(p[0]),M.sigmoid(p[1])),M.add(M.tanh(p[2]),M.sigmoid(p[3])));}";
    await caseClose(src, [x, w]);
    await caseClose(src, [x, w], SCHED);
    await caseClose(src, [x, w], AUTOTUNE);
  });

  it('autotuned matmul larger than one workgroup matches CPU (thread cap)', async () => {
    for (const N of [32, 64]) {
      const a = { data: grid(N, N, 1) };
      const b = { data: grid(N, N, 2) };
      await caseClose("(M,a,b)=>M.relu(M.matmul(a,b))", [a, b], SCHED);
      await caseClose("(M,a,b)=>M.relu(M.matmul(a,b))", [a, b], AUTOTUNE);
    }
  });

  it('large matmul chain splits into a multi-kernel plan and matches CPU', async () => {
    const x = { data: grid(4, 8, 1) };
    const ws = [];
    for (let i = 0; i < 9; i++) ws.push({ data: grid(8, 8, 10 + i) });
    const src = "(M,x,a,b,c,d,e,f,g,h,i)=>{let t=x;for(const w of [a,b,c,d,e,f,g,h,i])t=M.relu(M.matmul(t,w));return t;}";
    await caseClose(src, [x, ...ws]);
  });

  it('mixed-dtype packed kernel (index_select + matmul + bias) matches CPU', async () => {
    const idx = { data: [0, 2, 5], dtype: 'i32' };
    const W = { data: grid(10, 8, 1) };
    const a = { data: grid(8, 8, 2) };
    const b = { data: grid(1, 8, 3) };
    const c = { data: grid(8, 4, 4) };
    const d = { data: grid(1, 4, 5) };
    const src = "(M,idx,W,a,b,c,d)=>{const e=M.index_select(W,0,idx);const h=M.relu(M.add(M.matmul(e,a),b));return M.add(M.matmul(h,c),d);}";
    await caseClose(src, [idx, W, a, b, c, d]);
  });

  async function cmpEager(res, tol, name) {
    expect(res.cpu, `${name}: no cpu result`).toBeDefined();
    expect(res.gpu, `${name}: no gpu result`).toBeDefined();
    expect(res.gpu.length, `${name}: length`).toBe(res.cpu.length);
    let maxErr = 0, bad = -1;
    for (let i = 0; i < res.cpu.length; i++) {
      const e = Math.abs(res.cpu[i] - res.gpu[i]) / (1 + Math.abs(res.cpu[i]));
      if (e > maxErr) { maxErr = e; bad = i; }
    }
    expect(maxErr, `${name} idx ${bad}: cpu=${res.cpu[bad]} gpu=${res.gpu[bad]}`).toBeLessThan(tol);
  }

  it('eager Linear forward (W^T transpose view → device contiguous) matches CPU', async () => {
    if (!hasGpu) return;
    for (const [inF, outF, B] of [[16, 24, 4], [32, 16, 8], [8, 32, 1]]) {
      const res = await page.evaluate((a, b, c) => window.runEagerLinear(a, b, c, 7), inF, outF, B);
      await cmpEager(res, 3e-3, `Linear ${inF}x${outF} B${B}`);
    }
  }, 60000);

  it('eager LSTM forward (batchFirst transpose views) matches CPU', async () => {
    if (!hasGpu) return;
    const res = await page.evaluate(() => window.runEagerLSTM('lstm', 8, 12, 5, 3, 11));
    await cmpEager(res, 3e-3, 'LSTM');
  }, 60000);

  it('eager GRU forward (batchFirst transpose views) matches CPU', async () => {
    if (!hasGpu) return;
    const res = await page.evaluate(() => window.runEagerLSTM('gru', 8, 12, 5, 3, 13));
    await cmpEager(res, 3e-3, 'GRU');
  }, 60000);

  it('fused eager LSTM (B=1 persistent-RNN kernel, the webgpu cuDNN-equivalent) matches CPU', async () => {
    if (!hasGpu) return;
    for (const [E, H, T] of [[8, 12, 5], [64, 128, 20], [16, 256, 4]]) {
      const res = await page.evaluate((e, h, t) => window.runFusedLSTM(e, h, t, 31), E, H, T);
      await cmpEager(res, 3e-3, `fused LSTM ${E}x${H} T${T}`);
    }
  }, 60000);

  it('fused eager GRU (B=1 persistent-RNN kernel, the webgpu cuDNN-equivalent) matches CPU', async () => {
    if (!hasGpu) return;
    for (const [E, H, T, L] of [[8, 12, 5, 1], [64, 128, 20, 1], [16, 256, 4, 1], [8, 12, 6, 2]]) {
      const res = await page.evaluate((e, h, t, l) => window.runFusedGRU(e, h, t, l, 37), E, H, T, L);
      await cmpEager({ cpu: res.cpu, gpu: res.gpu }, 3e-3, `fused GRU ${E}x${H} T${T} L${L} out`);
      await cmpEager({ cpu: res.cpuH, gpu: res.gpuH }, 3e-3, `fused GRU ${E}x${H} T${T} L${L} hidden`);
    }
  }, 60000);

  it('Trainer(accelerator="webgpu").predict eager inference matches CPU (the Tera webgpu path)', async () => {
    if (!hasGpu) return;
    for (const [arch, inF, outF, B, nb] of [['mlp', 12, 5, 4, 3], ['lstm', 8, 6, 3, 2]]) {
      const res = await page.evaluate((a, i, o, b, n) => window.runTrainerPredict(a, i, o, b, n, 21), arch, inF, outF, B, nb);
      expect(res.gpu.length, `${arch}: batch count`).toBe(res.cpu.length);
      for (let k = 0; k < res.cpu.length; k++) {
        await cmpEager({ cpu: res.cpu[k], gpu: res.gpu[k] }, 3e-3, `${arch} predict batch ${k}`);
      }
    }
  }, 60000);

});
