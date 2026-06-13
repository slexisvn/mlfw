import { describe, it, expect, beforeAll, afterAll } from 'vitest';
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
window.__gpu = !!navigator.gpu;
window.__ready = true;
`;

const PAGE_HTML = `<!doctype html><html><head><meta charset="utf-8"></head><body>
<script type="module" src="./runner.js"></script></body></html>`;

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
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
          + "export { max_pool2d, avg_pool2d } from './src/nn/functional/pooling.js';\n",
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
});
