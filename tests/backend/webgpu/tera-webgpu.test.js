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

// Browser stubs mirroring scripts/build.js: the notebook/Tera browser bundle pulls in
// the framework whose node-only paths (the 'webgpu' npm package, query-engine distributed
// execution, and the Trainer's CUDA-graph node FFI) are never reached in a browser.
const webgpuPkgStub = {
  name: 'webgpu-pkg-stub',
  setup(b) {
    b.onResolve({ filter: /^webgpu$/ }, () => ({ path: 'webgpu-pkg', namespace: 'stub-wgpkg' }));
    b.onLoad({ filter: /.*/, namespace: 'stub-wgpkg' }, () => ({
      contents: 'export const create = () => { throw new Error("no webgpu npm in browser"); };\nexport const globals = undefined;\nexport default { create, globals };',
      loader: 'js',
    }));
  },
};
const throwProxy = (what) => `export default new Proxy({}, { get() { throw new Error("mlfw: ${what} is not available in the browser build"); } });`;
const distributedStub = {
  name: 'distributed-stub',
  setup(b) {
    b.onResolve({ filter: /[\\/]distributed[\\/]/ }, () => ({ path: 'd', namespace: 'stub-dist' }));
    b.onLoad({ filter: /.*/, namespace: 'stub-dist' }, () => ({ contents: throwProxy('distributed execution'), loader: 'js' }));
  },
};

const PAGE_RUNNER = `
import * as M from './bundle.js';
const flatHost = (t) => Array.from((t.contiguous ? t.contiguous() : t).data, v => typeof v === 'bigint' ? Number(v) : v);
const readDev = (t) => Array.from(t._impl.storage.rawData.subarray(0, t.numel), v => typeof v === 'bigint' ? Number(v) : v);

window.runTeraToDevice = async () => {
  await M.preloadWebGPU();
  const printed = [];
  const rt = new M.TeraRuntime({ output: (s) => printed.push(String(s)) });
  const src = [
    'x = tensor([[1.0, 2.0, 3.0, 4.0], [0.5, -0.5, 0.5, -0.5], [2.0, 0.0, -1.0, 1.0]])',
    'net = Linear(4, 3)',
    'ycpu = net(x)',
    'net.to(device="webgpu")',
    'ygpu = net(x.to(device="webgpu"))',
    'print(ygpu)',
  ].join('\\n');
  await rt.execute(src);
  await M.flushWebGPUEager();
  return { cpu: flatHost(rt.getVariable('ycpu')), gpu: readDev(rt.getVariable('ygpu')), printed: printed.join('\\n') };
};
window.__gpu = !!navigator.gpu;
window.__ready = true;
`;

const PAGE_HTML = `<!doctype html><html><head><meta charset="utf-8"></head><body>
<script type="module" src="./runner.js"></script></body></html>`;

describe.skipIf(!deps)('Tera interpreter on WebGPU via Chrome (.to(device="webgpu"))', () => {
  let browser, server, page, port, bundleCode, hasGpu;

  beforeAll(async () => {
    const built = await deps.esbuild.build({
      stdin: {
        contents: "export { TeraRuntime } from './src/cli/runtime.js';\n"
          + "export { preloadWebGPU } from './src/runtime/backend_registry.js';\n"
          + "export { flushWebGPUEager } from './src/runtime/webgpu.js';\n",
        resolveDir: PROJECT_ROOT,
        loader: 'js',
      },
      bundle: true, format: 'esm', platform: 'browser', conditions: ['browser'], target: ['es2022'],
      define: { 'process.env.NODE_ENV': '"production"' },
      plugins: [webgpuPkgStub, distributedStub],
      write: false,
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

  it('Chrome exposes WebGPU', () => {
    expect(hasGpu).toBe(true);
  });

  it('.to(device="webgpu") moves a Linear model to the device and infers, matching the host run', async () => {
    if (!hasGpu) return;
    const res = await page.evaluate(() => window.runTeraToDevice());
    expect(res.gpu.length).toBe(res.cpu.length);
    let maxErr = 0, bad = -1;
    for (let i = 0; i < res.cpu.length; i++) {
      const e = Math.abs(res.cpu[i] - res.gpu[i]) / (1 + Math.abs(res.cpu[i]));
      if (e > maxErr) { maxErr = e; bad = i; }
    }
    expect(maxErr, `.to(webgpu) idx ${bad}: cpu=${res.cpu[bad]} gpu=${res.gpu[bad]}`).toBeLessThan(3e-3);
  }, 60000);

  it('print() of a webgpu tensor flushes eager work so values are materialized (not stale)', async () => {
    if (!hasGpu) return;
    const res = await page.evaluate(() => window.runTeraToDevice());
    // formatValue prints "Tensor(shape=..., device=webgpu)\n[[...]]" — the values are on line 2.
    const jsonPart = res.printed.slice(res.printed.indexOf('\n') + 1);
    const printedNums = JSON.parse(jsonPart).flat();
    expect(printedNums.length).toBe(res.cpu.length);
    let pErr = 0;
    for (let i = 0; i < res.cpu.length; i++) pErr = Math.max(pErr, Math.abs(res.cpu[i] - printedNums[i]) / (1 + Math.abs(res.cpu[i])));
    expect(pErr, `print(webgpu) flush mismatch; printed=${res.printed}`).toBeLessThan(3e-3);
  }, 60000);

});
