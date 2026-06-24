import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { copyFileSync, readFileSync, rmSync } from 'node:fs';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');
const entry = resolve(root, 'src/index.js');
const outdir = resolve(root, 'dist');

const pkg = JSON.parse(readFileSync(resolve(root, 'package.json'), 'utf8'));
const external = Object.keys(pkg.dependencies ?? {});

rmSync(outdir, { recursive: true, force: true });

const NODE_STUBS = {
  'webgpu': `
export const create = () => {
  throw new Error('mlfw: the "webgpu" npm package is not available in the browser; use navigator.gpu instead');
};
export const globals = undefined;
export default { create, globals };
`,
};

const browserStubPlugin = {
  name: 'mlfw-browser-stubs',
  setup(b) {
    const keys = Object.keys(NODE_STUBS);
    const filter = new RegExp('^(' + keys.map((k) => k.replace(/[/\\^$*+?.()|[\]{}]/g, '\\$&')).join('|') + ')$');
    b.onResolve({ filter }, (a) => ({ path: a.path, namespace: 'mlfw-stub' }));
    b.onLoad({ filter: /.*/, namespace: 'mlfw-stub' }, (a) => ({
      contents: NODE_STUBS[a.path],
      loader: 'js',
    }));
  },
};

const distributedStubPlugin = {
  name: 'mlfw-distributed-stub',
  setup(b) {
    b.onResolve({ filter: /[\\/]distributed[\\/]/ }, () => ({ path: 'mlfw-distributed', namespace: 'mlfw-stub-dist' }));
    b.onLoad({ filter: /.*/, namespace: 'mlfw-stub-dist' }, () => ({
      contents: 'export default new Proxy({}, { get() { throw new Error("mlfw: distributed execution is not available in the browser build"); } });',
      loader: 'js',
    }));
  },
};

const sharedBrowser = {
  bundle: true,
  platform: 'browser',
  target: ['es2022'],
  plugins: [browserStubPlugin, distributedStubPlugin],
  define: { 'process.env.NODE_ENV': '"production"' },
  keepNames: true,
  logLevel: 'info',
  format: 'esm',
  minify: true,
};

const sharedNode = {
  bundle: true,
  platform: 'node',
  minify: true,
  keepNames: true,
  format: 'esm',
  external,
}

await Promise.all([
  build({
    ...sharedNode,
    entryPoints: [entry],
    outfile: resolve(outdir, 'index.js'),
  }),
  build({
    ...sharedNode,
    entryPoints: [resolve(root, 'src/cli/index.js')],
    outfile: resolve(outdir, 'cli.js'),
  }),
  build({
    ...sharedBrowser,
    entryPoints: [resolve(root, 'src/notebook/browser.js')],
    outfile: resolve(root, 'notebook/dist/mlfw.esm.js'),
  }),
  build({
    ...sharedBrowser,
    entryPoints: [resolve(root, 'src/cli/csv.js')],
    outfile: resolve(root, 'notebook/dist/csv.esm.js'),
  }),
]);

copyFileSync(resolve(root, 'vscode-ext/language-data.json'), resolve(root, 'notebook/dist/language-data.json'));