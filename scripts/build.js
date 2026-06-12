import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { copyFileSync } from 'node:fs';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');
const entry = resolve(root, 'src/index.js');
const outdir = resolve(root, 'dist');

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

// The query-engine browser plugin lazily imports distributed-execution modules
// that ship only with the node build. They are never reached in the browser, so
// resolve them to a stub that throws if ever invoked.
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

const shared = {
  bundle: true,
  platform: 'browser',
  target: ['es2022'],
  plugins: [browserStubPlugin, distributedStubPlugin],
  define: { 'process.env.NODE_ENV': '"production"' },
  logLevel: 'info',
};

await Promise.all([
  build({
    ...shared,
    entryPoints: [entry],
    format: 'esm',
    outfile: resolve(outdir, 'mlfw.esm.js'),
    sourcemap: true,
  }),
  build({
    ...shared,
    entryPoints: [entry],
    format: 'esm',
    outfile: resolve(outdir, 'mlfw.esm.min.js'),
    minify: true,
    sourcemap: true,
  }),
  build({
    ...shared,
    entryPoints: [entry],
    format: 'iife',
    globalName: 'mlfw',
    outfile: resolve(outdir, 'mlfw.global.js'),
    sourcemap: true,
  }),
  build({
    ...shared,
    entryPoints: [resolve(root, 'notebook/lib.js')],
    format: 'esm',
    outfile: resolve(root, 'notebook/dist/mlfw-lang.esm.js'),
    sourcemap: true,
  }),
]);

copyFileSync(resolve(root, 'vscode-ext/language-data.json'), resolve(root, 'notebook/dist/language-data.json'));

console.log('Browser bundles written to dist/ and notebook/dist/');
