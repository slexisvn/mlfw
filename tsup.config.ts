import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import type { Plugin } from 'esbuild';
import { defineConfig } from 'tsup';

const pkg = JSON.parse(readFileSync(resolve('package.json'), 'utf8')) as {
  dependencies?: Record<string, string>;
};

const dependencies = Object.keys(pkg.dependencies ?? {});

const NODE_STUBS: Record<string, string> = {
  webgpu: `
export const create = () => {
  throw new Error('mlfw: the "webgpu" npm package is not available in the browser; use navigator.gpu instead');
};
export const globals = undefined;
export default { create, globals };
`,
};

const browserStubPlugin: Plugin = {
  name: 'mlfw-browser-stubs',
  setup(build) {
    const keys = Object.keys(NODE_STUBS);
    const filter = new RegExp(
      '^(' +
        keys.map((key) => key.replace(/[/\\^$*+?.()|[\]{}]/g, '\\$&')).join('|') +
        ')$',
    );

    build.onResolve({ filter }, (args) => ({ path: args.path, namespace: 'mlfw-stub' }));
    build.onLoad({ filter: /.*/, namespace: 'mlfw-stub' }, (args) => ({
      contents: NODE_STUBS[args.path],
      loader: 'js',
    }));
  },
};

export default defineConfig([
  {
    entry: ['src/index.ts'],
    outDir: 'dist',
    format: ['esm'],
    platform: 'node',
    target: 'node18',
    bundle: true,
    splitting: false,
    minify: true,
    keepNames: true,
    clean: true,
    external: dependencies,
    dts: true,
    outExtension: () => ({ js: '.node.js' }),
  },
  {
    entry: ['src/index.ts'],
    outDir: 'dist',
    format: ['esm'],
    platform: 'browser',
    target: 'es2022',
    bundle: true,
    splitting: false,
    minify: true,
    keepNames: true,
    clean: false,
    noExternal: dependencies,
    define: { 'process.env.NODE_ENV': '"production"' },
    esbuildPlugins: [browserStubPlugin],
    outExtension: () => ({ js: '.browser.js' }),
  },
  {
    entry: { internals: 'docs/tools/internals-entry.ts' },
    outDir: 'dist',
    format: ['esm'],
    platform: 'node',
    target: 'node18',
    bundle: true,
    splitting: false,
    minify: false,
    keepNames: true,
    clean: false,
    external: dependencies,
    dts: false,
    outExtension: () => ({ js: '.node.js' }),
  },
]);
