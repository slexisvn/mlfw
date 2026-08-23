import { describe, it, expect, beforeAll } from 'vitest';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import * as esbuild from 'esbuild';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..');

const NODE_BUILTINS = [
  'fs', 'path', 'os', 'crypto', 'child_process', 'worker_threads', 'url', 'util',
  'stream', 'buffer', 'module', 'perf_hooks', 'v8', 'vm', 'zlib', 'net', 'http',
  'https', 'tty', 'readline', 'assert', 'events', 'process', 'querystring', 'dns',
  'cluster', 'timers', 'string_decoder', 'async_hooks', 'inspector', 'repl', 'tls',
  'dgram', 'sqlite', 'trace_events', 'wasi', 'diagnostics_channel',
];

const BUILTIN_IMPORT = new RegExp(
  `(?:^|[^\\w$])(?:import|export)[^;\\n]*?from\\s*['"](?:node:)?(${NODE_BUILTINS.join('|')})(?:/[^'"]*)?['"]`,
  'm',
);
const BARE_IMPORT = new RegExp(`(?:^|[^\\w$])import\\s*['"](?:node:)?(${NODE_BUILTINS.join('|')})['"]`, 'm');

const NODE_GLOBALS = [
  ['process', /(?<![\w$.'"`])process\s*(?:\.|\[)/],
  ['__dirname', /(?<![\w$.])__dirname\b/],
  ['__filename', /(?<![\w$.])__filename\b/],
  ['require()', /(?<![\w$.])require\s*\(/],
  ['global', /(?<![\w$.])global\s*(?:\.|\[)/],
  ['setImmediate', /(?<![\w$.])setImmediate\s*\(/],
  ['createRequire', /(?<![\w$.])createRequire\b/],
];

const BROWSER_GLOBALS = [
  ['window', /(?<![\w$.])window\s*(?:\.|\[)/],
  ['document', /(?<![\w$.])document\s*(?:\.|\[)/],
  ['localStorage', /(?<![\w$.])localStorage\b/],
];

const WEBGPU_STUB = {
  name: 'webgpu-stub',
  setup(build) {
    build.onResolve({ filter: /^webgpu$/ }, (args) => ({ path: args.path, namespace: 'webgpu-stub' }));
    build.onLoad({ filter: /.*/, namespace: 'webgpu-stub' }, () => ({
      contents: 'export const create = () => {}; export const globals = undefined;',
      loader: 'js',
    }));
  },
};

async function graphInputs(platform) {
  const result = await esbuild.build({
    absWorkingDir: ROOT,
    entryPoints: ['src/index.ts'],
    bundle: true,
    write: false,
    metafile: true,
    format: 'esm',
    platform,
    target: platform === 'browser' ? 'es2022' : 'node18',
    conditions: platform === 'browser' ? ['browser'] : [],
    external: platform === 'node' ? ['koffi', 'webgpu'] : [],
    plugins: platform === 'browser' ? [WEBGPU_STUB] : [],
    logLevel: 'silent',
  });
  return Object.keys(result.metafile.inputs)
    .filter((f) => !f.includes(':'))
    .map((f) => f.split('\\').join('/'));
}

function stripLineComments(source) {
  return source.split(/\r?\n/).map((line) => line.replace(/\/\/.*$/, '')).join('\n');
}

function sourcesOf(files) {
  return files.map((file) => [file, stripLineComments(readFileSync(resolve(ROOT, file), 'utf8'))]);
}

describe('browser/node build isolation', () => {
  let browserFiles;
  let nodeFiles;

  beforeAll(async () => {
    [browserFiles, nodeFiles] = await Promise.all([graphInputs('browser'), graphInputs('node')]);
  }, 120000);

  it('resolves both graphs from the same entry', () => {
    expect(browserFiles).toContain('src/index.ts');
    expect(nodeFiles).toContain('src/index.ts');
  });

  it('keeps every src/runtime/node module out of the browser graph', () => {
    expect(browserFiles.filter((f) => f.startsWith('src/runtime/node/'))).toEqual([]);
  });

  it('keeps every src/runtime/browser module out of the node graph', () => {
    expect(nodeFiles.filter((f) => f.startsWith('src/runtime/browser/'))).toEqual([]);
  });

  it('swaps a browser counterpart in for each node-only runtime module the browser graph needs', () => {
    const swapped = browserFiles.filter((f) => f.startsWith('src/runtime/browser/'));
    expect(swapped.length).toBeGreaterThan(0);
    for (const file of swapped) {
      expect(nodeFiles).toContain(file.replace('src/runtime/browser/', 'src/runtime/node/'));
    }
  });

  it('imports no node builtin anywhere in the browser graph', () => {
    const offenders = sourcesOf(browserFiles)
      .filter(([, src]) => BUILTIN_IMPORT.test(src) || BARE_IMPORT.test(src))
      .map(([file]) => file);
    expect(offenders).toEqual([]);
  });

  it('touches no node global anywhere in the browser graph', () => {
    const offenders = [];
    for (const [file, src] of sourcesOf(browserFiles)) {
      for (const [name, pattern] of NODE_GLOBALS) {
        for (const line of src.split('\n')) {
          if (pattern.test(line)) offenders.push(`${file}: ${name} :: ${line.trim()}`);
        }
      }
    }
    expect(offenders).toEqual([]);
  });

  it('touches no dom-only global anywhere in the graph, so the bundle runs in a worker', () => {
    const offenders = [];
    for (const [file, src] of sourcesOf(browserFiles)) {
      for (const [name, pattern] of BROWSER_GLOBALS) {
        if (pattern.test(src)) offenders.push(`${file}: ${name}`);
      }
    }
    expect(offenders).toEqual([]);
  });

  it('declares a browser branch for every #io subpath the source imports', () => {
    const pkg = JSON.parse(readFileSync(resolve(ROOT, 'package.json'), 'utf8'));
    const declared = Object.keys(pkg.imports);
    expect(declared.length).toBeGreaterThan(0);
    for (const key of declared) {
      expect(pkg.imports[key].browser, `${key} has no browser branch`).toBeTruthy();
      expect(pkg.imports[key].default, `${key} has no default branch`).toBeTruthy();
    }

    const used = new Set();
    for (const [, src] of sourcesOf(browserFiles.concat(nodeFiles))) {
      for (const match of src.matchAll(/from\s*'(#io\/[^']+)'|import\('(#io\/[^']+)'\)/g)) {
        used.add(match[1] ?? match[2]);
      }
    }
    const matches = (specifier) => declared.some((key) => (
      key.endsWith('/*') ? specifier.startsWith(key.slice(0, -1)) : specifier === key
    ));
    expect([...used].filter((specifier) => !matches(specifier))).toEqual([]);
  });
});
