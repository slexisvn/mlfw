import { SEARCH_BUDGET } from './catalog/tuning.js';
import type { CompileOptions } from './protocol.js';

const TARGET_FACTORY: Record<CompileOptions['target'], string> = {
  cpu: 'CPUTarget',
  wasm: 'WasmTarget',
  cuda: 'CUDATarget',
  webgpu: 'WebGPUTarget',
};

function settingsOf(options: CompileOptions): string {
  return JSON.stringify({
    verify: options.verify,
    fusion: { enabled: options.fusion, strategy: options.fusionStrategy },
    scheduling: { enabled: options.scheduling, autotune: options.autotune, ...SEARCH_BUDGET },
    optimization: { layout: options.layout },
  }, null, 2);
}

export function reproScript(source: string, options: CompileOptions): string {
  const compileCall = options.backward === 'off'
    ? 'compile(compilable, inputs, settings)'
    : `compileWithBackward(compilable, inputs, { ...settings, mode: ${JSON.stringify(options.backward)} })`;

  return `// Repro exported from the mlfw pass visualizer.
// Run it from the repository root:  npm run build && node repro.mjs
// It compiles exactly what the tool compiled and checks the answer against eager.
// The tool compiles from src/; this script loads dist/, so build first or you are testing older code.

import * as mlfw from './dist/index.node.js';

const { compile, compileWithBackward, manual_seed, noGrad, TraceLevel, ${TARGET_FACTORY[options.target]} } = mlfw;

const SOURCE = ${JSON.stringify(source)};
const DISABLED = ${JSON.stringify(options.disabledPasses)};
const SEED = 0;

const settings = {
  ...${settingsOf(options)},
  target: ${TARGET_FACTORY[options.target]}(),
  passContext: DISABLED.length === 0 ? null : {
    disabledPasses: new Set(DISABLED),
    requiredPasses: new Set(),
    optLevel: Infinity,
    config: new Map(),
    shouldRun: (pass) => !DISABLED.includes(pass.name),
  },
  trace: {
    level: TraceLevel.INFO,
    sink: (event) => {
      if (event.type === 'error') console.error('[error]', event.phase, event.funcName, event.message);
      if (event.type === 'pass_skipped') console.log('[skipped]', event.passName, event.irLevel);
    },
  },
};

const names = Object.keys(mlfw).filter((name) => name !== 'default');
let captured = null;
const run = (model, inputs) => { captured = { model, inputs }; };
new Function(...names, 'run', SOURCE)(...names.map((name) => mlfw[name]), run);
if (!captured) throw new Error('the source never called run(model, inputs)');

const { model, inputs } = captured;
const compilable = typeof model === 'function' ? new (class { forward = model; })() : model;

manual_seed(SEED);
const handle = ${compileCall};
if (handle._ready) await handle._ready;

manual_seed(SEED);
const compiled = await handle(...inputs);
const eager = await noGrad(() => compilable.forward(...inputs));

const values = (t) => [...(t.contiguous ? t.contiguous().data : t.data)].map(Number);
const flat = (out) => (Array.isArray(out) ? out : [out]).flatMap(values);
const a = flat(compiled);
const b = flat(eager);
let worst = 0;
for (let i = 0; i < Math.min(a.length, b.length); i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));

console.log('kernels    ', handle.result ? handle.result().listKernels() : '(training)');
console.log('worst diff ', worst);
console.log(worst > 1e-3 ? 'MISCOMPILE: the compiled answer disagrees with eager' : 'compiled and eager agree');
`;
}

export function download(name: string, text: string): void {
  const url = URL.createObjectURL(new Blob([text], { type: 'text/plain' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  link.click();
  URL.revokeObjectURL(url);
}
