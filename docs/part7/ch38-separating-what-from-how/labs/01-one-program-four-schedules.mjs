import {
  compile, CPUTarget, WasmTarget, CUDATarget, TraceLevel, randn, resetVarCounter,
} from '../../_internals.mjs';

// One graph, four targets. The block is the same object in all four printouts.
// Everything that differs is a loop.

async function scheduled(label, target, opts = {}) {
  const snaps = [];
  const explains = [];
  const x = randn([4096]);
  // The fresh-variable counter is process-global, so reset it and the four
  // printouts differ only where the schedule differs.
  resetVarCounter();
  const compiled = compile({ forward: (a) => a.mul(2.0) }, [x], {
    target,
    fusion: { enabled: false },
    scheduling: { enabled: true, ...opts },
    trace: {
      level: TraceLevel.DEBUG,
      explains: true,
      irSnapshot: { afterScheduling: true },
      sink: (e) => {
        if (e.type === 'ir_snapshot') snaps.push(e.text);
        else if (e.type === 'explain' && e.category === 'schedule') explains.push(e);
      },
    },
  });
  try {
    await compiled(x);
  } catch (e) {
    // CUDA and WebGPU compile without a device; only the launch needs one.
  }
  console.log(`=== ${label} ===`);
  const body = snaps[snaps.length - 1].split('\n');
  for (const line of body.slice(3, -1)) console.log(line);
  for (const e of explains) console.log(`  rule: ${e.decision}  (${e.reason})`);
  console.log();
}

await scheduled('CPU — 8 cores, vector width 8', CPUTarget());
await scheduled('WASM — 1 core, SIMD width 4', WasmTarget());
await scheduled('WASM — 4 cores, SIMD width 4', WasmTarget({ numCores: 4 }));
await scheduled('CUDA — 1024 threads per block', CUDATarget());

console.log('The four differ in every loop and in nothing else.');
console.log('  - the block name is `mul_block_0` in all four;');
console.log('  - its one iteration variable is bound to a different expression each time;');
console.log('  - its read set, write set and body are identical, character for character.');
console.log();
console.log('That is the whole claim of this part: the loops are a plan, and a');
console.log('target-specific rule writes the plan. Chapter 39 is about the data');
console.log('structure that lets the rule rewrite the plan without touching the block.');
