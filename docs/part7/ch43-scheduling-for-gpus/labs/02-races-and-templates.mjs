import {
  compile, CUDATarget, TraceLevel, randn,
} from '../../_internals.mjs';

// Two things the GPU path does that the scheduling language does not: replace a
// whole function body with a hand-written template, and refuse a schedule the
// primitives accepted because the backend cannot prove it race-free.

async function cuda(label, fn, inputs, show) {
  const snaps = [];
  const compiled = compile({ forward: fn }, inputs, {
    target: CUDATarget(),
    fusion: { enabled: false },
    scheduling: { enabled: true },
    trace: {
      level: TraceLevel.DEBUG,
      irSnapshot: { afterScheduling: true },
      sink: (e) => { if (e.type === 'ir_snapshot') snaps.push(e.text); },
    },
  });
  try {
    await compiled(...inputs);
  } catch (e) {
    // No GPU on this machine; every artefact below is produced at compile time.
  }
  console.log(`=== ${label} ===`);
  show({ tir: snaps[snaps.length - 1] ?? '', source: compiled.source() });
  console.log();
}

// ------------------------------------------- a schedule the primitives wrote

await cuda('elementwise: the rule uses the primitives', (a) => a.mul(2.0), [randn([4096])],
  ({ source }) => console.log(source.split('\n').slice(1).join('\n').trimEnd()));

// ------------------------------------------------- a body that was replaced

await cuda('matmul 128x128: the body was replaced wholesale',
  (a, b) => a.matmul(b), [randn([128, 128]), randn([128, 128])],
  ({ tir, source }) => {
    const lines = tir.split('\n');
    console.log(lines.slice(4, 20).join('\n'));
    console.log('    …');
    console.log(`\n  ${lines.length} lines of TIR, none of which any primitive produced.`);
    console.log(`  shared-memory buffers: ${(tir.match(/\(shared\)/g) || []).length}`);
    console.log(`  thread bindings      : ${[...tir.matchAll(/\[(blockIdx|threadIdx)\.[xyz]\]/g)].map((m) => m[0]).join(' ')}`);
    console.log(`  __syncthreads in CUDA: ${(source.match(/__syncthreads/g) || []).length}, printed in the TIR as`);
    for (const l of lines.filter((l) => /UnknownNode/.test(l))) console.log(`      ${l.trim()}`);
    console.log('\n  `applyDeterministicGpuMatmul` recognised the epilogue shape and');
    console.log('  assigned `schedule.func.body = buildRegisterBlockedMatmul(...)`');
    console.log('  (gpu_matmul_schedule.ts:120). Not one scheduling primitive was');
    console.log('  called. The TIR prints `[UnknownNode: SyncThreadsNode]` because');
    console.log('  the printer covers 17 of 21 node kinds (Chapter 32, finding 13),');
    console.log('  which is how you can tell at a glance that this nest was not');
    console.log('  built by the scheduling language.');
  });

// --------------------------------------------- a race the backend repaired

await cuda('row-sum then scale: an intermediate crosses threads',
  (a) => a.sum(1).mul(2.0), [randn([8, 1024])],
  ({ source }) => {
    console.log(source.split('\n').slice(1).join('\n').trimEnd());
    console.log('\n  `buf_6` is written by one thread per row and read by another');
    console.log('  after the fan-out. `threadSharedIntermediates` flagged it, and');
    console.log('  because it is not stored under a block binding the CUDA backend');
    console.log('  chose to repair rather than refuse: `_promoteCrossThreadToShared`');
    console.log('  moved it to __shared__ and turned barriers on.');
  });

// ------------------------------------------- a race the backend refused

await cuda('two chained reductions over 300 rows',
  (a) => a.sum(1).sum(0), [randn([300, 4])],
  ({ source }) => {
    console.log(source.split('\n').slice(1, 18).join('\n').trimEnd());
    console.log('    …');
    console.log('\n  No blockIdx, no threadIdx, ordinary `for` loops — inside a');
    console.log('  __global__ function. The schedule bound those loops to threads and');
    console.log('  the backend threw the bindings away: `crossBlockRAWBuffers` found a');
    console.log('  storage buffer written under one binding signature and read under');
    console.log('  another, which no barrier inside a kernel can order, so `_serialize`');
    console.log('  set the launch to 1x1x1 and emitted the loops (cuda/codegen.ts:562).');
    console.log('\n  This is the fourth answer to Chapter 42\'s question. The primitives');
    console.log('  said yes, the validator was not asked, and the backend — the only');
    console.log('  layer that knows what a grid is — said no.');
  });
