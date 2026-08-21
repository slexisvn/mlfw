import {
  lowerToTir, printTensorIR, Schedule, SchedulePolicy, BackendPipeline,
  CUDATarget, WebGPUTarget, launchGeometry, randn,
} from '../../_internals.mjs';

// On a GPU the outer loops do not run: they are the index space the hardware
// hands each thread. This lab binds them by hand, reads the launch geometry the
// bindings imply, and then watches a rule do the same thing.

const N = 4096;
const fresh = async () => new Schedule(await lowerToTir((x) => x.mul(2.0), [randn([N])], CUDATarget()));

// -------------------------------------------------------------- by hand

const sch = await fresh();
const [i] = sch.getLoops('mul_block_0');
const [outer, inner] = sch.split(i, 256);
sch.bindThread(inner, 'threadIdx.x');
sch.bindThread(outer, 'blockIdx.x');

console.log('=== split(i, 256), bindThread(inner, threadIdx.x), bindThread(outer, blockIdx.x) ===');
console.log(printTensorIR(sch.func).split('\n').slice(4, -1).join('\n'));

console.log('\n=== the launch geometry those two annotations imply ===\n');
const s = sch.state.summary();
for (const [tag, b] of Object.entries(s.threadBindings)) {
  console.log(`  ${tag.padEnd(12)} <- ${b.varName.padEnd(12)} extent ${b.extent}`);
}
console.log(`  blockDim ${JSON.stringify(s.blockDim)}   gridDim ${JSON.stringify(s.gridDim)}`);
console.log(`  launchGeometry: ${JSON.stringify(launchGeometry(sch.func))}`);
console.log(`  ${s.blockDim[0] * s.gridDim[0]} threads for ${N} elements`);

console.log('\n=== and the kernel ===');
console.log(new BackendPipeline(CUDATarget()).compile(sch.func).source);
console.log('  The two `for` loops are gone. `blockIdx.x` and `threadIdx.x` are');
console.log('  read once into the variables the loops used to bind, and the body');
console.log('  runs exactly once per thread. The grid is not in the source at all —');
console.log('  it is metadata the runtime passes to the launch.');

// ------------------------------------------------- what the tag has to be

console.log('\n=== bindThread checks the tag and nothing else ===\n');
for (const tag of ['threadIdx.x', 'threadIdx.w', 'blockIdx.z', 'warpIdx.x']) {
  const t = await fresh();
  try {
    t.bindThread(t.getLoops('mul_block_0')[0], tag);
    console.log(`  ${tag.padEnd(14)} accepted`);
  } catch (e) {
    console.log(`  ${tag.padEnd(14)} ${e.message}`);
  }
}
console.log('\n  No dependence question is asked. `bindThread` is the one annotation');
console.log('  primitive with no legality check at all (schedule.ts:611) — the');
console.log('  backends check instead, which is §43.6.');

// --------------------------------------------------- what the rule chooses

console.log('\n=== the same nest, left to ElementwiseGPURule ===\n');
for (const [label, target] of [['CUDA', CUDATarget()], ['WebGPU', WebGPUTarget()]]) {
  const r = new Schedule(await lowerToTir((x) => x.mul(2.0), [randn([N])], target));
  new SchedulePolicy(target).applyToAllBlocks(r);
  const sum = r.state.summary();
  const tags = Object.entries(sum.threadBindings)
    .map(([t, b]) => `${t}=${b.extent}`).join('  ');
  console.log(`  ${label.padEnd(8)} maxThreadsPerBlock ${String(target.maxThreadsPerBlock).padEnd(6)} -> ${tags}`);
}
console.log('\n  Both cap the block at min(maxThreadsPerBlock, 256) — the 256 is a');
console.log('  literal in bindFusedSpatialGPU (rules.ts:180), not a target field —');
console.log('  so a CUDA device advertising 1024 threads per block is given 256.');
