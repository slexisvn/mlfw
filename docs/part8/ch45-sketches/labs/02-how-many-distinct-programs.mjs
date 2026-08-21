import {
  lowerToTir, randn, CPUTarget, WebGPUTarget,
  getSketchesForBlock, Schedule, clonePrimFunc, printTensorIR, resetVarCounter,
  ScheduleValidator,
} from '../../_internals.mjs';

// A sketch advertises |V1| x ... x |Vk| points. That is an upper bound on the
// number of programs, not a count of them: two parameter values collapse to one
// program whenever the apply function clamps them, ignores them, or takes the
// same branch. This lab instantiates every point of five sketches and counts
// the distinct nests that come out.

// The loop-variable counter is global (schedule.ts:193), so every
// instantiation must start from the same counter value for two nests to be
// comparable by name. `enumerate` resets it before each point.
function enumerate(makeFunc, blockName, sketch, target, points) {
  const seen = new Map();
  let refused = 0, invalid = 0;
  for (const params of points) {
    resetVarCounter();
    const work = makeFunc();
    try {
      sketch.instantiate(params)(new Schedule(work), blockName, target);
    } catch (e) { refused++; continue; }
    if (ScheduleValidator.validate(work).length > 0) { invalid++; continue; }
    const text = printTensorIR(work);
    if (!seen.has(text)) seen.set(text, params);
  }
  return { distinct: seen.size, refused, invalid, examples: [...seen.values()] };
}

const product = (sketch) => {
  const out = [{}];
  let acc = out;
  for (const v of sketch.variables) {
    const next = [];
    for (const partial of acc) for (const c of v.candidates) next.push({ ...partial, [v.name]: c });
    acc = next;
  }
  return acc;
};

const report = (label, sketch, r) => {
  const nominal = sketch.variables.reduce((a, v) => a * v.candidates.length, 1);
  console.log(`  ${label.padEnd(30)} nominal ${String(nominal).padStart(5)}   distinct ${String(r.distinct).padStart(5)}   refused ${String(r.refused).padStart(5)}   invalid ${String(r.invalid).padStart(3)}`);
};

// ------------------------------------------------ CPU

const mm = await lowerToTir((a, b) => a.matmul(b), [randn([16, 16]), randn([16, 16])]);
const cpuSketches = getSketchesForBlock(mm, 'matmul_1', CPUTarget());
const ewSketch = getSketchesForBlock(mm, 'matmul_init_0', CPUTarget())[0];

console.log('=== a 16x16x16 matmul, CPU ===\n');
for (const s of cpuSketches) {
  const r = enumerate(() => clonePrimFunc(mm), 'matmul_1', s, CPUTarget(), product(s));
  report(s.name, s, r);
}
{
  const r = enumerate(() => clonePrimFunc(mm), 'matmul_init_0', ewSketch, CPUTarget(), product(ewSketch));
  report(`${ewSketch.name} (init block)`, ewSketch, r);
  console.log(`    the ${ewSketch.variables[0].candidates.length} widths produce ${r.distinct} programs: ${r.examples.map((p) => p.vector_width).join(' ')}`);
}

{
  const small = await lowerToTir((a, b) => a.mul(b), [randn([4, 4]), randn([4, 4])]);
  const ss = getSketchesForBlock(small, 'mul_block_0', CPUTarget())[0];
  const r = enumerate(() => clonePrimFunc(small), 'mul_block_0', ss, CPUTarget(), product(ss));
  report(`${ss.name} (4x4 mul)`, ss, r);
  console.log(`    the ${ss.variables[0].candidates.length} widths produce ${r.distinct} programs: ${r.examples.map((p) => p.vector_width).join(' ')}`);
}

console.log('\n  `mlt_cpu` loses nothing: every factor tuple gives a different set of');
console.log('  extents, and `split` never refuses, because the tuple multiplies back');
console.log('  to the extent. `ssrsrs_cpu` refuses every point, `decomposeReduction`');
console.log('  first. `elementwise_cpu` keeps all five widths on a 16-wide innermost');
console.log('  loop and loses two on a 4-wide one, because its apply function guards');
console.log('  the split with `extent >= vector_width` (sketch_generators.ts:71), and');
console.log('  every width that fails that test produces the same bare `parallelize`.');

// ------------------------------------------------ GPU

const gmm = await lowerToTir((a, b) => a.matmul(b), [randn([16, 16]), randn([16, 16])], WebGPUTarget());
console.log('\n\n=== the same matmul, WebGPU ===\n');
for (const s of getSketchesForBlock(gmm, 'matmul_1', WebGPUTarget())) {
  const r = enumerate(() => clonePrimFunc(gmm), 'matmul_1', s, WebGPUTarget(), product(s));
  report(s.name, s, r);
}
{
  const bigSrc = await lowerToTir((a, b) => a.mul(b), [randn([64, 64]), randn([64, 64])], WebGPUTarget());
  const s = getSketchesForBlock(bigSrc, 'mul_block_0', WebGPUTarget())[0];
  const r = enumerate(() => clonePrimFunc(bigSrc), 'mul_block_0', s, WebGPUTarget(), product(s));
  report(`${s.name} (4096 elts)`, s, r);
  console.log(`    candidates ${s.variables[0].candidates.join(' ')}  ->  distinct block sizes ${r.examples.map((p) => p.block_size).join(' ')}`);
  console.log(`    target.maxThreadsPerBlock = ${WebGPUTarget().maxThreadsPerBlock}, and gpuThreadCap clamps to min(that, 256)`);
}

console.log('\n  `BLOCK_SIZE_CANDIDATES` (sketch_generators.ts:7) offers six thread-block');
console.log('  sizes up to 1024. `gpuThreadCap` (sketch_generators.ts:10) then clamps');
console.log('  every one of them to at most 256, so 256, 512 and 1024 all name the');
console.log('  same kernel: two of the six advertised points are aliases of a third.');

// ------------------------------------------------ what the extra points cost

console.log('\n\n=== an alias is not free ===\n');
const s = getSketchesForBlock(gmm, 'matmul_init_0', WebGPUTarget())[0];
const pts = product(s);
console.log(`  ${s.name} on a 16x16 init block: ${pts.length} points, ${enumerate(() => clonePrimFunc(gmm), 'matmul_init_0', s, WebGPUTarget(), pts).distinct} distinct program.`);
console.log('  All six collapse, and not because of the thread cap: this block belongs');
console.log('  to a function that contains a reduction and has only 256 elements,');
console.log('  which is the `primFuncHasReduction` shortcut at sketch_generators.ts:102');
console.log('  — one thread block, whatever was asked for.');
console.log('  The search does not know they are aliases. It scores each point, keeps');
console.log('  each in the population, and — with a benchmark runner attached and');
console.log('  `topKForBenchmark` at its default of 5 — measures the same kernel five');
console.log('  times, once per name.');
console.log('  `EvolutionarySearch` memoises on `sketch.name + JSON.stringify(params)`');
console.log('  (search.ts:117), which is the parameter identity, not the program');
console.log('  identity, so the memo does not catch it either.');
