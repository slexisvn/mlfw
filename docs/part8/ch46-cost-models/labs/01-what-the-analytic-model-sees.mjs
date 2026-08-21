import {
  lowerToTir, randn, CPUTarget, WebGPUTarget, getSketchesForBlock, Schedule,
  clonePrimFunc, extractBlockMini, buildBlockMap, FeatureExtractor,
  STATEMENT_FEATURE_SCHEMA, AnalyticalCostModel, ScheduleValidator,
} from '../../_internals.mjs';

// The analytical cost model is seven bounded terms over twenty-three features.
// This lab prints the terms for one nest, then asks what happens to them as the
// schedule changes — first over the elementwise space, then over the whole of
// the multi-level tiling space.

const target = CPUTarget();
const cm = new AnalyticalCostModel(target);
const mm = await lowerToTir((a, b) => a.matmul(b), [randn([64, 64]), randn([64, 64])]);
const blockMap = buildBlockMap(mm.body);
const mini = extractBlockMini(mm, 'matmul_1', blockMap);
const mlt = getSketchesForBlock(mm, 'matmul_1', target).find((s) => s.name === 'mlt_cpu');

// -------------------------------------------------- 1. the two feature sets

console.log('=== two extractors, two disjoint feature sets ===\n');
const whole = FeatureExtractor.extract(mini);
console.log(`  FeatureExtractor.extract        -> ${Object.keys(whole).length} whole-function scalars, for the analytic model`);
console.log(`  FeatureExtractor.extractStatements -> ${FeatureExtractor.extractStatements(mini).length} vector(s) of ${STATEMENT_FEATURE_SCHEMA.length}, for the learned model`);
const READ_BY_ANALYTIC_CPU = ['numLoops', 'numParallelLoops', 'numVectorizedLoops', 'numSerialLoops',
  'innermostExtent', 'strideOneAccesses', 'nonStrideOneAccesses', 'arithmeticIntensity',
  'numMathOps', 'numExternCalls'];
const unread = Object.keys(whole).filter((k) => !READ_BY_ANALYTIC_CPU.includes(k)
  && !['threadBlockSize', 'gridSize'].includes(k));
console.log(`\n  of the ${Object.keys(whole).length} scalars, the CPU path of estimateFromFeatures reads ${READ_BY_ANALYTIC_CPU.length}:`);
console.log(`    ${READ_BY_ANALYTIC_CPU.join(', ')}`);
console.log(`  two more are GPU-only (threadBlockSize, gridSize), and ${unread.length} are extracted and never read:`);
console.log(`    ${unread.join(', ')}`);
console.log('\n  `totalIterations` is in that list. The model has no term for how much');
console.log('  work the nest does.');

// -------------------------------------------------- 2. the breakdown

console.log('\n\n=== the seven terms, on the lowered matmul nest ===\n');
const est = cm.estimateFromFeatures(whole);
const W = { parallelism: 2.0, vectorization: 1.5, memoryCoalescing: 2.0, occupancy: 1.0, arithmeticIntensity: 1.0, loopOverhead: -0.5, codeSize: -0.3 };
console.log('  term                  raw        weight    contribution');
for (const [k, v] of Object.entries(est.breakdown)) {
  console.log(`  ${k.padEnd(20)}  ${v.toFixed(6).padStart(9)}  ${String(W[k]).padStart(7)}  ${(v * W[k]).toFixed(6).padStart(14)}`);
}
console.log(`  ${'total'.padEnd(20)}  ${' '.repeat(9)}  ${' '.repeat(7)}  ${est.score.toFixed(6).padStart(14)}`);

// -------------------------------------------------- 3. a space with a gradient

console.log('\n\n=== the elementwise space: a real gradient, from one term ===\n');
const initMini = extractBlockMini(mm, 'matmul_init_0', blockMap);
const ew = getSketchesForBlock(mm, 'matmul_init_0', target)[0];
console.log('  vector_width   loops  par  vec  innermostExtent   vectorization term      score');
for (const w of ew.variables[0].candidates) {
  const work = clonePrimFunc(initMini);
  ew.instantiate({ vector_width: w })(new Schedule(work), 'matmul_init_0', target);
  const f = FeatureExtractor.extract(work);
  const e = cm.estimateFromFeatures(f);
  console.log(`  ${String(w).padStart(12)}   ${String(f.numLoops).padStart(5)}  ${String(f.numParallelLoops).padStart(3)}  ${String(f.numVectorizedLoops).padStart(3)}  ${String(f.innermostExtent).padStart(15)}   ${e.breakdown.vectorization.toFixed(4).padStart(18)}   ${e.score.toFixed(4).padStart(8)}`);
}
console.log(`\n  target.vectorWidth = ${target.vectorWidth}, and _scoreVectorization is`);
console.log('  `min(1, innermostExtent / vectorWidth)` (cost_model.ts:123). Every other');
console.log('  term is constant down the column, so the model\'s entire preference');
console.log('  over this space is "make the innermost extent at least the vector');
console.log('  width" — and Chapter 42 measured what the CPU backend does with a');
console.log('  `@vectorized` annotation: nothing. On WASM the term is meaningful.');

// -------------------------------------------------- 4. a space without one

console.log('\n\n=== the tiling space: no gradient at all ===\n');
const scores = new Map();
let n = 0;
for (const s0 of mlt.variables[0].candidates) {
  for (const s1 of mlt.variables[1].candidates) {
    const work = clonePrimFunc(mini);
    try {
      mlt.instantiate({ s0, s1, r0: mlt.variables[2].candidates[0] })(new Schedule(work), 'matmul_1', target);
    } catch (e) { continue; }
    if (ScheduleValidator.validate(work).length > 0) continue;
    const key = cm.score(work).toFixed(12);
    if (!scores.has(key)) scores.set(key, { s0, s1 });
    n++;
  }
}
console.log(`  points instantiated and validated: ${n}`);
console.log(`  distinct analytic scores:          ${scores.size}`);
for (const [k, v] of scores) console.log(`    ${k}   first reached at s0=${JSON.stringify(v.s0)} s1=${JSON.stringify(v.s1)}`);

const probe = (params) => {
  const work = clonePrimFunc(mini);
  mlt.instantiate(params)(new Schedule(work), 'matmul_1', target);
  return FeatureExtractor.extract(work);
};
const a = probe({ s0: [1, 1, 1, 64], s1: [1, 1, 1, 64], r0: [64] });
const b = probe({ s0: [64, 1, 1, 1], s1: [8, 2, 2, 2], r0: [64] });
console.log('\n  why: the features the model reads are the same for every point.');
console.log('  feature              [1,1,1,64]x[1,1,1,64]   [64,1,1,1]x[8,2,2,2]');
for (const k of READ_BY_ANALYTIC_CPU) {
  const x = typeof a[k] === 'number' ? a[k].toPrecision(6) : String(a[k]);
  const y = typeof b[k] === 'number' ? b[k].toPrecision(6) : String(b[k]);
  console.log(`  ${k.padEnd(20)} ${String(x).padStart(21)}   ${String(y).padStart(20)}`);
}
console.log('\n  The parallel loop in the second nest has extent 64 and in the first');
console.log('  extent 1, and `_scoreParallelism` on CPU is `numParallelLoops /');
console.log('  numLoops` (cost_model.ts:113) — a count ratio that never looks at an');
console.log('  extent. `innermostExtent` is the last loop the walk visits, which under');
console.log('  `mlt_cpu` is the reduction axis: the one axis the structure never');
console.log('  splits. So the term meant to score the vectorised loop is reading the');
console.log('  loop below it.');

// -------------------------------------------------- 5. the GPU path is different

console.log('\n\n=== the same question on a GPU target ===\n');
const gmm = await lowerToTir((a2, b2) => a2.matmul(b2), [randn([64, 64]), randn([64, 64])], WebGPUTarget());
const gmini = extractBlockMini(gmm, 'matmul_1', buildBlockMap(gmm.body));
const gcm = new AnalyticalCostModel(WebGPUTarget());
const gmlt = getSketchesForBlock(gmm, 'matmul_1', WebGPUTarget()).find((s) => s.name === 'mlt_gpu');
const gscores = new Set();
let gn = 0;
for (const s0 of gmlt.variables[0].candidates) {
  for (const s1 of gmlt.variables[1].candidates) {
    const work = clonePrimFunc(gmini);
    try {
      gmlt.instantiate({ s0, s1, r0: gmlt.variables[2].candidates[0] })(new Schedule(work), 'matmul_1', WebGPUTarget());
    } catch (e) { continue; }
    gscores.add(gcm.score(work).toFixed(12));
    gn++;
  }
}
console.log(`  mlt_gpu: ${gn} points, ${gscores.size} distinct scores`);
console.log('\n  On a GPU the model is not flat, because `_scoreParallelism` and');
console.log('  `_scoreOccupancy` both read `threadBlockSize` and `gridSize`, which the');
console.log('  extractor computes from the extents of the thread-bound loops');
console.log('  (features.ts:274). Binding a loop makes its extent visible to the');
console.log('  model; parallelising one does not.');
