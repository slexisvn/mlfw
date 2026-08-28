import {
  _traceCore, foldWeightParams, weightPredicate, MAX_FOLDABLE_ELEMENTS, firstFunction,
  compile, CPUTarget, WasmTarget, CUDATarget, WebGPUTarget,
  optimizationCandidates, selectWinner, candidateByName, gateCacheKey, graphSignature,
  BASELINE, DEFAULT_MIN_GAIN, randn, zeros, manual_seed,
} from '../../_internals.mjs';

const contiguous = (t) => t.contiguous().data;

console.log('=== folding a weight turns a parameter into a constant ===\n');
console.log(`  MAX_FOLDABLE_ELEMENTS = ${MAX_FOLDABLE_ELEMENTS}\n`);
console.log(`  ${'candidate'.padEnd(16)} ${'numel'.padStart(6)}  foldable?`);
manual_seed(7);
for (const [label, t] of [
  ['f32 [8,16]', randn([8, 16])],
  ['f32 [16]', randn([16])],
  ['f32 [32,32]', randn([32, 32])],
  ['f32 [33,32]', randn([33, 32])],
  ['i32 [8,16]', zeros([8, 16], { dtype: 'i32' })],
  ['f32 scalar', randn([])],
]) {
  console.log(`  ${label.padEnd(16)} ${String(t.numel).padStart(6)}  ${weightPredicate()(t)}`);
}
console.log('\n  Rank 1 is excluded, so a bias is never folded however small it is.');

console.log('\n=== what that does to the traced function ===\n');
for (const [label, dims] of [['all weights small', 16], ['one weight large', 256]]) {
  manual_seed(7);
  const W1 = randn([8, dims]);
  const b1 = randn([dims]);
  const W2 = randn([dims, 4]);
  const b2 = randn([4]);
  const traced = await _traceCore((x) => x.matmul(W1).add(b1).relu().matmul(W2).add(b2), [randn([2, 8])], { name: 'mlp' });
  const before = firstFunction(traced.graph).inputTypes.length;
  const folded = foldWeightParams(traced, contiguous);
  const after = firstFunction(folded.graph);
  const constants = [...after.ops()].filter((o) => o.opName === 'constant' && o.getAttr && o.getAttr('folded_weight')).length;
  console.log(`  ${label.padEnd(18)} arity ${before} -> ${after.inputTypes.length}`
    + `   captured ${traced.capturedParams.length} -> ${folded.capturedParams.length}   folded constants ${constants}`);
}

console.log('\n=== the optimization gate offers candidates the target admits ===\n');
console.log(`  DEFAULT_MIN_GAIN = ${DEFAULT_MIN_GAIN}\n`);
console.log(`  ${'target'.padEnd(18)} ${'tensorCore'.padEnd(11)} ${'blockedLayout'.padEnd(14)} candidates`);
for (const t of [CPUTarget(), WasmTarget(), CUDATarget(), WebGPUTarget(), CUDATarget({ supportsTensorCore: true, supportsBlockedLayout: true })]) {
  const c = optimizationCandidates(t);
  const name = t.supportsTensorCore && t.supportsBlockedLayout ? 'cuda +tc +layout' : t.name;
  console.log(`  ${name.padEnd(18)} ${String(!!t.supportsTensorCore).padEnd(11)} ${String(!!t.supportsBlockedLayout).padEnd(14)} ${c.length ? c.map((x) => x.name).join(', ') : '(none)'}`);
}

console.log('\n=== how a winner is chosen ===\n');
for (const [label, measurements] of [
  ['a clear win', [{ name: BASELINE, ms: 10, correct: true }, { name: 'layout', ms: 6, correct: true }]],
  ['a win below the floor', [{ name: BASELINE, ms: 10, correct: true }, { name: 'layout', ms: 9.7, correct: true }]],
  ['fast but wrong', [{ name: BASELINE, ms: 10, correct: true }, { name: 'layout', ms: 2, correct: false }]],
  ['two candidates', [{ name: BASELINE, ms: 10, correct: true }, { name: 'layout', ms: 8, correct: true }, { name: 'tensorize', ms: 5, correct: true }]],
]) {
  const d = selectWinner(measurements);
  console.log(`  ${label.padEnd(23)} winner ${d.winner.padEnd(10)} gain ${d.gain.toFixed(3)}`);
}
try { selectWinner([{ name: BASELINE, ms: 10, correct: false }]); }
catch (e) { console.log(`  ${'an incorrect baseline'.padEnd(23)} ${e.message}`); }
console.log(`\n  cache key: ${gateCacheKey(graphSignature(['dot', 'add', 'maximum'], [[64, 128]]), 'cuda_generic', optimizationCandidates(CUDATarget({ supportsTensorCore: true })))}`);
console.log(`  candidateByName(..., 'layout') on a CPU target -> ${JSON.stringify(candidateByName(optimizationCandidates(CPUTarget()), 'layout'))}`);

console.log('\n=== the gate, run for real, eight times over identical code ===\n');
console.log('  Each round uses a fresh input width so the decision cache does not answer for it.\n');
console.log(`  ${'N'.padStart(5)} ${'baseline'.padStart(10)} ${'layout'.padStart(10)} ${'gain'.padStart(7)}  winner`);
let picks = 0;
for (let k = 0; k < 8; k++) {
  const N = 128 + k;
  manual_seed(7);
  const W1 = randn([N, 128]);
  const b1 = randn([128]);
  const compiled = compile({ forward: (x) => x.matmul(W1).add(b1).relu() }, [randn([64, N])], {
    target: CPUTarget(), tuneOptimizations: true,
  });
  await compiled(randn([64, N]));
  const d = compiled.tuningReport()[0];
  if (d.winner !== BASELINE) picks++;
  console.log(`  ${String(N).padStart(5)} ${d.baselineMs.toFixed(3).padStart(10)} ${d.measurements[1].ms.toFixed(3).padStart(10)} ${d.gain.toFixed(3).padStart(7)}  ${d.winner}`);
}

manual_seed(7);
const Wsame = randn([128, 128]);
const bsame = randn([128]);
const sources = {};
for (const [name, optimization] of [['baseline', undefined], ['layout', { layout: true }]]) {
  const c = compile({ forward: (x) => x.matmul(Wsame).add(bsame).relu() }, [randn([64, 128])], { target: CPUTarget(), optimization });
  await c(randn([64, 128]));
  sources[name] = c.source();
}
console.log(`\n  the two configurations emit ${sources.baseline === sources.layout ? 'byte-identical' : 'different'} source`
  + ` (${sources.baseline.length} vs ${sources.layout.length} characters)`);
console.log(`  and the gate still preferred 'layout' in ${picks} of 8 rounds.`);
