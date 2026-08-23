import {
  lowerToTir, randn, CPUTarget, getSketchesForBlock, EvolutionarySearch, RandomSearch,
  Deadline, Autotuner, clonePrimFunc, TaskScheduler, GradientSchedulerPolicy,
  ScheduleSketch, SearchVariable,
  Schedule, extractBlockMini, buildBlockMap, ScheduleValidator,
} from '../../_internals.mjs';

const nextLcg = (s) => (s * 1664525 + 1013904223) & 0x7fffffff;
console.log('=== the low bits of a power-of-two LCG ===\n');
console.log('  multiplier 1664525 mod 4 =', 1664525 % 4, '   increment 1013904223 mod 4 =', 1013904223 % 4);
console.log('  so x mod 4 obeys  x <- (x + 3) mod 4,  and x mod 2 obeys  x <- (x + 1) mod 2\n');
for (const max of [2, 4, 8, 5, 48]) {
  let s = 42;
  const out = [];
  for (let i = 0; i < 16; i++) { s = nextLcg(s); out.push(s % max); }
  console.log(`  _rng(${String(max).padStart(2)}) from seed 42:  ${out.join(' ')}`);
}
console.log('\n  For a modulus that is a power of two the sequence has period equal to');
console.log('  that modulus and steps by a constant. For 5 and 48 the high bits are');
console.log('  involved and the values look random. The number of sketches is a');
console.log('  power of two on both CPU (four) and GPU (two).');

console.log('\n\n=== `_initPopulation` draws the sketch index with `_rng(sketches.length)` ===\n');
const mm = await lowerToTir((a, b) => a.matmul(b), [randn([64, 64]), randn([64, 64])]);
const sketches = getSketchesForBlock(mm, 'matmul_1', CPUTarget());
console.log(`  sketches: ${sketches.map((s, i) => `${i}=${s.name}(${s.variables.length}v)`).join('  ')}\n`);

for (const seed of [1, 7, 42, 123]) {
  let s = seed;
  const rng = (max) => { s = nextLcg(s); return s % max; };
  const picks = [];
  for (let i = 0; i < 12; i++) {
    const idx = rng(sketches.length);
    picks.push(idx);
    for (const v of sketches[idx].variables) rng(v.candidates.length);
  }
  const names = [...new Set(picks)].map((i) => sketches[i].name).join(', ');
  console.log(`  seed ${String(seed).padStart(3)}:  indices ${picks.join('')}   population drawn from {${names}}`);
}
console.log('\n  Each individual consumes one draw for the sketch plus one per variable');
console.log('  of the sketch it drew. A three-variable sketch costs four draws and');
console.log('  advances the index by 4 x 3 = 0 (mod 4) — a fixed point. The');
console.log('  one-variable sketch costs two draws and advances it by 2. So the index');
console.log('  settles within two individuals whatever the seed, and which sketch it');
console.log('  settles on is the whole of what the seed decides. The shipped default');
console.log('  is 42 (autotuner.ts:120).');

console.log('\n=== and what the whole tuner then returns ===\n');
for (const [label, opts] of [
  ['evolutionary, seed 42 (the default)', { seed: 42 }],
  ['evolutionary, seed 1', { seed: 1 }],
  ['evolutionary, seed 7', { seed: 7 }],
  ['random, seed 42', { seed: 42, strategy: 'random', numTrials: 16 }],
]) {
  const at = new Autotuner(CPUTarget(), { timeBudgetMs: 60000, useTuningDB: false, ...opts });
  const res = at.tune(clonePrimFunc(mm));
  const got = [...res.entries()].map(([b, r]) => `${b} -> ${r.sketchName}`).join('   ');
  console.log(`  ${label.padEnd(36)} ${got || '(no block tuned)'}`);
}
console.log('\n  At the shipped default seed every member of the initial population is');
console.log('  `ssrsrs_cpu`, which throws on every lowered block (§45.5), so the');
console.log('  population scores nothing, `candidates` comes back empty and');
console.log('  `matmul_1` — the only block in this function that costs anything —');
console.log('  gets no tuning result at all. `RandomSearch` is immune because it');
console.log('  iterates over the sketch list rather than sampling it (search.ts:62).');

console.log('\n\n=== elitism: best-so-far is monotone, and the memo is why ===\n');
const mini = extractBlockMini(mm, 'matmul_1', buildBlockMap(mm.body));
const mlt = sketches.find((s) => s.name === 'mlt_cpu');
const scoreOf = (sketch, params) => {
  const work = clonePrimFunc(mini);
  try { sketch.instantiate(params)(new Schedule(work), 'matmul_1', CPUTarget()); } catch (e) { return null; }
  if (ScheduleValidator.validate(work).length > 0) return null;
  return Math.log2(params.s0[0] * params.s1[0] + 1) - 0.1 * params.s0[3];
};
let calls = 0;
const seen = new Set();
const es = new EvolutionarySearch({ populationSize: 10, numGenerations: 6, seed: 3 });
const trace = [];
const r = es.search([mlt], (sketch, params) => {
  calls++;
  seen.add(JSON.stringify(params));
  const v = scoreOf(sketch, params);
  if (v !== null) trace.push(v);
  return v === null ? null : { score: v };
});
console.log(`  populationSize 10 x numGenerations 6 = 60 slots`);
console.log(`  evaluator calls: ${calls}   distinct parameter vectors: ${seen.size}`);
console.log(`  best found: ${r.candidates[0].score.toFixed(4)} at ${JSON.stringify(r.candidates[0].params)}`);
let running = -Infinity;
const perGen = [];
for (let g = 0; g * 10 < trace.length; g++) {
  const slice = trace.slice(g * 10, (g + 1) * 10);
  if (slice.length === 0) break;
  running = Math.max(running, ...slice);
  perGen.push(running.toFixed(4));
}
console.log(`  best-so-far after each block of 10 evaluations: ${perGen.join(' -> ')}`);
console.log('\n  Non-decreasing, and it has to be: `nextGen` starts as a copy of the');
console.log('  elites (search.ts:140), so the best individual of generation g is still');
console.log('  present in generation g+1. The `evalMemo` cache (search.ts:116) is what');
console.log('  makes that argument sound — a re-evaluated elite must score the same,');
console.log('  and here it is not re-evaluated at all.');

console.log('\n\n=== the deadline, and how far past it a search can run ===\n');
const dummy = new ScheduleSketch('s', [new SearchVariable('x', Array.from({ length: 97 }, (_, i) => i))], () => {});
for (const [label, make] of [
  ['RandomSearch(numTrials 1000)', (d) => new RandomSearch({ numTrials: 1000, seed: 1, deadline: d })],
  ['EvolutionarySearch(pop 8, gens 50)', (d) => new EvolutionarySearch({ populationSize: 8, numGenerations: 50, seed: 1, deadline: d })],
]) {
  let now = 0;
  let n = 0;
  const d = new Deadline(50, () => now);
  make(d).search([dummy], () => { n++; now += 10; return { score: 1 }; });
  console.log(`  ${label.padEnd(36)} budget 50 "ms", each evaluation costs 10: ${n} evaluations, clock at ${now}`);
}
console.log('\n  `RandomSearch` tests the deadline before every trial (search.ts:65), so it');
console.log('  starts none it cannot afford. `EvolutionarySearch` tests it once per');
console.log('  generation (search.ts:125) and then evaluates a whole population, so it');
console.log('  overruns by up to `populationSize` evaluations — plus one more pass,');
console.log('  because the final scoring loop (search.ts:157) has no deadline test at');
console.log('  all. Both are bounded; neither is tight.');

console.log('\n\n=== the task scheduler, with the gains it is designed for ===\n');
const policy = new GradientSchedulerPolicy();
const runOrder = (gain) => {
  const order = [];
  const mk = (w, name) => ({ weight: w, name, session: { plateaued: false, runRound() { return gain; } } });
  const tasks = [mk(1, 'weight-1'), mk(10, 'weight-10')];
  new TaskScheduler({ pick(ts) { const t = policy.pick(ts); if (t) order.push(t.name); return t; } })
    .run(tasks, new Deadline(Infinity), { maxRoundsPerTask: 4, plateauPatience: 5 });
  return order;
};
console.log(`  every round reports a finite gain of 1:   ${runOrder(1).join('  ')}`);
console.log(`  every round reports Infinity:             ${runOrder(Infinity).join('  ')}`);
console.log('\n  With finite gains the weight-10 task takes its rounds first, which is');
console.log('  the point of `priority = weight * gain` (task_scheduler.ts:26). With');
console.log('  infinite gains every priority is Infinity, `>` is false between them,');
console.log('  and the policy degenerates to list order.');
console.log('\n  `BlockTuningSession.runRound` reports its first improvement against a');
console.log('  starting best of -Infinity (session.ts:123), so the first productive');
console.log('  round of every task returns Infinity — and `gainEwma = 0.5 * Infinity +');
console.log('  0.5 * gainEwma` never decays. Every real task is in the second row.');
