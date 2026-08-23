import {
  lowerToTir, randn, CPUTarget, getSketchesForBlock, Schedule, clonePrimFunc,
  extractBlockMini, buildBlockMap, FeatureExtractor, STATEMENT_FEATURE_SCHEMA,
  LearnedCostModel, AnalyticalCostModel, GuidedCostModel, EvolutionarySearch,
} from '../../_internals.mjs';

const target = CPUTarget();

const mm = await lowerToTir((a, b) => a.matmul(b).relu(), [randn([16, 16]), randn([16, 16])]);
const stmts = FeatureExtractor.extractStatements(mm);
console.log('=== one PrimFunc becomes N statement vectors, then one row ===\n');
console.log(`  relu(a @ b) has ${stmts.length} BufferStore statements, each a ${STATEMENT_FEATURE_SCHEMA.length}-vector\n`);
const MAXED = ['depth', 'threadBlockSize', 'gridSize', 'underReduction', 'vectorized', 'parallelized', 'innermostExtent'];
const MEANED = ['arithmeticIntensity'];
const at = (name) => STATEMENT_FEATURE_SCHEMA.indexOf(name);
const how = (name) => (MAXED.includes(name) ? 'max' : MEANED.includes(name) ? 'mean' : 'sum');
const aggregate = (vs, name) => {
  const col = vs.map((v) => v[at(name)]);
  const h = how(name);
  return h === 'max' ? Math.max(...col) : h === 'mean' ? col.reduce((a, b) => a + b, 0) / col.length : col.reduce((a, b) => a + b, 0);
};
console.log('  feature              ' + stmts.map((_, i) => `stmt${i}`.padStart(11)).join('') + '   aggregate   how');
for (const name of ['iterCount', 'numMathOps', 'numReads', 'numWrites', 'depth', 'innermostExtent', 'vectorized', 'arithmeticIntensity']) {
  const col = stmts.map((v) => v[at(name)]);
  console.log(`  ${name.padEnd(20)} ${col.map((x) => x.toPrecision(4).padStart(11)).join('')}   ${aggregate(stmts, name).toPrecision(5).padStart(9)}   ${how(name)}`);
}
console.log('\n  `aggregateStatements` (cost_model.ts:20) sums every feature except the');
console.log('  seven in MAX_FEATURE_NAMES and the one in MEAN_FEATURE_NAMES, then');
console.log(`  appends the statement count, giving a ${STATEMENT_FEATURE_SCHEMA.length + 1}-dimensional row. Two schedules`);
console.log('  with the same row are, to this model, the same program.');

console.log('\n\n=== Theorem 46.3, executed: monotone transforms are invisible ===\n');
const mini = extractBlockMini(mm, 'matmul_1', buildBlockMap(mm.body));
const mlt = getSketchesForBlock(mm, 'matmul_1', target).find((s) => s.name === 'mlt_cpu');
const rawScore = (sketch, params) => {
  const work = clonePrimFunc(mini);
  try { sketch.instantiate(params)(new Schedule(work), 'matmul_1', target); } catch (e) { return null; }
  const f = FeatureExtractor.extract(work);
  return 100 / (1 + f.numSerialLoops) + 3 * f.numVectorizedLoops - Math.log2(Math.max(f.outermostExtent, 1));
};
const runWith = (g) => new EvolutionarySearch({ populationSize: 12, numGenerations: 4, seed: 11 })
  .search([mlt], (s, p) => { const v = rawScore(s, p); return v === null ? null : { score: g(v) }; }).candidates;
const baseRun = runWith((x) => x);
for (const [label, g] of [['s', (x) => x], ['3s - 1000', (x) => 3 * x - 1000],
  ['exp(s / 10)', (x) => Math.exp(x / 10)], ['-1 / (s + 100)', (x) => -1 / (x + 100)]]) {
  const c = runWith(g);
  const same = c.length === baseRun.length && c.every((x, i) => JSON.stringify(x.params) === JSON.stringify(baseRun[i].params));
  console.log(`  score transform ${label.padEnd(16)} best = ${JSON.stringify(c[0].params)}  identical ranking: ${same}`);
}
console.log('\n  Four models whose predictions differ by orders of magnitude, one search');
console.log('  trajectory. Nothing downstream of `evaluator` reads the value:');
console.log('  `scored.sort((a, b) => b.score - a.score)` (search.ts:134) and `_consider`');
console.log('  (session.ts:235) are both comparisons.');

console.log('\n\n=== Counterexample 46.5: error and regret are different objectives ===\n');
console.log('  Definition 46.2 scores higher-is-better; this table is in predicted');
console.log('  *cost*, lower-is-better, so the model picks an argmin.\n');
const truth = [1.0, 2.0, 100.0];
console.log('  model                prediction            MSE   picks   true cost   regret');
for (const [name, p] of Object.entries({
  'A  perfect        ': [1.0, 2.0, 100.0],
  'B  small error    ': [2.0, 1.0, 100.0],
  'C  enormous error ': [50.0, 51.0, 60.0],
})) {
  const mse = p.reduce((a, x, i) => a + (x - truth[i]) ** 2, 0) / truth.length;
  const pick = p.indexOf(Math.min(...p));
  console.log(`  ${name}  ${JSON.stringify(p).padEnd(18)}  ${mse.toFixed(2).padStart(9)}   ${String(pick).padStart(5)}   ${truth[pick].toFixed(1).padStart(9)}   ${(truth[pick] - Math.min(...truth)).toFixed(1).padStart(6)}`);
}
console.log('\n  C\'s squared error is more than three thousand times B\'s and its regret');
console.log('  is zero. The model is fitted with `(pred - measured)^2` (gbt.ts:45) and');
console.log('  used as a comparator; the two are related but not the same objective.');

const rows = [];
for (const n of [16, 32, 64, 128]) {
  const src = await lowerToTir((a, b) => a.mul(b).relu(), [randn([n, n]), randn([n, n])]);
  const blockMini = extractBlockMini(src, 'maximum_block_1', buildBlockMap(src.body));
  const sk = getSketchesForBlock(src, 'maximum_block_1', target)[0];
  for (const w of sk.variables[0].candidates) {
    const work = clonePrimFunc(blockMini);
    sk.instantiate({ vector_width: w })(new Schedule(work), 'maximum_block_1', target);
    const v = FeatureExtractor.extractStatements(work);
    let ms = 0;
    for (const s of v) ms += (s[at('iterCount')] / 1e6) * (1 + 4 / Math.max(s[at('innermostExtent')], 1));
    rows.push({ v, y: -ms, w, n });
  }
}
const bestIdx = rows.reduce((b, r, i) => (r.y > rows[b].y ? i : b), 0);
console.log('\n\n=== the learning curve on the representation the session builds ===\n');
console.log(`  ${rows.length} samples, ${rows[0].v.length} statement per sample, best is vector_width=${rows[bestIdx].w} at n=${rows[bestIdx].n}\n`);
console.log('  trees        MSE     model argmax   regret (ms)   discordant pairs');
for (const numTrees of [1, 2, 4, 8, 16, 32, 60]) {
  const m = new LearnedCostModel(null, { numTrees });
  for (const r of rows) m.addSample(r.v, r.y);
  m.train();
  const p = rows.map((r) => m.predict(r.v));
  const mse = p.reduce((a, x, i) => a + (x - rows[i].y) ** 2, 0) / rows.length;
  const pick = p.reduce((b, x, i) => (x > p[b] ? i : b), 0);
  let disc = 0, tot = 0;
  for (let i = 0; i < rows.length; i++) {
    for (let j = i + 1; j < rows.length; j++) {
      if (rows[i].y === rows[j].y) continue;
      tot++;
      if (Math.sign(p[i] - p[j]) !== Math.sign(rows[i].y - rows[j].y)) disc++;
    }
  }
  console.log(`  ${String(numTrees).padStart(5)}  ${mse.toExponential(3).padStart(10)}   ${`w=${rows[pick].w} n=${rows[pick].n}`.padStart(12)}   ${(rows[bestIdx].y - rows[pick].y).toFixed(4).padStart(11)}   ${(100 * disc / tot).toFixed(1).padStart(15)}%`);
}
console.log('\n  Five orders of magnitude of error, and the ranking converges with it.');
console.log('  That is the usual case, and it is why fitting is worth doing. What');
console.log('  Counterexample 46.5 rules out is the converse — that a lower error is');
console.log('  by itself evidence of a better search.');

console.log('\n\n=== why the session scores the mini function and not the real one ===\n');
const big = await lowerToTir((a, b) => a.mul(b).relu(), [randn([64, 64]), randn([64, 64])]);
const bigSketch = getSketchesForBlock(big, 'maximum_block_1', target)[0];
const bigMini = extractBlockMini(big, 'maximum_block_1', buildBlockMap(big.body));
const wholeRows = new Set(), miniRows = new Set();
console.log('  vector_width   innermostExtent per statement (whole func)   max   mini func');
for (const w of bigSketch.variables[0].candidates) {
  const whole = clonePrimFunc(big);
  bigSketch.instantiate({ vector_width: w })(new Schedule(whole), 'maximum_block_1', target);
  const wv = FeatureExtractor.extractStatements(whole);
  const mn = clonePrimFunc(bigMini);
  bigSketch.instantiate({ vector_width: w })(new Schedule(mn), 'maximum_block_1', target);
  const mv = FeatureExtractor.extractStatements(mn);
  wholeRows.add(JSON.stringify(wv.map((v) => v.slice())));
  miniRows.add(JSON.stringify(mv.map((v) => v.slice())));
  console.log(`  ${String(w).padStart(12)}   ${JSON.stringify(wv.map((v) => v[at('innermostExtent')])).padStart(41)}   ${String(aggregate(wv, 'innermostExtent')).padStart(3)}   ${JSON.stringify(mv.map((v) => v[at('innermostExtent')]))}`);
}
const collapse = (set) => set.size;
const wholeAgg = new Set(), miniAgg = new Set();
for (const w of bigSketch.variables[0].candidates) {
  for (const [src, out] of [[big, wholeAgg], [bigMini, miniAgg]]) {
    const work = clonePrimFunc(src);
    bigSketch.instantiate({ vector_width: w })(new Schedule(work), 'maximum_block_1', target);
    const vs = FeatureExtractor.extractStatements(work);
    out.add(JSON.stringify(STATEMENT_FEATURE_SCHEMA.map((n) => aggregate(vs, n))));
  }
}
console.log(`\n  distinct aggregated rows over the five widths — whole function: ${collapse(wholeAgg)}, mini function: ${collapse(miniAgg)}`);
console.log('\n  On the whole function the five schedules produce one row. The block');
console.log('  being tuned has innermost extent 1, 2, 4, 8, 16; the `mul` block beside');
console.log('  it has 64; and `innermostExtent` is aggregated by maximum, so the');
console.log('  parameter under search is erased. `BlockTuningSession` escapes this by');
console.log('  evaluating `extractBlockMini(...)` instead (session.ts:104), which');
console.log('  contains one block and therefore one statement.');
console.log('\n  The label does not get the same treatment. `_measure` benchmarks the');
console.log('  whole scheduled function and pairs its median with the mini function\'s');
console.log('  features (session.ts:229-231), so in a multi-block program every');
console.log('  sample is labelled with time the block did not spend.');

console.log('\n\n=== GuidedCostModel: the handover, at eight samples ===\n');
const analytic = new AnalyticalCostModel(target);
const learned = new LearnedCostModel();
const guided = new GuidedCostModel(analytic, learned);
console.log(`  confidenceSamples = ${guided.confidenceSamples}\n`);
console.log('  samples   trained   guided.score(mini)   = analytic?   = learned?');
for (let i = 0; i < 11; i++) {
  const g = guided.score(mini);
  const eqA = Math.abs(g - analytic.score(mini)) < 1e-12;
  const eqL = learned.trained && Math.abs(g - learned.predict(FeatureExtractor.extractStatements(mini))) < 1e-12;
  console.log(`  ${String(learned.sampleCount).padStart(7)}   ${String(learned.trained).padEnd(7)}   ${g.toFixed(6).padStart(18)}   ${String(eqA).padEnd(11)}   ${eqL}`);
  learned.addSample(rows[i % rows.length].v, rows[i % rows.length].y);
  learned.train();
}
console.log('\n  The switch is discontinuous in value — an analytic score of 2.5 becomes');
console.log('  a predicted negative millisecond count — and continuous in behaviour,');
console.log('  because only the order matters. It costs `topKForBenchmark`');
console.log('  measurements per round, 5 by default, so the learned model takes over');
console.log('  in round two of any task that is measured at all.');
