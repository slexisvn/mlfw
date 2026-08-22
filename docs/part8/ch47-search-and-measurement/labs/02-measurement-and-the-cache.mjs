import {
  lowerToTir, randn, CPUTarget, WasmTarget, Autotuner, robustStats,
  computeWorkloadKey, buildBlockMap, getSketchesForBlock, collectAllBlockNames,
  TuningDatabase, TuningRecord, CODEGEN_VERSION,
  compileGraph, buildFunction, TensorType, ScalarType, TraceLevel,
  Buffer, PrimFunc, BlockNode, BlockRealizeNode, BufferStoreNode, BufferLoadNode,
  VariableNode, IntImmNode, MathOpNode, ForNode, ForKind,
} from '../../_internals.mjs';

// Measuring is the only part of an autotuner that is not a pure function, and
// the tuning database is the only part that outlives the process. This lab
// looks at what the measurement summarises away, what the cache key keeps, and
// what a cache hit reproduces.

// -------------------------------------------------- 1. the statistics

console.log('=== robustStats: what a benchmark reports (benchmark.ts:71) ===\n');
console.log(`  ${'samples'.padEnd(30)} ${'median'.padStart(12)} ${'min'.padStart(7)} ${'trimmedMean'.padStart(13)} ${'cv'.padStart(8)}`);
for (const [label, s] of [
  ['ten clean runs', [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
  ['nine clean, one interrupted', [1, 1, 1, 1, 1, 1, 1, 1, 1, 100]],
  ['a slow first run (no warmup)', [50, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
  ['a machine under load', [1, 3, 2, 9, 1, 7, 2, 8, 1, 4]],
]) {
  const r = robustStats(s);
  console.log(`  ${label.padEnd(30)} ${r.median.toFixed(2).padStart(12)} ${r.min.toFixed(2).padStart(7)} ${r.trimmedMean.toFixed(4).padStart(13)} ${r.cv.toFixed(4).padStart(8)}`);
}
console.log('\n  The score the search learns from is `-medianMs` (session.ts:206), so');
console.log('  the outlier in row two costs nothing and the missing warmup in row');
console.log('  three costs nothing either. `cv` is computed and only acted on when');
console.log('  `maxCv > 0`; the default is 0 (autotuner.ts:130), so the re-measure');
console.log('  loop (benchmark.ts:184) always breaks after one round.');

// -------------------------------------------------- 2. the workload key

console.log('\n\n=== the workload key: what makes two blocks the same problem ===\n');
const shapes = [[8, 8, 8], [16, 8, 8], [8, 8, 16]];
console.log(`  ${'program'.padEnd(34)} ${'block'.padEnd(15)} key`);
for (const [M, K, N] of shapes) {
  const pf = await lowerToTir((a, b) => a.matmul(b), [randn([M, K]), randn([K, N])]);
  const bm = buildBlockMap(pf.body);
  for (const name of collectAllBlockNames(pf.body).slice().reverse()) {
    console.log(`  ${`matmul ${M}x${K} @ ${K}x${N}`.padEnd(34)} ${name.padEnd(15)} ${computeWorkloadKey(pf, name, CPUTarget(), bm)}`);
  }
}
const one = await lowerToTir((a, b) => a.matmul(b), [randn([8, 8]), randn([8, 8])]);
console.log(`\n  ${'the 8x8 block on WasmTarget()'.padEnd(34)} ${'matmul_1'.padEnd(15)} ${computeWorkloadKey(one, 'matmul_1', WasmTarget(), buildBlockMap(one.body))}`);
console.log(`  ${'... on CPUTarget({vectorWidth:4})'.padEnd(34)} ${'matmul_1'.padEnd(15)} ${computeWorkloadKey(one, 'matmul_1', CPUTarget({ vectorWidth: 4 }), buildBlockMap(one.body))}`);
console.log(`  CPUTarget().name is "${CPUTarget().name}" for both, and the key ends with`);
console.log('  `target.name` and `target.kind` (workload_key.ts:57).');

// The key is built from the block's declared buffers and its expression tree.
// Nothing in it describes the loop nest.
function nest(extent) {
  const A = new Buffer('A', [64], 'f32', 'global');
  const C = new Buffer('C', [64], 'f32', 'global');
  const i = new VariableNode('i', 'int32');
  const v = new VariableNode('v', 'int32');
  const store = new BufferStoreNode(C, [v], new MathOpNode('+', new BufferLoadNode(A, [v]), new IntImmNode(1)));
  const block = new BlockNode('e', [new BlockRealizeNode(v, i)], [{ buffer: A }], [{ buffer: C }], store);
  return new PrimFunc('f', [], new ForNode(i, new IntImmNode(0), new IntImmNode(extent), ForKind.SERIAL, block), new Map([['A', A], ['C', C]]));
}
console.log('\n  three hand-built nests over the same buffers, different trip counts:\n');
console.log('  loop extent   key         elementwise_cpu behaviour');
for (const e of [64, 32, 3]) {
  const pf = nest(e);
  const sk = getSketchesForBlock(pf, 'e', CPUTarget())[0];
  const usable = sk.variables[0].candidates.filter((w) => e >= w * 2);
  console.log(`  ${String(e).padStart(11)}   ${computeWorkloadKey(pf, 'e', CPUTarget())}    widths that split: {${usable.join(', ')}}`);
}
console.log('\n  One key, three iteration domains, three different sets of usable');
console.log('  parameters. `computeWorkloadKey` records buffer shapes, dtypes and the');
console.log('  block\'s expression tree (workload_key.ts:30-38); the loop nest is not');
console.log('  part of it. For a matmul the shapes determine the extents and the key');
console.log('  is adequate; for a block whose domain is not its buffer shape it is not.');

// ------------------------------------------- 2b. and the key is 32 bits wide

// The same block over buffers of a different size — so the description the key
// is built from really does differ.
function sized(n) {
  const A = new Buffer('A', [n], 'f32', 'global');
  const C = new Buffer('C', [n], 'f32', 'global');
  const i = new VariableNode('i', 'int32');
  const v = new VariableNode('v', 'int32');
  const store = new BufferStoreNode(C, [v], new MathOpNode('+', new BufferLoadNode(A, [v]), new IntImmNode(1)));
  const block = new BlockNode('e', [new BlockRealizeNode(v, i)], [{ buffer: A }], [{ buffer: C }], store);
  return new PrimFunc('f', [], new ForNode(i, new IntImmNode(0), new IntImmNode(n), ForKind.SERIAL, block), new Map([['A', A], ['C', C]]));
}

console.log('\n\n=== two different workloads, one key ===\n');
for (const n of [10039, 10040, 11827]) {
  console.log(`  the same block over buffers of shape [${String(n).padStart(5)}]   key = ${computeWorkloadKey(sized(n), 'e', CPUTarget())}`);
}
console.log('\n  `computeWorkloadKey` ends in `fnv1a(parts.join("|"))` (workload_key.ts:60),');
console.log('  a 32-bit hash rendered as eight hex digits, so two descriptions that do');
console.log('  differ still collide with probability about 2^-32 each. A birthday search');
console.log('  over the descriptions this compiler actually builds finds a pair well');
console.log('  inside the range of real tensor sizes: 10,039 and 11,827 elements are the');
console.log('  same problem to the database, and 10,040 is not.');
console.log('  Nothing can detect it. `TuningRecord` stores the key and never the');
console.log('  description it hashed (tuning_db.ts:29), so a lookup has nothing to');
console.log('  re-check against, and `lookup` returns the record for whichever workload');
console.log('  was tuned first.');

// -------------------------------------------------- 3. the database

console.log('\n\n=== the database: ranking, versioning, and what it keeps ===\n');
const db = new TuningDatabase(1);
db.store('k', new TuningRecord('k', 'elementwise_cpu', { vector_width: 8 }, 5.0, [{ primitive: 'parallelize', args: ['i'] }], 1));
db.store('k', new TuningRecord('k', 'elementwise_cpu', { vector_width: 2 }, 3.9, null, 1));
const measured = new TuningRecord('k', 'elementwise_cpu', { vector_width: 4 }, -0.9, null, 1);
measured.medianMs = 0.9;
db.store('k', measured);
console.log('  three records under one key, after `rankRecords` (tuning_db.ts:53):');
for (const r of db.lookupTopK('k', 5)) console.log(`    ${JSON.stringify(r.params).padEnd(22)} score ${String(r.score).padStart(5)}   medianMs ${r.medianMs}`);
console.log('\n  Measured records sort before unmeasured ones and among themselves by');
console.log('  ascending time; unmeasured ones by descending score. Two scales, never');
console.log('  compared, because the measured/unmeasured split is tested first.');

const blob = db.serialize();
const { codegenVersion, ...noCodegen } = blob;
const { scheduleSemanticsVersion, ...noSemantics } = blob;
const { codegenVersion: _c, scheduleSemanticsVersion: _s, ...preVersioning } = blob;
console.log(`\n  stamps written:                     ${blob.codegenVersion} / ${blob.scheduleSemanticsVersion}`);
console.log(`  reload with both unchanged:         ${TuningDatabase.deserialize(blob).size} records`);
console.log(`  reload with a foreign codegen:      ${TuningDatabase.deserialize({ ...blob, codegenVersion: 'mlfw-codegen-2' }).size} records`);
console.log(`  reload with a foreign semantics:    ${TuningDatabase.deserialize({ ...blob, scheduleSemanticsVersion: 'mlfw-schedule-1' }).size} records`);
console.log(`  reload with codegen absent:         ${TuningDatabase.deserialize(noCodegen).size} records`);
console.log(`  reload with semantics absent:       ${TuningDatabase.deserialize(noSemantics).size} records`);
console.log(`  reload of a genuine pre-versioning file (neither field): ${TuningDatabase.deserialize(preVersioning).size} records`);
const odd = new TuningDatabase(1);
odd.store('z', new TuningRecord('z', 's', {}, 1, null, 9));
console.log(`  a record claiming version 9, stored into a version-1 database: kept, version ${odd.lookup('z').version}`);
console.log('\n  Two guards, and only one of them has an `!== undefined` escape.');
console.log('  `codegenVersion` is exempt when absent (tuning_db.ts:130), which on its');
console.log('  own would have let a pre-versioning file through. `scheduleSemanticsVersion`');
console.log('  has no such exemption (tuning_db.ts:133), so a file missing it is rejected —');
console.log('  and a genuine pre-versioning file is missing both. The hole in the first');
console.log('  guard is real and is covered by the second.');
console.log('  The per-record `version` is still stored and never compared.');

// -------------------------------------------------- 4. what a cache hit reproduces

console.log('\n\n=== a cache hit, end to end ===\n');
const F = ScalarType.F32;
const T = (sh) => new TensorType(sh, F);
const mk = () => buildFunction('mm', [T([6, 4]), T([4, 6])], [T([6, 6])], (b, a) => {
  b.returnOp([b.relu(b.matmul(a[0], a[1]).getResult(0)).getResult(0)]);
});
const shared = new TuningDatabase();
const sched = { enabled: true, autotune: true, strategy: 'random', numTrials: 8, seed: 1, tuningDB: shared };
const normalise = (src) => src.replace(/_\d+/g, '_N');

const first = compileGraph(mk(), CPUTarget(), { scheduling: { ...sched } });
console.log(`  after the first compile the database holds ${shared.size} record(s)`);
const events = [];
const second = compileGraph(mk(), CPUTarget(), {
  scheduling: { ...sched },
  trace: { level: TraceLevel.VERBOSE, sink: (e) => events.push(e) },
});
const stats = events.filter((e) => e.type === 'autotune');
console.log(`  the second compile reports ${stats.map((e) => `${e.cacheHits}/${e.blockCount} cache hits`).join(', ')}`);
console.log(`  the two kernels agree up to loop-variable numbering: ${normalise(first.getSource('mm')) === normalise(second.getSource('mm'))}`);
console.log(`  byte-for-byte identical: ${first.getSource('mm') === second.getSource('mm')}`);

const A = new Float32Array(24), B = new Float32Array(24);
for (let i = 0; i < 24; i++) { A[i] = Math.cos(i * 0.9) * 1.2; B[i] = Math.sin(i * 0.4); }
const ref = new Float32Array(36), out = new Float32Array(36);
compileGraph(mk(), CPUTarget(), {}).run('mm', A, B, ref);
second.run('mm', A, B, out);
let worst = 0;
for (let i = 0; i < 36; i++) worst = Math.max(worst, Math.abs(ref[i] - out[i]));
console.log(`  and against the untuned baseline: max |difference| = ${worst.toExponential(2)}`);
console.log('\n  A cache hit reproduces the sketch name and the parameters');
console.log('  (autotuner.ts:230) and re-derives the sketch from the block, so the');
console.log('  kernel comes out the same up to the fresh-variable counter — which is');
console.log('  a module global and is not part of anything the database stores.');
console.log('  Chapter 48 is about the object that would have to carry it.');

// -------------------------------------------------- 5. the shipped default

console.log('\n\n=== is anything measured at all? ===\n');
const probe = new Autotuner(CPUTarget(), {});
console.log(`  new Autotuner(CPUTarget(), {}):  hardwareMeasure=${probe.config.hardwareMeasure}  measurer=${probe.config.measurer}  enableBenchmark=${probe.config.enableBenchmark}`);
console.log(`  benchmarkRunner: ${probe.benchmarkRunner}`);
const withBench = new Autotuner(CPUTarget(), { enableBenchmark: true });
console.log(`  new Autotuner(CPUTarget(), { enableBenchmark: true }): benchmarkRunner is a ${withBench.benchmarkRunner.constructor.name}`);
console.log('\n  `enableBenchmark` defaults to `hardwareMeasure || !!measurer`');
console.log('  (autotuner.ts:127), both of which are off for a CPU compile. So the');
console.log('  shipped CPU pipeline runs the search, never measures anything, and');
console.log('  `runRound` takes `candidates[0]` and declares itself plateaued');
console.log('  (session.ts:132-134): one round, no timings, the cost model alone.');
console.log('  `BenchmarkRunner` does work on a CPU target — it compiles the function');
console.log('  and times it (benchmark.ts:151) — it is simply not switched on.');
