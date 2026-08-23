import {
  lowerToTir, randn, CPUTarget, CUDATarget, Schedule, ScheduleTrace, clonePrimFunc, printTensorIR,
  deriveSketches, analyzePureMatmul, TuningDatabase, TuningRecord,
  compileGraph, buildFunction, TensorType, ScalarType,
} from '../../_internals.mjs';

const QUERIES = new Set(['getBlockSRef', 'getBlock', 'getLoops', 'getTrace', 'verify']);
const members = Object.getOwnPropertyNames(Schedule.prototype)
  .filter((n) => n !== 'constructor' && !n.startsWith('_'))
  .filter((n) => typeof Object.getOwnPropertyDescriptor(Schedule.prototype, n).value === 'function');
const primitives = members.filter((n) => !QUERIES.has(n));
const recordsOwnStep = (n) => /trace\.record\(/.test(Schedule.prototype[n].toString());

console.log('=== the public surface of `Schedule` ===\n');
console.log(`  ${members.length} methods = ${QUERIES.size} queries + ${primitives.length} primitives\n`);
const silent = primitives.filter((n) => !recordsOwnStep(n));
console.log(`  primitives that record a step of their own: ${primitives.length - silent.length}`);
console.log(`  primitives that do not:                     ${silent.length}   ${silent.join(', ')}`);
console.log('\n  Three of the four are composites and are recorded as their parts:');
console.log('  `tile` is splits and a reorder, `computeInline` and');
console.log('  `computeInlineBlock` both route through `_applyInline`, which records');
console.log('  under the name it was called with (schedule.ts:890). `tensorize` is not');
console.log('  a composite. It sets a function attribute (schedule.ts:1099) and');
console.log('  records nothing, so a schedule that tensorises has a trace that does');
console.log('  not describe it.');

console.log('\n\n=== `annotate` records whatever it is given ===\n');
{
  const base = await lowerToTir((a, b) => a.matmul(b), [randn([8, 8]), randn([8, 8])]);
  for (const [label, v] of [['4', 4], ["'unroll'", 'unroll'], ['[2, 4]', [2, 4]],
    ['1n  (BigInt)', 1n], ['() => 1', () => 1], ['undefined', undefined]]) {
    const sch = new Schedule(clonePrimFunc(base));
    sch.annotate(sch.getLoops('matmul_1')[0], 'pragma', v);
    let out;
    try {
      out = JSON.stringify(sch.trace.serialize()[0].args);
    } catch (e) {
      out = `${e.constructor.name}: ${e.message}`;
    }
    console.log(`  annotate(loop, 'pragma', ${label.padEnd(14)}) -> ${out}`);
  }
}
console.log('\n  Three of the six survive the round trip. A BigInt makes the whole');
console.log('  database unserialisable — `TuningDatabase.saveToFile` is one');
console.log('  `JSON.stringify` over every record (tuning_db.ts:146), so one such step');
console.log('  loses the file, not the step. A function and `undefined` are worse: they');
console.log('  serialise to `null`, and replaying `annotate(loop, "pragma", null)` sets');
console.log('  an annotation the recorded schedule never had.');
console.log('\n  `annotate` is one of the nine primitives with no caller in `src/`');
console.log('  (Chapter 38), so nothing in the compiler can reach this today. It is the');
console.log('  argument type, not the value, that the trace format never constrained:');
console.log('  `ScheduleArgs` is `readonly unknown[]` (trace.ts:1).');

console.log('\n\n=== `tensorize`, watched ===\n');
const mm = await lowerToTir((a, b) => a.matmul(b), [randn([16, 16]), randn([16, 16])]);
const ts = new Schedule(clonePrimFunc(mm));
ts.split(ts.getLoops('matmul_1')[0], 4);
const before = ts.trace.length;
ts.tensorize('wmma_16x16x16', { M: 16, N: 16, K: 16, a: 'A', b: 'B', c: 'C' });
console.log(`  trace steps before tensorize: ${before}   after: ${ts.trace.length}`);
console.log(`  the function did change:      TENSOR_INTRIN = ${JSON.stringify(ts.func.getAttr('tensor_intrin'))}`);
console.log('\n  Replaying this trace reproduces the split and loses the intrinsic. The');
console.log('  backend reads the attribute, so the two programs compile differently.');

console.log('\n\n=== the register-blocked GPU matmul ===\n');
const gpu = await lowerToTir((a, b) => a.matmul(b), [randn([128, 128]), randn([128, 128])], CUDATarget());
const plan = analyzePureMatmul(gpu);
const rich = deriveSketches(gpu, plan.reductionBlock, CUDATarget(), { richGpu: true })[0];
const work = clonePrimFunc(gpu);
const gsch = new Schedule(work);
rich.instantiate({ config_index: 3 })(gsch, plan.reductionBlock, CUDATarget());
console.log(`  sketch: ${rich.name}, ${rich.enumerate().length} configurations`);
console.log(`  after applying one: ${gsch.trace.length} trace steps, body replaced: ${printTensorIR(work).includes('rb_As')}`);
const record = new TuningRecord('gpu-key', rich.name, { config_index: 3 }, -1.0, gsch.trace.serialize(), 1);
console.log(`  the TuningRecord it would produce: traceData = ${JSON.stringify(record.traceData)}`);
console.log('\n  An empty trace replays to the unscheduled function. Whether that');
console.log('  matters depends on who reads the trace, which is the next section.');

console.log('\n\n=== what a cache hit is keyed on, and what it ignores ===\n');
const F = ScalarType.F32;
const T = (sh) => new TensorType(sh, F);
const mk = () => buildFunction('mm', [T([6, 4]), T([4, 6])], [T([6, 6])], (b, a) => {
  b.returnOp([b.relu(b.matmul(a[0], a[1]).getResult(0)).getResult(0)]);
});
const db = new TuningDatabase();
const sched = { enabled: true, autotune: true, strategy: 'random', numTrials: 8, seed: 1, tuningDB: db };
const normalise = (src) => src.replace(/_\d+/g, '_N');

const first = compileGraph(mk(), CPUTarget(), { scheduling: { ...sched } });
const entries = db.serialize().entries;
console.log('  records written by the first compile:');
for (const e of entries) {
  const steps = (e.traceData || []).map((st) => st.primitive).join(' ');
  console.log(`    key ${e.workloadKey}  sketch ${e.sketchName.padEnd(16)} params ${JSON.stringify(e.params).padEnd(38)}`);
  console.log(`    ${' '.repeat(13)}traceData: ${(e.traceData || []).length} steps  ${steps}`);
}

for (const e of entries) {
  const r = db.lookup(e.workloadKey);
  r.traceData = [{ primitive: 'thisWouldThrowIfAnyoneReplayedIt', args: [] }];
}
const second = compileGraph(mk(), CPUTarget(), { scheduling: { ...sched } });
console.log(`\n  every stored trace replaced with a nonsense step, then recompiled:`);
console.log(`    the compile succeeded:            ${!!second.getSource('mm')}`);
console.log(`    same kernel up to variable names: ${normalise(first.getSource('mm')) === normalise(second.getSource('mm'))}`);
console.log('\n  Nothing read them. `Autotuner.tune` serves a cache hit from the stored');
console.log('  `sketchName` and `params` (autotuner.ts:230) and `_buildTunedSchedule`');
console.log('  re-derives the sketch from the block and instantiates it');
console.log('  (autotuner.ts:323-327). `TuningRecord.traceData` is written by');
console.log('  `bestTrace()` (session.ts:151), stored, serialised to disk, and read by');
console.log('  no code in `src/`; `ScheduleTrace.replay` has no caller there either.');
console.log('\n  That is why the empty GPU trace above costs nothing today, and it is');
console.log('  also why the counter dependence of §48.5 has never been noticed: the');
console.log('  cache re-derives instead of replaying, so it needs the sketch to still');
console.log('  exist under the same name and to still accept the same parameters —');
console.log('  a different set of assumptions from the ones a trace would need.');

console.log('\n\n=== what a trace is bound to ===\n');
const other = await lowerToTir((a, b) => a.matmul(b), [randn([14, 14]), randn([14, 14])]);
const src = new Schedule(clonePrimFunc(mm));
src.split(src.getLoops('matmul_1')[0], 4);
const t = src.trace.serialize();
console.log(`  a trace recorded on a 16x16 matmul: ${JSON.stringify(t)}`);
const dst = new Schedule(other);
try {
  ScheduleTrace.deserialize(t).replay(dst);
  console.log(`  replayed onto a 14x14 matmul: succeeded, outer extent ${dst.getLoops('matmul_1')[0].extent.value}`);
  console.log(`  and a guard appeared, because 4 does not divide 14: ${printTensorIR(other).includes('if (')}`);
} catch (e) {
  console.log(`  replayed onto a 14x14 matmul: ${e.message}`);
}
console.log('\n  It transfers, because the loop is called `ls0_6` in both programs, and');
console.log('  `split` rounds up and guards when the factor does not divide the extent');
console.log('  (Theorem 40.2). What a trace does not carry is the target,');
console.log('  the block name, or the sketch and parameters it came from: those are');
console.log('  arguments to `sketch.instantiate(params)(schedule, blockName, target)`');
console.log('  and live in the `TuningRecord` beside the trace, not inside it.');
