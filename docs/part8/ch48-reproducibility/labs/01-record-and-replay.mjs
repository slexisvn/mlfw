import {
  lowerToTir, randn, CPUTarget, Schedule, ScheduleTrace,
  printTensorIR, resetVarCounter, clonePrimFunc, toKernel,
} from '../../_internals.mjs';

const target = CPUTarget();
const mk = () => lowerToTir((a, b) => a.matmul(b), [randn([8, 8]), randn([8, 8])]);

console.log('=== recording ===\n');
const p1 = await mk();
const s1 = new Schedule(p1);
s1.split(s1.getLoops('matmul_1')[0], 4);
{
  const l = s1.getLoops('matmul_1');
  s1.reorder(l[0], l[2], l[1]);
}
{
  const l = s1.getLoops('matmul_1');
  s1.parallelize(l[0]);
  s1.vectorize(l[2]);
}
const recorded = s1.trace.serialize();
for (const step of recorded) console.log(`  ${step.primitive.padEnd(12)} ${JSON.stringify(step.args)}`);
console.log(`\n  ${recorded.length} steps, and every argument is a string, a number or an`);
console.log(`  array of them: JSON round-trips exactly (${JSON.stringify(JSON.parse(JSON.stringify(recorded))) === JSON.stringify(recorded)}).`);
console.log('\n  `Schedule` records loop *names*, not loop objects (schedule.ts:314), and');
console.log('  `_resolveLoop` (schedule.ts:239) turns a name back into a `ForNode` by');
console.log('  walking the body. That is what makes a trace serialisable at all: a');
console.log('  `ForNode` is a live pointer into an IR that the next primitive replaces.');

const ir1 = printTensorIR(p1);
console.log('\n  the nest it produced:');
console.log(ir1.split('\n').filter((l) => /for /.test(l)).map((l) => '   ' + l.trim()).join('\n'));

console.log('\n\n=== replaying into a fresh copy of the same program ===\n');
const p2 = await mk();
const s2 = new Schedule(p2);
ScheduleTrace.deserialize(recorded).replay(s2);
console.log(`  identical IR: ${printTensorIR(p2) === ir1}`);
console.log(`  the replay recorded nothing of its own: ${s2.trace.length} steps`);
console.log('\n  `replay` sets `_replaying` around each call (trace.ts:54) and every');
console.log('  primitive guards its `trace.record` with it, so replaying a trace into');
console.log('  a schedule does not duplicate the trace.');

console.log('\n\n=== the same trace, replayed with the counter in another state ===\n');
const p3 = await mk();
new Schedule(clonePrimFunc(p3)).split('ls0_6', 2);
const s3 = new Schedule(p3);
try {
  ScheduleTrace.deserialize(recorded).replay(s3);
  console.log(`  identical IR: ${printTensorIR(p3) === ir1}`);
} catch (e) {
  console.log(`  replay threw: ${e.message}`);
}
console.log('\n  what changed:\n');
const namesAfterSplit = async (extraSplits) => {
  const p = await mk();
  for (let i = 0; i < extraSplits; i++) new Schedule(clonePrimFunc(p)).split('ls0_6', 2);
  const sch = new Schedule(p);
  const [o, i2] = sch.split(sch.getLoops('matmul_1')[0], 4);
  return `${o.loopVar.name} / ${i2.loopVar.name}`;
};
for (const k of [0, 1, 2, 3]) {
  console.log(`  ${k} unrelated split(s) beforehand:  split('ls0_6', 4) produces ${await namesAfterSplit(k)}`);
}
console.log('\n  `freshVar` numbers from a module-global counter (schedule.ts:193), so');
console.log('  the names a `split` introduces depend on how much scheduling has');
console.log('  happened in the process, not on the program being scheduled. The trace');
console.log('  above names `ls0_6_o_0`; replaying it after two extra variables have');
console.log('  been handed out produces `ls0_6_o_2`, and the next step asks for a loop');
console.log('  that is not there. `_resolveLoop` returns the string unchanged');
console.log('  (schedule.ts:254), so the error names a type rather than a name.');
console.log('\n  `resetVarCounter` is the only way to set the counter, it is exported');
console.log('  from the schedule module for exactly this reason, and `TuningRecord`');
console.log('  (tuning_db.ts:29) does not store it.');

console.log('\n\n=== a trace records what was done, not that it was right ===\n');
const p4 = await mk();
try {
  ScheduleTrace.deserialize([
    { primitive: 'split', args: ['ls0_6', 4] },
    { primitive: 'thisIsNotAPrimitive', args: [] },
  ]).replay(new Schedule(p4));
} catch (e) {
  console.log(`  a trace naming a primitive that does not exist: ${e.message}`);
}
console.log(`  and the split before it has already been applied: ${printTensorIR(p4).includes('_o_')}`);
console.log('\n  That is the only check `replay` performs (trace.ts:51). A step whose');
console.log('  arguments are wrong gets whatever the primitive does with them, and a');
console.log('  step that fails leaves every earlier step in place. There is no');
console.log('  transaction and no validation pass: `ScheduleValidator` is run by the');
console.log('  tuning session (session.ts:186), never by `replay`.');

const A = new Float32Array(64), B = new Float32Array(64);
for (let i = 0; i < 64; i++) { A[i] = Math.sin(i * 0.5); B[i] = Math.cos(i * 0.25); }
const ref = new Float32Array(64), got = new Float32Array(64);
toKernel(await mk(), target).call(A, B, ref);
toKernel(p2, target).call(A, B, got);
let worst = 0;
for (let i = 0; i < 64; i++) worst = Math.max(worst, Math.abs(ref[i] - got[i]));
console.log(`\n\n=== the replayed schedule computes the same product ===\n`);
console.log(`  max |difference| against the unscheduled nest: ${worst.toExponential(2)}`);
