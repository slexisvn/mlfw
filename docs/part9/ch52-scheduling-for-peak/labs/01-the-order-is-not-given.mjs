import {
  compile, CPUTarget, TraceLevel, randn, manual_seed,
} from '../../../../dist/index.node.js';

manual_seed(3);

const BIG = 4096;
const a = randn([2, BIG]);
const b = randn([2, BIG]);
const c = randn([2, BIG]);

const wideThenNarrow = (p, q, r) => {
  const wa = p.mul(p);
  const wb = q.mul(q);
  const wc = r.mul(r);
  const ra = wa.sum(1);
  const rb = wb.sum(1);
  const rc = wc.sum(1);
  return ra.add(rb).add(rc);
};

async function run(fn, inputs, scheduleForPeak) {
  const events = [];
  const compiled = compile({ forward: fn }, inputs, {
    target: CPUTarget(),
    fusion: { enabled: false },
    memory: { scheduleForPeak },
    trace: { level: TraceLevel.DEBUG, sink: (e) => events.push(e) },
  });
  const out = await compiled(...inputs);
  return {
    reorder: events.find((e) => e.type === 'function' && e.phase === 'memoryScheduling') ?? null,
    plan: events.find((e) => e.type === 'memory') ?? null,
    values: (await out.toArray()).flat(9),
  };
}

console.log('=== three wide intermediates, produced before any is consumed ===\n');
console.log('  const wa = p.mul(p), wb = q.mul(q), wc = r.mul(r);   // three 32 KiB buffers');
console.log('  const ra = wa.sum(1), rb = wb.sum(1), rc = wc.sum(1); // each collapses to 8 bytes');
console.log('  return ra.add(rb).add(rc);\n');

const asWritten = await run(wideThenNarrow, [a, b, c], false);
const reordered = await run(wideThenNarrow, [a, b, c], true);

console.log(`  scheduleForPeak=false   planned peak = ${String(asWritten.plan.peakMemory).padStart(6)} bytes   pass reported: ${asWritten.reorder ? 'reordered' : 'no change'}`);
console.log(`  scheduleForPeak=true    planned peak = ${String(reordered.plan.peakMemory).padStart(6)} bytes   pass reported: ${reordered.reorder ? `${reordered.reorder.originalPeakBytes} -> ${reordered.reorder.peakBytes}` : 'no change'}`);
console.log(`\n  ratio: ${(asWritten.plan.peakMemory / reordered.plan.peakMemory).toFixed(2)}x`);

const maxErr = Math.max(...asWritten.values.map((v, i) => Math.abs(v - reordered.values[i])));
console.log(`  the two orders compute the same numbers: max difference ${maxErr}`);
console.log(`  output: ${reordered.values.map((v) => v.toFixed(3)).join(', ')}`);

console.log('\n=== the same computation, already written in a good order ===\n');
const narrowEagerly = (p, q, r) => {
  const ra = p.mul(p).sum(1);
  const rb = q.mul(q).sum(1);
  const rc = r.mul(r).sum(1);
  return ra.add(rb).add(rc);
};
const already = await run(narrowEagerly, [a, b, c], true);
const alreadyOff = await run(narrowEagerly, [a, b, c], false);
console.log(`  scheduleForPeak=false   planned peak = ${String(alreadyOff.plan.peakMemory).padStart(6)} bytes   pass reported: ${alreadyOff.reorder ? 'reordered' : 'no change'}`);
console.log(`  scheduleForPeak=true    planned peak = ${String(already.plan.peakMemory).padStart(6)} bytes   pass reported: ${already.reorder ? `${already.reorder.originalPeakBytes} -> ${already.reorder.peakBytes}` : 'no change'}`);
console.log('\n  written this way the program is already near the floor without the pass, and the pass');
console.log('  shaves the remainder. Both source orders end within 64 bytes of each other once it runs,');
console.log('  which is the point: the pass makes the order you wrote stop mattering.');

console.log('\n=== a program with no independent work to move ===\n');
const chain = (t) => t.mul(2).add(1).relu().mul(3);
const chained = await run(chain, [a], true);
console.log(`  planned peak = ${chained.plan.peakMemory} bytes   pass reported: ${chained.reorder ? 'reordered' : 'no change'}`);
console.log('  every statement depends on the one before it, so the dependence graph admits exactly one order.');
