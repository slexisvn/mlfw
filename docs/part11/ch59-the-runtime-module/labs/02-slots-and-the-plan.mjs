import {
  compile, CPUTarget, TraceLevel, randn, manual_seed,
} from '../../_internals.mjs';

manual_seed(1);

const H = 16;
const x = randn([8, H]);
const A = randn([H, 64]);
const B = randn([64, H]);
const C = randn([H, H]);

// The graph splitter fires on a launch-boundary count. A CPU target does not declare
// one, so the labs ask for the same threshold CUDA ships with: two matmuls and the
// graph is partitioned into a plan of single-kernel steps.
const splitting = () => CPUTarget({ attrs: { graphSplit: { matmul: 2 } } });

async function planFor(fn, memory = {}) {
  let event = null;
  const compiled = compile({ forward: fn }, [x], {
    target: splitting(),
    memory,
    trace: { level: TraceLevel.DEBUG, sink: (e) => { if (e.type === 'memory') event = e; } },
  });
  const out = await compiled(x);
  const plan = compiled.result().module.executionPlan;
  return { plan, event, values: (await out.toArray()).flat(9) };
}

const chain = (t) => t.matmul(C).relu().matmul(C).relu().matmul(C);

console.log('=== a plan is slots, steps and a slot-to-buffer colouring ===\n');
const { plan } = await planFor(chain);
console.log(`  ${plan.numSlots} slots, ${plan.steps.length} steps, ${plan.intermediates.length} intermediates`);
console.log(`  argSlots ${JSON.stringify(plan.argSlots)}  (inputs, then captured weights, then the output)\n`);
console.log(`  ${'step'.padEnd(6)} ${'kernel'.padEnd(12)} in -> out`);
for (let k = 0; k < plan.steps.length; k++) {
  const s = plan.steps[k];
  console.log(`  ${String(k).padEnd(6)} ${s.name.padEnd(12)} [${s.inputSlots}] -> [${s.outputSlots}]`);
}

console.log('\n  slot  bytes  buffer');
const bytesOf = (slot) => {
  const it = plan.intermediates.find((i) => i.slot === slot);
  return it ? it.shape.reduce((a, d) => a * d, 1) * 4 : 0;
};
for (let s = 0; s < plan.numSlots; s++) {
  const pinned = plan.argSlots.includes(s);
  console.log(`  ${String(s).padStart(4)}  ${String(bytesOf(s)).padStart(5)}  ${String(plan.buffers.slotBuffer[s]).padStart(6)}${pinned ? '   (argument — pinned)' : ''}`);
}

console.log('\n=== what slot reuse and donation are each worth ===\n');
const shapes = {
  'a chain of six': (t) => { let u = t; for (let i = 0; i < 5; i++) u = u.matmul(C).relu(); return u.matmul(C); },
  'a widening chain': (t) => t.matmul(A).relu().matmul(B).relu().matmul(C),
  'two branches joined': (t) => { const u = t.matmul(C).relu(); const v = t.matmul(C).relu(); return u.add(v).matmul(C); },
};

const configs = [
  ['planReuse + planDonation (default)', {}],
  ['planReuse only', { planDonation: false }],
  ['neither', { planReuse: false }],
];

console.log(`  ${'graph'.padEnd(21)} ${'configuration'.padEnd(36)} ${'bytes'.padStart(7)} ${'buffers'.padStart(8)} ${'donated'.padStart(8)}`);
for (const [label, fn] of Object.entries(shapes)) {
  const reference = await planFor(fn, { planReuse: false });
  for (const [cfgLabel, memory] of configs) {
    const { plan: p, values } = await planFor(fn, memory);
    const total = p.buffers
      ? p.buffers.bufferBytes.reduce((a, b) => a + b, 0)
      : p.intermediates.reduce((a, it) => a + it.shape.reduce((q, d) => q * d, 1) * 4, 0);
    const buffers = p.buffers ? p.buffers.bufferBytes.filter((b) => b > 0).length : p.intermediates.length;
    const worst = Math.max(...values.map((v, i) => Math.abs(v - reference.values[i])));
    console.log(
      `  ${label.padEnd(21)} ${cfgLabel.padEnd(36)} ${String(total).padStart(7)} ${String(buffers).padStart(8)} ${String(p.buffers ? p.buffers.donated : 0).padStart(8)}`
      + `   max err ${worst.toExponential(0)}`,
    );
  }
}

console.log('\n=== the trace reports both columns ===\n');
const { event } = await planFor(chain);
console.log('  ', JSON.stringify(event.plan), ` over ${event.slots} slots`);
