import '../../../tools/freshness.mjs';
import {
  compile, CPUTarget, TraceLevel, randn, manual_seed,
} from '../../../../dist/index.node.js';

manual_seed(5);

const N = 32;
const x = randn([N, N]);
const y = randn([N, N]);

async function plan(fn, inputs, memory = {}) {
  let event = null;
  const compiled = compile({ forward: fn }, inputs, {
    target: CPUTarget(),
    fusion: { enabled: false },
    memory,
    trace: { level: TraceLevel.DEBUG, sink: (e) => { if (e.type === 'memory') event = e; } },
  });
  const out = await compiled(...inputs);
  const source = compiled.source();
  const arena = Number(source.match(/new ArrayBuffer\((\d+)\)/)?.[1] ?? 0);
  const loose = [...source.matchAll(/new Float32Array\((\d+)\)/g)]
    .map((m) => Number(m[1]) * 4).reduce((a, b) => a + b, 0);
  return { event, allocated: arena + loose, values: (await out.toArray()).flat(9) };
}

console.log('=== which programs offer a buffer to overwrite ===\n');
const cases = [
  ['elementwise chain', (t) => t.mul(2).add(1).relu(), [x]],
  ['elementwise, two inputs', (a, b) => a.mul(2).add(b).relu(), [x, y]],
  ['matmul then relu', (a, b) => a.matmul(b).relu(), [x, y]],
  ['an intermediate read twice', (a) => { const u = a.mul(2); return u.add(u); }, [x]],
];
for (const [label, fn, ins] of cases) {
  const r = await plan(fn, ins);
  console.log(`  ${label.padEnd(28)} temporaries=${String(r.event.totalTemporaries).padStart(2)}   in-place candidates=${r.event.totalInplace}`);
}

console.log('\n=== what the plan says, and what the program allocates ===\n');
const chain = (t) => t.mul(2).add(1).relu().mul(3).add(0.5).tanh();
const reference = await plan(chain, [x], { inplaceReuse: false });

const rows = [];
for (const poolAllocation of [false, true]) {
  for (const inplaceReuse of [true, false]) {
    const r = await plan(chain, [x], { poolAllocation, inplaceReuse });
    const maxErr = Math.max(...r.values.map((v, i) => Math.abs(v - reference.values[i])));
    rows.push({
      config: `poolAllocation=${String(poolAllocation).padEnd(5)} inplaceReuse=${String(inplaceReuse).padEnd(5)}`,
      planned: r.event.peakMemory,
      candidates: r.event.totalInplace,
      allocated: r.allocated,
      maxErr,
    });
  }
}

console.log(`  ${'configuration'.padEnd(44)} ${'plan says'.padStart(10)} ${'candidates'.padStart(11)} ${'allocates'.padStart(10)} ${'max err'.padStart(9)}`);
for (const r of rows) {
  console.log(`  ${r.config.padEnd(44)} ${String(r.planned).padStart(10)} ${String(r.candidates).padStart(11)} ${String(r.allocated).padStart(10)} ${r.maxErr.toExponential(0).padStart(9)}`);
}

const shipped = rows.find((r) => r.config.startsWith('poolAllocation=false') && r.config.includes('inplaceReuse=true '));
const off = rows.find((r) => r.config.startsWith('poolAllocation=false') && r.config.includes('inplaceReuse=false'));

console.log('\n=== reading the two columns against each other ===\n');
console.log(`  the shipped default plans for ${shipped.planned} bytes and allocates ${shipped.allocated}`);
console.log(`  turning the feature off plans for ${off.planned} bytes and allocates ${off.allocated}`);
console.log(`  so the reported peak falls by ${(off.planned / shipped.planned).toFixed(2)}x and the bytes the program allocates by ${(off.allocated / shipped.allocated).toFixed(2)}x`);
console.log(`  every configuration computes the same numbers: max error ${Math.max(...rows.map((r) => r.maxErr))}`);
