import '../../../tools/freshness.mjs';
import {
  compile, CPUTarget, TraceLevel, randn, manual_seed,
} from '../../../../dist/index.node.js';

manual_seed(11);

const N = 32;
const x = randn([N, N]);
const chain = (t) => t.mul(2).add(1).relu().mul(3);

async function build(memory) {
  let plan = null;
  const compiled = compile({ forward: chain }, [x], {
    target: CPUTarget(),
    fusion: { enabled: false },
    memory,
    trace: { level: TraceLevel.DEBUG, sink: (e) => { if (e.type === 'memory') plan = e; } },
  });
  const out = await compiled(x);
  return { plan, source: compiled.source(), out: (await out.toArray()).flat(9) };
}

function arenaOf(source) {
  const size = Number(source.match(/new ArrayBuffer\((\d+)\)/)?.[1] ?? 0);
  const placed = [...source.matchAll(/const (buf_\d+) = new Float32Array\(_mem_pool, (\d+), (\d+)\)/g)]
    .map((m) => ({ name: m[1], offset: Number(m[2]), elems: Number(m[3]) }))
    .sort((a, b) => a.offset - b.offset);
  const outside = [...source.matchAll(/const (buf_\d+) = new Float32Array\((\d+)\)/g)]
    .map((m) => ({ name: m[1], bytes: Number(m[2]) * 4 }));
  return { size, placed, outside };
}

const pooled = await build({ poolAllocation: true, inplaceReuse: false });
const arena = arenaOf(pooled.source);

console.log('=== one ArrayBuffer, every temporary at an offset inside it ===');
console.log(`  const _mem_pool = new ArrayBuffer(${arena.size});\n`);
const byOffset = new Map();
for (const b of arena.placed) {
  if (!byOffset.has(b.offset)) byOffset.set(b.offset, []);
  byOffset.get(b.offset).push(b);
}
console.log('  offset     bytes   buffers placed there');
for (const [offset, group] of [...byOffset.entries()].sort((a, b) => a[0] - b[0])) {
  const bytes = Math.max(...group.map((b) => b.elems * 4));
  const shared = group.length > 1 ? '   <-- sharing one address' : '';
  console.log(`  ${String(offset).padStart(6)} ${String(bytes).padStart(9)}   ${group.map((b) => b.name).join(', ')}${shared}`);
}
if (arena.outside.length > 0) {
  console.log('\n  allocated outside the arena:');
  for (const b of arena.outside) console.log(`  ${b.name.padEnd(10)} ${String(b.bytes).padStart(17)}`);
}

console.log('\n=== alignment ===');
const distinct = [...new Set(arena.placed.map((b) => b.offset))].sort((a, b) => a - b);
console.log(`  distinct offsets: ${distinct.join(', ')}`);
console.log(`  every offset is a multiple of 64: ${distinct.every((o) => o % 64 === 0)}`);
const scalarRun = distinct.filter((o) => o >= 8192);
console.log(`  the scalar run steps by ${scalarRun.slice(1).map((o, i) => o - scalarRun[i]).join(', ')} — a 4-byte scalar still costs a 64-byte slot`);

console.log('\n=== what the packing achieved ===');
const totalIfNothingShared = arena.placed.reduce((s, b) => s + Math.ceil(b.elems * 4 / 64) * 64, 0)
  + arena.outside.reduce((s, b) => s + b.bytes, 0);
console.log(`  planner's reported peak      : ${pooled.plan.peakMemory} bytes`);
console.log(`  arena actually emitted       : ${arena.size} bytes`);
console.log(`  sum of every temporary       : ${totalIfNothingShared} bytes`);
console.log(`  temporaries the planner saw  : ${pooled.plan.totalTemporaries}`);

console.log('\n=== best-fit against first-fit, same program ===');
for (const strategy of ['best-fit', 'first-fit']) {
  const r = await build({ poolAllocation: true, inplaceReuse: false, allocStrategy: strategy });
  const a = arenaOf(r.source);
  console.log(`  ${strategy.padEnd(10)} peak=${String(r.plan.peakMemory).padStart(6)}  arena=${String(a.size).padStart(6)}  offsets=[${a.placed.map((b) => b.offset).join(' ')}]`);
}

console.log('\n=== the packing did not change the answer ===');
const reference = await build({ poolAllocation: false, inplaceReuse: false });
const maxErr = Math.max(...pooled.out.map((v, i) => Math.abs(v - reference.out[i])));
console.log(`  max difference against an unpooled compile: ${maxErr}`);
