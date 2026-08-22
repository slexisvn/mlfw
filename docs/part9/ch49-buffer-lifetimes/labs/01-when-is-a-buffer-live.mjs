import {
  compile, CPUTarget, TraceLevel, randn,
} from '../../../../dist/index.node.js';

const N = 64;

async function tirOf(fn, inputs, opts = {}) {
  let text = null;
  let memory = null;
  const compiled = compile({ forward: fn }, inputs, {
    target: CPUTarget(),
    fusion: { enabled: false },
    memory: { inplaceReuse: false },
    ...opts,
    trace: {
      level: TraceLevel.DEBUG,
      irSnapshot: { afterScheduling: true },
      sink: (e) => {
        if (e.type === 'ir_snapshot' && text === null) text = e.text;
        if (e.type === 'memory') memory = e;
      },
    },
  });
  await compiled(...inputs);
  const allocated = [...compiled.source().matchAll(/new Float32Array\((\d+)\)/g)]
    .map((m) => Number(m[1]) * 4).reduce((a, b) => a + b, 0);
  return { text, memory, allocated };
}

function blocksInOrder(text) {
  const out = [];
  const re = /block (\w+) \{([\s\S]*?)\n(\s*)\}/g;
  for (const m of text.matchAll(re)) {
    const body = m[2];
    const lineOf = (kind) => body.split('\n').find((l) => l.trim().startsWith(`${kind}(`)) ?? '';
    const names = (kind) => [...new Set([...lineOf(kind).matchAll(/buf_\d+/g)].map((b) => b[0]))];
    out.push({ name: m[1], reads: names('reads'), writes: names('writes') });
  }
  return out;
}

function intervals(blocks) {
  const live = new Map();
  blocks.forEach((blk, idx) => {
    for (const buf of [...blk.reads, ...blk.writes]) {
      const iv = live.get(buf);
      if (!iv) live.set(buf, { buf, firstUse: idx, lastUse: idx });
      else if (idx > iv.lastUse) iv.lastUse = idx;
    }
  });
  return [...live.values()].sort((a, b) => a.firstUse - b.firstUse || a.buf.localeCompare(b.buf));
}

const x = randn([N, N]);
const chain = (t) => t.mul(2).add(1).relu().mul(3);

const { text, memory, allocated } = await tirOf(chain, [x]);
const blocks = blocksInOrder(text);
const ivs = intervals(blocks);

console.log('=== the linearized program: one index per block ===');
blocks.forEach((b, i) => {
  console.log(`  ${String(i).padStart(2)}  ${b.name.padEnd(18)} reads ${(b.reads.join(' ') || '-').padEnd(20)} writes ${b.writes.join(' ') || '-'}`);
});

const span = blocks.length - 1;
console.log('\n=== live intervals, one row per buffer ===');
console.log(`      buffer      [first,last]   ${'0'.padEnd(span + 1, ' ')}`.trimEnd());
for (const iv of ivs) {
  const bar = Array.from({ length: blocks.length }, (_, i) =>
    (i >= iv.firstUse && i <= iv.lastUse) ? '#' : '.').join('');
  console.log(`      ${iv.buf.padEnd(10)}  [${String(iv.firstUse).padStart(2)},${String(iv.lastUse).padStart(2)}]      ${bar}`);
}

console.log('\n=== which pairs may share storage (disjoint intervals) ===');
let shareable = 0;
let interfering = 0;
for (let i = 0; i < ivs.length; i++) {
  for (let j = i + 1; j < ivs.length; j++) {
    const a = ivs[i];
    const b = ivs[j];
    const overlap = a.firstUse <= b.lastUse && b.firstUse <= a.lastUse;
    if (overlap) interfering++;
    else shareable++;
  }
}
console.log(`  ${shareable} disjoint pair(s), ${interfering} interfering pair(s)`);

const maxLive = Math.max(...blocks.map((_, i) =>
  ivs.filter((iv) => i >= iv.firstUse && i <= iv.lastUse).length));
console.log(`  widest point of the program: ${maxLive} buffers live at once, out of ${ivs.length} total`);
console.log(`  the planner reported peak: ${memory.peakMemory} bytes over ${memory.totalTemporaries} temporaries`);
console.log(`  the emitted program allocates: ${allocated} bytes`);

console.log('\n=== the same program, fused ===');
const fused = await tirOf(chain, [x], { fusion: { enabled: true } });
const fusedBlocks = blocksInOrder(fused.text);
const fusedIvs = intervals(fusedBlocks);
console.log(`  ${fusedBlocks.length} block(s), ${fusedIvs.length} buffer(s) touched`);
console.log(`  the planner reported peak: ${fused.memory.peakMemory} bytes over ${fused.memory.totalTemporaries} temporaries`);
