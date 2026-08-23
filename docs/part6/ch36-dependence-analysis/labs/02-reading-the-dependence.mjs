import {
  compile, CPUTarget, TraceLevel, randn,
} from '../../../../dist/index.node.js';

async function analyse(label, fn, inputs) {
  const snaps = [];
  const compiled = compile({ forward: fn }, inputs, {
    target: CPUTarget(),
    fusion: { enabled: false },
    scheduling: { enabled: true },
    trace: {
      level: TraceLevel.DEBUG,
      irSnapshot: { afterLowering: true, afterScheduling: true },
      sink: (e) => { if (e.type === 'ir_snapshot') snaps.push({ label: e.label.split(':')[0], text: e.text }); },
    },
  });
  await compiled(...inputs);
  const before = snaps.find((s) => s.label === 'afterLowering').text;
  const after = snaps.find((s) => s.label === 'afterScheduling').text;

  const kindOf = new Map();
  for (const m of after.matchAll(/for (\S+) in 0\.\.\S+ (?:@(\w+) )?/g)) kindOf.set(m[1], m[2] || 'serial');

  console.log(`=== ${label} ===`);
  console.log(`  ${'block'.padEnd(16)} ${'loop'.padEnd(10)} ${'in write idx'.padEnd(13)} ${'self read'.padEnd(10)} carried   scheduled`);

  const loops = [];
  for (const line of before.split('\n')) {
    const t = line.trim();
    const f = t.match(/^for (\S+) in/);
    if (f) { loops.push({ name: f[1], depth: line.search(/\S/) }); continue; }
    const blk = t.match(/^block (\S+) \{/);
    if (blk) {
      const bodyStart = before.indexOf(line) + line.length;
      const store = before.slice(bodyStart).split('\n').find((l) => /^\s+\S+\[[^\]]*\] = /.test(l));
      if (!store) continue;
      const target = store.trim().match(/^(\S+)\[([^\]]*)\] = (.*)$/);
      const [, buf, idx, rhs] = target;
      const selfRead = rhs.includes(`${buf}[`);
      for (const l of loops) {
        const binds = before.slice(bodyStart).split('\n').slice(0, 8)
          .map((s) => s.trim().match(/^bind (\S+) = (\S+)$/)).filter(Boolean);
        const iv = binds.find((b) => b[2] === l.name);
        const inWrite = idx.split(',').some((s) => s.trim() === l.name || (iv && s.trim() === iv[1]));
        const carried = !inWrite && selfRead ? 'RAW' : (!inWrite ? 'WAW' : '-');
        const sched = [...kindOf.entries()]
          .filter(([n]) => n === l.name || n.startsWith(l.name + '_'))
          .map(([, k]) => k).join(' + ') || '(gone)';
        console.log(`  ${blk[1].padEnd(16)} ${l.name.padEnd(10)} ${String(inWrite).padEnd(13)} ${String(selfRead).padEnd(10)} ${carried.padEnd(9)} ${sched}`);
      }
      loops.length = 0;
    }
    if (t === '}') loops.pop();
  }
  console.log();
}

const big = randn([32, 32]);

await analyse('x * x', (x) => x.mul(x), [big]);
await analyse('x.sum(1)', (x) => x.sum(1), [big]);
await analyse('x @ y', (x, y) => x.matmul(y), [big, randn([32, 32])]);

console.log('The rule reads straight off the table: every loop marked RAW came back');
console.log('serial, and no loop marked RAW was ever split, vectorised or made');
console.log('parallel. The loops that stayed serial without carrying a dependence');
console.log('stayed serial for a different reason — the schedule ran out of things');
console.log('worth parallelising, which is Part VII, not this chapter.');
console.log('');
console.log('This hand version only looks at whether a variable is present. The real');
console.log('one solves for the distance, which is what lets it answer "independent"');
console.log('for subscripts that do contain the variable — a[2i] against a[2i+1],');
console.log('for example, where no solution exists because 2 does not divide 1.');
