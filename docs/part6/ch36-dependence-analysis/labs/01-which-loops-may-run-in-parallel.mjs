import {
  compile, CPUTarget, TraceLevel, randn, zeros, scan,
} from '../../../../dist/index.node.js';

// Scheduling is off by default. Turned on, the CPU policy asks one question of
// every loop before it marks it parallel or vectorised: does this loop carry a
// dependence? These are the answers, read off the scheduled IR.

async function kinds(label, fn, inputs) {
  const snaps = [];
  const compiled = compile({ forward: fn }, inputs, {
    target: CPUTarget(),
    fusion: { enabled: false },
    scheduling: { enabled: true },
    trace: {
      level: TraceLevel.DEBUG,
      irSnapshot: { afterScheduling: true },
      sink: (e) => { if (e.type === 'ir_snapshot') snaps.push(e.text); },
    },
  });
  await compiled(...inputs);
  console.log(`=== ${label} ===`);
  let depth = 0;
  for (const line of snaps[0].split('\n')) {
    const t = line.trim();
    if (/^for /.test(t)) {
      const m = t.match(/^for (\S+) in 0\.\.(\S+) (?:@(\w+) )?/);
      console.log(`  ${'  '.repeat(depth)}${m[1].padEnd(14)} extent ${String(m[2]).padEnd(6)} ${m[3] ? '@' + m[3] : 'serial'}`);
      depth++;
    } else if (/^block /.test(t)) {
      console.log(`  ${'  '.repeat(depth)}[${t.replace(/ \{$/, '').slice(6)}]`);
      depth++;
    } else if (t === '}') {
      depth = Math.max(0, depth - 1);
    }
  }
  console.log();
}

const big = randn([64, 64]);
const other = randn([64, 64]);

await kinds('elementwise: x * x + x', (x) => x.mul(x).add(x), [big]);
await kinds('reduction: x.sum(1)', (x) => x.sum(1), [big]);
await kinds('contraction: x @ y', (x, y) => x.matmul(y), [big, other]);

await kinds(
  'recurrence: scan(c -> 0.9*c + x_t)',
  (xs, c0) => scan((c, xt) => { const nc = c.mul(0.9).add(xt); return [nc, nc]; }, c0, xs)[1],
  [randn([8, 4]), zeros([4])],
);

console.log('Four programs, three different answers to one question.');
console.log('  - Both elementwise loops are independent: each output element is');
console.log('    written once and read by nothing in the same nest.');
console.log('  - In the reduction the spatial loop is independent and the reduce');
console.log('    loop is not: every iteration reads the accumulator the previous');
console.log('    one wrote. Distance 0 in the spatial axis, unknown in the other.');
console.log('  - In the contraction the same thing happens on the k loop.');
console.log('  - The scan time loop is never asked: it is emitted as @recurrence,');
console.log('    a declaration by the lowering rule that the loop is sequential.');
