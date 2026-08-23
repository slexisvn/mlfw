import {
  compile, CPUTarget, TraceLevel, randn, ops,
} from '../../../../dist/index.node.js';

async function pad(label, low, high, shape) {
  const snaps = [];
  const x = randn(shape);
  const compiled = compile({ forward: (t) => ops.pad(t, low, high) }, [x], {
    target: CPUTarget(),
    fusion: { enabled: false },
    trace: {
      level: TraceLevel.DEBUG,
      irSnapshot: { afterLowering: true },
      sink: (e) => { if (e.type === 'ir_snapshot') snaps.push(e.text); },
    },
  });
  await compiled(x);
  const guarded = snaps[0].split('\n').find((l) => l.includes('= if ('));
  const js = compiled.source().split('\n').find((l) => l.includes('] = ')).trim();
  const count = (s) => (s.match(/>=|<(?!=)|>(?!=)|<=/g) || []).length;
  console.log(`  ${label}`);
  if (!guarded) {
    console.log('    no pad op reached lowering: canonicalisation removed it');
    console.log(`    emitted  (${count(js)} comparisons): ${js.slice(js.indexOf('] = ') + 4)}`);
    console.log();
    return;
  }
  const tir = guarded.trim();
  console.log(`    lowered  (${count(tir)} comparisons): ${tir.slice(tir.indexOf('= if (') + 6, tir.indexOf(') {'))}`);
  console.log(`    emitted  (${count(js)} comparisons): ${js.slice(js.indexOf('] = ') + 4)}`);
  console.log();
}

console.log('=== the same rule, three padding patterns ===\n');
await pad('pad rows only    [4,3] -> [6,3]', [1, 0], [1, 0], [4, 3]);
await pad('pad columns only [4,3] -> [4,5]', [0, 1], [0, 1], [4, 3]);
await pad('pad both         [4,3] -> [6,5]', [1, 1], [1, 1], [4, 3]);
await pad('pad nothing      [4,3] -> [4,3]', [0, 0], [0, 0], [4, 3]);

console.log('  Four comparisons are emitted per two-dimensional pad, whatever the');
console.log('  padding is. Interval arithmetic over the loop extents removes exactly');
console.log('  the ones on the unpadded axes. For the column loop of extent 3 the');
console.log('  variable is bound to [0,2]; i - 0 then has bound [0,2] so i >= 0 is');
console.log('  proved, and i - 3 has bound [-3,-1] so i < 3 is proved. On the padded');
console.log('  row axis the variable is bound to [0,5], so i - 1 has bound [-1,4],');
console.log('  which straddles zero and decides nothing. The test survives.');

console.log('\n\n=== the guard that could also have gone ===\n');

const x = randn([4, 3]);
const padded = compile({ forward: (t) => ops.pad(t, [1, 0], [1, 0]) }, [x], {
  target: CPUTarget(), fusion: { enabled: false },
});
await padded(x);
console.log(padded.source().split('\n').map((l) => '  ' + l).join('\n'));
console.log('\n  The surviving test is false for exactly two of the six rows. A loop');
console.log('  split into 0..1, 1..5 and 5..6 would need no test at all — that is');
console.log('  loop partitioning, and LoopPartitionPass exists. It matches only the');
console.log('  guard shape a split with a non-dividing extent produces (Chapter 40),');
console.log('  not this one, and it is off by default.');
