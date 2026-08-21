import {
  compile, CPUTarget, TraceLevel, randn,
} from '../../../../dist/index.node.js';

// A block is the unit the scheduler moves. This lab prints three of them and
// reads off the four parts every block has: a name, a list of iteration
// variables with their bindings, a declared read set, and a declared write set.

async function tir(fn, inputs) {
  const snaps = [];
  const compiled = compile({ forward: fn }, inputs, {
    target: CPUTarget(),
    fusion: { enabled: false },
    trace: {
      level: TraceLevel.DEBUG,
      irSnapshot: { afterLowering: true },
      sink: (e) => { if (e.type === 'ir_snapshot') snaps.push(e.text); },
    },
  });
  await compiled(...inputs);
  return snaps[0];
}

function show(label, text) {
  console.log(`=== ${label} ===`);
  console.log(text.split('\n').slice(1, -1).map((l) => '  ' + l).join('\n'));
  console.log();
}

const a = randn([2, 3]);
const b = randn([3, 2]);

show('one elementwise block', await tir((x) => x.mul(x), [a]));
show('a reduction: two blocks, one buffer', await tir((x) => x.sum(1), [a]));
show('a contraction: an init block and an accumulation block', await tir((x, y) => x.matmul(y), [a, b]));

// What the block header does and does not tell you.

const reduceText = await tir((x) => x.sum(1), [a]);
const blocks = [...reduceText.matchAll(/block (\w+) \{([\s\S]*?)\n\s*\}/g)];

console.log('=== what a block header carries ===\n');
for (const [, name, body] of blocks) {
  const binds = [...body.matchAll(/bind (\w+) = (.+)/g)].map(([, v, e]) => `${v}=${e.trim()}`);
  const reads = (body.match(/reads\(\[(.*?)\]\)/) || [, '(none)'])[1];
  const writes = (body.match(/writes\(\[(.*?)\]\)/) || [, '(none)'])[1];
  console.log(`  ${name}`);
  console.log(`    iter vars : ${binds.join(', ')}`);
  console.log(`    reads     : ${reads}`);
  console.log(`    writes    : ${writes}`);
}

console.log('\n  Two things the header does not carry:');
console.log('    - the region touched inside each buffer (printed as "[...]");');
console.log('    - the kind of each iteration variable (DataPar or CommReduce),');
console.log('      which is the field the scheduler consults before anything else.');
console.log('    Both exist on the node. Neither is printed.');
