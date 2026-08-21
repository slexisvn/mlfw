import {
  compile, CPUTarget, TraceLevel, randn, ops,
} from '../../../../dist/index.node.js';

// Every elementwise operation in the registry is lowered by the same function.
// The nest, the block, the read set and the store are produced once; the rule
// supplies one callback that builds the leaf expression.

function normalise(text) {
  const names = new Map();
  return text
    .replace(/buf_(\d+)/g, (_, n) => {
      if (!names.has(n)) names.set(n, `b${names.size}`);
      return names.get(n);
    })
    .replace(/v(\d+)_\d+/g, 'i$1')
    .replace(/i(\d+)_\d+/g, 'i$1');
}

async function leaf(label, fn, inputs) {
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
  const lines = normalise(snaps[0]).split('\n');
  const nests = lines.filter((l) => /^\s+for i\d+ in/.test(l)).map((l) => l.trim().replace(/ \{$/, ''));

  let start = -1;
  for (let i = 0; i < lines.length; i++) if (/^\s+b\d+\[[^\]]*\] = /.test(lines[i])) start = i;
  let depth = 0;
  let store = '';
  for (let i = start; i < lines.length; i++) {
    store += (store ? ' ' : '') + lines[i].trim();
    depth += (lines[i].match(/\{/g) || []).length - (lines[i].match(/\}/g) || []).length;
    if (depth <= 0) break;
  }
  console.log(`  ${label.padEnd(22)} ${nests.slice(-2).join(' / ').padEnd(32)} ${store}`);
}

const a = randn([2, 3]);
const b = randn([2, 3]);
const c = randn([2, 3]);

console.log('=== the same nest, six different leaves ===\n');
console.log(`  ${'op'.padEnd(22)} ${'innermost two loops'.padEnd(32)} store\n`);

await leaf('add  (binary infix)', (x, y) => x.add(y), [a, b]);
await leaf('div  (binary infix)', (x, y) => x.div(y), [a, b]);
await leaf('neg  (unary infix)', (x) => x.neg(), [a]);
await leaf('exp  (extern call)', (x) => x.exp(), [a]);
await leaf('maximum (extern call)', (x, y) => x.maximum(y), [a, b]);
await leaf('where (select)', (x, y, z) => ops.where(x.gt(y), y, z), [a, b, c]);

console.log('\n  Four of these six leaves come from one table of scalar operator names.');
console.log('  Whether the leaf prints as infix or as a call is decided by whether the');
console.log('  name is in BINARY_ARITH; nothing else about the rule changes.');
console.log('\n  Look at neg. The printer emits the operator only when the node has a');
console.log('  second operand, so a unary minus prints as a bare parenthesis. The node');
console.log('  is right, and the backend reads the node rather than the text:');
const negCompiled = compile({ forward: (x) => x.neg() }, [a], { target: CPUTarget(), fusion: { enabled: false } });
await negCompiled(a);
console.log('    ' + negCompiled.source().split('\n').find((l) => l.includes('] = ')).trim());

console.log('\n\n=== broadcasting is not a loop ===\n');
console.log(`  ${'op'.padEnd(22)} ${'innermost two loops'.padEnd(32)} store\n`);
await leaf('[2,3] + [2,3]', (x, y) => x.add(y), [a, b]);
await leaf('[2,3] + [1,3]', (x, y) => x.add(y), [a, randn([1, 3])]);
await leaf('[2,3] + [2,1]', (x, y) => x.add(y), [a, randn([2, 1])]);
await leaf('[2,3] + scalar', (x) => x.add(2.0), [a]);

console.log('\n  A broadcast operand is read with a literal 0 in the broadcast axis.');
console.log('  The size-1 buffer is never expanded, and the nest is the same in all four.');
