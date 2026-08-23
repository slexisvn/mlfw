import {
  compile, compileWithBackward, CPUTarget, TraceLevel, tensor, randn, ones, ops, nn,
} from '../../../../dist/index.node.js';

async function pair(label, fn, inputs) {
  const snaps = new Map();
  const compiled = compile({ forward: fn }, inputs, {
    target: CPUTarget(),
    fusion: { enabled: false },
    trace: {
      level: TraceLevel.DEBUG,
      irSnapshot: { afterGraphPasses: true, afterLowering: true },
      sink: (e) => { if (e.type === 'ir_snapshot') snaps.set(e.label.split(':')[0], e.text); },
    },
  });
  await compiled(...inputs);
  const graphOps = [...snaps.get('afterGraphPasses').matchAll(/%\d+ = ([a-z_0-9]+)\(/g)]
    .map((m) => m[1]).filter((n) => n !== 'constant');
  const tir = snaps.get('afterLowering');
  const blocks = [...tir.matchAll(/block ([a-z_]+?)_?\d* \{/g)].map((m) => m[1]);
  const loops = (tir.match(/for \w+ in/g) || []).length;
  row(label, graphOps.join(' '), blocks.join(' ') || '(none)', loops);
}

const W1 = 24, W2 = 32, W3 = 30;

function wrap(text, width) {
  const out = [];
  let line = '';
  for (const word of text.split(' ')) {
    if (line && (line + ' ' + word).length > width) { out.push(line); line = word; }
    else line = line ? line + ' ' + word : word;
  }
  out.push(line);
  return out;
}

function row(a, b, c, loops) {
  const bs = wrap(b, W2);
  const cs = wrap(c, W3);
  const h = Math.max(bs.length, cs.length);
  for (let i = 0; i < h; i++) {
    console.log(`  ${(i ? '' : a).padEnd(W1)} ${(bs[i] || '').padEnd(W2)} ${(cs[i] || '').padEnd(W3)} ${i ? '' : loops}`);
  }
}

function header() {
  console.log(`  ${'user code'.padEnd(W1)} ${'ops reaching lowering'.padEnd(W2)} ${'blocks emitted'.padEnd(W3)} loops
`);
}

const m = randn([4, 6]);
const n = randn([6, 4]);

console.log('=== one op in, one rule out ===\n');
header();

await pair('x.add(y)', (a, b) => a.add(b), [m, m]);
await pair('x.exp()', (a) => a.exp(), [m]);
await pair('ops.where(x>y, x, y)', (a, b) => ops.where(a.gt(b), a, b), [m, m]);
await pair('ops.clamp(x, 0, 1)', (a) => ops.clamp(a, 0, 1), [m]);
await pair('x.sum(1)', (a) => a.sum(1), [m]);
await pair('x.mean(1)', (a) => a.mean(1), [m]);
await pair('x.argmax(1)', (a) => a.argmax(1), [m]);
await pair('x.matmul(y)', (a, b) => a.matmul(b), [m, n]);
await pair('x.transpose(1,0)', (a) => a.transpose(1, 0), [m]);
await pair('x.reshape([3,8])', (a) => a.reshape([3, 8]), [m]);
await pair('x.flip(0)', (a) => a.flip(0), [m]);
await pair('ops.pad(x,[1,0],[1,0])', (a) => ops.pad(a, [1, 0], [1, 0]), [m]);
await pair('ops.cat([x, x], 0)', (a) => ops.cat([a, a], 0), [m]);
await pair('ops.pool2d max 2x2', (a) => ops.pool2d(a.reshape([1, 1, 4, 6]), 'max', [2, 2], [2, 2], [[0, 0], [0, 0]]), [m]);
await pair('ops.one_hot(argmax, 6)', (a) => ops.one_hot(a.argmax(1), 6), [m]);
await pair('ops.conv2d 1x1x4x6', (a, w) => ops.conv2d(a.reshape([1, 1, 4, 6]), w, [1, 1], [[0, 0], [0, 0]], [1, 1], 1), [m, randn([2, 1, 3, 3])]);

console.log('\n=== ops that never reach lowering ===\n');
header();

await pair('x.softmax(1)', (a) => a.softmax(1), [m]);
await pair('x.sigmoid()', (a) => a.sigmoid(), [m]);
await pair('x.gelu()', (a) => a.gelu(), [m]);

console.log('\n  No lowering rule exists for softmax, sigmoid or gelu, and none is needed:');
console.log('  the decomposition pass of Chapter 21 has already replaced them with');
console.log('  operations that do have one. A missing rule is a bug only if the op');
console.log('  can still be present when the lowering phase starts.');

console.log('\n\n=== a rule whose index comes from the data ===\n');
header();
await pair('table.gather(0, idx)', (t, i) => t.gather(0, i), [randn([5, 3]), tensor([[0, 2, 3], [1, 1, 4]], 'i32')]);

console.log('\n\n=== a rule only the backward pass reaches ===\n');

const conv = new nn.Conv2d(1, 2, 3, { padding: 1 });
const img = randn([1, 1, 6, 6]);
const seen = [];
const cf = compileWithBackward({ forward: (t) => conv.forward(t) }, [img], {
  target: CPUTarget(),
  fusion: { enabled: false },
  trace: {
    level: TraceLevel.DEBUG,
    irSnapshot: { afterGraphPasses: true, afterLowering: true },
    sink: (e) => { if (e.type === 'ir_snapshot') seen.push(e); },
  },
});
const out = await cf(img);
await cf.backward(ones(out.shape));

console.log(`  ${'function'.padEnd(20)} ${'ops reaching lowering'.padEnd(46)} blocks emitted\n`);
for (let i = 0; i < seen.length; i += 2) {
  const graphOps = [...seen[i].text.matchAll(/%\d+ = ([a-z_0-9]+)\(/g)].map((m) => m[1]).filter((n) => n !== 'constant');
  const blocks = [...seen[i + 1].text.matchAll(/block ([a-z_]+?)_?\d* \{/g)].map((m) => m[1]);
  const name = seen[i + 1].label.split(':')[1];
  console.log(`  ${name.padEnd(20)} ${[...new Set(graphOps)].join(' ').padEnd(46)} ${[...new Set(blocks)].join(' ')}`);
}

console.log('\n  reverse has no producer in the tracing layer — x.flip() traces to gather.');
console.log('  The conv VJP builds one to flip the kernel, so the rule runs on every');
console.log('  model that backpropagates through a convolution and on no other.');
