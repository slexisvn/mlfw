import {
  compile, CPUTarget, TraceLevel, tensor, ops,
} from '../../../../dist/index.node.js';

async function study(label, fn, inputs) {
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
  const graph = snaps.get('afterGraphPasses');
  const tir = snaps.get('afterLowering');
  console.log(`=== ${label} ===`);
  console.log('  graph:');
  for (const l of graph.split('\n').filter((l) => /%\d+ = /.test(l))) console.log('    ' + l.trim());
  console.log('  TIR:');
  for (const l of tir.split('\n').filter((l) => /for |block |buffer_map|\] = /.test(l))) console.log('    ' + l.trim());
  const bufs = (tir.match(/buf_\d+/g) || []);
  console.log(`  distinct buffers: ${new Set(bufs).size}`
    + `   loop nests: ${(tir.match(/block \w+ \{/g) || []).length}`);
  console.log();
}

const row = tensor([[10, 20, 30]]);

await study(
  'broadcast_in_dim -> mul   (consumer is elementwise)',
  (r) => ops.broadcast_in_dim(r, [4, 3], [0, 1]).mul(2.0),
  [row],
);

await study(
  'broadcast_in_dim -> sum   (consumer is a reduction)',
  (r) => ops.broadcast_in_dim(r, [4, 3], [0, 1]).sum(0),
  [row],
);

console.log('The broadcast op is identical in both graphs. In the first it produces');
console.log('no statement at all: the [1,3] buffer is read as buf_1[0, j] from inside');
console.log("the consumer's nest. In the second the 4x3 result is materialised —");
console.log('12 extra elements written and read back — because a reduce cannot');
console.log('absorb the index rewrite.');

async function tirOnly(fn, inputs) {
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

console.log('\n\n=== two graphs, one loop nest ===\n');
const m = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]);
const implicit = await tirOnly((p, q) => p.add(q), [m, row]);
const explicit = await tirOnly((p, q) => p.add(ops.broadcast_in_dim(q, [4, 3], [0, 1])), [m, row]);
console.log('  x.add(row)                                  -> add_block');
console.log('  x.add(broadcast_in_dim(row, [4,3], [0,1]))  -> add_block');
console.log(`\n  identical TIR: ${implicit === explicit}`);
console.log('\n  Lowering is a function, not a bijection. Two graphs that Part IV would');
console.log('  treat as different programs — different op counts, different fusion');
console.log('  candidates — arrive here as the same loop nest, and nothing downstream');
console.log('  can tell them apart. That is the price of the level change, and the');
console.log('  reason every decision that needs the graph has already been made.');
