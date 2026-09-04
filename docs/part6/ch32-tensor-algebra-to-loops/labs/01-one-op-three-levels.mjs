import {
  compile, CPUTarget, TraceLevel, tensor, Linear, ReLU, Sequential, manual_seed,
} from '../../../../dist/index.node.js';

async function levels(model, inputs, opts = {}) {
  const snaps = new Map();
  const compiled = compile(model, inputs, {
    target: CPUTarget(),
    fusion: { enabled: false },
    ...opts,
    trace: {
      level: TraceLevel.DEBUG,
      irSnapshot: { afterGraphPasses: true, afterLowering: true },
      sink: (e) => { if (e.type === 'ir_snapshot') snaps.set(e.label.split(':')[0], e.text); },
    },
  });
  await compiled(...inputs);
  return { graph: snaps.get('afterGraphPasses'), tir: snaps.get('afterLowering'), src: compiled.source() };
}

const indent = (text) => text.split('\n').map((l) => '  ' + l).join('\n');

const a = tensor([[1, 2], [3, 4]]);
const b = tensor([[5, 6], [7, 8]]);
const one = await levels({ forward: (p, q) => p.add(q) }, [a, b]);

console.log('=== 1. graph IR — whole tensors, no indices ===');
console.log(indent(one.graph));
console.log('\n=== 2. TIR — loops, buffers, one scalar store ===');
console.log(indent(one.tir));
console.log('\n=== 3. emitted CPU source — flat offsets ===');
console.log(indent(one.src));

manual_seed(0);
const model = new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
const x = tensor([[0.5, -1.5], [1.0, 2.0]]);

console.log('\n\n=== the running example, as a loop nest ===');
console.log('  Sequential(Linear(2,8), ReLU(), Linear(8,1)) — the program from Chapter 1\n');
const run = await levels(model, [x], { fusion: { enabled: true } });
console.log(indent(run.tir));

console.log('\n\n=== the same program, counted at each level ===\n');

const counts = (r) => ({
  ops: (r.graph.match(/%\d+ = "?tera\./g) || []).length,
  loops: (r.tir.match(/for \w+ in/g) || []).length,
  blocks: (r.tir.match(/block \w+ \{/g) || []).length,
  tirLines: r.tir.split('\n').length,
  srcLines: r.src.split('\n').length,
});

console.log('  fusion   graph ops   TIR loops   TIR blocks   TIR lines   source lines');
for (const fusion of [false, true]) {
  manual_seed(0);
  const r = await levels(
    new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1)),
    [x],
    { fusion: { enabled: fusion } },
  );
  const c = counts(r);
  console.log(
    `  ${String(fusion).padEnd(8)}${String(c.ops).padStart(9)}${String(c.loops).padStart(12)}`
    + `${String(c.blocks).padStart(13)}${String(c.tirLines).padStart(12)}${String(c.srcLines).padStart(15)}`,
  );
}
