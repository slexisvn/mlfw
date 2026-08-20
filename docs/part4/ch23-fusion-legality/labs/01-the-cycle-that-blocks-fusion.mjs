import {
  tensor, Module, compile, CPUTarget, TraceLevel, manual_seed,
} from '../../../../dist/index.node.js';

manual_seed(0);

const grid = (f) => tensor(Array.from({ length: 16 }, (_, i) => f(i))).reshape([4, 4]);
const x = grid((i) => i / 16);
const y = grid((i) => (i % 5) / 5);
const w = grid((i) => (i % 3) / 3);

class CycleCreating extends Module {
  forward(a, b, c) {
    const p = a.add(b);
    const q = p.matmul(c);
    return p.mul(q);
  }
}

class NoCycle extends Module {
  forward(a, b, c) {
    const p = a.add(b);
    const q = c.matmul(c);
    return p.mul(q);
  }
}

async function show(label, Klass) {
  let ir = null;
  const compiled = compile(new Klass(), [x, y, w], {
    target: CPUTarget(),
    trace: {
      level: TraceLevel.DEBUG,
      irSnapshot: { afterGraphPasses: true },
      sink: (e) => { if (e.type === 'ir_snapshot') ir = e.text; },
    },
  });
  await compiled._ready;
  const fusions = (ir.match(/= fusion\(/g) || []).length;
  console.log(`=== ${label} ===`);
  console.log(ir);
  console.log(`fusion regions: ${fusions}\n`);
}

await show('q depends on p: fusing p with its consumer would create a cycle', CycleCreating);
await show('q does not depend on p: the same two operations fuse', NoCycle);
