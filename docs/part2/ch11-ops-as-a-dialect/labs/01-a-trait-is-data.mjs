import {
  tensor, Module, compile, CPUTarget, TraceLevel, trace, printModule, manual_seed,
} from '../../../../dist/index.node.js';

manual_seed(0);

const x = tensor([[1, 2], [3, 4]]);
const y = tensor([[5, 6], [7, 8]]);

class Commutative extends Module {
  forward(a, b) { return a.add(b).mul(b.add(a)); }
}

class NotCommutative extends Module {
  forward(a, b) { return a.sub(b).mul(b.sub(a)); }
}

async function run(label, Klass) {
  console.log(`=== ${label} ===`);
  console.log(printModule(await trace((a, b) => new Klass().forward(a, b), [x, y])));

  const passes = [];
  let optimized = null;
  const compiled = compile(new Klass(), [x, y], {
    target: CPUTarget(),
    trace: {
      level: TraceLevel.DEBUG,
      irSnapshot: { afterGraphPasses: true },
      sink: (e) => {
        if (e.type === 'pass' && e.changed) passes.push(`${e.passName}: ${e.opCountBefore} -> ${e.opCountAfter}`);
        if (e.type === 'ir_snapshot') optimized = e.text;
      },
    },
  });
  await compiled._ready;

  console.log(`passes that changed something: ${passes.join(', ')}`);
  console.log(optimized);
  console.log();
}

await run('add(a, b) and add(b, a)', Commutative);
await run('sub(a, b) and sub(b, a)', NotCommutative);
