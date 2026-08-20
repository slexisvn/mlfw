import {
  tensor, Module, Linear, compile, CPUTarget, TraceLevel, trace, printModule, manual_seed,
} from '../../../../dist/index.node.js';

manual_seed(0);

async function show(label, Klass, inputs) {
  console.log(`=== ${label} ===`);
  console.log('traced:');
  console.log(printModule(await trace((...a) => new Klass().forward(...a), inputs)));

  const passes = [];
  let optimized = null;
  const compiled = compile(new Klass(), inputs, {
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
  console.log(`passes: ${passes.join(', ')}`);
  console.log('after graph passes:');
  console.log(optimized);
  console.log();
}

class Identity extends Module {
  forward(t) { return t.add(0).mul(1); }
}

class TransposedDot extends Module {
  constructor() { super(); this.l = new Linear(2, 3); }
  forward(t) { return this.l.forward(t); }
}

class FoldableConstants extends Module {
  forward(t) { return t.mul(2 * 3).add(10 - 4); }
}

const x = tensor([[1, 2], [3, 4]]);
await show('x + 0, then * 1', Identity, [x]);
await show('a Linear layer: transpose feeding a dot', TransposedDot, [x]);
await show('constants the host language already folded', FoldableConstants, [x]);
