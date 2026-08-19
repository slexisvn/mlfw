import {
  tensor, Linear, ReLU, Sequential, compile, CPUTarget, TraceLevel, manual_seed,
} from '../../../../dist/index.node.js';

manual_seed(0);

const model = new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
const x = tensor([[0.5, -1.5], [1.0, 2.0]]);

const seen = new Map();

const compiled = compile(model, [x], {
  target: CPUTarget(),
  trace: {
    level: TraceLevel.DEBUG,
    irSnapshot: { afterGraphPasses: true, afterLowering: true },
    sink: (event) => { if (event.type === 'ir_snapshot') seen.set(event.label, event.text); },
  },
});
await compiled._ready;

for (const [label, text] of seen) {
  console.log(`=== ${label} ===`);
  console.log(text);
  console.log();
}

console.log('=== generated code ===');
console.log(compiled.source());
