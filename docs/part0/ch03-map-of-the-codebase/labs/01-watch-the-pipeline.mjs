import {
  tensor, Linear, ReLU, Sequential, compile, CPUTarget, TraceLevel, manual_seed,
} from '../../../../dist/index.node.js';

manual_seed(0);

const model = new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
const x = tensor([[0.5, -1.5], [1.0, 2.0]]);

const compiled = compile(model, [x], {
  target: CPUTarget(),
  trace: {
    level: TraceLevel.VERBOSE,
    sink: (event) => {
      if (event.type === 'phase') {
        const when = event.action === 'start' ? '>' : '<';
        const ms = event.durationMs === undefined ? '' : ` (${event.durationMs.toFixed(2)} ms)`;
        console.log(`${when} phase ${event.phase}${ms}`);
      } else if (event.type === 'pass' && event.changed) {
        console.log(`    pass ${event.passName}: ${event.opCountBefore} ops -> ${event.opCountAfter} ops`);
      }
    },
  },
});

await compiled._ready;
