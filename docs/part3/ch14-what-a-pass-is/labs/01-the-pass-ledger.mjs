import {
  tensor, Linear, ReLU, Sequential, compile, CPUTarget, TraceLevel, manual_seed,
} from '../../../../dist/index.node.js';

manual_seed(0);

const model = new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
const x = tensor([[0.5, -1.5], [1.0, 2.0]]);

const runs = [];

const compiled = compile(model, [x], {
  target: CPUTarget(),
  trace: {
    level: TraceLevel.VERBOSE,
    sink: (e) => { if (e.type === 'pass') runs.push(e); },
  },
});
await compiled._ready;

console.log('pass                    verdict     ops');
for (const e of runs) {
  const verdict = e.changed ? 'CHANGED' : 'UNCHANGED';
  const ops = e.opCountBefore < 0 ? '' : `${e.opCountBefore} -> ${e.opCountAfter}`;
  console.log(`${e.passName.padEnd(23)} ${verdict.padEnd(11)} ${ops}`);
}

const changed = runs.filter(e => e.changed).length;
console.log(`\n${runs.length} pass runs, ${changed} changed something, ${runs.length - changed} did not`);
