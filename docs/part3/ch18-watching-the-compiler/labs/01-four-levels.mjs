import {
  tensor, Linear, ReLU, Sequential, compile, CPUTarget, TraceLevel, manual_seed,
} from '../../../../dist/index.node.js';

const x = tensor([[0.5, -1.5], [1.0, 2.0]]);

function build() {
  manual_seed(0);
  return new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
}

const LEVELS = ['SILENT', 'INFO', 'VERBOSE', 'DEBUG'];
const byLevel = new Map();

for (const name of LEVELS) {
  const counts = new Map();
  const compiled = compile(build(), [x], {
    target: CPUTarget(),
    trace: {
      level: TraceLevel[name],
      irSnapshot: { afterGraphPasses: true, afterLowering: true },
      sink: (e) => counts.set(e.type, (counts.get(e.type) ?? 0) + 1),
    },
  });
  await compiled._ready;
  byLevel.set(name, counts);
}

const types = [...new Set([...byLevel.values()].flatMap(c => [...c.keys()]))];
console.log(`event type     ${LEVELS.map(l => l.padStart(8)).join('')}`);
for (const type of types) {
  console.log(`${type.padEnd(15)}${LEVELS.map(l => String(byLevel.get(l).get(type) ?? 0).padStart(8)).join('')}`);
}
console.log(`${'TOTAL'.padEnd(15)}${LEVELS.map(l => String([...byLevel.get(l).values()].reduce((a, b) => a + b, 0)).padStart(8)).join('')}`);

console.log('\n=== what one event of each type looks like at DEBUG ===');
const shown = new Set();
const compiled = compile(build(), [x], {
  target: CPUTarget(),
  trace: {
    level: TraceLevel.DEBUG,
    irSnapshot: { afterGraphPasses: true },
    sink: (e) => {
      if (shown.has(e.type)) return;
      shown.add(e.type);
      const { timestamp, level, text, ...rest } = e;
      console.log(`  ${JSON.stringify(rest)}${text === undefined ? '' : `  (+ ${text.split('\n').length} lines of IR)`}`);
    },
  },
});
await compiled._ready;
