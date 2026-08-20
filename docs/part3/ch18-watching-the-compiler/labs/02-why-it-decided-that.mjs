import {
  tensor, Linear, ReLU, Sequential, compile, CPUTarget, TraceLevel, manual_seed,
} from '../../../../dist/index.node.js';

const x = tensor([[0.5, -1.5], [1.0, 2.0]]);

function build() {
  manual_seed(0);
  return new Sequential(new Linear(2, 16), new ReLU(), new Linear(16, 16), new ReLU(), new Linear(16, 1));
}

for (const strategy of ['priority', 'dominator']) {
  const explains = [];
  const compiled = compile(build(), [x], {
    target: CPUTarget(),
    fusion: { strategy },
    scheduling: { enabled: true },
    trace: {
      level: TraceLevel.DEBUG,
      sink: (e) => { if (e.type === 'explain') explains.push(e); },
    },
  });
  await compiled._ready;

  console.log(`=== fusion strategy: ${strategy} ===`);
  for (const e of explains.filter(e => e.category === 'fusion')) {
    console.log(`  ${e.subject.padEnd(14)} ${e.decision.padEnd(10)} ` +
      `because: ${e.reason ?? '(no reason recorded)'}`);
  }

  const schedule = new Map();
  for (const e of explains.filter(e => e.category === 'schedule')) {
    schedule.set(e.decision, (schedule.get(e.decision) ?? 0) + 1);
  }
  console.log(`  and ${[...schedule].map(([d, n]) => `${n} block(s) scheduled by rule '${d}'`).join(', ')}\n`);
}

console.log('=== the same compile with explains off (INFO) ===');
let count = 0;
const compiled = compile(build(), [x], {
  target: CPUTarget(),
  scheduling: { enabled: true },
  trace: { level: TraceLevel.INFO, sink: (e) => { if (e.type === 'explain') count++; } },
});
await compiled._ready;
console.log(`  explain events: ${count}`);
