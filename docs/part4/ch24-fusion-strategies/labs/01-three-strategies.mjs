import {
  tensor, Module, compile, CPUTarget, TraceLevel, manual_seed,
} from '../../../../dist/index.node.js';

manual_seed(0);

const N = 1 << 18;
const a = tensor(Array.from({ length: N }, (_, i) => (i % 97) / 97));
const b = tensor(Array.from({ length: N }, (_, i) => (i % 89) / 89));

class Diamond extends Module {
  forward(x, y) {
    const s = x.add(y);
    return s.mul(x).add(s);
  }
}

async function best(fn, reps) {
  const times = [];
  for (let i = 0; i < reps; i++) {
    const t0 = performance.now();
    await fn();
    times.push(performance.now() - t0);
  }
  return Math.min(...times);
}

async function study(label, fusionConfig) {
  let ir = null;
  const explains = [];
  const compiled = compile(new Diamond(), [a, b], {
    target: CPUTarget(),
    fusion: fusionConfig,
    trace: {
      level: TraceLevel.DEBUG,
      irSnapshot: { afterGraphPasses: true },
      sink: (e) => {
        if (e.type === 'ir_snapshot') ir = e.text;
        if (e.type === 'explain' && e.category === 'fusion') explains.push(e);
      },
    },
  });
  await compiled._ready;
  await compiled(a, b);
  const ms = await best(() => compiled(a, b), 20);

  console.log(`=== ${label} ===`);
  for (const e of explains) console.log(`  cost model: ${e.subject} -> ${e.decision}${e.reason ? `: ${e.reason}` : ''}`);
  const inside = ir.split('\n').filter(l => /^ {8}%/.test(l));
  const held = inside.map(l => (l.match(/= "?tera\.(\w+)/) || [, '?'])[1]);
  console.log(`  ${(ir.match(/= "tera\.fusion"\(/g) || []).length} fusion region(s) holding: ${held.join(', ') || '(nothing)'}`);
  console.log(`  ${ms.toFixed(3)} ms\n`);
}

await study("strategy 'priority' (the default)", { strategy: 'priority' });
await study("strategy 'dominator'", { strategy: 'dominator' });
await study("strategy 'greedy' (the original FusionPass)", { strategy: 'greedy' });
await study("strategy 'dominator', shared-memory budget raised to 16 MiB",
  { strategy: 'dominator', cost: { maxSharedMemory: 1 << 24 } });
