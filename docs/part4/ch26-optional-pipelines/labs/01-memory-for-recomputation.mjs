import {
  tensor, Module, compile, CPUTarget, TraceLevel, manual_seed,
} from '../../../../dist/index.node.js';

manual_seed(0);

const N = 1 << 16;
const a = tensor(Array.from({ length: N }, (_, i) => (i % 97) / 97 + 0.5));

class Reused extends Module {
  forward(x) {
    const p = x.exp(), q = x.log(), r = x.neg();
    return p.add(q).add(r).add(p).add(q).add(r);
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

async function study(label, budget) {
  let ir = null, report = null, memory = null, warning = null;
  const compiled = compile(new Reused(), [a], {
    target: CPUTarget(),
    fusion: { enabled: false },
    optimization: {
      rematerialization: budget !== null,
      rematConfig: budget === null ? {} : { memoryBudget: budget },
    },
    trace: {
      level: TraceLevel.DEBUG,
      irSnapshot: { afterGraphPasses: true },
      sink: (e) => {
        if (e.type === 'ir_snapshot') ir = e.text;
        if (e.type === 'pass_detail' && e.passName === 'RematerializationPass') report = e;
        if (e.type === 'memory') memory = e;
        if (e.type === 'warning' && e.phase === 'rematerialization') warning = e;
      },
    },
  });
  await compiled._ready;
  await compiled(a);
  const ms = await best(() => compiled(a), 20);

  const ops = [...ir.matchAll(/= "?tera\.(\w+)/g)].map(m => m[1]);
  console.log(`=== ${label} ===`);
  console.log(`  graph: ${ops.length} operations -- ${ops.join(', ')}`);
  if (report) console.log(`  pass: ${report.iterations} rematerialization(s), live pressure ${report.peakPressure} bytes against a budget of ${report.budget}`);
  console.log(`  planned peak memory: ${memory.peakMemory} bytes across ${memory.totalTemporaries} temporaries`);
  if (warning) console.log(`  warning: ${warning.message}`);
  console.log(`  ${ms.toFixed(3)} ms\n`);
}

await study('rematerialization off (the default)', null);
await study('memory budget 512 KiB', 1 << 19);
await study('memory budget 128 KiB', 1 << 17);
