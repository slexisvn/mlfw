import {
  tensor, Linear, ReLU, Sequential, compile, CPUTarget, TraceLevel, manual_seed,
} from '../../../../dist/index.node.js';

const x = tensor(Array.from({ length: 64 * 128 }, (_, i) => ((i % 31) / 31) - 0.5)).reshape([64, 128]);

function build() {
  manual_seed(0);
  return new Sequential(new Linear(128, 256), new ReLU(), new Linear(256, 128), new ReLU());
}

function targetWithEpilogue(enabled) {
  const target = CPUTarget();
  target.enableEpilogueFusion = enabled;
  return target;
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

async function study(label, enabled) {
  let ir = null;
  const compiled = compile(build(), [x], {
    target: targetWithEpilogue(enabled),
    trace: {
      level: TraceLevel.DEBUG,
      irSnapshot: { afterGraphPasses: true },
      sink: (e) => { if (e.type === 'ir_snapshot') ir = e.text; },
    },
  });
  await compiled._ready;
  await compiled(x);
  const ms = await best(() => compiled(x), 20);

  console.log(`=== ${label} ===`);
  console.log(ir);
  console.log(`  ${ms.toFixed(3)} ms\n`);
}

await study('enableEpilogueFusion: false (the CPU default)', false);
await study('enableEpilogueFusion: true (what CUDA gets)', true);
