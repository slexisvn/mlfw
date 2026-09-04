import {
  tensor, Linear, ReLU, Sequential, compile, CPUTarget, TraceLevel, manual_seed,
} from '../../../../dist/index.node.js';

const x = tensor(Array.from({ length: 64 * 128 }, (_, i) => ((i % 31) / 31) - 0.5)).reshape([64, 128]);

function build() {
  manual_seed(0);
  return new Sequential(new Linear(128, 256), new ReLU(), new Linear(256, 128));
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

async function study(label, { layout, layoutAwareOps }) {
  const target = CPUTarget();
  if (layoutAwareOps) target.layoutAwareOps = new Set(layoutAwareOps);

  let ir = null;
  let detail = null;
  const compiled = compile(build(), [x], {
    target,
    optimization: { layout },
    trace: {
      level: TraceLevel.DEBUG,
      irSnapshot: { afterGraphPasses: true },
      sink: (e) => {
        if (e.type === 'ir_snapshot') ir = e.text;
        if (e.type === 'pass_detail' && e.passName === 'LayoutTransformPass') detail = e;
      },
    },
  });
  await compiled._ready;
  await compiled(x);
  const ms = await best(() => compiled(x), 20);

  const transforms = (ir.match(/tera\.layout_transform/g) || []).length;
  console.log(`=== ${label} ===`);
  console.log(`  target.layoutAwareOps = {${[...target.layoutAwareOps].join(', ')}}`);
  console.log(`  pass report: ${detail ? `${detail.conversions} conversion(s) proposed, ${detail.uniqueTransforms} kept` : 'the pass reported nothing'}`);
  console.log(`  layout_transform operations in the graph: ${transforms}`);
  console.log(`  ${ms.toFixed(3)} ms\n`);
}

await study('optimization.layout off (the default)', { layout: false });
await study('optimization.layout on, target declares nothing', { layout: true });
await study('optimization.layout on, target declares dot layout-aware', { layout: true, layoutAwareOps: ['dot'] });
