import {
  tensor, Linear, ReLU, Sequential, compile, CPUTarget, TraceLevel, manual_seed,
} from '../../../../dist/index.node.js';

const x = tensor([[0.5, -1.5], [1.0, 2.0]]);

function build() {
  manual_seed(0);
  return new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
}

async function run(label, disabled) {
  const off = new Set(disabled);
  let ir = null;
  const compiled = compile(build(), [x], {
    target: CPUTarget(),
    trace: {
      level: TraceLevel.DEBUG,
      irSnapshot: { afterGraphPasses: true },
      sink: (e) => { if (e.type === 'ir_snapshot') ir = e.text; },
    },
    passContext: { shouldRun: (pass) => !off.has(pass.name) },
  });
  await compiled._ready;

  const y = await compiled(x);
  console.log(`=== ${label} ===`);
  console.log(ir);
  console.log(`output: ${JSON.stringify(y.toArray())}`);
  console.log(`kernel lines: ${compiled.source().split('\n').length}\n`);
}

await run('everything on', []);
await run('dce off', ['dce']);
await run('every simplification off', ['canonicalize', 'algebraic_simplify', 'constant_fold', 'cse', 'dce']);
