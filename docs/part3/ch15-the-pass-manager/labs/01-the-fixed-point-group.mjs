import {
  tensor, Module, compile, CPUTarget, TraceLevel, manual_seed,
} from '../../../../dist/index.node.js';

manual_seed(0);

const x = tensor([[1, 2], [3, 4]]);

class ThereAndBackAgain extends Module {
  forward(a) { return a.transpose(1, 0).transpose(1, 0).add(0); }
}

async function run(label, maxIterations) {
  const runs = [];
  let ir = null;
  const compiled = compile(new ThereAndBackAgain(), [x], {
    target: CPUTarget(),
    optimization: { maxSimplifyIterations: maxIterations },
    trace: {
      level: TraceLevel.DEBUG,
      irSnapshot: { afterGraphPasses: true },
      sink: (e) => {
        if (e.type === 'pass') runs.push(e);
        if (e.type === 'ir_snapshot') ir = e.text;
      },
    },
  });
  await compiled._ready;

  console.log(`=== maxSimplifyIterations: ${maxIterations} ===`);
  let round = 0;
  for (const e of runs) {
    if (e.passName === 'canonicalize') console.log(`  -- round ${++round} --`);
    if (round === 0) continue;
    if (e.passName === 'PriorityFusionPass') break;
    console.log(`  ${e.passName.padEnd(20)} ${e.changed ? 'CHANGED  ' : 'UNCHANGED'} ${e.opCountBefore} -> ${e.opCountAfter}`);
  }
  console.log(ir);
}

await run('capped', 1);
await run('default', 8);
