import {
  tensor, Module, ops, compile, trace, printModule, CPUTarget, TraceLevel, manual_seed,
} from '../../../../dist/index.node.js';

manual_seed(0);

const x = tensor([[1, 2], [3, 4]]);
const index = tensor([[0, 1], [1, 0]], 'int32');

class DeadPureChain extends Module {
  forward(a) {
    const dead = a.exp().log().mul(3).add(7);
    return a.add(1);
  }
}

class DeadSideEffect extends Module {
  forward(a, i) {
    const dead = ops.scatter_add(a, 0, i, a);
    return a.add(1);
  }
}

async function run(label, Klass, inputs) {
  console.log(`=== ${label} ===`);
  const traced = await trace((...args) => new Klass().forward(...args), inputs);
  console.log(`traced: ${[...traced.functions().next().value.ops()].map(o => o.opName).join(', ')}`);

  let ir = null;
  let erased = 0;
  const compiled = compile(new Klass(), inputs, {
    target: CPUTarget(),
    fusion: { enabled: false },
    trace: {
      level: TraceLevel.DEBUG,
      irSnapshot: { afterGraphPasses: true },
      sink: (e) => {
        if (e.type === 'pass_detail' && e.passName === 'dce') erased += e.erasedCount;
        if (e.type === 'ir_snapshot') ir = e.text;
      },
    },
  });
  await compiled._ready;
  console.log(`dce erased ${erased} operation(s)`);
  console.log(ir);
}

await run('a dead chain of pure operations', DeadPureChain, [x]);
await run('a dead operation that writes', DeadSideEffect, [x, index]);
