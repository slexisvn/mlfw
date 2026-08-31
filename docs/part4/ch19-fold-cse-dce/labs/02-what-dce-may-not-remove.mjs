import {
  tensor, Module, ops, compile, trace, printModule, CPUTarget, TraceLevel, manual_seed,
} from '../../../../dist/index.node.js';
import {
  buildFunction, TensorType, ScalarType, DCEPass, IRPrinter, opHasSideEffects,
} from '../../../../dist/internals.node.js';

manual_seed(0);

const x = tensor([[1, 2], [3, 4]]);
const index = tensor([[0, 1], [1, 0]], 'int32');

class DeadPureChain extends Module {
  forward(a) {
    const dead = a.exp().log().mul(3).add(7);
    return a.add(1);
  }
}

class DeadFunctionalWrite extends Module {
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

function runOpaque() {
  console.log('=== a dead operation the compiler cannot see into ===');
  const t = new TensorType([2, 2], ScalarType.F32);
  const func = buildFunction('DeadCustomCall', [t], [t], (b, args) => {
    b.customCall('write_somewhere', [args[0]], [t]);
    b.returnOp([b.neg(args[0]).getResult(0)]);
  });

  const call = [...func.ops()].find((op) => op.opName === 'custom_call');
  console.log(`custom_call declares an effect: ${opHasSideEffects(call)}`);

  const before = [...func.ops()].length;
  new DCEPass().run(func);
  const after = [...func.ops()].length;
  console.log(`dce erased ${before - after} operation(s)`);
  console.log(new IRPrinter().printFunction(func));
}

await run('a dead chain of pure operations', DeadPureChain, [x]);
await run('a dead operation that writes only its own output', DeadFunctionalWrite, [x, index]);
runOpaque();
