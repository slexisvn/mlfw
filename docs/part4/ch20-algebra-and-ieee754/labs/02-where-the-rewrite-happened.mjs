import {
  tensor, Module, compile, trace, printModule, CPUTarget, TraceLevel, manual_seed,
} from '../../../../dist/index.node.js';

manual_seed(0);

const x = tensor([[-0.0, 1.5]]);

class AddZero extends Module { forward(a) { return a.add(0); } }
class MulZero extends Module { forward(a) { return a.mul(0); } }

async function inspect(label, Klass) {
  console.log(`=== ${label} ===`);
  const traced = await trace((a) => new Klass().forward(a), [x]);
  console.log('traced:      ' + [...traced.functions().next().value.ops()].map(o => o.opName).join(' -> '));

  let ir = null;
  const compiled = compile(new Klass(), [x], {
    target: CPUTarget(),
    trace: {
      level: TraceLevel.DEBUG,
      irSnapshot: { afterGraphPasses: true },
      sink: (e) => { if (e.type === 'ir_snapshot') ir = e.text; },
    },
  });
  await compiled._ready;

  const body = ir.split('\n').slice(2, -2).map(l => l.trim()).join(' | ');
  console.log('after passes: ' + body);
  console.log('kernel:');
  for (const line of compiled.source().split('\n').filter(l => l.includes('buf_') && l.includes('='))) {
    console.log('   ' + line.trim());
  }
  console.log();
}

await inspect('x + 0', AddZero);
await inspect('x * 0', MulZero);
