import {
  tensor, compileWithBackward, CPUTarget, TraceLevel, ones,
} from '../../../../dist/index.node.js';

async function settle(v) { return v && v.then ? await v : v; }

function irLines(ir) {
  return ir.split('\n').filter(l => /func |%\d+ = |return\(/.test(l)).map(l => '  ' + l.trim());
}

async function study(label, fwd, inputs) {
  const snaps = [];
  const cf = compileWithBackward({ forward: fwd }, inputs, {
    target: CPUTarget(),
    fusion: { enabled: false },
    trace: {
      level: TraceLevel.DEBUG,
      irSnapshot: { afterGraphPasses: true },
      sink: (e) => { if (e.type === 'ir_snapshot') snaps.push(e.text); },
    },
  });
  const out = await settle(cf(...inputs));
  const grads = await settle(cf.backward(ones(out.shape)));
  console.log(`=== ${label} ===`);
  console.log(irLines(snaps[1]).join('\n'));
  console.log('  gradients returned: ' + grads.length);
  grads.forEach((g, i) => console.log(`    [${i}] shape ${JSON.stringify(g.shape)} = ${JSON.stringify(g.toArray())}`));
  console.log();
}

const x = tensor([[1.0, 2.0]]);

await study('one value, two consumers:  (x * x) + x', (a) => a.mul(a).add(a).sum(), [x]);

await study('an input the output does not depend on', (a, b) => a.sum(), [x, tensor([[9.0, 9.0]])]);

await study('one value, three consumers:  x*x + x*x + x', (a) => a.mul(a).add(a.mul(a)).add(a).sum(), [x]);
