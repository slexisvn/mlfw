import {
  tensor, compileWithBackward, CPUTarget, TraceLevel, ones,
} from '../../../../dist/index.node.js';

// Each VJP rule emits a small subgraph into the backward function.
// Differentiate one operation at a time and read what each rule wrote.

const x = tensor([[0.5, 2.0]]);

const CASES = [
  ['exp',                        (a) => a.exp().sum()],
  ['log',                        (a) => a.log().sum()],
  ['tanh',                       (a) => a.tanh().sum()],
  ['relu  (traces to maximum)',  (a) => a.relu().sum()],
  ['mul(x, x)',                  (a) => a.mul(a).sum()],
  ['div',                        (a) => a.div(a.exp()).sum()],
];

async function settle(v) { return v && v.then ? await v : v; }

const sigOf = (ir) => (ir.split('\n')[1] || '').trim().replace(/^func /, '');

function bodyOf(ir) {
  return ir.split('\n')
    .filter(l => /^\s{4}(%\d+ = |return\()/.test(l))
    .map(l => l.trim());
}

for (const [name, fwd] of CASES) {
  const snaps = [];
  const cf = compileWithBackward({ forward: fwd }, [x], {
    target: CPUTarget(),
    fusion: { enabled: false },
    trace: {
      level: TraceLevel.DEBUG,
      irSnapshot: { afterGraphPasses: true },
      sink: (e) => { if (e.type === 'ir_snapshot') snaps.push(e.text); },
    },
  });
  const out = await settle(cf(x));
  const grads = await settle(cf.backward(ones(out.shape)));

  console.log(`=== d/dx  ${name} ===`);
  console.log('  forward  ' + sigOf(snaps[0]));
  console.log('  backward ' + sigOf(snaps[1]));
  for (const line of bodyOf(snaps[1])) console.log('    ' + line);
  console.log('  grad at x = [0.5, 2.0]: ' + JSON.stringify(grads[0].toArray()[0]) + '\n');
}
