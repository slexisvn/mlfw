import {
  tensor, Linear, ReLU, Sequential, compileWithBackward, CPUTarget, TraceLevel, ones, manual_seed,
} from '../../../../dist/index.node.js';

// The same derivative, packaged two ways:
//   'separate' -- one forward function plus one backward function
//   'joint'    -- a single function returning outputs and gradients together

const x = tensor([[0.5, -1.5], [1.0, 2.0]]);

function model() {
  manual_seed(0);
  return new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
}

async function settle(v) { return v && v.then ? await v : v; }

const sigOf = (ir) => (ir.split('\n')[1] || '').trim().replace(/^func /, '').replace(/ \{$/, '');
const modOf = (ir) => (ir.split('\n')[0] || '').trim().replace(/^module /, '').replace(/ \{$/, '');
const opsOf = (ir) => (ir.match(/^\s+%\d+ = /gm) || []).length;

async function best(fn, reps) {
  const t = [];
  for (let i = 0; i < reps; i++) { const t0 = performance.now(); await fn(); t.push(performance.now() - t0); }
  return Math.min(...t);
}

for (const mode of ['separate', 'joint']) {
  const snaps = [];
  const cf = compileWithBackward(model(), [x], {
    target: CPUTarget(),
    mode,
    trace: {
      level: TraceLevel.DEBUG,
      irSnapshot: { afterGraphPasses: true },
      sink: (e) => { if (e.type === 'ir_snapshot') snaps.push(e.text); },
    },
  });

  const out = await settle(cf(x));
  const grads = await settle(cf.backward(ones(out.shape)));

  console.log(`=== mode: '${mode}' ===`);
  console.log(`  compiled modules: ${snaps.length}`);
  for (const s of snaps) {
    console.log(`    ${modOf(s).padEnd(10)} ${opsOf(s)} ops   ${sigOf(s)}`);
  }
  console.log('  forward: ' + JSON.stringify(out.toArray()));
  console.log('  gradient shapes: ' + grads.map(g => JSON.stringify(g.shape)).join(' '));
  console.log('  d/dx: ' + JSON.stringify(grads[0].toArray()));

  const w = ones(out.shape);
  const fwdOnly = async () => { await settle(cf(x)); };
  const bwdOnly = async () => { await settle(cf.backward(w)); };
  const step = async () => { await fwdOnly(); await bwdOnly(); };
  await step();

  const tF = await best(fwdOnly, 30);
  const tB = await best(bwdOnly, 30);
  console.log(`  forward() alone:    ${tF.toFixed(3)} ms`);
  console.log(`  backward() alone:   ${tB.toFixed(3)} ms`);
  console.log(`  forward+backward:   ${(await best(step, 30)).toFixed(3)} ms\n`);
}
