import {
  tensor, Linear, compileWithBackward, CPUTarget, TraceLevel, ones, manual_seed,
} from '../../../../dist/index.node.js';

// `rematPolicy` is consulted once per forward operation whose result the
// backward pass could want: true means "do not save it, rebuild it".
// It is duck-typed -- any object with shouldRematerialize(op) will do --
// so the saved set can be driven from outside the compiler.

const x = tensor([[0.5, 1.0, 1.5, 2.0]]);
const fwd = (a) => a.exp().tanh().mul(a).sqrt().sum();

async function settle(v) { return v && v.then ? await v : v; }

const sigOf = (ir) => (ir.split('\n')[1] || '').trim().replace(/^func /, '').replace(/ \{$/, '');
const argsOf = (ir) => (sigOf(ir).match(/%\d+:/g) || []).length;
const opsOf = (ir) => (ir.match(/^\s+%\d+ = /gm) || []).length;

async function study(label, rematPolicy) {
  const snaps = [];
  const cf = compileWithBackward({ forward: fwd }, [x], {
    target: CPUTarget(),
    fusion: { enabled: false },
    ...(rematPolicy ? { rematPolicy } : {}),
    trace: {
      level: TraceLevel.DEBUG,
      irSnapshot: { afterGraphPasses: true },
      sink: (e) => { if (e.type === 'ir_snapshot') snaps.push(e.text); },
    },
  });
  const out = await settle(cf(x));
  const grads = await settle(cf.backward(ones(out.shape)));

  const saved = argsOf(snaps[1]) - 1;          // minus the incoming cotangent
  const fwdOuts = (sigOf(snaps[0]).split('->')[1].match(/tensor</g) || []).length;
  console.log(`${label.padEnd(30)} forward returns ${String(fwdOuts).padStart(2)}   ` +
              `backward: ${String(saved).padStart(2)} saved, ${String(opsOf(snaps[1])).padStart(2)} ops`);
  return { grad: grads[0].toArray()[0], bwdIr: snaps[1] };
}

console.log('=== f(x) = sum(sqrt(tanh(exp(x)) * x)),  x is [1, 4] ===\n');

const a = await study('default policy', undefined);
const b = await study('save everything', { shouldRematerialize: () => false });
const c = await study('recompute everything', { shouldRematerialize: () => true });
const d = await study("recompute only 'exp'", { shouldRematerialize: (op) => op.opName === 'exp' });

const same = (p, q) => p.every((v, i) => Math.abs(v - q[i]) < 1e-6);
console.log('\n=== do they agree? ===');
console.log('  default          ' + JSON.stringify(a.grad.map(v => +v.toFixed(6))));
console.log('  vs save-all      ' + (same(a.grad, b.grad) ? 'identical' : 'DIFFERENT'));
console.log('  vs recompute-all ' + (same(a.grad, c.grad) ? 'identical' : 'DIFFERENT'));
console.log('  vs exp-only      ' + (same(a.grad, d.grad) ? 'identical' : 'DIFFERENT'));

console.log('\n=== what "recompute everything" put in the backward function ===');
console.log(c.bwdIr.split('\n').filter(l => /func |%\d+ = |return\(/.test(l)).map(l => '  ' + l.trim()).join('\n'));

console.log('\n=== how the saved set grows with depth ===');
console.log('  (a stack of Linear layers: a `dot` result is never rematerialized)');
console.log('  layers   forward outputs   backward saved args');
const wide = tensor([Array.from({ length: 16 }, (_, i) => (i % 5) / 5 - 0.4)]);
for (const depth of [1, 2, 4, 8]) {
  manual_seed(0);
  const layers = Array.from({ length: depth }, () => new Linear(16, 16));
  const deep = (a0) => {
    let v = a0;
    for (const l of layers) v = l.forward(v).tanh();
    return v.sum();
  };
  const snaps = [];
  const cf = compileWithBackward({ forward: deep }, [wide], {
    target: CPUTarget(),
    trace: {
      level: TraceLevel.DEBUG,
      irSnapshot: { afterGraphPasses: true },
      sink: (e) => { if (e.type === 'ir_snapshot') snaps.push(e.text); },
    },
  });
  await settle(cf(wide));
  const fwdOuts = (sigOf(snaps[0]).split('->')[1].match(/tensor</g) || []).length;
  console.log(`  ${String(depth).padStart(6)}   ${String(fwdOuts).padStart(15)}   ${String(argsOf(snaps[1]) - 1).padStart(19)}`);
}
