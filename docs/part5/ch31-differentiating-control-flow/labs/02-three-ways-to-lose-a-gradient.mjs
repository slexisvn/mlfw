import {
  tensor, compileWithBackward, CPUTarget, TraceLevel, ones,
} from '../../../../dist/index.node.js';

// A gradient can go missing three ways. Two of them are silent and produce
// zeros; they differ in what the backward graph spends getting there.

const x = tensor([[1.7, -2.3, 0.4, 3.9]]);

async function settle(v) { return v && v.then ? await v : v; }
const countOps = (ir) => (ir.match(/^\s+%\d+ = /gm) || []).length;

async function study(label, fwd) {
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
  try {
    const out = await settle(cf(x));
    const grads = await settle(cf.backward(ones(out.shape)));
    const g = grads[0].toArray()[0].map(v => +v.toFixed(4));
    console.log(`  ${label.padEnd(34)} ${String(countOps(snaps[1])).padStart(2)} bwd ops   ${JSON.stringify(g)}`);
  } catch (e) {
    console.log(`  ${label.padEnd(34)} THREW: ${e.message.split('\n')[0].slice(0, 90)}`);
  }
}

console.log('=== x = [1.7, -2.3, 0.4, 3.9] ===\n');

console.log('a rule exists and the derivative is real:');
await study('sum(x)', (a) => a.sum());
await study('sum(x * x)', (a) => a.mul(a).sum());

console.log('\na rule exists and returns a numeric zero:');
await study('sum(floor(x))', (a) => a.floor().sum());
await study('sum(sign(x))', (a) => a.sign().sum());
await study('sum(x * floor(x))', (a) => a.mul(a.floor()).sum());

console.log('\nthe operation is a declared barrier (compare), so the');
console.log('sweep never reaches its operands at all:');
await study('sum(maximum(x, 0))   [= relu]', (a) => a.relu().sum());
await study('sum(where(x > 0, x, -x))', (a) => a.abs().sum());

console.log('\n=== what happened to the zeros ===');
console.log('  floor\'s rule emits a full tensor of zeros. In sum(floor(x)) that is');
console.log('  the only contribution, so constant folding materialises the broadcast');
console.log('  and DCE sweeps the rest: the backward graph is one folded constant.');
console.log('  In sum(x * floor(x)) the zero is added to a real contribution, and');
console.log('  the addition survives -- x + 0 is not an identity on floats, so');
console.log('  AddZero declines without a fast-math licence. Part IV cleans up after');
console.log('  Part V only as far as IEEE 754 lets it.');

// The same three programs, with the simplification passes switched off, so the
// waste the rules actually emit is visible.
console.log('\n=== the same programs with canonicalize/simplify disabled ===');
for (const [label, fwd] of [
  ['sum(floor(x))', (a) => a.floor().sum()],
  ['sum(x * floor(x))', (a) => a.mul(a.floor()).sum()],
]) {
  const snaps = [];
  const off = new Set(['canonicalize', 'algebraic_simplify', 'constant_fold', 'cse', 'dce']);
  const cf = compileWithBackward({ forward: fwd }, [x], {
    target: CPUTarget(),
    fusion: { enabled: false },
    passContext: { shouldRun: (pass) => !off.has(pass.name) },
    trace: {
      level: TraceLevel.DEBUG,
      irSnapshot: { afterGraphPasses: true },
      sink: (e) => { if (e.type === 'ir_snapshot') snaps.push(e.text); },
    },
  });
  await settle(cf(x));
  console.log(`  ${label.padEnd(20)} ${String(countOps(snaps[1])).padStart(2)} bwd ops as the rules emitted them`);
}
