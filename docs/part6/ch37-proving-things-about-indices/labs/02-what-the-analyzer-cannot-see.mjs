import {
  compile, CPUTarget, TraceLevel, tensor, randn, nn,
} from '../../../../dist/index.node.js';

async function emitted(label, fn, inputs) {
  const compiled = compile({ forward: fn }, inputs, { target: CPUTarget(), fusion: { enabled: false } });
  await compiled(...inputs);
  const line = compiled.source().split('\n').filter((l) => l.includes('] = ')).pop().trim();
  console.log(`  ${label.padEnd(28)} ${line.slice(line.indexOf('] = ') + 4)}`);
  return compiled;
}

console.log('=== 1. an index the analyser can bound completely ===\n');
await emitted('x.transpose(1,0)', (t) => t.transpose(1, 0), [randn([4, 3])]);
console.log('\n  Both subscripts are loop variables with literal extents. Nothing to');
console.log('  guard, nothing to prove, no arithmetic beyond the flattening.');

console.log('\n\n=== 2. an index the analyser bounds well enough for tdiv, not enough to fold ===\n');
await emitted('x.reshape([12]) from [4,3]', (t) => t.reshape([12]), [randn([4, 3])]);
console.log('\n  The dividend was proved non-negative, which is what turns floor');
console.log('  division into truncating division. The stronger fact — that');
console.log('  (f tdiv 3) * 3 + (f tmod 3) is f — is not in the rewrite set.');

console.log('\n\n=== 3. an index the analyser cannot bound at all ===\n');
const lookup = new nn.Embedding(5, 3);
await emitted('embedding lookup', (i) => lookup.forward(i), [tensor([0, 2, 3], 'i32')]);
console.log('\n  The subscript is a value loaded from a buffer. There is no expression');
console.log('  to take intervals over, so the bound is (-inf, +inf) and no guard is');
console.log('  emitted or removed: the kernel indexes with whatever the data says.');

console.log('\n\n=== where that one gets answered instead ===\n');
const emb = new nn.Embedding(5, 3);
const good = tensor([0, 2, 4], 'i32');
const compiledEmb = compile(emb, [good], { target: CPUTarget() });
console.log(`  in range  [0,2,4] -> shape ${JSON.stringify((await compiledEmb(good)).shape)}`);
try {
  await compiledEmb(tensor([0, 2, 9], 'i32'));
  console.log('  out of range [0,2,9] -> no error');
} catch (e) {
  console.log(`  out of range [0,2,9] -> ${e.message}`);
}

console.log('\n  The check runs on the host, once per call, over the argument the');
console.log('  compiler could not reason about. It is derived at compile time by');
console.log('  walking the graph for indexed-table ops and tracing the index operand');
console.log('  back to a function argument through shape-only operations.');
console.log('  An index computed inside the graph rather than passed in gets no');
console.log('  check at either end.');
