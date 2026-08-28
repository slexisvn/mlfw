import {
  _traceCore, printFunction, firstFunction, dispatcher, DispatchKey,
  randn, zeros, where, manual_seed,
} from '../../_internals.mjs';

manual_seed(7);

const W1 = randn([8, 16]);
const b1 = randn([16]);
const W2 = randn([16, 4]);
const b2 = randn([4]);
const mlp = (x) => x.matmul(W1).add(b1).relu().matmul(W2).add(b2);

console.log('=== the running example, recorded ===\n');
const traced = await _traceCore(mlp, [randn([2, 8])], { name: 'mlp' });
const func = firstFunction(traced.graph);
console.log(`  user inputs      ${traced.numUserInputs}`);
console.log(`  captured tensors ${traced.capturedParams.map((t) => t.shape.join('x')).join(', ')}`);
console.log(`  function arity   ${func.inputTypes.length} = ${traced.numUserInputs} user + ${traced.capturedParams.length} captured\n`);
console.log(printFunction(func).split('\n').map((l) => '  ' + l).join('\n'));

console.log('\n  Four tensors the user never passed became four extra parameters.');
console.log('  Nothing in the source said "these are weights" — they were simply read.');

console.log('\n=== a symbolic tensor has a shape and a dtype, and no value ===\n');
let seen = null;
await _traceCore((x) => { seen = x; return x.relu(); }, [randn([2, 8])], { name: 'probe' });
console.log(`  class     ${seen.constructor.name}`);
console.log(`  shape     [${seen.shape}]   dtype ${seen.dtype}   device ${seen.device}`);
console.log(`  isSymbolic ${seen.isSymbolic}   keys ${String(seen.dispatchKeySet)}`);
for (const [label, fn] of [
  ['.data', () => seen.data],
  ['item()', () => seen.item()],
  ['toArray()', () => seen.toArray()],
  ['iterating it', () => [...seen]],
]) {
  try { fn(); console.log(`  ${label.padEnd(14)} returned a value (unexpected)`); }
  catch (e) { console.log(`  ${label.padEnd(14)} ${e.message.split('\n')[0]}`); }
}

console.log('\n=== what a trace can and cannot see ===\n');
const cases = [
  ['tensor arithmetic', (x) => x.mul(2).add(1)],
  ['a branch on a host value', (x) => (2 > 1 ? x.relu() : x.tanh())],
  ['a JavaScript loop', (x) => { let u = x; for (let i = 0; i < 3; i++) u = u.add(1); return u; }],
  ['where(), the tensor branch', (x) => where(x.gt(zeros([2, 8])), x, x.mul(0.1))],
];
for (const [label, fn] of cases) {
  const t = await _traceCore(fn, [randn([2, 8])], { name: 'case' });
  const ops = [...firstFunction(t.graph).ops()].map((o) => o.opName).filter((n) => n !== 'return');
  console.log(`  ${label.padEnd(26)} ${ops.length} ops: ${ops.join(' ')}`);
}
console.log('\n  The loop is gone: three adds, not a loop. The branch is gone: one relu, not a');
console.log('  choice. A trace records the operations that ran, and host control flow ran at');
console.log('  trace time and left no trace of itself.');

try {
  await _traceCore((x) => (x.sum().item() > 0 ? x.relu() : x.tanh()), [randn([2, 8])], { name: 'databranch' });
} catch (e) {
  console.log('\n  A branch on tensor *contents* is the case that cannot be silent:\n');
  console.log(e.message.split('\n').map((l) => '    ' + l).join('\n'));
}

console.log('\n=== the same tensor object is captured once ===\n');
const shared = randn([4, 4]);
const twice = await _traceCore((x) => x.matmul(shared).matmul(shared), [randn([4, 4])], { name: 'twice' });
console.log(`  read twice, captured ${twice.capturedParams.length} time(s)`);
const distinct = await _traceCore((x) => x.matmul(shared).matmul(randn([4, 4])), [randn([4, 4])], { name: 'distinct' });
console.log(`  two different tensors, captured ${distinct.capturedParams.length} time(s)`);

console.log('\n=== a scalar tensor becomes a constant, not a parameter ===\n');
const scalarT = randn([]);
const withScalar = await _traceCore((x) => x.mul(scalarT), [randn([4, 4])], { name: 'scalar' });
console.log(`  captured params ${withScalar.capturedParams.length}, ops: ${[...firstFunction(withScalar.graph).ops()].map((o) => o.opName).join(' ')}`);

console.log('\n=== tracing is a dispatch layer, and it is installed on demand ===\n');
const traceKernels = dispatcher.listOps().filter((o) => dispatcher.findOp(o).entry.hasKernel(DispatchKey.TRACING)).length;
console.log(`  operators with a TRACING kernel after the traces above: ${traceKernels}`);
console.log('  _traceCore() calls _ensureTracing() once, then runs the user function inside');
console.log('  withIncludedKeys(TRACING) — which is why every operation inside it is recorded');
console.log('  and no operation outside it is.');
