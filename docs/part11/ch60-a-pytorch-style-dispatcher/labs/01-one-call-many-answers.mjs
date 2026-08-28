import {
  dispatcher, DispatchKey, DispatchKeySet, computeKeySet, KernelFunction,
  randn, manual_seed, noGrad,
} from '../../_internals.mjs';

manual_seed(3);

console.log('=== the table ===\n');
const ops = dispatcher.listOps();
console.log(`  ${ops.length} operators registered, first eight: ${ops.slice(0, 8).map((o) => o.replace('mlc::', '')).join(' ')}`);

const perKey = new Map();
for (const key of ops) {
  for (const k of dispatcher.findOp(key).entry.registeredKeys()) perKey.set(k, (perKey.get(k) || 0) + 1);
}
console.log('\n  key             ops with a kernel');
for (const [k, n] of [...perKey].sort((a, b) => a[0] - b[0])) {
  console.log(`  ${String(DispatchKey[k]).padEnd(16)} ${String(n).padStart(3)}`);
}
const declared = Object.entries(DispatchKey)
  .filter(([n, v]) => typeof v === 'number' && n !== 'NUM_KEYS' && !perKey.has(v))
  .map(([n]) => n);
console.log(`\n  declared with no kernel anywhere: ${declared.join(', ')}`);
console.log(`  fallbacks registered:             ${dispatcher.fallbacks.registeredKeys().length === 0 ? '(none)' : dispatcher.fallbacks.registeredKeys().map((k) => DispatchKey[k]).join(', ')}`);

console.log('\n=== a tensor carries its own keys, and the call unions them ===\n');
const plain = randn([2, 2]);
const tracked = randn([2, 2]).requiresGrad_(true);
console.log('  a plain tensor         ', String(plain.dispatchKeySet));
console.log('  requiresGrad_(true)    ', String(tracked.dispatchKeySet));

const add = dispatcher.findOp('add');
console.log(`\n  schema  ${add.schema.name}(${add.schema.args.map((a) => `${a.kind} ${a.name}`).join(', ')}) -> ${add.schema.returns.map((r) => r.kind).join(', ')}`);
console.log(`  tensor argument positions: ${JSON.stringify(add.tensorArgIndices)}`);

for (const [label, args] of [['add(plain, plain)', [plain, plain]], ['add(tracked, plain)', [tracked, plain]], ['add(plain, tracked)', [plain, tracked]]]) {
  const ks = computeKeySet(args, add.schema);
  console.log(`  ${label.padEnd(21)} ${String(ks).padEnd(38)} highest = ${DispatchKey[ks.highestPriority()]}`);
}

console.log('\n=== the chain: each kernel removes its key and redispatches ===\n');
const original = new Map();
for (const k of add.entry.registeredKeys()) original.set(k, add.entry.lookupKernel(k));
const visited = [];
for (const [k, kernel] of original) {
  add.entry.registerKernel(k, KernelFunction.fromUnboxed((ks, ...args) => {
    visited.push(DispatchKey[k]);
    return kernel.callUnboxed(ks, ...args);
  }));
}

const trip = (label, fn) => { visited.length = 0; fn(); console.log(`  ${label.padEnd(26)} ${visited.join(' -> ')}`); };
trip('add(plain, plain)', () => plain.add(plain));
trip('add(tracked, plain)', () => tracked.add(plain));
trip('inside noGrad(...)', () => noGrad(() => tracked.add(plain)));

for (const [k, kernel] of original) add.entry.registerKernel(k, kernel);

console.log('\n  noGrad does not remove the autograd key: the autograd kernel still runs,');
console.log('  reads a module-global flag, and redispatches. §60.7 says why that matters.');

console.log('\n=== priority is a number, and the ordering is total ===\n');
const rows = [
  ['backend only', DispatchKeySet.fromKeys(DispatchKey.CPU)],
  ['backend + autograd', DispatchKeySet.fromKeys(DispatchKey.CPU, DispatchKey.AUTOGRAD_CPU)],
  ['+ tracing', DispatchKeySet.fromKeys(DispatchKey.CPU, DispatchKey.AUTOGRAD_CPU, DispatchKey.TRACING)],
  ['+ autocast', DispatchKeySet.fromKeys(DispatchKey.CPU, DispatchKey.AUTOCAST, DispatchKey.AUTOGRAD_CPU, DispatchKey.TRACING)],
];
for (const [label, ks] of rows) {
  const order = [...ks].map((k) => DispatchKey[k]);
  console.log(`  ${label.padEnd(20)} ${String(ks.count()).padStart(2)} keys, visited in order: ${order.join(' -> ')}`);
}

console.log('\n=== a key with no kernels is a layer that has not been installed yet ===\n');
const { registerTracingDispatch } = await import('../../_internals.mjs');
const countFor = (key) => dispatcher.listOps().filter((o) => dispatcher.findOp(o).entry.hasKernel(key)).length;
console.log(`  TRACING kernels before registerTracingDispatch(): ${countFor(DispatchKey.TRACING)}`);
registerTracingDispatch();
console.log(`  TRACING kernels after:                            ${countFor(DispatchKey.TRACING)}`);
console.log('\n  Chapter 61 is what those kernels do.');
