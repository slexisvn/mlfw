import {
  compile, jitCompile, jitCacheClear, CPUTarget, WasmTarget,
  randn, manual_seed,
} from '../../_internals.mjs';

manual_seed(13);

console.log('=== the eager cache is keyed on shape, dtype, scalars and target ===\n');
jitCacheClear();
const cpu = CPUTarget();
const wasm = WasmTarget();
const requests = [
  ['add', [randn([4, 4]), randn([4, 4])], null, cpu],
  ['add', [randn([4, 4]), randn([4, 4])], null, cpu],
  ['add', [randn([8, 4]), randn([8, 4])], null, cpu],
  ['add', [randn([4, 4]), randn([4, 4])], null, wasm],
  ['mul', [randn([4, 4]), randn([4, 4])], null, cpu],
  ['sum', [randn([4, 4])], { dim: 0 }, cpu],
  ['sum', [randn([4, 4])], { dim: 1 }, cpu],
  ['sum', [randn([4, 4])], { dim: 0 }, cpu],
];
const kernels = new Set();
console.log(`  ${'request'.padEnd(34)} kernel`);
for (const [op, args, scalars, target] of requests) {
  const entry = jitCompile(op, args, scalars, target);
  kernels.add(entry.funcName + '@' + target.name);
  const label = `${op}(${args.map((a) => a.shape.join('x')).join(', ')})${scalars ? ' ' + JSON.stringify(scalars) : ''} on ${target.name.split('_')[0]}`;
  console.log(`  ${label.padEnd(34)} ${entry.funcName}`);
}
console.log(`\n  ${requests.length} requests, ${kernels.size} compiled kernels.`);
console.log('  A shape change, a scalar change and a target change each miss; a repeat hits.');

console.log('\n=== so an eager loop over changing shapes compiles once per shape ===\n');
jitCacheClear();
const seenNames = new Set();
for (const n of [64, 64, 65, 64, 66]) {
  const a = randn([n, n]);
  const t0 = performance.now();
  const { funcName } = jitCompile('add', [a, a], null, cpu);
  const ms = performance.now() - t0;
  const fresh = !seenNames.has(funcName);
  seenNames.add(funcName);
  console.log(`  ${String(n).padStart(4)}x${n}  ${ms.toFixed(3).padStart(8)}ms  ${funcName.padEnd(12)} ${fresh ? 'compiled' : 'cache hit'}`);
}

console.log('\n=== the compiled cache is keyed on a signature, then on the guards ===\n');
const W = randn([8, 4]);
class Linear { forward(x) { return x.matmul(W).relu(); } }

for (const [label, dynamicShapes] of [['static', null], ['dynamic dim 0', [new Set([0])]]]) {
  const compiled = compile(new Linear(), [randn([6, 8])], { target: CPUTarget(), dynamicShapes });
  const results = [];
  for (const shape of [[6, 8], [7, 8], [512, 8]]) {
    const out = await compiled(randn(shape));
    results.push(`${shape.join('x')} -> ${out.shape.join('x')}`);
  }
  console.log(`  ${label.padEnd(14)} ${results.join('   ')}`);
}

console.log('\n=== what a missing guard costs ===\n');
const dynamic = compile(new Linear(), [randn([6, 8])], { target: CPUTarget(), dynamicShapes: [true] });
for (const shape of [[6, 8], [12, 8], [6, 16]]) {
  const out = await dynamic(randn(shape));
  const values = (await out.toArray()).flat(9);
  const finite = values.every(Number.isFinite);
  console.log(`  input [${String(shape).padEnd(6)}] -> output [${out.shape.join('x')}]  ${finite ? 'finite' : 'NOT FINITE — ' + values.filter((v) => !Number.isFinite(v)).length + ' of ' + values.length + ' values are NaN'}`);
}
console.log('\n  The third row asked the kernel to contract a length-16 axis against a length-8');
console.log('  weight. Every guard passed, no kernel was recompiled, and the result is garbage.');
console.log('  Under the default (static) tracing the same call recompiles and is correct:\n');
const staticOne = compile(new Linear(), [randn([6, 8])], { target: CPUTarget() });
try {
  const out = await staticOne(randn([6, 16]));
  const values = (await out.toArray()).flat(9);
  console.log(`  static  [6,16   ] -> [${out.shape.join('x')}]  ${values.every(Number.isFinite) ? 'finite' : 'NOT FINITE'}`);
} catch (e) {
  console.log(`  static  [6,16   ] -> threw: ${e.message.split('\n')[0]}`);
}
