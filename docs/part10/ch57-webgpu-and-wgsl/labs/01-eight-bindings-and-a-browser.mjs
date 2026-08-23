import {
  compile, compileGraph, buildFunction, TensorType, ScalarType,
  WebGPUTarget, CUDATarget, randn, manual_seed,
} from '../../_internals.mjs';

manual_seed(57);
const T = (s) => new TensorType(s, ScalarType.F32);

const kernelsOf = (graph, target, opts = {}) => {
  const res = compileGraph(graph, target, opts);
  return res.module.kernels.names().map((n) => ({ name: n, ...res.module.kernels.get(n) }));
};

console.log('=== a buffer is a binding, and there are not many bindings ===\n');
const few = buildFunction('few', [T([16]), T([16])], [T([16])],
  (b, a) => b.returnOp([b.add(a[0], a[1]).getResult(0)]));
for (const k of kernelsOf(few, WebGPUTarget())) {
  console.log(k.source.split('\n').filter((l) => l.startsWith('@group')).map((l) => `  ${l}`).join('\n'));
}

const N = 8;
const many = buildFunction('many', Array.from({ length: N }, () => T([16])), [T([16])], (b, a) => {
  let v = a[0];
  for (let i = 1; i < N; i++) v = b.add(v, a[i]).getResult(0);
  b.returnOp([v]);
});
console.log(`\n  the same shape with ${N + 1} buffers:\n`);
for (const k of kernelsOf(many, WebGPUTarget(), { fusion: { enabled: false } })) {
  console.log(k.source.split('\n').filter((l) => l.startsWith('@group')).map((l) => `  ${l}`).join('\n'));
  for (const b of k.metadata.bindings) {
    if (!b.packed) continue;
    console.log(`    ${b.name}: ${b.packed.length} buffers, ${b.packedSize} elements`);
    console.log(`      ${b.packed.map((p) => `${p.name}@${p.offset}`).join('  ')}`);
  }
  const access = k.source.split('\n').find((l) => /_pw_\w+\[/.test(l));
  console.log(`\n  and an access reads: ${access.trim()}`);
  const scalars = [...new Set([...k.source.matchAll(/var (_s\d+): /g)].map((m) => m[1]))];
  console.log(`  every intermediate collapsed to a per-thread scalar: ${scalars.join(' ') || '(none)'}`);
}
console.log('\n  A WebGPU compute shader receives its buffers through an explicit');
console.log('  binding table, and a device is only required to offer eight storage');
console.log('  buffers per stage. Past six the backend concatenates every statically-');
console.log('  sized buffer of the same element type and access mode into one storage');
console.log('  array and gives each a base offset — Chapter 53\'s flat index, one');
console.log('  level further out.');

console.log('\n=== a dynamic shape is a uniform, not a parameter ===\n');
const x = randn([4, 8]);
const dyn = compile({ forward: (a) => a.mul(2.0).add(1.0) }, [x], { target: WebGPUTarget(), dynamic_shapes: [true] });
const dsrc = dyn.source() ?? '';
console.log(dsrc.split('\n').filter((l) => /struct ShapeParams|_ds_|uniform/.test(l)).slice(0, 8).map((l) => `  ${l.trim()}`).join('\n'));
console.log(`\n  and a use: ${(dsrc.split('\n').find((l) => l.includes('_shapes.')) ?? '').trim()}`);
console.log('\n  WGSL has no scalar kernel parameters, so the shape parameters become');
console.log('  a uniform struct in one more binding, and every reference to an extent');
console.log('  is an i32 cast of a u32 field.');

console.log('\n=== the same packing problem, solved twice more inside one kernel ===\n');
const sm = buildFunction('sm', [T([8, 64])], [T([8, 64])],
  (b, a) => b.returnOp([b.softmax(a[0], 1).getResult(0)]));
for (const k of kernelsOf(sm, WebGPUTarget())) {
  const slots = [...new Set([...k.source.matchAll(/var (_lt\d+|_s\d+): /g)].map((m) => m[1]))];
  const wg = [...k.source.matchAll(/var<workgroup> (\w+): array<(\w+), (\d+)>/g)];
  console.log(`  ${k.name.padEnd(8)} private slots: ${slots.join(' ') || '(none)'}`);
  console.log(`  ${''.padEnd(8)} workgroup:     ${wg.map((m) => `${m[1]}[${m[3]}]`).join(' ') || '(none)'}`);
  const declared = wg.reduce((s, m) => s + Number(m[3]) * 4, 0);
  console.log(`  ${''.padEnd(8)} declared workgroup bytes ${declared}, reported sharedMemBytes ${k.metadata.sharedMemBytes}`);
}
console.log('\n  `_assignLocalSlots` computes a live interval for every kernel-local');
console.log('  buffer and hands out slots from a min-heap keyed on when each frees —');
console.log('  which is Chapters 49 and 50 again, at codegen time, over WGSL variable');
console.log('  names. A buffer with at most one element per thread and no cross-thread');
console.log('  sharing gets something better than a slot: a single scalar, `_s0`.');
console.log('');
console.log('  The reported `sharedMemBytes` counts only buffers that arrived already');
console.log('  in shared scope. Everything the backend itself promoted to');
console.log('  `var<workgroup>` is missing from the figure, which is why the two');
console.log('  numbers above disagree. The CUDA backend sums both.');

console.log('\n=== WGSL has no infinite literal ===\n');
const mx = buildFunction('mx', [T([4, 16])], [T([4])], (b, a) => {
  const neg = b.scalarConstant(-Infinity, ScalarType.F32).getResult(0);
  b.returnOp([b.reduce(a[0], neg, [1], 'max').getResult(0)]);
});
for (const [label, target] of [['cuda', CUDATarget()], ['webgpu', WebGPUTarget()]]) {
  for (const k of kernelsOf(mx, target)) {
    const line = k.source.split('\n').find((l) => /INFINITY|0x1\.fff/.test(l));
    if (line) console.log(`  ${label.padEnd(8)} ${line.trim()}`);
  }
}
console.log('');
console.log('  A max-reduction\'s identity is -infinity, and WGSL forbids a non-finite');
console.log('  literal, so the backend emits the largest finite f32 instead. On any');
console.log('  input whose values are finite the two agree. On an input that already');
console.log('  contains -infinity — a causal attention mask is the everyday case —');
console.log('  they do not: CUDA propagates it and WebGPU clamps it to -3.4e38.');

console.log('\n=== the reasons this backend gives up ===\n');
const cases = {
  elementwise: () => buildFunction('ew', [T([64, 64])], [T([64, 64])], (b, a) => b.returnOp([b.tanh(a[0]).getResult(0)])),
  softmax: () => buildFunction('sm2', [T([8, 64])], [T([8, 64])], (b, a) => b.returnOp([b.softmax(a[0], 1).getResult(0)])),
  layer_norm: () => buildFunction('ln', [T([8, 64]), T([64]), T([64])], [T([8, 64])], (b, a) => b.returnOp([b._inferAndBuild('layer_norm', [a[0], a[1], a[2]], { axis: 1, epsilon: 1e-5 }).getResult(0)])),
  matmul: () => buildFunction('mm', [T([64, 64]), T([64, 64])], [T([64, 64])], (b, a) => b.returnOp([b.matmul(a[0], a[1]).getResult(0)])),
};
console.log(`  ${'graph'.padEnd(12)} ${'kernel'.padEnd(9)} ${'threads'.padStart(8)}  diagnosis`);
for (const [name, mk] of Object.entries(cases)) {
  for (const k of kernelsOf(mk(), WebGPUTarget())) {
    const m = k.metadata;
    const threads = m.workgroupSize.reduce((a, b) => a * b) * m.dispatchSize.reduce((a, b) => a * b);
    console.log(`  ${name.padEnd(12)} ${k.name.padEnd(9)} ${String(threads).padStart(8)}  ${m.launchDiagnosis ? m.launchDiagnosis.reason : '-'}`);
  }
}
console.log('\n  WebGPUTarget declares `{ enabled: true }` in its scheduling attributes,');
console.log('  so every kernel here is scheduled — and the ones that then turn out to');
console.log('  need a value across workgroups are serialized, with the reason recorded');
console.log('  in the same place the CUDA backend records it, and read by the same');
console.log('  nobody.');
