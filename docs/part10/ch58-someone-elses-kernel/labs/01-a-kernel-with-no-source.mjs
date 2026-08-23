import {
  compileGraph, buildFunction, TensorType, ScalarType,
  lowerToTir, detectPureMatmul, CUBLAS_PROVIDER,
  registerExternalCodegenProvider, unregisterExternalCodegenProvider,
  activeExternalCodegenProviders, isExternalCodegenEnabled,
  registerExternalCodegen, unregisterExternalCodegen, getExternalCodegen,
  FuncAttr, TargetKind, CUDATarget, randn, manual_seed,
} from '../../_internals.mjs';

manual_seed(58);
const T = (s) => new TensorType(s, ScalarType.F32);

const kernelsOf = (graph, target, opts = {}) => {
  const res = compileGraph(graph, target, opts);
  return res.module.kernels.names().map((n) => ({ name: n, ...res.module.kernels.get(n) }));
};

const mm = () => buildFunction('mm', [T([64, 32]), T([32, 48])], [T([64, 48])],
  (b, a) => b.returnOp([b.matmul(a[0], a[1]).getResult(0)]));

console.log('=== the same graph, two backends for one operation ===\n');
console.log(`  ${'matmulBackend'.padEnd(14)} ${'kernel'.padEnd(10)} ${'source chars'.padStart(12)}  metadata`);
for (const backend of ['native', 'cublas']) {
  for (const k of kernelsOf(mm(), CUDATarget(), { matmulBackend: backend })) {
    console.log(`  ${backend.padEnd(14)} ${k.name.padEnd(10)} ${String(k.source.length).padStart(12)}  ${JSON.stringify(k.metadata).slice(0, 120)}`);
  }
}
console.log('\n  With `matmulBackend: cublas` the backend emits zero characters. The');
console.log('  compiled kernel is a descriptor — three dimensions, three operand');
console.log('  positions — and the runtime is expected to know what to do with it.');
console.log('  A CompiledKernel whose `source` is empty is a perfectly ordinary');
console.log('  CompiledKernel; nothing downstream needs a special case.');

console.log('\n=== what `detectPureMatmul` will and will not recognise ===\n');
const cases = [
  ['a plain matmul', (a, b) => a.matmul(b), [randn([8, 4]), randn([4, 6])]],
  ['matmul then relu', (a, b) => a.matmul(b).relu(), [randn([8, 4]), randn([4, 6])]],
  ['a batched matmul', (a, b) => a.matmul(b), [randn([2, 8, 4]), randn([2, 4, 6])]],
  ['an elementwise chain', (a, b) => a.mul(b), [randn([8, 4]), randn([8, 4])]],
];
for (const [label, fn, inputs] of cases) {
  const f = await lowerToTir(fn, inputs, CUDATarget());
  const info = detectPureMatmul(f);
  console.log(`  ${label.padEnd(22)} ${info ? `M=${info.M} N=${info.N} K=${info.K}  operands ${info.aIdx},${info.bIdx} -> ${info.cIdx}` : 'not a pure matmul'}`);
}
console.log('\n  The detector walks the lowered function, requires every block\'s name to');
console.log('  contain "matmul", requires f32 and rank 2 and static extents, and reads');
console.log('  the operand positions out of the buffer map. It is a pattern match on a');
console.log('  block name, which is the compiler\'s oldest way of asking what an');
console.log('  operation is and the one Part IV argues against everywhere else.');

console.log('\n=== a provider decides when, and a codegen entry decides what ===\n');
for (const [label, cfg] of [['default', {}], ['matmulBackend: cublas', { matmulBackend: CUBLAS_PROVIDER }]]) {
  const target = CUDATarget();
  const active = activeExternalCodegenProviders({ ...cfg, target }, target).map((p) => p.name);
  console.log(`  ${label.padEnd(22)} active providers: ${active.length ? active.join(', ') : '(none)'}` +
    `   enabled(cublas)=${isExternalCodegenEnabled(CUBLAS_PROVIDER, { ...cfg, target }, target)}`);
}
console.log('');
console.log('  A provider carries four things: a name, a predicate over the config and');
console.log('  the target, a list of graph passes to run when it is active, and an');
console.log('  annotate step that walks the lowered module and attaches an attribute to');
console.log('  the functions it claims. Emission is a separate registry keyed by the');
console.log('  same name, which the backend pipeline consults before anything else.');

console.log('\n=== a provider registered for the wrong target is not an error ===\n');
const original = getExternalCodegen(CUBLAS_PROVIDER);
unregisterExternalCodegen(CUBLAS_PROVIDER);
registerExternalCodegen(CUBLAS_PROVIDER, {
  targetKind: TargetKind.CPU,
  runtimeKind: 'js',
  compile: () => ({ source: '/* never reached on a CUDA target */', metadata: { kind: 'js' } }),
});
for (const k of kernelsOf(mm(), CUDATarget(), { matmulBackend: CUBLAS_PROVIDER })) {
  console.log(`  the graph still compiles: ${k.name}, ${k.source.length} characters of ${k.metadata.kind}`);
  console.log(`  the attribute is still on the function, and it was ignored.`);
}
unregisterExternalCodegen(CUBLAS_PROVIDER);
registerExternalCodegen(CUBLAS_PROVIDER, original);
console.log('\n  `BackendPipeline.compile` reads the attribute, looks the name up, and');
console.log('  checks `entry.targetKind === this.target.kind`. If either the lookup or');
console.log('  the check fails it falls through to ordinary codegen with no warning —');
console.log('  so a provider registered against the wrong target kind, or removed, is');
console.log('  a silent loss of the library call rather than a compilation error.');

console.log('\n=== the interface is open, and is one operation wide ===\n');
const NAME = 'book_example';
registerExternalCodegenProvider({
  name: NAME,
  suppressesEpilogueFusion: true,
  enabled: (config) => config.matmulBackend === NAME,
  annotate: (tirModule) => {
    for (const primFunc of tirModule) {
      const info = detectPureMatmul(primFunc);
      if (info) primFunc.setAttr(FuncAttr.EXTERNAL_CODEGEN, { name: NAME, info });
    }
  },
});
registerExternalCodegen(NAME, {
  targetKind: TargetKind.CUDA,
  runtimeKind: 'cuda',
  compile: (primFunc, target, info) => ({
    source: '',
    metadata: { kind: 'cuda', bookExample: info, outputIndices: [info.cIdx] },
  }),
});
for (const k of kernelsOf(mm(), CUDATarget(), { matmulBackend: NAME })) {
  console.log(`  ${k.name}: ${k.source.length} characters, metadata ${JSON.stringify(k.metadata)}`);
}
unregisterExternalCodegenProvider(NAME);
unregisterExternalCodegen(NAME);
console.log('\n  Twenty lines to route every pure matmul in a compilation to a library.');
console.log('  And then the shape of the interface: `ExternalKernelInfo` is');
console.log('  `{ M, N, K, transB?, aIdx, bIdx, cIdx }`. A convolution, an attention');
console.log('  kernel or an RNN has nothing to say in that vocabulary — the generic');
console.log('  mechanism carries one operation\'s signature in its type.');

console.log('\n=== what the split costs ===\n');
const fused = () => buildFunction('mmr', [T([64, 32]), T([32, 48])], [T([64, 48])],
  (b, a) => b.returnOp([b.relu(b.matmul(a[0], a[1]).getResult(0)).getResult(0)]));
for (const backend of ['native', 'cublas']) {
  const ks = kernelsOf(fused(), CUDATarget(), { matmulBackend: backend });
  console.log(`  ${backend.padEnd(8)} ${ks.length} kernel(s): ${ks.map((k) => `${k.name}(${k.source.length} chars)`).join(', ')}`);
}
console.log('\n  With the native backend the relu is fused into the matmul\'s epilogue');
console.log('  and the product never leaves the kernel. With cuBLAS it cannot be:');
console.log('  the provider declares `suppressesEpilogueFusion`, the graph is split,');
console.log('  and the 64x48 product is written to global memory by one kernel and');
console.log('  read back by the next. That round trip is the price of the library, and');
console.log('  it is why "call the fast kernel" is a graph-level decision rather than');
console.log('  a codegen one.');
