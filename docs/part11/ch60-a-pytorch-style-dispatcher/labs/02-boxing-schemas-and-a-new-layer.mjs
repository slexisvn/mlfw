import {
  dispatcher, DispatchKey, DispatchKeySet, KernelFunction, IValue, Library,
  parseSchema, withExcludedKeys, withIncludedKeys, guardStack,
  randn, tensor, manual_seed,
} from '../../_internals.mjs';

manual_seed(4);

console.log('=== a schema is a parsed string, and it decides which arguments are looked at ===\n');
for (const text of [
  'add(Tensor self, Tensor other) -> Tensor',
  'sum.dim(Tensor self, int[] dim, bool keepdim=False) -> Tensor',
  'topk(Tensor self, int k, int dim=-1) -> (Tensor, Tensor)',
  'to(Tensor self, Device device, Dtype dtype) -> Tensor',
]) {
  const s = parseSchema(text, 'mlc');
  const args = s.args.map((a) => `${a.kind}:${a.name || '_'}${a.defaultValue !== null ? '=' + a.defaultValue : ''}`).join(' ');
  console.log(`  ${s.key().padEnd(12)} ${args.padEnd(52)} tensors at ${JSON.stringify(s.tensorArgIndices)} returns ${s.returns.length}`);
}
const odd = parseSchema('mystery(Complex z, Tensor t) -> Tensor', 'mlc');
console.log(`\n  an unknown type becomes Scalar, silently: ${odd.args.map((a) => `${a.name}: ${a.kind}`).join(', ')} — tensors at ${JSON.stringify(odd.tensorArgIndices)}`);

console.log('\n=== boxed and unboxed are two calling conventions for the same kernel ===\n');
const unboxed = KernelFunction.fromUnboxed((_ks, a, b) => a + b);
const boxed = KernelFunction.fromBoxed((_ks, stack) => [IValue.int(stack[0].value + stack[1].value)]);
for (const [label, k] of [['fromUnboxed', unboxed], ['fromBoxed', boxed]]) {
  console.log(`  ${label.padEnd(12)} isBoxed=${String(k.isBoxed).padEnd(6)} isUnboxed=${String(k.isUnboxed).padEnd(6)}`
    + ` callUnboxed(2,3) = ${k.callUnboxed(null, 2, 3)}`
    + `   callBoxed([2,3]) = ${k.callBoxed(null, [IValue.int(2), IValue.int(3)])[0].value}`);
}
console.log('\n  Each convention is synthesised from the other when it is missing, so a kernel\n  registered one way is callable the other way.');

console.log('\n=== boxing infers a tag from the JavaScript value ===\n');
const TAGS = ['TENSOR', 'INT', 'FLOAT', 'BOOL', 'INT_LIST', 'TENSOR_LIST', 'STRING', 'NONE', 'DEVICE', 'DTYPE'];
const tagProbe = KernelFunction.fromBoxed((_ks, stack) => [IValue.intList(stack.map((iv) => iv.tag))]);
const values = [randn([2]), 3, 2.0, 2.5, true, 'mean', null, undefined, [1, 2], [randn([1])]];
const labels = ['a Tensor', '3', '2.0', '2.5', 'true', "'mean'", 'null', 'undefined', '[1, 2]', '[Tensor]'];
const tags = tagProbe.callUnboxed(null, ...values);
for (let i = 0; i < labels.length; i++) console.log(`  ${labels[i].padEnd(11)} -> ${TAGS[tags[i]]}`);
console.log('\n  2.0 is an INT, because the tag is chosen by Number.isInteger.');

console.log('\n=== a new op, a new backend kernel, and a new layer — from outside src/ ===\n');
const lib = new Library('lab', 'DEF');
lib.def('checksum(Tensor self, float k) -> float');
lib.impl('checksum', DispatchKey.CPU, (_ks, t, k) => {
  let acc = 0;
  for (const v of t.data) acc += v * k;
  return acc;
});

const handle = dispatcher.findOp('lab::checksum');
const x = tensor([1, 2, 3, 4], { shape: [2, 2] });
const CPU_ONLY = DispatchKeySet.fromKeys(DispatchKey.CPU);
console.log('  checksum(x, 10)        ', dispatcher.dispatch(handle, CPU_ONLY, x, 10));

const LOG_KEY = DispatchKey.CUSTOM_1;
const log = [];
handle.entry.registerKernel(LOG_KEY, KernelFunction.fromUnboxed((ks, ...args) => {
  log.push(`${args.length} args, remaining ${String(ks)}`);
  return dispatcher.redispatch(handle, ks, ...args);
}));
const WITH_LAYER = DispatchKeySet.fromKeys(DispatchKey.CPU, LOG_KEY);
console.log('  with a CUSTOM_1 layer  ', dispatcher.dispatch(handle, WITH_LAYER, x, 10), ' log:', JSON.stringify(log));
console.log(`  CUSTOM_1 is ${LOG_KEY} and CPU is ${DispatchKey.CPU}, so the layer sorts first — the priority order is the numbering`);

console.log('\n=== the guard stack rewrites the key set of every call inside it ===\n');
console.log(`  depth outside any guard: ${guardStack.depth}`);
withIncludedKeys(DispatchKeySet.fromKeys(LOG_KEY), () => {
  log.length = 0;
  const r = dispatcher.dispatch(handle, CPU_ONLY, x, 10);
  console.log(`  withIncludedKeys: depth ${guardStack.depth}, result ${r}, the layer ran ${log.length} time(s) although the call site asked for CPU alone`);
});
withExcludedKeys(DispatchKeySet.fromKeys(LOG_KEY), () => {
  log.length = 0;
  const r = dispatcher.dispatch(handle, WITH_LAYER, x, 10);
  console.log(`  withExcludedKeys: depth ${guardStack.depth}, result ${r}, the layer ran ${log.length} time(s) although the call site asked for it`);
});
console.log(`  depth after both:        ${guardStack.depth}`);
console.log('\n  A guard applies for its whole dynamic extent, so an included key must have a');
console.log('  kernel on every operator reachable inside it:');
withIncludedKeys(DispatchKeySet.fromKeys(LOG_KEY), () => {
  try { randn([2, 2]).mul(2); } catch (e) { console.log(`    ${e.message}`); }
});

console.log('\n=== what happens when nothing matches ===\n');
try {
  dispatcher.dispatch(handle, DispatchKeySet.fromKeys(DispatchKey.WASM), x, 10);
} catch (e) {
  console.log('  no WASM kernel     ->', e.message);
}
handle.entry.setCatchAll(KernelFunction.fromUnboxed(() => 0));
console.log('  after setCatchAll  ->', dispatcher.dispatch(handle, DispatchKeySet.fromKeys(DispatchKey.WASM), x, 10),
  '(a catch-all makes a key reachable, not correct)');
console.log('\n  Neither escape hatch is used by the framework: no operator sets a catch-all,');
console.log('  and the fallback table is empty, so every call the framework makes resolves in');
console.log('  the per-operator table or throws.');
