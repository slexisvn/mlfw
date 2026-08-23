import {
  compile, CPUTarget, randn, manual_seed,
} from '../../_internals.mjs';

manual_seed(54);

const x = randn([4, 8]);
const y = randn([4, 8]);
const w = randn([8, 8]);

const build = (fn, inputs, opts = {}) => {
  const k = compile({ forward: fn }, inputs, { target: CPUTarget(), ...opts });
  const src = (k.source() ?? '').split('\n').filter((l) => !l.startsWith('//')).join('\n');
  return { k, src };
};

const stats = (src) => ({
  lines: src.split('\n').filter((l) => l.trim()).length,
  loops: (src.match(/for \(let /g) ?? []).length,
  arrays: (src.match(/new \w+Array\(/g) ?? []).length,
  views: (src.match(/new \w+Array\(_mem_pool,/g) ?? []).length,
  pool: Number((src.match(/new ArrayBuffer\((\d+)\)/) ?? [, 0])[1]),
  accs: (src.match(/let _acc_\d+/g) ?? []).length,
  frounds: (src.match(/Math\.fround/g) ?? []).length,
});

const cases = [
  ['one elementwise op', (a) => a.mul(2.0), [x]],
  ['a chain of five', (a) => a.mul(2.0).add(1.0).relu().tanh().mul(0.5), [x]],
  ['two inputs', (a, b) => a.mul(b).add(a), [x, y]],
  ['a reduction', (a) => a.sum(1), [x]],
  ['matmul then relu', (a, b) => a.matmul(b).relu(), [x, w]],
];

console.log('=== what the CPU backend emits ===\n');
console.log(`  ${'program'.padEnd(20)} ${'lines'.padStart(5)} ${'loops'.padStart(5)} ${'arrays'.padStart(6)} ${'accs'.padStart(4)} ${'fround'.padStart(6)}`);
for (const [label, fn, inputs] of cases) {
  const s = stats(build(fn, inputs).src);
  console.log(`  ${label.padEnd(20)} ${String(s.lines).padStart(5)} ${String(s.loops).padStart(5)} ${String(s.arrays).padStart(6)} ${String(s.accs).padStart(4)} ${String(s.frounds).padStart(6)}`);
}

console.log('\n=== the chain of five, in full ===\n');
console.log(build((a) => a.mul(2.0).add(1.0).relu().tanh().mul(0.5), [x]).src);

console.log('=== the reduction, in full ===\n');
console.log(build((a) => a.sum(1), [x]).src);

const chain = (a) => a.mul(2.0).add(1.0).relu().mul(3.0).tanh();
console.log('=== a scalar constant is a buffer everywhere except in the source ===\n');
for (const poolAllocation of [false, true]) {
  const { src } = build(chain, [x], { fusion: { enabled: false }, memory: { poolAllocation } });
  const s = stats(src);
  const declared = [...src.matchAll(/const (buf_\d+) = new \w+Array\(/g)].map((m) => m[1]);
  const read = declared.filter((n) => new RegExp(`${n}\\[`).test(src));
  console.log(`  poolAllocation=${String(poolAllocation).padEnd(5)}  pool bytes=${String(s.pool).padStart(4)}` +
    `  declared=${String(declared.length).padStart(2)}  of which read=${read.length}` +
    `  distinct addresses used=${new Set([...src.matchAll(/new \w+Array\(_mem_pool, (\d+),/g)].map((m) => m[1])).size || 'n/a'}`);
}
console.log('\n  The five-operation chain has four scalar constants and four full-size');
console.log('  temporaries. On the default path the constants are folded into the');
console.log('  expressions that use them and the four temporaries are aliased onto');
console.log('  one, so a single `new Float32Array(32)` survives. Turn the pool on and');
console.log('  the aliasing happens by shared byte offset instead, so all four names');
console.log('  are declared as views onto offset 0 — and the four constants are given');
console.log('  arena slots the emitted program never reads.\n');

console.log('=== and every one of them computes what eager computes ===\n');
for (const [label, fn, inputs] of cases) {
  const { k } = build(fn, inputs);
  const got = (await (await k(...inputs)).toArray()).flat(9);
  const want = (await (await fn(...inputs)).toArray()).flat(9);
  const err = Math.max(...got.map((v, i) => Math.abs(v - want[i])));
  console.log(`  ${label.padEnd(20)} max err ${err.toExponential(0)}`);
}
