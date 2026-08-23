import {
  lowerToTir, toLIR, emit, encodeWat, compile, Schedule,
  WasmTarget, randn, manual_seed,
} from '../../_internals.mjs';

manual_seed(55);

const SIMD = { scheduling: { enabled: true } };
const N = 256;
const x = randn([N]);
const y = randn([N]);

const vecOps = (src) => (src.match(/f32x4\.\w+|i32x4\.\w+|v128\.\w+/g) ?? []);

console.log('=== which loops the backend agrees to vectorise ===\n');
const cases = [
  ['elementwise, two inputs', (a, b) => a.mul(b).add(a), [x, y]],
  ['a longer chain', (a) => a.mul(2.0).add(1.0).relu().tanh(), [x]],
  ['a transcendental', (a) => a.exp(), [x]],
  ['a comparison', (a, b) => a.maximum(b), [x, y]],
  ['a reduction', (a) => a.sum(), [x]],
  ['a 2-D reduction', (a) => a.sum(1), [randn([16, 16])]],
];
console.log(`  ${'program'.padEnd(24)} ${'v128 ops'.padStart(8)} ${'v128.load'.padStart(9)} ${'extract_lane'.padStart(12)}  vectorised?`);
for (const [label, fn, ins] of cases) {
  const src = compile({ forward: fn }, ins, { target: WasmTarget(), ...SIMD }).source() ?? '';
  const ops = vecOps(src);
  const loads = ops.filter((o) => o === 'v128.load').length;
  const lanes = ops.filter((o) => o.endsWith('extract_lane')).length;
  console.log(`  ${label.padEnd(24)} ${String(ops.length).padStart(8)} ${String(loads).padStart(9)} ${String(lanes).padStart(12)}  ${loads > 0 ? 'yes' : 'no, scalar loop'}`);
}

console.log('\n  A loop is vectorised only when all five legality conditions hold: the');
console.log('  lane variables are all bound inside the body, every guard is lane-');
console.log('  invariant, every subscript is affine in the lane variable, every guarded');
console.log('  load stays in range, and there is no loop-carried dependence. The plain');
console.log('  reduction fails the last one and falls back to the scalar loop, silently');
console.log('  and correctly.');
console.log('');
console.log('  `extract_lane` is the escape hatch inside a vectorised loop: WebAssembly');
console.log('  has no f32x4.exp, so a transcendental loads a vector, pulls the four');
console.log('  lanes out, calls the imported scalar function on each, and rebuilds the');
console.log('  vector. The loads and the stores are vectorised; the mathematics is not.');

console.log('\n=== and where it fires, the answers are unchanged ===\n');
for (const [label, fn, ins] of cases) {
  const scalar = compile({ forward: fn }, ins, { target: WasmTarget() });
  const vector = compile({ forward: fn }, ins, { target: WasmTarget(), ...SIMD });
  const flat = (v) => (Array.isArray(v) ? v.flat(9) : [v]);
  const a = flat(await (await scalar(...ins)).toArray());
  const b = flat(await (await vector(...ins)).toArray());
  const err = Math.max(...a.map((v, i) => Math.abs(v - b[i])));
  console.log(`  ${label.padEnd(24)} scalar vs vectorised: ${err === 0 ? 'bit-identical' : `max err ${err.toExponential(2)}`}`);
}
console.log('\n  Five of the six are bit-identical — level N0 — because a lane-wise');
console.log('  elementwise loop performs exactly the operations the scalar loop did.');
console.log('  The 2-D reduction is not, and that is not a rounding accident: its');
console.log('  reduction axis was annotated @vectorized by the rule policy, so it took');
console.log('  the vectorised accumulator, which keeps four partial sums. The next');
console.log('  section is that difference, made large enough to see.');

console.log('\n=== a vectorised reduction is not the same sum ===\n');

const reduceKernel = async (n, vectorize) => {
  const f = await lowerToTir((a) => a.sum(), [randn([n])], WasmTarget());
  if (vectorize) {
    const sch = new Schedule(f);
    sch.vectorize(sch.getLoops('reduce_acc_1')[0]);
    return emit(toLIR(sch.func, WasmTarget()), WasmTarget());
  }
  return emit(toLIR(f, WasmTarget()), WasmTarget());
};

const runReduce = async (kernel, values) => {
  const inst = await WebAssembly.instantiate(await WebAssembly.compile(encodeWat(kernel.source)), { math: {} });
  const mem = new Float32Array(inst.exports.memory.buffer);
  const off = kernel.metadata.bufferOffsets;
  mem.set(values, off.get('buf_1') / 4);
  inst.exports.traced(0, 0);
  return mem[off.get('buf_3') / 4];
};

for (const n of [64, 66]) {
  const values = Array.from({ length: n }, (_, i) => (i === 0 ? 2 ** 24 : 1));
  const scalar = await runReduce(await reduceKernel(n, false), values);
  const vector = await runReduce(await reduceKernel(n, true), values);
  console.log(`  n=${String(n).padEnd(3)}  x = [2^24, 1, 1, ...]`);
  console.log(`         scalar loop      ${scalar}`);
  console.log(`         four lanes       ${vector}   (differs by ${vector - scalar})`);
}

const vk = await reduceKernel(64, true);
console.log('');
console.log(`  the vectorised kernel uses: ${[...new Set(vecOps(vk.source))].join(' ')}`);
console.log(`  and ${vk.source.split('\n').filter((l) => l.trim().startsWith('f32x4.extract_lane')).length} extract_lane instructions to fold the four partial sums together.`);
console.log('');
console.log('  Every element of x after the first is 1, and 2^24 + 1 rounds back to');
console.log('  2^24 in f32 — so the scalar loop absorbs all 63 of them and returns');
console.log('  2^24. The four-lane loop keeps four partial sums; three of them are');
console.log('  pure counts of ones, which survive, and the combine adds them at the');
console.log('  end. Same operands, same operator, different association.');
console.log('');
console.log('  This is level N2 on the book\'s ladder, and it is the reason the');
console.log('  vectorised accumulator is gated on `+` alone: `_accumInstr` throws');
console.log('  outright for an integer max or min reduction rather than reassociating');
console.log('  one silently.');

console.log('\n=== a vector binding nothing reads ===\n');
const vsrc = compile({ forward: (a, b) => a.mul(b).add(a) }, [x, y], { target: WasmTarget(), ...SIMD }).source() ?? '';
const vlets = [...new Set([...vsrc.matchAll(/local\.set \$(\w+_vlet)/g)].map((m) => m[1]))];
for (const name of vlets) {
  const sets = (vsrc.match(new RegExp(`local\\.set \\$${name}\\b`, 'g')) ?? []).length;
  const gets = (vsrc.match(new RegExp(`local\\.get \\$${name}\\b`, 'g')) ?? []).length;
  console.log(`  ${name}: ${sets} set, ${gets} get`);
}
const lines = vsrc.split('\n').map((l) => l.trim());
const first = lines.findIndex((l) => l === 'f32x4.splat');
const last = lines.findIndex((l) => /local\.set \$\w+_vlet/.test(l));
if (first >= 0 && last > first) {
  console.log(`\n  the ${last - first + 1} instructions that produce it:\n`);
  console.log(lines.slice(first - 2, last + 1).map((l) => `    ${l}`).join('\n'));
}
console.log('\n  The loop\'s scope binding is the flat index, and in vector mode the');
console.log('  emitter builds a vector *of* that index — splatting the outer part and');
console.log('  filling the four lanes with consecutive values — in float lanes, because');
console.log('  the vector mode\'s dtype is the one the arithmetic uses. Then the actual');
console.log('  addressing goes through the scalar address CSE instead, and the vector');
console.log('  is never read. It is dead work inside the hot loop.');

console.log('\n=== the other axis: one loop across workers ===\n');
const par = compile({ forward: (a, b) => a.mul(b).add(a) }, [randn([4096]), randn([4096])],
  { target: WasmTarget({ numCores: 4 }), scheduling: { enabled: true } });
const psrc = par.source() ?? '';
console.log(`  the module takes _par_start/_par_end parameters: ${/_par_start/.test(psrc)}`);
console.log('  When a loop is annotated @parallel and the backend can prove the');
console.log('  partition is safe, the emitted function takes a half-open range and the');
console.log('  runtime hands each worker a slice of the same linear memory. That is');
console.log('  the one form of thread-level parallelism any of the four backends');
console.log('  reaches on a CPU.');
