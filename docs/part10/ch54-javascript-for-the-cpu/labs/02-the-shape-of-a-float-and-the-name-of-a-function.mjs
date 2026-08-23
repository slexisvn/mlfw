import {
  compile, CPUTarget, WasmTarget, CUDATarget, WebGPUTarget, tensor, randn, manual_seed,
} from '../../_internals.mjs';

manual_seed(54);
const f = Math.fround;

console.log('=== JavaScript has no float32, so the backend has to add one ===\n');
const A = f(1 + 2 ** -12);
const C = f(2 ** -24);
console.log(`  a = b = 1 + 2^-12 = ${A}`);
console.log(`  c       = 2^-24   = ${C}`);
console.log('');
console.log(`  a*b in f64, exactly            ${A * A}`);
console.log(`  rounded to f32 once, at the end  fround(a*b + c) = ${f(A * A + C)}`);
console.log(`  rounded after every operation    fround(fround(a*b) + c) = ${f(f(A * A) + C)}`);
console.log(`  they differ by ${Math.abs(f(f(A * A) + C) - f(A * A + C))}, which is one f32 ulp at this magnitude (2^-23 = ${2 ** -23})`);

const a = tensor([[A]]);
const c = tensor([[C]]);
const fn = (p, q) => p.mul(p).add(q);
const eager = (await fn(a, c).toArray()).flat(9)[0];
const k = compile({ forward: fn }, [a, c], { target: CPUTarget() });
const compiled = (await (await k(a, c)).toArray()).flat(9)[0];
console.log('');
console.log(`  eager                            ${eager}`);
console.log(`  compiled                         ${compiled}`);
console.log(`  the compiled kernel rounds per operation: ${compiled === f(f(A * A) + C)}`);

console.log('\n=== where Math.fround appears, and where it does not ===\n');
const x = randn([2, 3]);
for (let n = 1; n <= 6; n++) {
  let g = (t) => t;
  for (let i = 0; i < n; i++) { const prev = g; g = (t) => prev(t).mul(1.0000001); }
  const src = compile({ forward: g }, [x], { target: CPUTarget() }).source() ?? '';
  const body = src.split('\n').find((l) => l.includes('] =')) ?? '';
  console.log(`  ${n} multiplies -> ${String((body.match(/Math\.fround/g) ?? []).length).padStart(2)} roundings`);
}
console.log('\n  n operations, n-1 roundings. The outermost one is dropped because the');
console.log('  value is about to be stored into a Float32Array, and that store rounds.');
console.log('  Every inner rounding is load-bearing: it is what makes the emitted');
console.log('  JavaScript compute in the width the program declared.');

console.log('\n=== the externs that need no rounding ===\n');
const exact = ['max', 'min', 'abs', 'floor', 'ceil', 'round', 'sqrt'];
console.log(`  exact in f32, emitted bare:   ${exact.join(' ')}`);
console.log('  inexact, wrapped in a rounding: exp log sin cos tanh pow log2 ...');
console.log('');
for (const [label, fn2] of [
  ['relu (Math.max)', (t) => t.relu().add(1.0)],
  ['sqrt', (t) => t.abs().sqrt().add(1.0)],
  ['tanh', (t) => t.tanh().add(1.0)],
  ['exp', (t) => t.exp().add(1.0)],
]) {
  const src = compile({ forward: fn2 }, [x], { target: CPUTarget() }).source() ?? '';
  const body = (src.split('\n').find((l) => l.includes('] =')) ?? '').trim();
  console.log(`  ${label.padEnd(16)} ${body.replace(/^buf_\d+\[[^\]]*\] *= */, '').replace(/;$/, '').replace(/buf_1\[[^\]]*\]/g, 'x')}`);
}
console.log('\n  A function whose f32 result is exactly the f32 rounding of its exact');
console.log('  result needs no correction; one that is computed in f64 and only then');
console.log('  narrowed does. `Math.max` is in the first class, `Math.tanh` in the second.');

console.log('\n=== erf: three backends share an approximation, one does not ===\n');

const AS_A = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429];
const AS_P = 0.3275911;
const erfAS = (v) => {
  const sign = v < 0 ? -1 : 1;
  const av = Math.abs(v);
  const t = 1 / (1 + AS_P * av);
  let poly = 0;
  for (let i = AS_A.length - 1; i >= 0; i--) poly = poly * t + AS_A[i];
  return sign * (1 - poly * t * Math.exp(-av * av));
};
const erfSeries = (v) => {
  const sign = v < 0 ? -1 : 1;
  const av = Math.abs(v);
  let sum = av;
  let term = av;
  for (let n = 1; n < 400; n++) {
    term *= -av * av / n;
    const cn = term / (2 * n + 1);
    sum += cn;
    if (Math.abs(cn) < 1e-19 * Math.abs(sum)) break;
  }
  return sign * 2 / Math.sqrt(Math.PI) * sum;
};

const probe = tensor([[0.0451, 0.5, 1.0, 2.0]]);
const show = (label, vals) => console.log(`  ${label.padEnd(9)} ` + vals.map((v) => v.toPrecision(9).padStart(14)).join(''));
show('eager', (await probe.erf().toArray()).flat(9));
for (const [label, target] of [['cpu', CPUTarget()], ['wasm', WasmTarget()]]) {
  const kf = compile({ forward: (t) => t.erf() }, [probe], { target });
  show(label, (await (await kf(probe)).toArray()).flat(9));
}
for (const [label, target] of [['cuda', CUDATarget()], ['webgpu', WebGPUTarget()]]) {
  const kf = compile({ forward: (t) => t.erf() }, [probe], { target });
  const src = kf.source() ?? '';
  const line = (src.split('\n').find((l) => /erff|select\(-1/.test(l)) ?? '').trim();
  console.log(`  ${label.padEnd(9)} emits ${line.replace(/^buf_\d+\[[^\]]*\] *= */, '').slice(0, 70)}`);
}

let worst = 0;
let worstAt = 0;
for (let i = 0; i <= 32000; i++) {
  const v = i / 10000;
  const d = Math.abs(erfAS(v) - erfSeries(v));
  if (d > worst) { worst = d; worstAt = v; }
}
console.log('');
console.log('  the shared approximation is Abramowitz & Stegun 7.1.26; against a Taylor');
console.log(`  reference its worst absolute error on [0, 3.2] is ${worst.toExponential(3)} at x = ${worstAt.toFixed(4)},`);
console.log(`  which is ${(worst / 2 ** -23).toFixed(2)} f32 ulps at magnitude 1.`);
console.log('');
console.log('  eager, cpu and wasm agree to the last bit because all three read the');
console.log('  same constants out of util/special_math.ts, and webgpu inlines them into');
console.log('  WGSL. CUDA calls erff from the device math library instead, so it is the');
console.log('  one backend whose erf is not this approximation — close, and not equal.');

console.log('\n=== the emitted function name is the model\'s name, unchecked ===\n');
class Shadow {}
Object.defineProperty(Shadow, 'name', { value: 'Math' });
const model = new Shadow();
model.forward = (t) => t.tanh();

const shadowed = compile(model, [x], { target: CPUTarget() });
console.log((shadowed.source() ?? '').split('\n').filter((l) => !l.startsWith('//')).join('\n'));
try {
  await shadowed(x);
  console.log('  it ran');
} catch (e) {
  console.log(`  it threw: ${e.constructor.name}: ${e.message}`);
}
console.log('');
console.log('  `function Math(...)` binds the name `Math` inside its own scope, so the');
console.log('  `Math.tanh` in the body resolves to the kernel rather than to the global.');
console.log('  The backend emits `function ${func.name}(...)` with no sanitisation, and');
console.log('  the name arrives from the traced model\'s constructor.');

const ok = new (class Net {})();
ok.forward = (t) => t.tanh();
const fine = compile(ok, [x], { target: CPUTarget() });
const got = (await (await fine(x)).toArray()).flat(9);
const want = (await x.tanh().toArray()).flat(9);
console.log(`\n  the same program under a different class name: max err ${Math.max(...got.map((v, i) => Math.abs(v - want[i])))}`);
