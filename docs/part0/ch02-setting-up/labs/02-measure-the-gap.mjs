import {
  randn, Module, compile, CPUTarget, manual_seed,
} from '../../../../dist/index.node.js';

manual_seed(0);

class Chain extends Module {
  forward(t) {
    return t.mul(2).add(1).relu().mul(0.5).add(3).tanh()
            .mul(1.5).add(0.25).relu().mul(0.8).add(2).tanh();
  }
}

class CheapChain extends Module {
  forward(t) {
    return t.mul(2).add(1).relu().mul(0.5).add(3).mul(1.1)
            .mul(1.5).add(0.25).relu().mul(0.8).add(2).mul(1.2);
  }
}

function median(fn, repeats, inner) {
  const samples = [];
  for (let i = 0; i < repeats; i++) {
    const t0 = performance.now();
    for (let k = 0; k < inner; k++) fn();
    samples.push((performance.now() - t0) / inner);
  }
  samples.sort((a, b) => a - b);
  return samples[repeats >> 1];
}

async function measure(Klass, n, inner) {
  const x = randn([n, n]);
  const model = new Klass();
  const compiled = compile(model, [x], { target: CPUTarget() });
  await compiled._ready;

  for (let i = 0; i < 200; i++) { model.forward(x); compiled(x); }

  const eagerMs = median(() => model.forward(x), 15, inner);
  const compiledMs = median(() => compiled(x), 15, inner);

  const a = model.forward(x);
  const b = compiled(x);
  let diff = 0;
  for (let i = 0; i < a.data.length; i++) diff = Math.max(diff, Math.abs(a.data[i] - b.data[i]));

  return { eagerMs, compiledMs, diff };
}

console.log('twelve operations, two `tanh` among them\n');
console.log('  size      eager    compiled   ratio   max abs diff');

for (const [n, inner] of [[16, 200], [128, 50], [512, 5]]) {
  const r = await measure(Chain, n, inner);
  console.log(
    `${String(n).padStart(6)}` +
    `${(r.eagerMs * 1000).toFixed(1).padStart(11)} us` +
    `${(r.compiledMs * 1000).toFixed(1).padStart(9)} us` +
    `${(r.eagerMs / r.compiledMs).toFixed(2).padStart(8)}x` +
    `${r.diff.toExponential(1).padStart(14)}`
  );
}

console.log('\nthe same twelve operations with the two `tanh` replaced by multiplications\n');
console.log('  size      eager    compiled   ratio   max abs diff');

for (const [n, inner] of [[16, 200], [128, 50], [512, 5]]) {
  const r = await measure(CheapChain, n, inner);
  console.log(
    `${String(n).padStart(6)}` +
    `${(r.eagerMs * 1000).toFixed(1).padStart(11)} us` +
    `${(r.compiledMs * 1000).toFixed(1).padStart(9)} us` +
    `${(r.eagerMs / r.compiledMs).toFixed(2).padStart(8)}x` +
    `${r.diff.toExponential(1).padStart(14)}`
  );
}
