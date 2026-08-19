import { randn, Module, compile, CPUTarget, manual_seed } from '../../../../dist/index.node.js';

manual_seed(0);

class Cheap extends Module {
  forward(t) {
    return t.mul(2).add(1).relu().mul(0.5).add(3).mul(1.1)
            .mul(1.5).add(0.25).relu().mul(0.8).add(2).mul(1.2);
  }
}

class WithTanh extends Module {
  forward(t) {
    return t.mul(2).add(1).relu().mul(0.5).add(3).tanh()
            .mul(1.5).add(0.25).relu().mul(0.8).add(2).tanh();
  }
}

function median(fn, reps) {
  const samples = [];
  for (let i = 0; i < reps; i++) {
    const t0 = performance.now();
    fn();
    samples.push(performance.now() - t0);
  }
  samples.sort((a, b) => a - b);
  return samples[reps >> 1];
}

const N = 1024;
const x = randn([N, N]);

console.log(`twelve elementwise operations on a ${N}x${N} tensor (${(N * N * 4 / 1048576).toFixed(0)} MB)\n`);
console.log('  chain              eager ms   compiled ms     speedup');

for (const [name, Klass] of [['10 cheap + 2 tanh', WithTanh], ['12 cheap ops     ', Cheap]]) {
  const model = new Klass();
  const compiled = compile(model, [x], { target: CPUTarget() });
  await compiled._ready;

  for (let i = 0; i < 5; i++) { model.forward(x); compiled(x); }

  const eagerMs = median(() => model.forward(x), 9);
  const compiledMs = median(() => compiled(x), 9);

  console.log(`  ${name}${eagerMs.toFixed(1).padStart(11)}${compiledMs.toFixed(1).padStart(14)}${(eagerMs / compiledMs).toFixed(2).padStart(12)}x`);
}
