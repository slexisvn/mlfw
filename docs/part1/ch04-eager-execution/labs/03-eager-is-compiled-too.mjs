import { randn, manual_seed } from '../../../../dist/index.node.js';

manual_seed(0);

function timeOnce(fn) {
  const t0 = performance.now();
  fn();
  return performance.now() - t0;
}

function medianBatched(fn, reps, inner) {
  const s = [];
  for (let i = 0; i < reps; i++) {
    const t0 = performance.now();
    for (let k = 0; k < inner; k++) fn();
    s.push((performance.now() - t0) / inner);
  }
  s.sort((a, b) => a - b);
  return s[reps >> 1];
}

console.log('the first call at a new shape pays for code generation; later calls do not\n');
console.log('   shape        first call      steady state       ratio');

for (const n of [8, 9, 10, 11]) {
  const a = randn([n, n]);
  const b = randn([n, n]);
  const first = timeOnce(() => a.add(b));
  for (let i = 0; i < 1000; i++) a.add(b);
  const steady = medianBatched(() => a.add(b), 15, 2000);
  console.log(
    `  ${n}x${n}`.padEnd(12) +
    `${first.toFixed(3).padStart(12)} ms` +
    `${(steady * 1000).toFixed(2).padStart(15)} us` +
    `${Math.round(first / steady).toString().padStart(12)}x`
  );
}

const c = randn([8, 8]);
const d = randn([8, 8]);
console.log(`\n  a fresh pair of tensors at an already-seen shape: ${timeOnce(() => c.add(d)).toFixed(3)} ms`);
