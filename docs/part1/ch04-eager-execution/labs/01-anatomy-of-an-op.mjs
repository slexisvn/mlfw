import { randn, manual_seed } from '../../../../dist/index.node.js';

manual_seed(0);

function median(fn, reps, inner) {
  const samples = [];
  for (let i = 0; i < reps; i++) {
    const t0 = performance.now();
    for (let k = 0; k < inner; k++) fn();
    samples.push((performance.now() - t0) / inner);
  }
  samples.sort((a, b) => a - b);
  return samples[reps >> 1];
}

function timeOp(build, n, inner) {
  const run = build(n);
  for (let i = 0; i < Math.max(20, inner); i++) run();
  return median(run, 15, inner);
}

const addAt = (n) => { const a = randn([n, n]); const b = randn([n, n]); return () => a.add(b); };
const tanhAt = (n) => { const a = randn([n, n]); return () => a.tanh(); };

console.log('one eager add, by size');
console.log('      n      elements       us/call      ns/element');
const rows = [];
for (const [n, inner] of [[1, 3000], [4, 3000], [16, 2000], [64, 1000], [256, 200], [1024, 20]]) {
  const ms = timeOp(addAt, n, inner);
  rows.push([n, ms]);
  console.log(
    `${String(n).padStart(7)}${String(n * n).padStart(14)}` +
    `${(ms * 1000).toFixed(2).padStart(14)}${(ms * 1e6 / (n * n)).toFixed(2).padStart(16)}`
  );
}

const alpha = rows[0][1] * 1000;
const big = rows[rows.length - 1];
const beta = big[1] * 1e6 / (big[0] * big[0]);
console.log(`\nfixed cost per call   alpha = ${alpha.toFixed(2)} us`);
console.log(`marginal cost         beta  = ${beta.toFixed(2)} ns/element`);
console.log(`break-even size             = ${Math.round(alpha * 1000 / beta)} elements` +
            ` (a ${Math.round(Math.sqrt(alpha * 1000 / beta))}x${Math.round(Math.sqrt(alpha * 1000 / beta))} tensor)`);

const N = 1024;
const addMs = timeOp(addAt, N, 20);
const tanhMs = timeOp(tanhAt, N, 20);
const bytes = 3 * N * N * 4;
console.log(`\nat ${N}x${N} (${(N * N * 4 / 1048576).toFixed(0)} MB per tensor)`);
console.log(`  add   ${addMs.toFixed(2)} ms   -> ${(bytes / addMs / 1e6).toFixed(2)} GB/s of traffic`);
console.log(`  tanh  ${tanhMs.toFixed(2)} ms   -> ${(tanhMs / addMs).toFixed(1)}x the cost of add`);
