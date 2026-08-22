import { randn, manual_seed } from '../../../../dist/index.node.js';
import { summarize, format } from '../../../tools/measure.mjs';

manual_seed(0);

function sample(fn, reps, inner) {
  const samples = [];
  for (let i = 0; i < reps; i++) {
    const t0 = performance.now();
    for (let k = 0; k < inner; k++) fn();
    samples.push((performance.now() - t0) / inner);
  }
  return summarize(samples);
}

function timeOp(build, n, inner) {
  const run = build(n);
  for (let i = 0; i < Math.max(20, inner); i++) run();
  return sample(run, 21, inner);
}

// Weighted least squares for T = alpha + beta*m.
// Weights are 1/T^2, so the fit minimizes *relative* error: without them the
// 1048576-element point would be the only one the residual sum can see, and
// the "fit" would degenerate into reading beta off the largest row.
function fitAffine(points) {
  let sw = 0, sm = 0, st = 0, smm = 0, smt = 0;
  for (const [m, t] of points) {
    const w = 1 / (t * t);
    sw += w; sm += w * m; st += w * t; smm += w * m * m; smt += w * m * t;
  }
  const det = sw * smm - sm * sm;
  const beta = (sw * smt - sm * st) / det;
  const alpha = (st - beta * sm) / sw;
  let worst = 0;
  for (const [m, t] of points) worst = Math.max(worst, Math.abs(alpha + beta * m - t) / t);
  return { alpha, beta, worst };
}

const addAt = (n) => { const a = randn([n, n]); const b = randn([n, n]); return () => a.add(b); };
const tanhAt = (n) => { const a = randn([n, n]); return () => a.tanh(); };

console.log('one eager add, by size   (median of 21 rounds)');
console.log('      n      elements       us/call      ns/element        rel. IQR');
const points = [];
for (const [n, inner] of [[1, 3000], [4, 3000], [16, 2000], [64, 1000], [256, 200], [1024, 20]]) {
  const s = timeOp(addAt, n, inner);
  points.push([n * n, s.median]);
  console.log(
    `${String(n).padStart(7)}${String(n * n).padStart(14)}` +
    `${(s.median * 1000).toFixed(2).padStart(14)}${(s.median * 1e6 / (n * n)).toFixed(2).padStart(16)}` +
    `${(s.rsd * 100).toFixed(1).padStart(14)}%`
  );
}

const { alpha, beta, worst } = fitAffine(points);
const alphaUs = alpha * 1000;
const betaNs = beta * 1e6;
const breakEven = alphaUs * 1000 / betaNs;
console.log(`\nweighted least-squares fit of T(m) = alpha + beta*m over all ${points.length} points`);
console.log(`fixed cost per call   alpha = ${alphaUs.toFixed(2)} us`);
console.log(`marginal cost         beta  = ${betaNs.toFixed(2)} ns/element`);
console.log(`worst relative residual     = ${(worst * 100).toFixed(1)}%`);
console.log(`break-even size             = ${Math.round(breakEven)} elements` +
            ` (a ${Math.round(Math.sqrt(breakEven))}x${Math.round(Math.sqrt(breakEven))} tensor)`);

// The two-point shortcut, printed next to the fit so the difference is visible.
// It is not a fit: it reads alpha off the smallest row and beta off the largest,
// so it cannot disagree with either, and it cannot report a residual.
const naiveAlpha = points[0][1] * 1000;
const naiveBeta = points[points.length - 1][1] * 1e6 / points[points.length - 1][0];
console.log(`\nfor comparison, the two-point shortcut alpha=T(1), beta=T(N)/N:`);
console.log(`  alpha = ${naiveAlpha.toFixed(2)} us, beta = ${naiveBeta.toFixed(2)} ns/element`);
console.log(`  it reports no residual because it passes through both points by construction`);
console.log(`  ns/element is still falling at m=${points[points.length - 1][0]}, so T(N)/N is an`);
console.log(`  upper bound on the asymptotic marginal cost, not an estimate of it`);

const N = 1024;
const addS = timeOp(addAt, N, 20);
const tanhS = timeOp(tanhAt, N, 20);
const elems = N * N;
const addBytes = 3 * elems * 4;   // two reads + one write
const tanhBytes = 2 * elems * 4;  // one read + one write
console.log(`\nat ${N}x${N} (${(elems * 4 / 1048576).toFixed(0)} MB per tensor)`);
console.log(`  add   ${format(addS)}`);
console.log(`         ${(addBytes / 1048576).toFixed(0)} MB moved -> ${(addBytes / addS.median / 1e6).toFixed(2)} GB/s`);
console.log(`  tanh  ${format(tanhS)}`);
console.log(`         ${(tanhBytes / 1048576).toFixed(0)} MB moved -> ${(tanhBytes / tanhS.median / 1e6).toFixed(2)} GB/s`);
console.log(`  tanh costs ${(tanhS.median / addS.median).toFixed(1)}x an add while moving ${(tanhBytes / addBytes).toFixed(2)}x the bytes`);
