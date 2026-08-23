import {
  lowerToTir, randn, CPUTarget, getSketchesForBlock, Schedule, clonePrimFunc,
  toKernel, AnalyticalCostModel, FeatureExtractor,
} from '../../_internals.mjs';

const N = 256;
const target = CPUTarget();
const base = await lowerToTir((a, b) => a.matmul(b), [randn([N, N]), randn([N, N])]);
const mlt = getSketchesForBlock(base, 'matmul_1', target).find((s) => s.name === 'mlt_cpu');

const A = new Float32Array(N * N), B = new Float32Array(N * N);
for (let i = 0; i < N * N; i++) { A[i] = Math.sin(i * 0.7); B[i] = Math.cos(i * 0.3); }
const ref = new Float32Array(N * N);
for (let i = 0; i < N; i++) for (let k = 0; k < N; k++) { const a = A[i * N + k]; for (let j = 0; j < N; j++) ref[i * N + j] += a * B[k * N + j]; }

const POINTS = [
  ['(no tiling — the lowered nest)', null],
  ['s0=[1,1,1,256] s1=[1,1,1,256]', { s0: [1, 1, 1, 256], s1: [1, 1, 1, 256], r0: [256] }],
  ['s0=[256,1,1,1] s1=[256,1,1,1]', { s0: [256, 1, 1, 1], s1: [256, 1, 1, 1], r0: [256] }],
  ['s0=[8,1,4,8]   s1=[4,2,8,4]  ', { s0: [8, 1, 4, 8], s1: [4, 2, 8, 4], r0: [256] }],
  ['s0=[8,2,2,8]   s1=[8,2,2,8]  ', { s0: [8, 2, 2, 8], s1: [8, 2, 2, 8], r0: [256] }],
  ['s0=[1,1,256,1] s1=[1,1,1,256]', { s0: [1, 1, 256, 1], s1: [1, 1, 1, 256], r0: [256] }],
  ['s0=[32,1,8,1]  s1=[1,2,8,16] ', { s0: [32, 1, 8, 1], s1: [1, 2, 8, 16], r0: [256] }],
  ['s0=[2,2,2,32]  s1=[2,2,2,32] ', { s0: [2, 2, 2, 32], s1: [2, 2, 2, 32], r0: [256] }],
];

const cm = new AnalyticalCostModel(target);
const rows = [];
for (const [label, params] of POINTS) {
  const pf = clonePrimFunc(base);
  if (params) mlt.instantiate(params)(new Schedule(pf), 'matmul_1', target);
  const f = FeatureExtractor.extract(pf);
  const { call } = toKernel(pf, target);
  const C = new Float32Array(N * N);
  for (let i = 0; i < 5; i++) { C.fill(0); call(A, B, C); }
  const ts = [];
  for (let i = 0; i < 9; i++) { C.fill(0); const t0 = performance.now(); call(A, B, C); ts.push(performance.now() - t0); }
  ts.sort((a, b) => a - b);
  let worst = 0;
  for (let i = 0; i < N * N; i++) worst = Math.max(worst, Math.abs(C[i] - ref[i]));
  rows.push({ label, ms: ts[ts.length >> 1], worst, loops: f.numLoops, score: cm.score(pf) });
}

console.log('=== eight points of `mlt_cpu` on a 256x256 matmul ===\n');
console.log('  schedule                        loops   max |err| vs scalar   model score   median ms (MEASURED)');
for (const r of rows) {
  console.log(`  ${r.label}   ${String(r.loops).padStart(5)}   ${r.worst.toExponential(2).padStart(19)}   ${r.score.toFixed(6).padStart(11)}   ${r.ms.toFixed(3).padStart(19)}`);
}

const ms = rows.map((r) => r.ms);
const spread = Math.max(...ms) / Math.min(...ms);
console.log(`\n  measured spread: ${spread.toFixed(2)}x between the fastest and the slowest`);
console.log('  (MEASURED — this number changes from run to run and machine to machine)');

console.log('\n=== the reproducible half ===\n');
const errs = new Set(rows.map((r) => r.worst.toExponential(2)));
console.log(`  distinct error bounds across the eight schedules: ${errs.size}  ${[...errs].join(' ')}`);
console.log(`  distinct loop counts:                              ${new Set(rows.map((r) => r.loops)).size}  {${[...new Set(rows.map((r) => r.loops))].join(', ')}}`);
console.log(`  distinct cost-model scores:                        ${new Set(rows.map((r) => r.score.toFixed(9))).size}  {${[...new Set(rows.map((r) => r.score.toFixed(6)))].join(', ')}}`);
console.log('\n  Every point computes the same product to the same accuracy — the');
console.log('  primitives are sound, so the space contains no wrong answers. Every');
console.log('  tiled point has the same number of loops, because a four-level split');
console.log('  is three splits whatever the factors are. And every tiled point gets');
console.log('  the same score, which is Chapter 46\'s subject: the objective the');
console.log('  search maximises is constant on the whole 2,304-point space of');
console.log('  `mlt_cpu` — the largest sketch this block has that can actually run.');
