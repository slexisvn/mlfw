import {
  enumerateFactorizations, getTileStructure, levelCounts, CPU_TILING_SSRSRS,
  getSketchesForBlock, analyzeBlockStructure, collectAllBlockNames,
  lowerToTir, randn, CPUTarget, WebGPUTarget, printTensorIR,
} from '../../_internals.mjs';

function primeExponents(n) {
  const e = new Map();
  for (let p = 2; p * p <= n; p++) while (n % p === 0) { e.set(p, (e.get(p) || 0) + 1); n /= p; }
  if (n > 1) e.set(n, (e.get(n) || 0) + 1);
  return e;
}
const choose = (n, k) => { let r = 1; for (let i = 0; i < k; i++) r = (r * (n - i)) / (i + 1); return Math.round(r); };
const orderedFactorizations = (n, L) => [...primeExponents(n).values()].reduce((acc, e) => acc * choose(e + L - 1, L - 1), 1);

console.log('=== ordered factorizations of an extent into L levels ===\n');
console.log('  extent  levels   Theorem 44.2   enumerateFactorizations   uncapped');
for (const [n, L] of [[8, 3], [12, 3], [96, 2], [7, 4], [64, 4], [64, 5], [256, 4], [1024, 4], [2048, 4], [4096, 4], [720, 4]]) {
  const capped = enumerateFactorizations(n, L).length;
  const uncapped = enumerateFactorizations(n, L, 1e6).length;
  console.log(`  ${String(n).padStart(6)}  ${String(L).padStart(6)}   ${String(orderedFactorizations(n, L)).padStart(12)}   ${String(capped).padStart(23)}   ${String(uncapped).padStart(8)}`);
}
console.log('\n  The closed form and the uncapped enumerator agree everywhere. The');
console.log('  middle column is what the search actually gets: `maxCandidates`');
console.log('  defaults to 48 (factorization.ts:45), so from 64 upwards the space');
console.log('  is subsampled rather than enumerated.');

console.log('\n\n=== the three tile structures (tile_structure.ts) ===\n');
for (const [label, st] of [['CPU  (default)', getTileStructure(CPUTarget())],
                           ['GPU  (default)', getTileStructure(WebGPUTarget())],
                           ['CPU  SSRSRS   ', CPU_TILING_SSRSRS]]) {
  const { spatialLevels, reductionLevels } = levelCounts(st);
  const order = st.order.map(([k, l]) => `${k}${l}`).join(' ');
  const roles = Object.entries(st.roles).map(([k, v]) => `${k}=${v}`).join(' ');
  console.log(`  ${label}  ${st.name.padEnd(11)} order: ${order.padEnd(20)} S-levels ${spatialLevels}  R-levels ${reductionLevels}`);
  console.log(`  ${' '.repeat(label.length)}  ${' '.repeat(11)} roles: ${roles}`);
}
console.log('\n  The default CPU structure has one reduction level, and one level');
console.log('  means one factor: `enumerateFactorizations(K, 1)` returns `[[K]]`');
console.log('  (factorization.ts:46). The contraction axis of a matmul is never');
console.log('  split by `mlt_cpu`. Only `ssrsrs_cpu` has two R levels.');

const pf = await lowerToTir((a, b) => a.matmul(b), [randn([64, 64]), randn([64, 64])]);
console.log('\n\n=== a 64x64x64 matmul, as the autotuner sees it ===\n');
console.log(printTensorIR(pf).split('\n').filter(l => /for |block /.test(l)).map(l => '  ' + l.trim()).join('\n'));

console.log('\n  block                 axes              sketch           search variables                   points');
let joint = 1;
for (const name of collectAllBlockNames(pf.body)) {
  const st = analyzeBlockStructure(pf, name);
  const sketches = getSketchesForBlock(pf, name, CPUTarget());
  let blockTotal = 0;
  for (const s of sketches) {
    const vars = s.variables.map(v => `${v.name}[${v.candidates.length}]`).join(' ') || '(none)';
    const pts = s.variables.reduce((a, v) => a * v.candidates.length, 1);
    blockTotal += pts;
    console.log(`  ${name.padEnd(20)}  ${`S=${st.spatial} R=${st.reduction} reads=${st.reads}`.padEnd(16)}  ${s.name.padEnd(15)}  ${vars.padEnd(33)}  ${String(pts).padStart(6)}`);
  }
  console.log(`  ${' '.repeat(20)}  ${' '.repeat(16)}  ${'—'.padEnd(15)}  ${'block total'.padEnd(33)}  ${String(blockTotal).padStart(6)}`);
  joint *= blockTotal;
}
console.log(`\n  Two blocks, tuned independently, so the function's joint space is the`);
console.log(`  product: ${joint.toLocaleString('en-US')} distinct schedules for one matrix multiply.`);

console.log('\n\n=== the cap is a truncation, not a sample ===\n');
const full = enumerateFactorizations(4096, 4, 1e6);
const capped = enumerateFactorizations(4096, 4);
const lead = (ts) => [...new Set(ts.map(t => t[0]))].sort((a, b) => a - b);
console.log(`  extent 4096, 4 levels: ${full.length} tuples exist, ${capped.length} are offered.`);
console.log(`  leading factors present in all ${full.length}: ${lead(full).join(' ')}`);
console.log(`  leading factors present in the ${capped.length} offered: ${lead(capped).join(' ')}`);
console.log(`  largest tuple offered: ${JSON.stringify(capped[capped.length - 1])}`);
console.log('\n  `enumerateFactorizations` stops the recursion at `maxCandidates * 8`');
console.log('  tuples (factorization.ts:48) and only then subsamples. The recursion');
console.log('  walks divisors in ascending order, so the tuples it never reaches are');
console.log('  exactly the ones with a large outermost factor — the coarse-grained');
console.log('  tilings. For a 256-extent axis nothing is lost; for 4096 the whole');
console.log('  top of the range is.');
