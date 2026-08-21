import {
  lowerToTir, printTensorIR, Schedule, toKernel, randn, BackendPipeline, CUDATarget,
} from '../../_internals.mjs';

const show = (sch, from = 4) => console.log(printTensorIR(sch.func).split('\n').slice(from, -1).join('\n'));

// ----------------------------------------------------------- computeInline

const inline = new Schedule(await lowerToTir((x) => x.mul(x).add(1.0), [randn([2, 3])]));
console.log(`=== two blocks, one intermediate: ${inline.state.allBlockNames().join(', ')} ===`);
show(inline);

inline.computeInline('mul_block_0');
console.log('\n=== computeInline("mul_block_0") ===');
show(inline);
console.log('\n  The producer nest is gone, its store has become a subexpression of');
console.log('  the consumer, and `buf_5` is no longer read or written. The read set');
console.log('  of add_block_1 went from {buf_5, buf_4} to {buf_4, buf_1}: the block');
console.log('  lost the intermediate and inherited what the producer read, which is');
console.log('  what it now loads. `retargetBufferReads` is the one place the read set');
console.log('  of Chapter 33 is actually maintained.');
console.log('\n  Note the operand is now duplicated: `buf_1[..] * buf_1[..]` was one');
console.log('  load of buf_5 and is now two loads of buf_1. Inlining trades memory');
console.log('  traffic for recomputation, and nothing here counts the trade.');

// ------------------------------------------------------------- cacheRead

const cr = new Schedule(await lowerToTir((a, b) => a.matmul(b), [randn([4, 6]), randn([6, 5])]));
cr.cacheRead('matmul_1', cr.getBlock('matmul_1').reads[0].buffer.name, 'local');
console.log('\n=== cacheRead("matmul_1", "buf_1", "local") ===');
show(cr, 15);
console.log('\n  A staging nest, an `allocate`, and the block reads the cache now.');
console.log('  The copy is over the WHOLE buffer: 4x6 elements, hoisted above the');
console.log('  whole nest, because cacheRead builds its loops from `buf.shape`');
console.log('  (schedule.ts:1014) and not from the region the block reads. On a GPU');
console.log('  the useful version of this primitive stages one tile inside the');
console.log('  k-loop; that version would need computeAt, which §41.6 gets to.');

// ------------------------------------------------------------ cacheWrite

const cw = new Schedule(await lowerToTir((a, b) => a.matmul(b), [randn([4, 6]), randn([6, 5])]));
cw.cacheWrite('matmul_1', 'buf_5', 'local');
console.log('\n=== cacheWrite("matmul_1", "buf_5", "local") ===');
show(cw, 15);
console.log('\n  Same shape, mirrored: the block now writes the cache and a flush');
console.log('  nest copies it back afterwards.');

// ---------------------------------------------------- what runs, and what not

console.log('\n=== do the four memory primitives change the answer? ===\n');
const A = new Float32Array([...Array(24).keys()].map((i) => i + 1));
const B = new Float32Array([...Array(30).keys()].map((i) => (i % 5) - 2));
const run = (sch) => {
  const out = new Float32Array(20);
  toKernel(sch.func).call(A, B, out);
  return [...out].join(' ');
};
const base = new Schedule(await lowerToTir((a, b) => a.matmul(b), [randn([4, 6]), randn([6, 5])]));
console.log(`  baseline   ${run(base)}`);
console.log(`  cacheRead  ${run(cr)}`);
console.log(`  cacheWrite ${run(cw)}`);

console.log('\n  All three agree — on this backend. The same cacheWrite schedule');
console.log('  compiled for CUDA does not:\n');
const cuda = new BackendPipeline(CUDATarget()).compile(cw.func).source.split('\n');
for (const l of cuda.filter((l) => /cachew\[20\]|cachew\[vls|_o0 = buf/.test(l))) console.log(`      ${l.trim()}`);
console.log('\n  Two things. `float …cachew[20];` is not zeroed, and the block');
console.log('  accumulates into it — the init block above still zeroes buf_5, which');
console.log('  the flush then overwrites, so the zeroing is dead and the accumulator');
console.log('  starts at whatever the stack held. It works on CPU only because a');
console.log('  fresh Float32Array is zero-filled. And the flush block declares');
console.log('  `const int x = x;`, because cacheWrite reuses one VariableNode as');
console.log('  both the loop variable and the block iteration variable');
console.log('  (schedule.ts:767). Neither is caught by anything, and neither');
console.log('  matters today: no code in src/ calls cacheWrite.');

// ------------------------------------------------------ decomposeReduction

const dr = new Schedule(await lowerToTir((x) => x.sum(1), [randn([2, 8])]));
console.log('\n=== decomposeReduction on a lowered reduction ===\n');
try {
  dr.decomposeReduction('reduce_acc_1');
  console.log('  accepted');
} catch (e) {
  console.log(`  ${e.message}`);
}

const rf = new Schedule(await lowerToTir((x) => x.sum(1), [randn([2, 8])]));
rf.rfactor('reduce_acc_1', rf.getLoops('reduce_acc_1')[1].loopVar.name, 4);
try {
  rf.decomposeReduction('reduce_acc_1_rf_p');
  console.log(`  after rfactor, on '${'reduce_acc_1_rf_p'}': accepted`);
  console.log(`  blocks now: ${rf.state.allBlockNames().join(', ')}`);
} catch (e) {
  console.log(`  after rfactor: ${e.message}`);
}

console.log('\n  `decomposeReduction` splits an init-bearing block into an init');
console.log('  block and an update block, and it opens by requiring `initBody`');
console.log('  (schedule.ts:718). No lowering rule sets that field — Chapter 33');
console.log('  finding 12 — so on compiler-produced TIR the primitive always');
console.log('  throws. It works on a block rfactor built, because rfactor is the');
console.log('  only thing in the compiler that sets initBody.');
console.log('\n  That matters beyond this lab: createSSRSRSTilingSketch');
console.log('  (autotune/tiling.ts:131) opens with schedule.decomposeReduction,');
console.log('  so the SSRSRS tiling structure cannot be applied to any block the');
console.log('  lowering rules produce. Part VIII returns to it.');
