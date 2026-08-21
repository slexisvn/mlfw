import {
  lowerToTir, Schedule, ScheduleValidator, BackendPipeline,
  CPUTarget, WasmTarget, randn, compile,
} from '../../_internals.mjs';

// Three layers of this compiler answer "may the reduction axis of a matmul run
// in parallel?" and they do not agree. This lab asks all three.

const build = async () => new Schedule(await lowerToTir((a, b) => a.matmul(b), [randn([4, 6]), randn([6, 5])]));

// -------------------------------------- what the block declares about its axes

const decl = await build();
console.log('=== the block\'s own declaration ===\n');
for (const iv of decl.getBlock('matmul_1').iterVars) {
  console.log(`  ${iv.iterVar.name.padEnd(10)} bound to ${String(iv.binding.name).padEnd(8)} kind ${iv.kind}`);
}
console.log('\n  `markCommReduce` tagged the contraction axis when the lowering rule');
console.log('  built the block (Chapter 33). Nothing has verified the tag.');

// ------------------------------------------------ layer 1: the primitive

console.log('\n=== layer 1: what each primitive says about that axis ===\n');
for (const primitive of ['parallelize', 'vectorize', 'unroll']) {
  const sch = await build();
  const k = sch.getLoops('matmul_1')[2];
  try {
    sch[primitive](k);
    console.log(`  ${primitive.padEnd(12)} ACCEPTED   loop kind is now @${k.kind}`);
  } catch (e) {
    console.log(`  ${primitive.padEnd(12)} REFUSED    ${e.message}`);
  }
}
console.log('\n  Same loop, same dependence, opposite answers. `parallelize` passes');
console.log('  IterVarPolicy.SPATIAL, which admits DataPar only; `vectorize` passes');
console.log('  ACCUMULABLE, which admits CommReduce too (legality.ts:17). The');
console.log('  dependence is found in both cases and overruled in one.');

// --------------------------------------------------- layer 2: the validator

const vec = await build();
vec.vectorize(vec.getLoops('matmul_1')[2]);

console.log('\n=== layer 2: what ScheduleValidator says about the result ===\n');
for (const e of ScheduleValidator.validate(vec.func)) console.log(`  ${e}`);
console.log('\n  The validator disagrees with the primitive that produced this IR.');
console.log('  `Schedule.verify()` calls it and nothing in the pipeline calls');
console.log('  `Schedule.verify()`; the one production caller is the autotuner\'s');
console.log('  session (autotune/session.ts:186). A schedule the rule policy built');
console.log('  is never validated; a schedule the search built always is.');

// ---------------------------------------------------- layer 3: the backend

console.log('\n=== layer 3: what each backend does with it ===\n');
const cpu = new BackendPipeline(CPUTarget()).compile(vec.func).source;
console.log('  CPU  : SIMD in the emitted JavaScript? ' +
  (/v128|f32x4/.test(cpu) ? 'yes' : 'no — the annotation is inert here'));

const wasmDirect = new BackendPipeline(WasmTarget()).compile(vec.func).source;
console.log(`  WASM : ${wasmDirect.split('\n').filter((l) => /f32x4/.test(l)).length} SIMD opcodes`
  + ' — `_vectorizationIsLegal` ends with `!loopCarriedDependenceIn(body)`');
console.log('         (backend/wasm/codegen.ts:1606) and declines.');

// The shipping pipeline reaches the same annotation by a different route, and
// runs AccumulatorDetectionPass first, which gives the backend a node it can
// vectorise correctly.
const x = randn([8, 64]);
const shipped = compile({ forward: (a) => a.sum(1) }, [x], {
  target: WasmTarget({ numCores: 4 }), fusion: { enabled: false }, scheduling: { enabled: true },
});
const got = await shipped(x);
const wat = shipped.source();
console.log(`\n  WASM, through compile(): ${wat.split('\n').filter((l) => /f32x4/.test(l)).length} SIMD opcodes,`
  + ` ${wat.split('\n').filter((l) => /extract_lane/.test(l)).length} extract_lane`);
const expect = await x.sum(1);
const close = (await got.toArray()).flat().every((v, i) => Math.abs(v - expect.toArray().flat()[i]) < 1e-4);
console.log(`  and the answer matches the reference: ${close}`);
console.log('\n  `ReductionWasmRule` vectorises the reduction axis on purpose');
console.log('  (rules.ts:492). Ahead of codegen, AccumulatorDetectionPass has');
console.log('  turned the accumulation into an LIRAccumulatorNode, and the WASM');
console.log('  backend has a SIMD path for exactly that node: four lanes of');
console.log('  partial sums and a horizontal reduce at the end — the four');
console.log('  `extract_lane` opcodes. It is rfactor by 4, performed by the');
console.log('  backend, on the strength of the same associativity licence');
console.log('  Chapter 41 charged for.');
console.log('\n  So all three layers end up correct, for three different reasons:');
console.log('  layer 1 trusted a declaration, layer 2 was never asked, and layer 3');
console.log('  re-derived the dependence and handled it. Only the third is a proof.');
