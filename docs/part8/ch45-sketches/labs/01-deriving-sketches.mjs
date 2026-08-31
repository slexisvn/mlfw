import {
  lowerToTir, randn, CPUTarget, WebGPUTarget, CUDATarget,
  getSketchesForBlock, deriveSketches, analyzeBlockStructure, collectAllBlockNames,
  buildBlockDAG, findFusibleConsumer, classifyBlock, findBlock,
  analyzePureMatmul, Schedule, clonePrimFunc, printTensorIR, ScheduleValidator, toKernel,
  Buffer, PrimFunc, SeqNode, BlockNode, BufferStoreNode, BufferLoadNode,
  VariableNode, IntImmNode, MathOpNode, ForNode, ForKind,
} from '../../_internals.mjs';

const CASES = [
  ['a matmul', (a, b) => a.matmul(b), [randn([16, 16]), randn([16, 16])]],
  ['matmul then relu', (a, b) => a.matmul(b).relu(), [randn([16, 16]), randn([16, 16])]],
  ['two elementwise ops', (a, b) => a.mul(b).add(1.0), [randn([16, 16]), randn([16, 16])]],
  ['a sum over one axis', (a) => a.sum(1), [randn([16, 16])]],
];

console.log('=== which sketch each block gets, and why (derivation.ts:60-71) ===\n');
console.log('  program              block              S  R  reads  reduction   CPU sketches                              GPU sketches');
for (const [label, fn, inputs] of CASES) {
  const pf = await lowerToTir(fn, inputs);
  const dag = buildBlockDAG(pf);
  let first = true;
  for (const name of collectAllBlockNames(pf.body).slice().reverse()) {
    const st = analyzeBlockStructure(pf, name);
    const cpu = getSketchesForBlock(pf, name, CPUTarget(), undefined, { dag }).map((s) => s.name).join(' ');
    const gpu = getSketchesForBlock(pf, name, WebGPUTarget(), undefined, { dag }).map((s) => s.name).join(' ');
    console.log(`  ${(first ? label : '').padEnd(20)} ${name.padEnd(18)} ${st.spatial}  ${st.reduction}  ${String(st.reads).padStart(5)}  ${String(st.hasReduction).padEnd(9)}   ${cpu.padEnd(41)} ${gpu}`);
    first = false;
  }
}
console.log('\n  Three rules, tried in priority order. Priority 10 wants a reduction,');
console.log('  at least one spatial axis and at least two declared reads — the');
console.log('  "compute intensive" test, which only a matmul passes. Priority 20');
console.log('  takes any other reduction. Priority 30 matches everything left, so');
console.log('  the derivation is total. The classification is structural: nothing');
console.log('  in it looks at the block name.');

const pf = await lowerToTir((a, b) => a.matmul(b), [randn([16, 16]), randn([16, 16])]);
console.log('\n\n=== the holes of `mlt_cpu` on a 16x16x16 matmul ===\n');
const mlt = getSketchesForBlock(pf, 'matmul_1', CPUTarget()).find((s) => s.name === 'mlt_cpu');
for (const v of mlt.variables) {
  const shown = v.candidates.slice(0, 5).map((c) => JSON.stringify(c)).join(' ');
  console.log(`  ${v.name}  ${String(v.candidates.length).padStart(2)} candidates:  ${shown}${v.candidates.length > 5 ? ' ...' : ''}`);
}

console.log('\n=== two points, and the nest each produces ===');
for (const params of [{ s0: [1, 1, 1, 16], s1: [1, 1, 1, 16], r0: [16] },
  { s0: [2, 2, 2, 2], s1: [1, 2, 4, 2], r0: [16] }]) {
  const work = clonePrimFunc(pf);
  mlt.instantiate(params)(new Schedule(work), 'matmul_1', CPUTarget());
  console.log(`\n  ${JSON.stringify(params)}`);
  const lines = printTensorIR(work).split('\n');
  const start = lines.findIndex((l) => l.includes('ls0_6'));
  for (const l of lines.slice(start).filter((l) => /for |bind vls0|bind vrs0/.test(l))) console.log('   ' + l.trim());
}
console.log('\n  Nine loops in both cases: a four-level split is three calls to `split`');
console.log('  whatever the factors are, and `applyRoles` then parallelises the');
console.log('  outermost spatial level and vectorises the innermost one. The factors');
console.log('  appear only as extents and as multipliers in the two bindings — the');
console.log('  mixed-radix reconstruction of Proposition 45.5, and no guard appears,');
console.log('  because at every step the split factor divides the extent exactly.');

console.log('\n\n=== `ssrsrs_cpu`: the standard reduction tiling, and what stops it ===\n');
const ssrsrs = getSketchesForBlock(pf, 'matmul_1', CPUTarget()).find((s) => s.name === 'ssrsrs_cpu');
console.log(`  offered with ${ssrsrs.variables.reduce((a, v) => a * v.candidates.length, 1)} points, r0 = ${ssrsrs.variables[2].candidates.map((c) => JSON.stringify(c)).join(' ')}`);
try {
  ssrsrs.instantiate({ s0: [1, 1, 1, 16], s1: [1, 1, 1, 16], r0: [1, 16] })(new Schedule(clonePrimFunc(pf)), 'matmul_1', CPUTarget());
  console.log('  applied: no error');
} catch (e) {
  console.log(`  applied: ${e.message}`);
}
console.log('\n  `createSSRSRSTilingSketch` opens with `decomposeReduction` (tiling.ts:131)');
console.log('  and Chapter 41 showed that primitive throws on every block the lowering');
console.log('  rules emit. So the only structure in this compiler that would split a');
console.log('  contraction axis is unreachable, and the reachable one has a single');
console.log('  reduction level, which means one factor, which means no split at all.');

console.log('\n\n=== `fused`: producer-consumer fusion, and what stops it ===\n');
const relu = await lowerToTir((a, b) => a.matmul(b).relu(), [randn([16, 16]), randn([16, 16])]);
const rdag = buildBlockDAG(relu);
console.log('  block DAG of `relu(a @ b)`:');
for (const b of rdag.blocks) console.log(`    ${b.name.padEnd(18)} declared reads [${b.reads}]  writes [${b.writes}]`);
const info = classifyBlock(relu, 'matmul_1');
const blk = findBlock(relu.body, 'matmul_1');
console.log(`\n  matmul_1: enclosing spatial loop variables [${info.loops.filter((l) => !info.reductionLoopVars.has(l.loopVar.name)).map((l) => l.loopVar.name)}]`);
console.log(`            store subscript variables        [${blk.body.indices.map((i) => i.name)}]`);
console.log(`            findFusibleConsumer -> ${findFusibleConsumer(relu, rdag, 'matmul_1', classifyBlock)}`);

function directNest() {
  const A = new Buffer('A', [8], 'f32', 'global'), T = new Buffer('T', [8], 'f32', 'global'), C = new Buffer('C', [8], 'f32', 'global');
  const i = new VariableNode('i', 'int32'), j = new VariableNode('j', 'int32');
  const p = new BlockNode('prod', [{ iterVar: i, binding: i }], [{ buffer: A }], [{ buffer: T }],
    new BufferStoreNode(T, [i], new MathOpNode('*', new BufferLoadNode(A, [i]), new IntImmNode(2))));
  const c = new BlockNode('cons', [{ iterVar: j, binding: j }], [{ buffer: T }], [{ buffer: C }],
    new BufferStoreNode(C, [j], new MathOpNode('+', new BufferLoadNode(T, [j]), new IntImmNode(1))));
  const body = new SeqNode([
    new ForNode(i, new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, p),
    new ForNode(j, new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, c),
  ]);
  return new PrimFunc('direct', [], body, new Map([['A', A], ['T', T], ['C', C]]));
}
const dn = directNest();
console.log('\n  the same test on a hand-built pair whose store subscript IS its loop');
console.log(`  variable: findFusibleConsumer('prod') -> ${findFusibleConsumer(dn, buildBlockDAG(dn), 'prod', classifyBlock)}`);
console.log('\n  `findFusibleConsumer` compares the producer\'s store subscripts against');
console.log('  its enclosing loop variable names (block_dag.ts:119). Every block the');
console.log('  lowering rules emit binds `v...` iteration variables to `...` loop');
console.log('  variables, so the two name lists never match and the comparison always');
console.log('  fails. The mechanism works; the shape it was written for is not the');
console.log('  shape this compiler produces.');

console.log('\n\n=== what the primitives refuse, and what they do not ===\n');
{
  const probe = (label, fn) => {
    const work = clonePrimFunc(pf);
    const sch = new Schedule(work);
    try {
      fn(sch);
      const errs = ScheduleValidator.validate(work);
      console.log(`  ${label.padEnd(46)} ACCEPTED${errs.length ? '  (validator: ' + errs.length + ' error' + (errs.length > 1 ? 's' : '') + ')' : ''}`);
    } catch (e) {
      console.log(`  ${label.padEnd(46)} ${e.message}`);
    }
  };
  probe('fuseLoops(m, k)  — not a direct child', (s2) => { const l = s2.getLoops('matmul_1'); s2.fuseLoops(l[0], l[2]); });
  probe('decomposeReduction on a lowered block', (s2) => s2.decomposeReduction('matmul_1'));
  probe('bindThread(m, "warpIdx.x")', (s2) => s2.bindThread(s2.getLoops('matmul_1')[0], 'warpIdx.x'));
  probe('rfactor over the reduction axis c0_8', (s2) => s2.rfactor('matmul_1', 'c0_8', 2));
  probe('rfactor over the SPATIAL axis ls0_6', (s2) => s2.rfactor('matmul_1', 'ls0_6', 2));
}
console.log('\n  `rfactor` checks that the named loop exists, that its extent is constant,');
console.log('  that the factor divides it, and that the block body is an accumulating');
console.log('  store over an associative operator (schedule.ts:633-654). It never checks');
console.log('  that the axis it is factoring is a *reduction* axis, and the block');
console.log('  abstraction records exactly that (Chapter 33). Reassociating a spatial');
console.log('  axis is not a reordering of a sum; it is a different program.');
{
  const A = new Float32Array(256).map((_, i) => (i % 7) - 3);
  const B = new Float32Array(256).map((_, i) => (i % 5) - 2);
  const ref = new Float32Array(256), got = new Float32Array(256);
  toKernel(clonePrimFunc(pf)).call(A, B, ref);
  const bad = clonePrimFunc(pf);
  new Schedule(bad).rfactor('matmul_1', 'ls0_6', 2);
  try {
    toKernel(bad).call(A, B, got);
    let worst = 0;
    for (let i = 0; i < 256; i++) worst = Math.max(worst, Math.abs(got[i] - ref[i]));
    console.log(`  running the spatially-rfactored kernel: max |error| = ${worst}`);
  } catch (e) {
    console.log(`  running the spatially-rfactored kernel: ${e.constructor.name}: ${e.message}`);
  }
}
console.log('\n  Nothing in this compiler asks for it — `createRfactorSketch` offers only');
console.log('  axes drawn from `blockInfo.reductionLoopVars` (sketch_generators.ts:34) —');
console.log('  and the search path runs `ScheduleValidator`, which rejects the result.');
console.log('  It is a precondition the primitive relies on its callers to hold.');

console.log('\n\n=== the GPU escape hatch: a sketch that replaces the body ===\n');
const gpu = await lowerToTir((a, b) => a.matmul(b), [randn([128, 128]), randn([128, 128])], CUDATarget());
const plan = analyzePureMatmul(gpu);
const rich = deriveSketches(gpu, plan.reductionBlock, CUDATarget(), { richGpu: true });
console.log(`  analyzePureMatmul: block ${plan.reductionBlock}, M=${plan.dims.M} N=${plan.dims.N} K=${plan.dims.K}`);
console.log(`  sketches: ${rich.map((s) => `${s.name}[${s.variables[0].candidates.length}]`).join(' ')}`);
console.log(`  and it carries an enumerate(): ${typeof rich[0].enumerate === 'function'}, ${rich[0].enumerate().length} configurations`);
const built = clonePrimFunc(gpu);
const bsch = new Schedule(built);
rich[0].instantiate({ config_index: 0 })(bsch, plan.reductionBlock, CUDATarget());
console.log(`  after applying config 0: ${bsch.trace.length} trace steps recorded`);
console.log('  the body it produced:');
console.log(printTensorIR(built).split('\n').slice(4, 14).map((l) => '   ' + l).join('\n'));
console.log('\n  This is not a schedule. `createMatmulRegisterBlockGPUSketch` assigns');
console.log('  `schedule.func.body = body` (gpu_matmul_sketch.ts:20) with a nest built');
console.log('  from scratch by `buildRegisterBlockedMatmul`. It is Chapter 43\'s finding');
console.log('  reached from the other side — the compiler\'s best GPU kernel is written,');
console.log('  not scheduled — and the consequence visible here is the one Chapter 48');
console.log('  collects: zero trace steps.');
