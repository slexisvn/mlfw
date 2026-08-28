import {
  _traceCore, compile, firstFunction, CPUTarget, TraceLevel,
  ShapeEnv, randn, manual_seed,
} from '../../_internals.mjs';

manual_seed(11);

const W = randn([8, 4]);
const linear = (x) => x.matmul(W).relu();

const showGuard = (g) => (g.type === 'divisible' ? `${g.sym} % ${g.divisor} == 0` : `${g.lhs} ${g.op} ${g.rhs}`);

console.log('=== every dimension gets a symbol; only some keep it ===\n');
for (const [label, dynamicShapes] of [
  ['static (the default)', null],
  ['dynamic_shapes: [{0}]', [new Set([0])]],
  ['dynamic_shapes: [true]', [true]],
]) {
  const core = await _traceCore(linear, [randn([6, 8])], { name: 'lin', dynamicShapes });
  const f = firstFunction(core.graph);
  const env = core.shapeEnv;
  console.log(`  ${label}`);
  console.log(`    input type   ${f.inputTypes[0].dtype}[${f.inputTypes[0].shape}]      (-1 is the DYNAMIC marker)`);
  console.log(`    symbols      ${[...env.symbols].map(([n, i]) => `${n}=hint ${i.hint} (arg ${i.inputIdx}, dim ${i.dimIdx})`).join(', ')}`);
  console.log(`    guards       ${env.guards.map(showGuard).join(' , ')}`);
  console.log(`    output shape ${JSON.stringify(core.outputSymShapes)}\n`);
}

console.log('=== an example of batch 1 is silently specialised ===\n');
for (const n of [1, 2, 7]) {
  const core = await _traceCore(linear, [randn([n, 8])], { name: 'lin', dynamicShapes: [new Set([0])] });
  const f = firstFunction(core.graph);
  console.log(`  example batch ${n}: type [${f.inputTypes[0].shape}]  guards ${core.shapeEnv.guards.map(showGuard).join(', ')}`);
}
console.log('\n  produceShapeSpec requires concreteShape[i] > 1 before it will keep a dimension');
console.log('  symbolic, so tracing on a batch of one produces a static kernel and recompiles');
console.log('  for every other batch size.');

console.log('\n=== a guard set answers "may I reuse this kernel?" ===\n');
for (const [label, dims] of [['batch dimension only', new Set([0])], ['every dimension', new Set([0, 1])]]) {
  const env = new ShapeEnv();
  env.produceShapeSpec(0, [6, 8], dims);
  for (const [name] of env.symbols) if (dims.has(env.symbols.get(name).dimIdx)) env.guardRelation(name, 'gt', 0);
  console.log(`  ${label}: guards ${env.guards.map(showGuard).join(', ')}`);
  for (const shape of [[6, 8], [1, 8], [4096, 8], [6, 16]]) {
    env.bindInputShapes([{ shape }]);
    const { passed, failedGuard } = env.evaluateGuards();
    console.log(`    [${String(shape).padEnd(8)}] -> ${passed ? 'reuse' : `recompile (failed: ${showGuard(failedGuard)})`}`);
  }
}
console.log('\n  Under "every dimension" the [6,16] input passes: nothing in the guard set says');
console.log('  the contracting dimension has to stay 8. Lab 02 runs that kernel.');

console.log('\n=== how often the compiler actually runs ===\n');
function counted(dynamicShapes) {
  let compiles = 0;
  const compiled = compile({ forward: linear }, [randn([6, 8])], {
    target: CPUTarget(),
    dynamicShapes,
    trace: {
      level: TraceLevel.DEBUG,
      sink: (e) => { if (e.type === 'phase' && e.phase === 'codegen' && e.action === 'start') compiles++; },
    },
  });
  return { compiled, count: () => compiles };
}
const sequence = [[6, 8], [7, 8], [32, 8], [7, 8], [6, 8], [1024, 8]];
for (const [label, dyn] of [['static', null], ['dynamic dim 0', [new Set([0])]]]) {
  const { compiled, count } = counted(dyn);
  const atBuild = count();
  const shapes = [];
  for (const s of sequence) shapes.push((await compiled(randn(s))).shape.join('x'));
  console.log(`  ${label.padEnd(14)} ${count()} compilations for ${sequence.length} calls over `
    + `${new Set(sequence.map((s) => s.join('x'))).size} distinct shapes (${atBuild} at construction)`);
  console.log(`  ${' '.repeat(14)} outputs: ${shapes.join(' ')}`);
}

console.log('\n=== the symbolic dimension becomes a kernel argument ===\n');
const { compiled: dyn } = counted([new Set([0])]);
await dyn(randn([6, 8]));
console.log('  ' + dyn.source().split('\n').find((l) => l.includes('function')).trim());
console.log('\n  The trailing integer parameters are the resolved symbols; the runtime reads them');
console.log('  off the shapes of the tensors it was passed (Chapter 59 §59.4).');
