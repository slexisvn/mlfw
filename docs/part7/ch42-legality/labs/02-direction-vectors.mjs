import {
  lowerToTir, Schedule, toKernel, randn, Direction,
  collectBufferAccesses, dependences,
  Buffer, PrimFunc, ForNode, BlockNode, BlockRealizeNode,
  BufferStoreNode, BufferLoadNode, VariableNode, IntImmNode, MathOpNode, ForKind,
} from '../../_internals.mjs';

const DIR = (m) => (m === Direction.EQ ? '='
  : m === Direction.LT ? '<'
  : m === Direction.GT ? '>'
  : m === (Direction.LT | Direction.EQ) ? '≤'
  : m === (Direction.GT | Direction.EQ) ? '≥'
  : '*');

function report(label, func) {
  const info = collectBufferAccesses(func.body);
  const deps = dependences(info.byBuffer);
  console.log(`=== ${label} ===\n`);
  for (const dep of deps.filter((d) => d.loops.length > 0)) {
    const loops = dep.loops.map((l) => l.name).join(', ');
    const dirs = dep.masks.map(DIR).join(', ');
    console.log(`  ${dep.kind.padEnd(4)} on ${dep.buffer.name.padEnd(6)} over (${loops})  direction (${dirs})`);
  }
  const across = deps.filter((d) => d.loops.length === 0).length;
  if (across > 0) console.log(`  (+ ${across} between accesses in different nests, which share no loop)`);
}

const permutations = (xs) => (xs.length <= 1 ? [xs]
  : xs.flatMap((x, i) => permutations([...xs.slice(0, i), ...xs.slice(i + 1)]).map((p) => [x, ...p])));

async function tryEveryOrder(label, make, blockName) {
  const names = (await make()).getLoops(blockName).map((l) => l.loopVar.name);
  console.log(`\n=== every permutation of (${names.join(', ')}) ===\n`);
  for (const order of permutations(names.map((_, i) => i))) {
    const sch = await make();
    const loops = sch.getLoops(blockName);
    try {
      sch.reorder(...order.map((i) => loops[i]));
      console.log(`  (${order.map((i) => names[i]).join(', ')})  accepted`);
    } catch (e) {
      console.log(`  (${order.map((i) => names[i]).join(', ')})  ${e.message}`);
    }
  }
}

const mm = () => lowerToTir((a, b) => a.matmul(b), [randn([4, 6]), randn([6, 5])])
  .then((pf) => new Schedule(pf));

report('a matmul, C[m,n] += A[m,k] * B[k,n]', (await mm()).func);
console.log('\n  One pair of dependences on the accumulator, both with `=` on the');
console.log('  two spatial axes: iteration (m,n,k) and iteration (m,n,k\') touch');
console.log('  buf_5[m,n] for every k, k\'. The contraction axis is `*` — every');
console.log('  direction is possible — because the two accesses are at the same');
console.log('  subscript and the axis does not appear in it.');
await tryEveryOrder('matmul', mm, 'matmul_1');
console.log('\n  All six accepted. The `=` on every level above the `*` is what');
console.log('  makes that safe, not the `*` itself: a dependence runs earlier to');
console.log('  later, so its first non-`=` component is `<`, and one constraining');
console.log('  only iterations that agree on (m,n) cannot be reordered by moving k.');
console.log('\n  Note which layer answered. `reorder` asks blockAbstractionPermits');
console.log('  first, and these axes are declared DataPar/DataPar/CommReduce, so');
console.log('  the dependence above was never consulted. It would have accepted');
console.log('  all six as well, but it was not asked.');

function stencilFunc(tagged) {
  const A = new Buffer('A', [5, 5], 'f32', 'global');
  const i = new VariableNode('i', 'int32');
  const j = new VariableNode('j', 'int32');
  const read = new BufferLoadNode(A, [i, new MathOpNode('+', j, new IntImmNode(1))]);
  const store = new BufferStoreNode(A,
    [new MathOpNode('+', i, new IntImmNode(1)), j],
    new MathOpNode('+', read, new IntImmNode(1)));
  const iters = tagged
    ? [new BlockRealizeNode(i, i), new BlockRealizeNode(j, j)]
    : [{ iterVar: i, binding: i }, { iterVar: j, binding: j }];
  const block = new BlockNode('st', iters, [{ buffer: A }], [{ buffer: A }], store);
  let nest = block;
  for (const v of [j, i]) nest = new ForNode(v, new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, nest);
  return new PrimFunc('stencil', [], nest, new Map([['A', A]]));
}

console.log('\n');
report('a stencil, A[i+1,j] = A[i,j+1] + 1', stencilFunc(false));
console.log('\n  Direction (<, >): iteration (i, j) writes A[i+1, j], and iteration');
console.log('  (i+1, j-1) reads it — later in i, earlier in j.');
console.log('  Lexicographically positive under (i, j) — the leading entry is `<` —');
console.log('  and lexicographically negative under (j, i), where the leading entry');
console.log('  would be `>`. That is Theorem 42.4 exactly.');

await tryEveryOrder('stencil, axes untyped', async () => new Schedule(stencilFunc(false)), 'st');
console.log('\n  Refused, with the kind and the buffer named. This is the only shape');
console.log('  in Part VII that the dependence analysis turns down.');

await tryEveryOrder('the same stencil, axes declared DataPar', async () => new Schedule(stencilFunc(true)), 'st');
console.log('\n  Accepted. Nothing about the program changed — only the block header.');
console.log('  `reorderLegality` (legality.ts:48) asks `blockAbstractionPermits`');
console.log('  first, and a block whose axes are all DataPar permits any');
console.log('  permutation, so the dependence above is never consulted. The');
console.log('  declaration is false: the two iterations are not independent.');

const runStencil = (func) => {
  const A = new Float32Array(25).fill(0);
  toKernel(func).call(A);
  return [...A].slice(5, 21).map((v) => String(v).padStart(2)).join(' ');
};
const good = new Schedule(stencilFunc(true));
const bad = new Schedule(stencilFunc(true));
const l = bad.getLoops('st');
bad.reorder(l[1], l[0]);
console.log('\n=== what the accepted permutation costs ===\n');
console.log(`  (i, j)  ${runStencil(good.func)}`);
console.log(`  (j, i)  ${runStencil(bad.func)}`);
console.log('\n  Different answers from the same program. Corollary 33.7 is not a');
console.log('  hypothetical: a block that declares an axis DataPar when it is not');
console.log('  gets a transformation the analysis had already refused, and nothing');
console.log('  downstream notices. What protects the compiler is that the five');
console.log('  call sites of `markCommReduce` are the five rules that accumulate,');
console.log('  and every other rule emits a genuinely elementwise store.');
