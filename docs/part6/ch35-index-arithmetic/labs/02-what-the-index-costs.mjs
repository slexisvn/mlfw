import {
  compile, CPUTarget, randn,
} from '../../../../dist/index.node.js';

async function build(inShape, outShape) {
  const x = randn(inShape);
  const compiled = compile({ forward: (t) => t.reshape(outShape) }, [x], {
    target: CPUTarget(),
    fusion: { enabled: false },
  });
  await compiled(x);
  const line = compiled.source().split('\n').find((l) => l.includes('] = ')).trim();
  const rhs = line.slice(line.indexOf('] = ') + 4);
  return { x, compiled, rhs, elems: inShape.reduce((p, q) => p * q, 1) };
}

async function timeIt(compiled, x) {
  for (let i = 0; i < 40; i++) await compiled(x);
  const runs = [];
  for (let r = 0; r < 9; r++) {
    const t0 = performance.now();
    for (let i = 0; i < 20; i++) await compiled(x);
    runs.push((performance.now() - t0) / 20);
  }
  runs.sort((p, q) => p - q);
  return runs[4];
}

const a = await build([8192, 12], [8192, 4, 3]);
const b = await build([32768, 3], [16384, 6]);

console.log('=== same element count, two index expressions ===\n');
console.log(`  A  [8192,12] -> [8192,4,3]   ${a.elems} elements`);
console.log(`     ${a.rhs}`);
console.log(`  B  [32768,3] -> [16384,6]    ${b.elems} elements`);
console.log(`     ${b.rhs}`);

const ops = (s) => ({
  div: (s.match(/\//g) || []).length,
  mod: (s.match(/%/g) || []).length,
  mul: (s.match(/\*/g) || []).length,
  add: (s.match(/\+/g) || []).length,
});

console.log('\n  arithmetic per element');
console.log(`  ${'case'.padEnd(6)} ${'div'.padStart(4)} ${'mod'.padStart(4)} ${'mul'.padStart(4)} ${'add'.padStart(4)}`);
for (const [name, r] of [['A', a], ['B', b]]) {
  const o = ops(r.rhs);
  console.log(`  ${name.padEnd(6)} ${String(o.div).padStart(4)} ${String(o.mod).padStart(4)} ${String(o.mul).padStart(4)} ${String(o.add).padStart(4)}`);
}

const ta = await timeIt(a.compiled, a.x);
const tb = await timeIt(b.compiled, b.x);

console.log('\n  median of 9 runs of 20 calls (machine-specific):');
console.log(`    A  ${ta.toFixed(3)} ms`);
console.log(`    B  ${tb.toFixed(3)} ms      ${(tb / ta).toFixed(2)}x`);

console.log('\n  Both kernels copy the same bytes. The difference is index arithmetic');
console.log('  the simplifier could not remove.');

console.log('\n\n=== the identity that is left on the table ===\n');
console.log(`  B computes  (f / 3 | 0) * 3 + f % 3  where f = ${'i1 + i0 * 6'}`);
console.log('  Truncating division was chosen because the analyser proved f >= 0.');
console.log('  For a non-negative f, (f tdiv c) * c + (f tmod c) is exactly f, so the');
console.log('  whole expression is the flat index it started from. The proof that');
console.log('  licenses tdiv is the same proof that would license removing it.');
