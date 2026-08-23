import {
  lowerToTir, printTensorIR, Schedule, toKernel, randn,
} from '../../_internals.mjs';

const K = 8;
const build = async () => new Schedule(await lowerToTir((x) => x.sum(1), [randn([1, K])]));
const accLoop = (sch) => sch.getLoops('reduce_acc_1')[1].loopVar.name;

const before = await build();
console.log('=== the reduction as lowered ===');
console.log(printTensorIR(before.func).split('\n').slice(9, -1).join('\n'));

const after = await build();
const partial = after.rfactor('reduce_acc_1', accLoop(after), 4);
console.log(`\n=== rfactor(k, 4) — partial buffer ${partial.name}${JSON.stringify(partial.shape)} ===`);
console.log(printTensorIR(after.func).split('\n').slice(9, -1).join('\n'));
console.log(`\n  blocks now: ${after.state.allBlockNames().join(', ')}`);
console.log('  The accumulation block became two: `_rf_p` runs 4 independent');
console.log('  partial sums, `_rf_c` adds the 4 partials. Both carry an `init {}`,');
console.log('  which no lowering rule ever sets (Chapter 33) and rfactor always does.');

const serial = toKernel((await build()).func).call;
const factored = toKernel(after.func).call;

function sum(kernel, values) {
  const out = new Float32Array(1);
  kernel(new Float32Array(values), out);
  return out[0];
}

const rows = [
  ['1..8, exact in every order', [1, 2, 3, 4, 5, 6, 7, 8]],
  ['1e17 and its negation, four apart', [1e17, 1, 1, 1, -1e17, 1, 1, 1]],
];

console.log('\n=== does the answer change? ===\n');
console.log(`  ${'input'.padEnd(34)} ${'serial'.padEnd(10)} rfactor(4)`);
for (const [label, values] of rows) {
  console.log(`  ${label.padEnd(34)} ${String(sum(serial, values)).padEnd(10)} ${sum(factored, values)}`);
}

console.log('\n  The second row is Theorem 41.2 failing. Serial order pairs 1e17');
console.log('  with three 1s, loses all three to rounding, then subtracts 1e17 and');
console.log('  adds the remaining three: 3. rfactor with factor 4 pairs element i');
console.log('  with element i+4, so 1e17 meets -1e17 first and cancels exactly,');
console.log('  and the six surviving 1s sum to 6.');
console.log('  Both answers are correct additions of the same eight f32 values in');
console.log('  different orders. Neither is the real sum, which is 6 exactly and');
console.log('  which the serial order cannot reach.');

console.log('\n=== what rfactor refuses ===\n');
for (const factor of [1, 3, 8, 16]) {
  const sch = await build();
  try {
    sch.rfactor('reduce_acc_1', accLoop(sch), factor);
    console.log(`  factor ${String(factor).padEnd(3)} accepted`);
  } catch (e) {
    console.log(`  factor ${String(factor).padEnd(3)} ${e.message}`);
  }
}
console.log('\n  1 < factor < K and factor | K. The bound at both ends is not a');
console.log('  legality condition — a factor of 1 or K would be a correct if');
console.log('  pointless schedule — it is the guard that keeps the partial buffer');
console.log('  from being degenerate.');

const argmax = new Schedule(await lowerToTir((x) => x.argmax(1), [randn([1, K])]));
console.log(`\n  blocks of x.argmax(1): ${argmax.state.allBlockNames().join(', ')}`);
try {
  argmax.rfactor('arg_acc_1', argmax.getLoops('arg_acc_1')[1].loopVar.name, 4);
  console.log('  rfactor on argmax: accepted');
} catch (e) {
  console.log(`  rfactor on argmax: ${e.message}`);
}
console.log('\n  argmax is associative and commutative as a whole, and its block');
console.log('  body is not a single accumulating store — it is two conditional');
console.log('  stores over two buffers. The structural test rejects it before the');
console.log('  algebraic one is reached, so the operator set is never consulted.');
