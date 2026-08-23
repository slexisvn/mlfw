import {
  lowerToTir, printTensorIR, Schedule, toKernel, randn, compile, CUDATarget,
} from '../../_internals.mjs';

const N = 12;
const build = async () => new Schedule(await lowerToTir((x) => x.mul(2.0), [randn([N])]));
const tail = (sch) => printTensorIR(sch.func).split('\n').slice(4, -1).join('\n');

const input = new Float32Array(N).map((_, i) => i + 1);
const runOn = (sch) => {
  const out = new Float32Array(N + 4).fill(-1);
  toKernel(sch.func).call(input, out);
  return out;
};

const base = await build();
console.log('=== unscheduled ===');
console.log(tail(base));
console.log('  output:', [...runOn(base)].join(' '));

const guarded = await build();
guarded.split(guarded.getLoops('mul_block_0')[0], 5);
console.log('\n=== split(i, 5): ceil(12/5) = 3 outer, 5 inner, 15 >= 12 ===');
console.log(tail(guarded));
console.log('  output:', [...runOn(guarded)].join(' '));

function dropGuards(node) {
  let removed = 0;
  const walk = (n) => {
    if (!n || typeof n !== 'object') return;
    if (n.type === 'IfThenElseNode' && !n.elseBody) {
      const inner = n.thenBody;
      n.replaceWith(inner);
      removed++;
      walk(inner);
      return;
    }
    for (const k of ['body', 'initBody', 'thenBody', 'elseBody']) if (n[k]) walk(n[k]);
    if (n.stmts) for (const s of n.stmts) walk(s);
  };
  walk(node);
  return removed;
}

const unguarded = await build();
unguarded.split(unguarded.getLoops('mul_block_0')[0], 5);
console.log(`\n=== the same schedule with ${dropGuards(unguarded.func.body)} guard removed by hand ===`);
console.log(tail(unguarded));
console.log('  output:', [...runOn(unguarded)].join(' '));
console.log('\n  Columns 12 to 14 are past the end of a 12-element buffer. They');
console.log('  started at -1 and they are not -1 any more: iterations 12, 13 and 14');
console.log('  of the padded space read past the input (giving NaN) and wrote past');
console.log('  the output. Against a typed array the read is undefined and the');
console.log('  write is discarded; against a pooled arena (Chapter 50) both land in');
console.log('  the next buffer.');

const roundTrip = await build();
const [o, i] = roundTrip.split(roundTrip.getLoops('mul_block_0')[0], 5);
roundTrip.fuseLoops(o, i);
console.log('\n=== fuseLoops(split(i, 5)) — not the identity ===');
console.log(tail(roundTrip));
console.log('  output:', [...runOn(roundTrip)].join(' '));
console.log('\n  Three things did not come back:');
console.log('    - the loop runs 15 times, not 12;');
console.log('    - the guard is still there, now testing a divide-and-remultiply;');
console.log('    - the binding is `(f // 5) * 5 + f % 5`, which is `f` for every');
console.log('      non-negative `f`, and which no rewrite in the simplifier removes.');
console.log('  That last expression is Chapter 35\'s finding 8 arriving from the');
console.log('  other direction: there a reshape produced it, here the scheduler did.');
console.log('  trace:', JSON.stringify(roundTrip.getTrace().serialize()));

const y = randn([12, 5]);
const cuda = compile({ forward: (a) => a.mul(2.0) }, [y], {
  target: CUDATarget(), fusion: { enabled: false }, scheduling: { enabled: true },
});
try {
  await cuda(y);
} catch (e) {
}
console.log('\n=== the same expression, out of the shipping compiler ===');
console.log(cuda.source().split('\n').slice(1).join('\n').trimEnd());
console.log('\n  `ElementwiseGPURule` fused the two loops and bound the result to');
console.log('  threadIdx.x, so `v0_7 = f / 5` and `v1_8 = f % 5`; the subscript is');
console.log('  then `v0_7 * 5 + v1_8`. That is `f`, computed with a division, a');
console.log('  modulo, a multiply and an add, per element, having survived');
console.log('  SimplifyPass and the whole TIR pipeline. The analyzer did prove the');
console.log('  index non-negative — the emitted operators are C `/` and `%`, the');
console.log('  truncating pair, which it substitutes only under that proof.');
