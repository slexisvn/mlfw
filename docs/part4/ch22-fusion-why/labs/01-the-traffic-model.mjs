import {
  tensor, Module, compile, CPUTarget, TraceLevel, manual_seed,
} from '../../../../dist/index.node.js';
import { summarize, ratio } from '../../../tools/measure.mjs';

manual_seed(0);

class Chain extends Module {
  forward(x, y) { return x.add(y).mul(x).sub(y).add(x); }
}

function inputs(n) {
  return [
    tensor(Array.from({ length: n }, (_, i) => (i % 97) / 97)),
    tensor(Array.from({ length: n }, (_, i) => (i % 89) / 89)),
  ];
}

async function build(args, fusionEnabled) {
  const explains = [];
  let ir = null;
  const compiled = compile(new Chain(), args, {
    target: CPUTarget(),
    fusion: { enabled: fusionEnabled },
    trace: {
      level: TraceLevel.DEBUG,
      irSnapshot: { afterGraphPasses: true },
      sink: (e) => {
        if (e.type === 'explain' && e.category === 'fusion') explains.push(e);
        if (e.type === 'ir_snapshot') ir = e.text;
      },
    },
  });
  await compiled._ready;
  return { compiled, explains, ir };
}

async function sample(fn, reps) {
  const times = [];
  for (let i = 0; i < reps; i++) {
    const t0 = performance.now();
    await fn();
    times.push(performance.now() - t0);
  }
  return summarize(times);
}

const big = inputs(1 << 20);
const on = await build(big, true);
console.log('=== the graph, fused ===');
console.log(on.ir);
console.log('the cost model: ' + on.explains.map(e => `${e.subject} -> ${e.decision}`).join(', '));

console.log('\n=== traffic per call, counted from the graph ===');
console.log('  unfused: 4 kernels x (2 reads + 1 write) = 12 tensor round-trips');
console.log('  fused:   1 loop nest x (2 reads + 1 write) =  3 tensor round-trips');
console.log('  the model removes 9 round-trips, independent of tensor size\n');

console.log('elements   tensor    unfused   fused    speedup   traffic saved   IQR overlap');
for (const shift of [10, 14, 16, 18, 20, 22]) {
  const n = 1 << shift;
  const args = inputs(n);
  const off = await build(args, false);
  const fused = await build(args, true);
  await off.compiled(...args); await fused.compiled(...args);
  const sOff = await sample(() => off.compiled(...args), 25);
  const sOn = await sample(() => fused.compiled(...args), 25);
  const savedMiB = (9 * n * 4) / 1024 / 1024;
  const r = ratio(sOff, sOn);
  console.log(
    `${String(n).padStart(8)}  ${(n * 4 / 1024).toFixed(0).padStart(6)} KiB  ` +
    `${sOff.median.toFixed(3).padStart(7)}  ${sOn.median.toFixed(3).padStart(7)}  ` +
    `${r.value.toFixed(2).padStart(7)}x  ${savedMiB.toFixed(2).padStart(9)} MiB  ` +
    `${(r.overlapping ? 'YES - noise' : 'no').padStart(12)}`
  );
}
