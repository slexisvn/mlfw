import {
  lowerToTir, printTensorIR, Schedule, randn,
} from '../../_internals.mjs';

const build = async () => new Schedule(await lowerToTir((a, b) => a.matmul(b), [randn([12, 8]), randn([8, 6])]));

function nest(sch) {
  const lines = printTensorIR(sch.func).split('\n');
  let start = 0;
  for (let i = 0; i < lines.length; i++) if (/^ {2}for /.test(lines[i])) start = i;
  return lines.slice(start, -1).join('\n');
}

console.log('=== the nest as lowered ===');
console.log(nest(await build()));

{
  const sch = await build();
  const [m] = sch.getLoops('matmul_1');
  sch.split(m, 4);
  console.log('\n=== split(m, 4) — 12 = 3 x 4, so no guard ===');
  console.log(nest(sch));
}

{
  const sch = await build();
  const [m] = sch.getLoops('matmul_1');
  sch.split(m, 5);
  console.log('\n=== split(m, 5) — 12 = 2 x 5 + 2, so a guard ===');
  console.log(nest(sch));
}

{
  const sch = await build();
  const [m, n] = sch.getLoops('matmul_1');
  sch.fuseLoops(m, n);
  console.log('\n=== fuseLoops(m, n) — one loop of 72, and two divisions ===');
  console.log(nest(sch));
}

{
  const sch = await build();
  const [m, n, k] = sch.getLoops('matmul_1');
  sch.reorder(k, m, n);
  console.log('\n=== reorder(k, m, n) — the reduction axis moved outermost ===');
  console.log(nest(sch));
}

{
  const sch = await build();
  sch.tile('matmul_1', [0, 1], [4, 3]);
  console.log('\n=== tile("matmul_1", [0, 1], [4, 3]) ===');
  console.log(nest(sch));
  console.log('\n  trace:', JSON.stringify(sch.getTrace().serialize()));
  console.log('  `tile` records nothing of its own: it is two splits and a reorder,');
  console.log('  and each of those records itself. Replaying the trace replays the');
  console.log('  three steps, not the composite.');
}
