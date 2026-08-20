import {
  tensor, Module, compile, CPUTarget, TraceLevel, manual_seed,
} from '../../../../dist/index.node.js';

manual_seed(0);

const small = tensor([[1, 2], [3, 4]]);
const DEPTHS = [32, 64, 128, 256];

function chainOf(depth) {
  return class Chain extends Module {
    forward(a) {
      let v = a;
      for (let i = 0; i < depth; i++) v = v.add(a).mul(a);
      return v;
    }
  };
}

const models = new Map(DEPTHS.map(d => [d, new (chainOf(d))()]));

async function fusionTime(depth) {
  let ms = 0;
  let ops = 0;
  const compiled = compile(models.get(depth), [small], {
    target: CPUTarget(),
    verify: 'off',
    trace: {
      level: TraceLevel.VERBOSE,
      sink: (e) => {
        if (e.type !== 'pass' || e.passName !== 'PriorityFusionPass') return;
        ms = e.durationMs;
        ops = e.opCountBefore;
      },
    },
  });
  await compiled._ready;
  return { ops, ms };
}

for (let i = 0; i < 8; i++) for (const d of DEPTHS) await fusionTime(d);

const samples = new Map(DEPTHS.map(d => [d, []]));
for (let round = 0; round < 9; round++) {
  for (const d of DEPTHS) samples.get(d).push(await fusionTime(d));
}

const points = DEPTHS.map(d => {
  const runs = samples.get(d);
  return { ops: runs[0].ops, ms: Math.min(...runs.map(r => r.ms)) };
});

console.log('graph ops   PriorityFusionPass (ms)   exponent over previous point');
for (let i = 0; i < points.length; i++) {
  const p = points[i];
  const e = i === 0 ? '-'
    : (Math.log(p.ms / points[i - 1].ms) / Math.log(p.ops / points[i - 1].ops)).toFixed(2);
  console.log(`${String(p.ops).padStart(9)}   ${p.ms.toFixed(3).padStart(10)}   ${String(e).padStart(28)}`);
}

const first = points[0], last = points[points.length - 1];
const exponent = Math.log(last.ms / first.ms) / Math.log(last.ops / first.ops);
console.log(`\nover the whole ${(last.ops / first.ops).toFixed(0)}x span: exponent ${exponent.toFixed(2)}`);
console.log('1.0 would be linear, 2.0 quadratic');
