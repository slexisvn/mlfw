import {
  tensor, Linear, ReLU, Sequential, compile, CPUTarget, TraceLevel, manual_seed,
} from '../../../../dist/index.node.js';
import { summarize, ratio } from '../../../tools/measure.mjs';

const x = tensor([[0.5, -1.5], [1.0, 2.0]]);
const LEVELS = ['off', 'boundaries', 'each-pass'];

function build(depth, out = 32) {
  manual_seed(0);
  const layers = [new Linear(2, 32)];
  for (let i = 0; i < depth; i++) layers.push(new ReLU(), new Linear(32, i === depth - 1 ? out : 32));
  return new Sequential(...layers);
}

async function once(model, level) {
  const t0 = performance.now();
  const compiled = compile(model, [x], { target: CPUTarget(), verify: level });
  await compiled._ready;
  return performance.now() - t0;
}

function corrupt() {
  const done = new WeakSet();
  return {
    shouldRun(pass) {
      if (pass.name !== 'cse' || done.has(pass)) return true;
      done.add(pass);
      const inner = pass.run.bind(pass);
      pass.run = (func, analyses) => {
        inner(func, analyses);
        const ops = [...func.ops()];
        const victim = ops.find(o => o.opName === 'add');
        const dots = ops.filter(o => o.opName === 'dot');
        const donor = dots[dots.length - 1];
        if (victim && donor && !donor.getResult(0).type.equals(victim.getResult(0).type)) {
          victim.getResult(0).type = donor.getResult(0).type;
        }
        return 1;
      };
      return true;
    },
  };
}

console.log('\n=== one pass writes a result type that does not follow ===');
for (const level of ['each-pass', 'boundaries', 'off']) {
  const reported = [];
  console.log(`  verify: ${level}`);
  try {
    const compiled = compile(build(2, 1), [x], {
      target: CPUTarget(),
      verify: level,
      passContext: corrupt(),
      trace: { level: TraceLevel.INFO, sink: (e) => { if (e.type === 'error') reported.push(e); } },
    });
    await compiled._ready;
    console.log('    compiled, no complaint');
  } catch (e) {
    console.log(`    threw:  ${e.message.split(';')[0]}`);
  }
  for (const e of reported) console.log(`    blamed: ${e.passName ?? '(nobody)'} -- ${e.message.split(';')[0]}`);
}

const model = build(24);
for (let i = 0; i < 40; i++) await once(model, 'each-pass');

const samples = Object.fromEntries(LEVELS.map(l => [l, []]));
for (let round = 0; round < 40; round++) {
  for (const level of LEVELS) samples[level].push(await once(model, level));
}

console.log('=== compile time by verification level (49 graph ops, 40 interleaved rounds) ===');
console.log('  level        median     ratio    IQR                min      max');
const stats = Object.fromEntries(LEVELS.map(l => [l, summarize(samples[l])]));
const baseline = stats.off;
for (const level of LEVELS) {
  const s = stats[level];
  const r = ratio(baseline, s);
  console.log(
    `  ${level.padEnd(11)} ${s.median.toFixed(2).padStart(6)} ms` +
    `  ${(s.median / baseline.median).toFixed(2)}x` +
    `   ${s.q1.toFixed(2)}-${s.q3.toFixed(2)}` +
    `   ${s.min.toFixed(2).padStart(7)}  ${s.max.toFixed(2).padStart(7)}` +
    (level !== 'off' && r.overlapping ? '   [inside the noise]' : '')
  );
}
console.log('');
console.log('for comparison, minimum of 40 rounds (the best case, not a central estimate):');
for (const level of LEVELS) {
  console.log(`  ${level.padEnd(11)} ${stats[level].min.toFixed(2).padStart(6)} ms   ${(stats[level].min / baseline.min).toFixed(2)}x`);
}
