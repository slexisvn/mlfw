import {
  tensor, LSTM, compileWithBackward, CPUTarget, TraceLevel, ones, manual_seed,
} from '../../../../dist/index.node.js';

// An LSTM traces to a `scan`: one operation holding a loop body, whose size
// does not depend on the sequence length. Its backward does not.

async function settle(v) { return v && v.then ? await v : v; }

const countOps = (ir) => (ir.match(/^\s+%\d+ = /gm) || []).length;
const hasScan = (ir) => /= scan\(/.test(ir);

function sequence(T) {
  return tensor(Array.from({ length: T * 2 }, (_, i) => ((i % 7) / 7) - 0.5)).reshape([1, T, 2]);
}

console.log('=== forward and backward graph size against sequence length ===');
console.log('  T    forward ops  scan?   backward ops  scan?   bwd/T');

let prev = null;
for (const T of [2, 4, 8, 16]) {
  manual_seed(0);
  const m = new LSTM(2, 3, 1, true);
  const x = sequence(T);
  const snaps = [];
  const cf = compileWithBackward({ forward: (a) => m.forward(a)[0] }, [x], {
    target: CPUTarget(),
    trace: {
      level: TraceLevel.DEBUG,
      irSnapshot: { afterGraphPasses: true },
      sink: (e) => { if (e.type === 'ir_snapshot') snaps.push(e.text); },
    },
  });
  const out = await settle(cf(x));
  await settle(cf.backward(ones(out.shape)));

  const f = countOps(snaps[0]);
  const b = countOps(snaps[1]);
  console.log(
    `  ${String(T).padStart(2)}   ${String(f).padStart(11)}  ${String(hasScan(snaps[0])).padEnd(6)}  ` +
    `${String(b).padStart(12)}  ${String(hasScan(snaps[1])).padEnd(6)}  ${(b / T).toFixed(1)}`
  );
  if (prev) prev.push(b);
  prev = [b];
}

console.log('\n=== the gradient is still right ===');
manual_seed(0);
const m = new LSTM(2, 3, 1, true);
const T = 3;
const x = sequence(T);
const cf = compileWithBackward({ forward: (a) => m.forward(a)[0].sum() }, [x], { target: CPUTarget() });
const out = await settle(cf(x));
const grads = await settle(cf.backward(ones(out.shape)));
const analytic = grads[0].toArray().flat(3);

const EPS = 2e-3;
const flatIn = x.toArray().flat(3);
const at = async (k, d) => {
  const arr = flatIn.slice();
  arr[k] += d;
  const xk = tensor(arr).reshape([1, T, 2]);
  const plain = compileWithBackward({ forward: (a) => m.forward(a)[0].sum() }, [xk], { target: CPUTarget() });
  return (await settle(plain(xk))).toArray();
};

let worst = 0;
for (let k = 0; k < flatIn.length; k++) {
  const numeric = ((await at(k, EPS)) - (await at(k, -EPS))) / (2 * EPS);
  const rel = Math.abs(numeric - analytic[k]) / (1 + Math.abs(numeric));
  if (rel > worst) worst = rel;
}
console.log(`  ${flatIn.length} partials checked against central differences`);
console.log(`  largest relative error: ${worst.toExponential(1)}`);
