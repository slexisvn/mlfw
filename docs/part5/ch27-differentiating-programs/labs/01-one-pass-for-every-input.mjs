import {
  tensor, Linear, ReLU, Sequential, compileWithBackward, compile, CPUTarget, ones, manual_seed,
} from '../../../../dist/index.node.js';
import { summarize } from '../../../tools/measure.mjs';

function model() {
  manual_seed(0);
  return new Sequential(new Linear(8, 16), new ReLU(), new Linear(16, 1));
}

function loss(m) {
  return (x) => m.forward(x).sum();
}

async function settle(v) {
  return v && v.then ? await v : v;
}

async function reverseMode(m, x) {
  const cf = compileWithBackward({ forward: loss(m) }, [x], { target: CPUTarget() });
  const out = await settle(cf(x));
  const grads = await settle(cf.backward(ones(out.shape)));
  return { grad: grads[0], run: async () => { await settle(cf(x)); await settle(cf.backward(ones(out.shape))); } };
}

async function finiteDifferences(m, x, eps) {
  const forward = compile({ forward: loss(m) }, [x], { target: CPUTarget() });
  await forward._ready;
  const base = x.toArray();
  const n = base[0].length;
  const at = async (k, delta) => {
    const row = base[0].slice();
    row[k] += delta;
    const y = await settle(forward(tensor([row])));
    return y.toArray();
  };
  const grad = [];
  for (let k = 0; k < n; k++) {
    grad.push(((await at(k, eps)) - (await at(k, -eps))) / (2 * eps));
  }
  return { grad, evaluations: 2 * n };
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

const m = model();
const x = tensor([Array.from({ length: 8 }, (_, i) => ((i % 5) / 5) - 0.4)]);

const rev = await reverseMode(m, x);
const fd = await finiteDifferences(m, x, 1e-3);

const analytic = rev.grad.toArray()[0];

console.log('=== the same eight partial derivatives, two ways ===');
console.log('  k   reverse mode        central differences   rel. error');
for (let k = 0; k < analytic.length; k++) {
  const rel = Math.abs(fd.grad[k] - analytic[k]) / (1 + Math.abs(fd.grad[k]));
  console.log(
    `  ${k}   ${analytic[k].toFixed(9).padStart(14)}      ${fd.grad[k].toFixed(9).padStart(14)}   ${rel.toExponential(1)}`
  );
}

console.log('\n=== what each one cost ===');
console.log(`  reverse mode:         1 forward + 1 backward`);
console.log(`  central differences:  ${fd.evaluations} forward evaluations`);

console.log('');
console.log('=== how the cost scales with the number of inputs (medians of 15 rounds) ===');
console.log('  inputs   1 forward   reverse   differences   ratio   rev/fwd');
for (const n of [8, 32, 128]) {
  manual_seed(0);
  const mn = new Sequential(new Linear(n, 16), new ReLU(), new Linear(16, 1));
  const xn = tensor([Array.from({ length: n }, (_, i) => ((i % 5) / 5) - 0.4)]);

  const r = await reverseMode(mn, xn);
  await r.run();
  const sRev = await sample(r.run, 15);

  const fwd = compile({ forward: loss(mn) }, [xn], { target: CPUTarget() });
  await fwd._ready;
  await settle(fwd(xn));
  const sFwd = await sample(async () => { await settle(fwd(xn)); }, 15);

  const base = xn.toArray()[0];
  const eps = 1e-3;
  const runFd = async () => {
    for (let k = 0; k < n; k++) {
      for (const sign of [1, -1]) {
        const shifted = base.slice();
        shifted[k] += sign * eps;
        await settle(fwd(tensor([shifted])));
      }
    }
  };
  await runFd();
  const sFd = await sample(runFd, 5);

  console.log(
    `  ${String(n).padStart(6)}   ${sFwd.median.toFixed(3).padStart(9)}   ${sRev.median.toFixed(3).padStart(7)}   ` +
    `${sFd.median.toFixed(3).padStart(11)}   ${(sFd.median / sRev.median).toFixed(1).padStart(5)}x   ` +
    `${(sRev.median / sFwd.median).toFixed(1).padStart(6)}x`
  );
}
