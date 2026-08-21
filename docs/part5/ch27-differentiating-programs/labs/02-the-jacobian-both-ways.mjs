import {
  tensor, Linear, Sequential, compileWithBackward, compile, CPUTarget, manual_seed,
} from '../../../../dist/index.node.js';

// f : R^4 -> R^3. Its Jacobian is a 3x4 matrix.
// Reverse mode produces one ROW per backward pass (a cotangent times J).
// Finite differences produce one COLUMN per pair of forward evaluations.
// Both fill the same matrix; which is cheaper depends only on 3 vs 4.

const N_IN = 4;
const N_OUT = 3;

manual_seed(0);
const f = new Sequential(new Linear(N_IN, 6), new Linear(6, N_OUT));
const fwd = (x) => f.forward(x).tanh();

const x = tensor([[0.3, -0.7, 1.1, 0.2]]);

async function settle(v) { return v && v.then ? await v : v; }

function oneHot(i, n) {
  return tensor([Array.from({ length: n }, (_, k) => (k === i ? 1 : 0))]);
}

// --- rows, by reverse mode -------------------------------------------------
const cf = compileWithBackward({ forward: fwd }, [x], { target: CPUTarget() });
await settle(cf(x));

const rows = [];
for (let i = 0; i < N_OUT; i++) {
  const grads = await settle(cf.backward(oneHot(i, N_OUT)));
  rows.push(grads[0].toArray()[0]);
}

// --- columns, by central differences ---------------------------------------
const plain = compile({ forward: fwd }, [x], { target: CPUTarget() });
await plain._ready;
const EPS = 1e-3;
const base = x.toArray()[0];

const columns = [];
for (let j = 0; j < N_IN; j++) {
  const at = async (delta) => {
    const row = base.slice();
    row[j] += delta;
    return (await settle(plain(tensor([row])))).toArray()[0];
  };
  const hi = await at(EPS);
  const lo = await at(-EPS);
  columns.push(hi.map((v, i) => (v - lo[i]) / (2 * EPS)));
}

const show = (m) => m.map(r => '  [' + r.map(v => v.toFixed(6).padStart(10)).join(' ') + ' ]').join('\n');

console.log('=== J by reverse mode: one row per backward pass ===');
console.log(show(rows));
console.log(`  ${N_OUT} backward pass(es)`);

console.log('\n=== J by central differences: one column per input ===');
const asRows = Array.from({ length: N_OUT }, (_, i) => columns.map(c => c[i]));
console.log(show(asRows));
console.log(`  ${2 * N_IN} forward evaluation(s)`);

let maxRel = 0;
for (let i = 0; i < N_OUT; i++) {
  for (let j = 0; j < N_IN; j++) {
    const rel = Math.abs(rows[i][j] - asRows[i][j]) / (1 + Math.abs(asRows[i][j]));
    if (rel > maxRel) maxRel = rel;
  }
}
console.log(`\nlargest relative disagreement: ${maxRel.toExponential(1)}`);

console.log('\n=== the rule that falls out ===');
console.log(`  outputs = ${N_OUT}, inputs = ${N_IN}`);
console.log(`  reverse mode costs ~${N_OUT} sweep(s); a per-input mode costs ~${N_IN}`);
console.log('  training has outputs = 1 and inputs = every parameter, so reverse wins by that ratio');
