import {
  compileWithBackward, compile, CPUTarget, firstFunction,
  randn, zeros, ones, manual_seed, noGrad,
} from '../../_internals.mjs';

const D = 16;
const H = 64;
const LR = 0.05;
const STEPS = 40;

const init = () => {
  manual_seed(21);
  return { W1: randn([D, H]).mul(0.2), b1: zeros([H]), W2: randn([H, 1]).mul(0.2), b2: zeros([1]) };
};
const ORDER = ['W1', 'b1', 'W2', 'b2'];
const loss = (p, x, y) => {
  const d = x.matmul(p.W1).add(p.b1).relu().matmul(p.W2).add(p.b2).sub(y);
  return d.mul(d).mean();
};

// A stride-aware SGD step: read the gradient through its logical layout, not its storage.
const sgd = (t, g) => {
  const w = t._impl.storage.data;
  const grad = g.contiguous().data;
  for (let i = 0; i < w.length; i++) w[i] -= LR * grad[i];
  t._impl.bumpVersion();
};

function eagerRun(x, y) {
  const p = init();
  for (const k of ORDER) p[k].requiresGrad_(true);
  const t0 = performance.now();
  const curve = [];
  for (let s = 0; s < STEPS; s++) {
    const l = loss(p, x, y);
    curve.push(l.item());
    for (const k of ORDER) p[k].grad = null;
    l.backward();
    noGrad(() => { for (const k of ORDER) sgd(p[k], p[k].grad); });
  }
  return { ms: performance.now() - t0, curve, p };
}

async function compiledRun(x, y, mode) {
  const p = init();
  const cf = compileWithBackward({ forward: (xx, yy) => loss(p, xx, yy) }, [x, y], { target: CPUTarget(), mode });
  let warm = cf(x, y);
  if (warm?.then) warm = await warm;
  cf.backward(ones(warm.shape));
  const t0 = performance.now();
  const curve = [];
  for (let s = 0; s < STEPS; s++) {
    let l = cf(x, y);
    if (l?.then) l = await l;
    curve.push(l.item());
    let g = cf.backward(ones(l.shape));
    if (g?.then) g = await g;
    for (let i = 0; i < ORDER.length; i++) sgd(p[ORDER[i]], g[2 + i]);
  }
  return { ms: performance.now() - t0, curve, p, cf };
}

manual_seed(5);
const x = randn([64, D]);
const y = randn([64, 1]);

console.log('=== what compileWithBackward produces ===\n');
for (const mode of ['separate', 'joint']) {
  const p = init();
  const cf = compileWithBackward({ forward: (xx, yy) => loss(p, xx, yy) }, [x, y], { target: CPUTarget(), mode });
  let out = cf(x, y);
  if (out?.then) out = await out;
  let grads = cf.backward(ones(out.shape));
  if (grads?.then) grads = await grads;
  const units = cf.compiledUnits().map((u) => `${u.name}(${u.result.listKernels().join(', ')})`).join('  ');
  const ops = (g) => (g ? [...(typeof g.functions === 'function' ? firstFunction(g) : g).ops()].length : 0);
  console.log(`  mode=${mode.padEnd(9)} units: ${units}`);
  console.log(`  ${' '.repeat(14)}forward graph ${ops(cf.forwardGraph())} ops, backward graph ${ops(cf.backwardGraph())} ops`);
  console.log(`  ${' '.repeat(14)}loss ${out.item().toFixed(6)}   gradients returned: ${grads.map((t) => (t ? t.shape.join('x') || 'scalar' : 'null')).join(', ')}`);
  console.log(`  ${' '.repeat(14)}captured: ${cf.capturedParams().map((t) => t.shape.join('x')).join(', ')}\n`);
}
console.log('  Two gradients for the two user inputs, then one per captured parameter,');
console.log('  in the capture order Chapter 61 fixed.');

console.log('\n=== the compiled gradient agrees with the eager one, and with the slope ===\n');
{
  const pe = init();
  for (const k of ORDER) { pe[k].requiresGrad_(true); pe[k].grad = null; }
  loss(pe, x, y).backward();

  const pc = init();
  const cf = compileWithBackward({ forward: (xx, yy) => loss(pc, xx, yy) }, [x, y], { target: CPUTarget() });
  let out = cf(x, y);
  if (out?.then) out = await out;
  let grads = cf.backward(ones(out.shape));
  if (grads?.then) grads = await grads;

  console.log(`  ${'parameter'.padEnd(10)} ${'shape'.padStart(8)} ${'max |compiled - eager|'.padStart(23)}`);
  let gg = 0;
  for (let i = 0; i < ORDER.length; i++) {
    const a = pe[ORDER[i]].grad.contiguous().data;
    const b = grads[2 + i].contiguous().data;
    let worst = 0;
    for (let j = 0; j < a.length; j++) { worst = Math.max(worst, Math.abs(a[j] - b[j])); gg += b[j] * b[j]; }
    console.log(`  ${ORDER[i].padEnd(10)} ${(pe[ORDER[i]].shape.join('x') || 'scalar').padStart(8)} ${worst.toExponential(2).padStart(23)}`);
  }

  // A directional finite difference along the gradient: L(p + h g) - L(p - h g) ~ 2h |g|^2.
  // One scalar per step size, so the roundoff floor and the truncation error separate cleanly.
  console.log(`\n  the slope along the gradient itself, |g|^2 = ${gg.toFixed(6)}\n`);
  console.log(`  ${'h'.padStart(8)} ${'(L(p+hg) - L(p-hg)) / 2h'.padStart(25)} ${'relative error'.padStart(15)}`);
  const stores = ORDER.map((k) => pc[k]._impl.storage.data);
  const saved = stores.map((s) => Float32Array.from(s));
  const shift = (h) => {
    for (let i = 0; i < ORDER.length; i++) {
      const g = grads[2 + i].contiguous().data;
      for (let j = 0; j < stores[i].length; j++) stores[i][j] = saved[i][j] + h * g[j];
      pc[ORDER[i]]._impl.bumpVersion();
    }
  };
  for (const h of [1e-1, 1e-2, 1e-3, 1e-4]) {
    shift(h); let hi = cf(x, y); if (hi?.then) hi = await hi; const a = hi.item();
    shift(-h); let lo = cf(x, y); if (lo?.then) lo = await lo; const b = lo.item();
    const slope = (a - b) / (2 * h);
    console.log(`  ${h.toExponential(0).padStart(8)} ${slope.toFixed(6).padStart(25)} ${(Math.abs(slope - gg) / gg).toExponential(2).padStart(15)}`);
  }
  shift(0);
  console.log('\n  Truncation error falls with h until roundoff takes over. Chapter 65 is about');
  console.log('  choosing that step size on purpose rather than by inspection.');
}

console.log('\n=== 40 steps of SGD, eager against compiled ===\n');
const eager = eagerRun(x, y);
const separate = await compiledRun(x, y, 'separate');
const joint = await compiledRun(x, y, 'joint');

const worstCurve = (a, b) => Math.max(...a.map((v, i) => Math.abs(v - b[i])));
const worstParams = (a, b) => Math.max(...ORDER.flatMap((k) => {
  const u = a[k].contiguous().data;
  const v = b[k].contiguous().data;
  return [...u].map((q, i) => Math.abs(q - v[i]));
}));

console.log(`  ${'path'.padEnd(20)} ${'time'.padStart(9)} ${'loss first'.padStart(11)} ${'loss last'.padStart(11)} ${'max |Δloss|'.padStart(12)} ${'max |Δparam|'.padStart(13)}`);
console.log(`  ${'eager autograd'.padEnd(20)} ${(eager.ms.toFixed(1) + 'ms').padStart(9)} ${eager.curve[0].toFixed(6).padStart(11)} ${eager.curve.at(-1).toFixed(6).padStart(11)} ${'—'.padStart(12)} ${'—'.padStart(13)}`);
for (const [label, run] of [['compiled, separate', separate], ['compiled, joint', joint]]) {
  console.log(`  ${label.padEnd(20)} ${(run.ms.toFixed(1) + 'ms').padStart(9)} ${run.curve[0].toFixed(6).padStart(11)} ${run.curve.at(-1).toFixed(6).padStart(11)}`
    + ` ${worstCurve(eager.curve, run.curve).toExponential(2).padStart(12)} ${worstParams(eager.p, run.p).toExponential(2).padStart(13)}`);
}
console.log(`\n  speedup: separate ${(eager.ms / separate.ms).toFixed(2)}x, joint ${(eager.ms / joint.ms).toFixed(2)}x`);
console.log('  One f32 ulp is 1.19e-7 relative, so the two paths agree to the last bit that');
console.log('  a float32 accumulation can carry over 40 steps.');

console.log('\n=== where the crossover is ===\n');
console.log(`  ${'batch'.padStart(6)} ${'eager'.padStart(10)} ${'compiled'.padStart(10)} ${'speedup'.padStart(9)} ${'max |Δloss|'.padStart(12)}`);
for (const B of [1, 4, 16, 64, 256, 1024]) {
  manual_seed(5);
  const xb = randn([B, D]);
  const yb = randn([B, 1]);
  const e = eagerRun(xb, yb);
  const c = await compiledRun(xb, yb, 'separate');
  console.log(`  ${String(B).padStart(6)} ${(e.ms.toFixed(1) + 'ms').padStart(10)} ${(c.ms.toFixed(1) + 'ms').padStart(10)}`
    + ` ${(e.ms / c.ms).toFixed(2).padStart(8)}x ${worstCurve(e.curve, c.curve).toExponential(1).padStart(12)}`);
}

console.log('\n=== and what the first call costs ===\n');
console.log(`  ${'path'.padEnd(20)} ${'first call'.padStart(11)} ${'steady step'.padStart(12)}`);
console.log(`  ${'eager autograd'.padEnd(20)} ${'—'.padStart(11)} ${((eager.ms / STEPS).toFixed(3) + 'ms').padStart(12)}`);
for (const mode of ['separate', 'joint']) {
  const p = init();
  const t0 = performance.now();
  const cf = compileWithBackward({ forward: (xx, yy) => loss(p, xx, yy) }, [x, y], { target: CPUTarget(), mode });
  let l = cf(x, y); if (l?.then) l = await l;
  let g = cf.backward(ones(l.shape)); if (g?.then) g = await g;
  const first = performance.now() - t0;
  const t1 = performance.now();
  for (let s = 0; s < STEPS; s++) {
    let ll = cf(x, y); if (ll?.then) ll = await ll;
    let gg = cf.backward(ones(ll.shape)); if (gg?.then) gg = await gg;
  }
  const steady = (performance.now() - t1) / STEPS;
  console.log(`  ${('compiled, ' + mode).padEnd(20)} ${(first.toFixed(1) + 'ms').padStart(11)} ${(steady.toFixed(3) + 'ms').padStart(12)}`);
}
console.log('\n  The first call is a whole compilation. Everything Parts II to X do, once.');
