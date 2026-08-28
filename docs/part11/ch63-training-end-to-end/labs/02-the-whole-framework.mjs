import {
  tensor, manual_seed, unseed, sub, mul, sum,
  Linear, ReLU, MSELoss, SGD, Adam, TensorDataset, DataLoader,
  LightningModule, Trainer,
} from '../../_internals.mjs';

const D = 16;
const N = 256;

class Net extends LightningModule {
  constructor() {
    super();
    this.l1 = new Linear(D, 64);
    this.r = new ReLU();
    this.l2 = new Linear(64, 1);
    this.lf = new MSELoss();
  }
  forward(x) { return this.l2.forward(this.r.forward(this.l1.forward(x))); }
  trainingStep(batch) { const [x, y] = batch; return this.lf.forward(this.forward(x), y); }
  configureOptimizers() { return new Adam(this.parameters(), { lr: 0.02 }); }
}

const xs = [];
const ys = [];
for (let i = 0; i < N; i++) {
  let s = 0;
  for (let d = 0; d < D; d++) { const v = ((i * (d + 3)) % 17) / 17; xs.push(v); s += v * (d % 3 === 0 ? 1 : -0.5); }
  ys.push(s);
}
const X = tensor(xs, { shape: [N, D] });
const Y = tensor(ys, { shape: [N, 1] });
const loader = () => new DataLoader(new TensorDataset(tensor(xs, { shape: [N, D] }), tensor(ys, { shape: [N, 1] })), { batchSize: 32 });
const mse = (m) => { const d = sub(m.forward(X), Y); return sum(mul(d, d)).item() / N; };
const quiet = { logger: false, enableCheckpointing: false, enableProgress: false };

console.log('=== compile: true is one option on the trainer ===\n');
console.log(`  ${'run'.padEnd(16)} ${'time'.padStart(9)} ${'mse before'.padStart(12)} ${'mse after'.padStart(12)}`);
const finalMse = {};
for (const compileFlag of [false, true]) {
  manual_seed(2);
  const net = new Net();
  net.parameters();
  const before = mse(net);
  const t0 = performance.now();
  await new Trainer({ maxEpochs: 20, compile: compileFlag, ...quiet }).fit(net, loader());
  const ms = performance.now() - t0;
  finalMse[compileFlag] = mse(net);
  console.log(`  ${('compile=' + compileFlag).padEnd(16)} ${(ms.toFixed(0) + 'ms').padStart(9)} ${before.toFixed(6).padStart(12)} ${finalMse[compileFlag].toExponential(3).padStart(12)}`);
  unseed();
}
console.log(`\n  Both converge. They do not converge to the same place: ${(finalMse[false] / finalMse[true]).toFixed(0)}x apart after`);
console.log('  20 epochs from an identical initialisation. Lab 01 showed the compiled and');
console.log('  eager gradients agreeing to 2.4e-7 — two f32 ulps — so the divergence is not in the gradient.');

console.log('\n=== one step, from the same initialisation, to find where they part ===\n');
class Tiny extends LightningModule {
  constructor() { super(); this.l1 = new Linear(4, 8); this.r = new ReLU(); this.l2 = new Linear(8, 1); this.lf = new MSELoss(); }
  forward(x) { return this.l2.forward(this.r.forward(this.l1.forward(x))); }
  trainingStep(batch) { const [x, y] = batch; return this.lf.forward(this.forward(x), y); }
  configureOptimizers() { return new SGD(this.parameters(), { lr: 0.1 }); }
}
const M = 8;
const txs = [];
const tys = [];
for (let i = 0; i < M; i++) { let s = 0; for (let d = 0; d < 4; d++) { const v = ((i * (d + 3)) % 17) / 17; txs.push(v); s += v; } tys.push(s); }
const tinyLoader = () => new DataLoader(new TensorDataset(tensor(txs, { shape: [M, 4] }), tensor(tys, { shape: [M, 1] })), { batchSize: M });

const applied = {};
for (const compileFlag of [false, true]) {
  manual_seed(2);
  const net = new Tiny();
  const params = [...net.parameters()];
  const before = params.map((p) => Float32Array.from(p._impl.storage.data));
  await new Trainer({ maxEpochs: 1, compile: compileFlag, ...quiet }).fit(net, tinyLoader());
  applied[compileFlag] = params.map((p, i) => [...p._impl.storage.data].map((v, j) => (before[i][j] - v) / 0.1));
  unseed();
}
console.log('  the update each path applied to the 8x4 first-layer weight, divided by the');
console.log('  learning rate — that is, the gradient the optimizer believed it had:\n');
for (const flag of [false, true]) {
  console.log(`  compile=${String(flag).padEnd(6)} ${applied[flag][0].slice(0, 8).map((v) => v.toFixed(5).padStart(9)).join(' ')}`);
}
const same = applied[false][0].every((v, i) => Math.abs(v - applied[true][0][i]) < 1e-4);
console.log(`\n  identical: ${same}`);

console.log('\n=== the gradients themselves, and their strides ===\n');
for (const compileFlag of [false, true]) {
  manual_seed(2);
  const net = new Tiny();
  let captured = null;
  const orig = net.configureOptimizers.bind(net);
  net.configureOptimizers = () => {
    const opt = orig();
    const step = opt.step.bind(opt);
    opt.step = () => {
      if (!captured) {
        const p = [...net.parameters()][0];
        captured = {
          shape: p.grad.shape.join('x'),
          strides: p.grad.strides.join(','),
          contiguous: p.grad.isContiguous,
          logical: [...p.grad.contiguous().data].slice(0, 8),
          storage: [...p.grad._impl.storage.data].slice(0, 8),
        };
      }
      return step();
    };
    return opt;
  };
  await new Trainer({ maxEpochs: 1, compile: compileFlag, ...quiet }).fit(net, tinyLoader());
  console.log(`  compile=${String(compileFlag).padEnd(6)} grad ${captured.shape} strides (${captured.strides}) contiguous=${captured.contiguous}`);
  console.log(`  ${' '.repeat(14)}logical  ${captured.logical.map((v) => v.toFixed(5).padStart(9)).join(' ')}`);
  console.log(`  ${' '.repeat(14)}storage  ${captured.storage.map((v) => v.toFixed(5).padStart(9)).join(' ')}`);
  unseed();
}

console.log('\n  The two paths compute the same gradient. One of them hands it back as a');
console.log('  transposed view, whose storage is the same numbers in a different order —');
console.log('  and the optimizers read p.grad._impl.storage.data, not the strided tensor.');

console.log('\n=== the smallest case that shows it ===\n');
manual_seed(1);
const lin = new Linear(3, 2);
const ps = [...lin.parameters()];
const W = ps[0];
sum(lin.forward(tensor([1, 2, 3, 4, 5, 6], { shape: [2, 3] }))).backward();
console.log(`  weight  ${W.shape.join('x')}  strides (${W.strides.join(',')})`);
console.log(`  W.grad  ${W.grad.shape.join('x')}  strides (${W.grad.strides.join(',')})  contiguous=${W.grad.isContiguous}`);
console.log(`  gradient, read through its strides: ${[...W.grad.contiguous().data].join(' ')}`);
console.log(`  gradient, read from its storage:    ${[...W.grad._impl.storage.data].join(' ')}`);
const wBefore = Float32Array.from(W._impl.storage.data);
new SGD(ps, { lr: 1 }).step();
console.log(`  update SGD applied:                 ${[...W._impl.storage.data].map((v, i) => (wBefore[i] - v).toFixed(0)).join(' ')}`);
console.log('\n  Every element of a 2x3 weight should move by the gradient at that element.');
console.log('  Four of the six moved by the gradient at a different element.');
