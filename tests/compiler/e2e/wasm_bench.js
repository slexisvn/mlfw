import { performance } from 'node:perf_hooks';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { CPUTarget, WasmTarget } from '../../../src/backend/target.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { RuntimeTensor } from '../../../src/compiler/runtime/runtime.js';

const f32 = ScalarType.F32;
const T = s => new TensorType(s, f32);
const rand = n => Float32Array.from({length: n}, () => (Math.random() - 0.5) * 0.1);
const ones = n => new Float32Array(n).fill(1);
const zeros = n => new Float32Array(n).fill(0);

function bench(name, compileFn, runFn, warmup, iters) {
  const compiled = compileFn();
  for (let i = 0; i < warmup; i++) runFn(compiled);
  const t0 = performance.now();
  for (let i = 0; i < iters; i++) runFn(compiled);
  const totalMs = performance.now() - t0;
  return { name, totalMs, iters, avgUs: (totalMs / iters) * 1000 };
}

function benchPair(label, funcName, buildModel, makeInputs, warmup, iters) {
  const wasmResult = bench(label + ' [WASM]',
    () => compileGraph(buildModel(), WasmTarget(), { fusion: { enabled: false } }),
    r => { const { inputs, output } = makeInputs(); r.run(funcName, ...inputs, output); },
    warmup, iters);

  const cpuResult = bench(label + ' [CPU/JS]',
    () => compileGraph(buildModel(), CPUTarget(), { fusion: { enabled: false } }),
    r => { const { inputs, output } = makeInputs(); r.run(funcName, ...inputs, output); },
    warmup, iters);

  const ratio = cpuResult.avgUs / wasmResult.avgUs;
  console.log(label);
  console.log('  WASM:  ' + wasmResult.avgUs.toFixed(1) + ' us/iter (' + wasmResult.totalMs.toFixed(1) + 'ms / ' + iters + ')');
  console.log('  JS:    ' + cpuResult.avgUs.toFixed(1) + ' us/iter (' + cpuResult.totalMs.toFixed(1) + 'ms / ' + iters + ')');
  console.log('  Ratio: ' + (ratio > 1 ? 'WASM ' + ratio.toFixed(2) + 'x faster' : 'JS ' + (1/ratio).toFixed(2) + 'x faster'));
  console.log();
  return { label, wasm: wasmResult.avgUs, js: cpuResult.avgUs, ratio };
}

console.log('================================================================');
console.log('  WASM vs JS Runtime Benchmark');
console.log('================================================================');
console.log();

const results = [];

results.push(benchPair('vadd 1K', 'vadd',
  () => buildFunction('vadd', [T([1024]), T([1024])], [T([1024])], (b, [x, y]) => {
    b.returnOp([b.add(x, y).getResult(0)]);
  }),
  () => ({ inputs: [RuntimeTensor.fromArray(rand(1024), [1024]), RuntimeTensor.fromArray(rand(1024), [1024])], output: RuntimeTensor.zeros([1024]) }),
  50, 500));

results.push(benchPair('vadd 64K', 'vadd',
  () => buildFunction('vadd', [T([65536]), T([65536])], [T([65536])], (b, [x, y]) => {
    b.returnOp([b.add(x, y).getResult(0)]);
  }),
  () => ({ inputs: [RuntimeTensor.fromArray(rand(65536), [65536]), RuntimeTensor.fromArray(rand(65536), [65536])], output: RuntimeTensor.zeros([65536]) }),
  10, 100));

results.push(benchPair('vadd 256K', 'vadd',
  () => buildFunction('vadd', [T([262144]), T([262144])], [T([262144])], (b, [x, y]) => {
    b.returnOp([b.add(x, y).getResult(0)]);
  }),
  () => ({ inputs: [RuntimeTensor.fromArray(rand(262144), [262144]), RuntimeTensor.fromArray(rand(262144), [262144])], output: RuntimeTensor.zeros([262144]) }),
  5, 50));

results.push(benchPair('chain add+mul+exp+neg 4K', 'chain',
  () => buildFunction('chain', [T([4096]), T([4096])], [T([4096])], (b, [x, y]) => {
    const a = b.add(x, y);
    const m = b.mul(a.getResult(0), x);
    const e = b.exp(m.getResult(0));
    b.returnOp([b.neg(e.getResult(0)).getResult(0)]);
  }),
  () => ({ inputs: [RuntimeTensor.fromArray(rand(4096), [4096]), RuntimeTensor.fromArray(rand(4096), [4096])], output: RuntimeTensor.zeros([4096]) }),
  20, 200));

results.push(benchPair('matmul 32x32', 'matmul',
  () => buildFunction('matmul', [T([32, 32]), T([32, 32])], [T([32, 32])], (b, [x, w]) => {
    b.returnOp([b.matmul(x, w).getResult(0)]);
  }),
  () => ({ inputs: [RuntimeTensor.fromArray(rand(1024), [32, 32]), RuntimeTensor.fromArray(rand(1024), [32, 32])], output: RuntimeTensor.zeros([32, 32]) }),
  20, 200));

results.push(benchPair('matmul 64x64', 'matmul',
  () => buildFunction('matmul', [T([64, 64]), T([64, 64])], [T([64, 64])], (b, [x, w]) => {
    b.returnOp([b.matmul(x, w).getResult(0)]);
  }),
  () => ({ inputs: [RuntimeTensor.fromArray(rand(4096), [64, 64]), RuntimeTensor.fromArray(rand(4096), [64, 64])], output: RuntimeTensor.zeros([64, 64]) }),
  10, 100));

results.push(benchPair('matmul 128x128', 'matmul',
  () => buildFunction('matmul', [T([128, 128]), T([128, 128])], [T([128, 128])], (b, [x, w]) => {
    b.returnOp([b.matmul(x, w).getResult(0)]);
  }),
  () => ({ inputs: [RuntimeTensor.fromArray(rand(16384), [128, 128]), RuntimeTensor.fromArray(rand(16384), [128, 128])], output: RuntimeTensor.zeros([128, 128]) }),
  5, 20));

results.push(benchPair('softmax [16,256]', 'softmax',
  () => buildFunction('softmax', [T([16, 256])], [T([16, 256])], (b, [x]) => {
    b.returnOp([b.softmax(x, -1).getResult(0)]);
  }),
  () => ({ inputs: [RuntimeTensor.fromArray(rand(4096), [16, 256])], output: RuntimeTensor.zeros([16, 256]) }),
  20, 200));

results.push(benchPair('layernorm [4,64,128]', 'layernorm',
  () => buildFunction('layernorm', [T([4, 64, 128]), T([128]), T([128])], [T([4, 64, 128])], (b, [x, g, bt]) => {
    b.returnOp([b.layernorm(x, g, bt, -1, 1e-5).getResult(0)]);
  }),
  () => ({ inputs: [RuntimeTensor.fromArray(rand(32768), [4, 64, 128]), RuntimeTensor.fromArray(ones(128), [128]), RuntimeTensor.fromArray(zeros(128), [128])], output: RuntimeTensor.zeros([4, 64, 128]) }),
  10, 50));

results.push(benchPair('conv [1,4,16,16] 3x3', 'conv',
  () => buildFunction('conv', [T([1, 4, 16, 16]), T([4, 4, 3, 3])], [T([1, 4, 16, 16])], (b, [x, w]) => {
    b.returnOp([b.conv(x, w, [1, 1], [[1, 1], [1, 1]]).getResult(0)]);
  }),
  () => ({ inputs: [RuntimeTensor.fromArray(rand(1024), [1, 4, 16, 16]), RuntimeTensor.fromArray(rand(144), [4, 4, 3, 3])], output: RuntimeTensor.zeros([1, 4, 16, 16]) }),
  10, 50));

results.push(benchPair('attention [1,8,32]', 'attention',
  () => buildFunction('attention', [T([1, 8, 32]), T([1, 8, 32]), T([1, 8, 32])], [T([1, 8, 32])], (b, [Q, K, V]) => {
    const s = b.dot(Q, K, [2], [2], [0], [0]);
    const sm = b.softmax(s.getResult(0), -1);
    b.returnOp([b.dot(sm.getResult(0), V, [2], [1], [0], [0]).getResult(0)]);
  }),
  () => ({ inputs: [RuntimeTensor.fromArray(rand(256), [1, 8, 32]), RuntimeTensor.fromArray(rand(256), [1, 8, 32]), RuntimeTensor.fromArray(rand(256), [1, 8, 32])], output: RuntimeTensor.zeros([1, 8, 32]) }),
  20, 100));

results.push(benchPair('MLP 4x[32,64,32] gelu', 'mlp',
  () => buildFunction('mlp', [T([4, 32]), T([32, 64]), T([64, 32])], [T([4, 32])], (b, [x, w1, w2]) => {
    const h = b.gelu(b.matmul(x, w1).getResult(0));
    b.returnOp([b.matmul(h.getResult(0), w2).getResult(0)]);
  }),
  () => ({ inputs: [RuntimeTensor.fromArray(rand(128), [4, 32]), RuntimeTensor.fromArray(rand(2048), [32, 64]), RuntimeTensor.fromArray(rand(2048), [64, 32])], output: RuntimeTensor.zeros([4, 32]) }),
  20, 100));

results.push(benchPair('reduce sum [256,512]->[256]', 'reduce',
  () => buildFunction('reduce', [T([256, 512])], [T([256])], (b, [x]) => {
    b.returnOp([b.reduce(x, b.scalarConstant(0, f32).getResult(0), [1], 'sum').getResult(0)]);
  }),
  () => ({ inputs: [RuntimeTensor.fromArray(rand(131072), [256, 512])], output: RuntimeTensor.zeros([256]) }),
  10, 50));

console.log('================================================================');
console.log('  Summary');
console.log('================================================================');
console.log();
console.log('Model'.padEnd(35) + 'WASM (us)'.padStart(12) + 'JS (us)'.padStart(12) + 'Winner'.padStart(20));
console.log('-'.repeat(79));
for (const r of results) {
  const winner = r.ratio > 1 ? 'WASM ' + r.ratio.toFixed(2) + 'x' : 'JS ' + (1/r.ratio).toFixed(2) + 'x';
  console.log(r.label.padEnd(35) + r.wasm.toFixed(1).padStart(12) + r.js.toFixed(1).padStart(12) + winner.padStart(20));
}

const wasmWins = results.filter(r => r.ratio > 1).length;
console.log();
console.log('WASM faster: ' + wasmWins + '/' + results.length + ' benchmarks');
