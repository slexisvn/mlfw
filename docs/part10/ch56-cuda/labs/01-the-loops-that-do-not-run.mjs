import {
  lowerToTir, toLIR, emit, printTensorIR, Schedule, compileGraph,
  buildFunction, TensorType, ScalarType,
  CUDATarget, randn, manual_seed,
} from '../../_internals.mjs';

manual_seed(56);

const T = (s) => new TensorType(s, ScalarType.F32);

const fresh = async () => new Schedule(await lowerToTir((a) => a.mul(2.0), [randn([1024])], CUDATarget()));

console.log('=== the same kernel, serial and bound ===\n');
const serial = emit(toLIR((await fresh()).func, CUDATarget()), CUDATarget());
console.log(serial.source);
console.log(`  blockDim ${JSON.stringify(serial.metadata.blockDim)}  gridDim ${JSON.stringify(serial.metadata.gridDim)}  = ${serial.metadata.blockDim.reduce((a, b) => a * b) * serial.metadata.gridDim.reduce((a, b) => a * b)} thread\n`);

const bound = await fresh();
const [loop] = bound.getLoops('mul_block_0');
const [outer, inner] = bound.split(loop, 256);
bound.bindThread(inner, 'threadIdx.x');
bound.bindThread(outer, 'blockIdx.x');
const boundK = emit(toLIR(bound.func, CUDATarget()), CUDATarget());
console.log(boundK.source);
console.log(`  blockDim ${JSON.stringify(boundK.metadata.blockDim)}  gridDim ${JSON.stringify(boundK.metadata.gridDim)}  = ${boundK.metadata.blockDim.reduce((a, b) => a * b) * boundK.metadata.gridDim.reduce((a, b) => a * b)} threads`);
console.log('\n  The two `for` loops are gone. Their variables are now `const int` reads');
console.log('  of the hardware index registers, and the body runs once per thread. The');
console.log('  grid is not in the source at all: it is metadata handed to the launch.');

console.log('\n=== when three bindings want the same tag at two extents ===\n');
const guardCase = async () => {
  const gf = await lowerToTir((a) => a.mul(2.0).sum(1), [randn([8, 32])], CUDATarget());
  const gs = new Schedule(gf);
  gs.bindThread(gs.getLoops('mul_block_0')[1], 'threadIdx.x');
  gs.bindThread(gs.getLoops('reduce_init_1')[0], 'threadIdx.x');
  gs.bindThread(gs.getLoops('reduce_acc_2')[0], 'threadIdx.x');
  return toLIR(gs.func, CUDATarget());
};
const gk = emit(await guardCase(), CUDATarget());
console.log(gk.source);
console.log(`  blockDim ${JSON.stringify(gk.metadata.blockDim)}`);
console.log('\n  A thread tag has one extent per launch — the largest any binding asked');
console.log('  for, here 32. The two bindings that asked for 8 are wrapped in');
console.log('  `if (threadIdx.x < 8)`, so the other 24 threads do nothing. A binding');
console.log('  whose extent is dynamic reports extent 0 and is never guarded at all.');
console.log('');
console.log('  Two other things happened in that kernel without being asked for: the');
console.log('  intermediate `buf_6` was promoted to `__shared__` because one thread');
console.log('  writes what another reads, and `__syncthreads()` was inserted after');
console.log('  each thread-bound region. Chapter 56\'s second lab is about that decision.');

console.log('\n=== the limit the backend refuses to exceed ===\n');
const tooMany = new Schedule(await lowerToTir((a) => a.mul(2.0), [randn([64, 64])], CUDATarget()));
const [bi, bj] = tooMany.getLoops('mul_block_0');
tooMany.bindThread(bi, 'threadIdx.x');
tooMany.bindThread(bj, 'threadIdx.y');
try {
  emit(toLIR(tooMany.func, CUDATarget()), CUDATarget());
  console.log('  64 x 64 threads per block: accepted');
} catch (e) {
  console.log(`  64 x 64 threads per block: ${e.message}`);
}
console.log('\n  Thrown at codegen rather than discovered at launch, which is the right');
console.log('  place: a kernel that cannot start is a compilation failure. The shared-');
console.log('  memory budget is treated differently — it is a reason to give up on a');
console.log('  transformation rather than an error, which the next lab measures.');

console.log('\n=== how many threads the shipped defaults actually ask for ===\n');
const cases = {
  'elementwise': () => buildFunction('ew', [T([256, 256])], [T([256, 256])], (b, a) => b.returnOp([b.tanh(b.mul(a[0], a[0]).getResult(0)).getResult(0)])),
  'softmax': () => buildFunction('sm', [T([8, 64])], [T([8, 64])], (b, a) => b.returnOp([b.softmax(a[0], 1).getResult(0)])),
  'layer_norm': () => buildFunction('ln', [T([8, 64]), T([64]), T([64])], [T([8, 64])], (b, a) => b.returnOp([b._inferAndBuild('layer_norm', [a[0], a[1], a[2]], { axis: 1, epsilon: 1e-5 }).getResult(0)])),
  'matmul': () => buildFunction('mm', [T([64, 64]), T([64, 64])], [T([64, 64])], (b, a) => b.returnOp([b.matmul(a[0], a[1]).getResult(0)])),
  'sum to scalar': () => buildFunction('ra', [T([64, 64])], [T([])], (b, a) => {
    const z = b.scalarConstant(0, ScalarType.F32).getResult(0);
    b.returnOp([b.reduce(a[0], z, [0, 1], 'sum').getResult(0)]);
  }),
};

for (const [label, opts] of [['shipped defaults', {}], ['scheduling: { enabled: true }', { scheduling: { enabled: true } }]]) {
  console.log(`  --- ${label} ---`);
  console.log(`  ${'graph'.padEnd(15)} ${'kernel'.padEnd(8)} ${'threads'.padStart(8)} ${'shared'.padStart(7)}  diagnosis`);
  for (const [name, mk] of Object.entries(cases)) {
    const res = compileGraph(mk(), CUDATarget(), opts);
    for (const kn of res.module.kernels.names()) {
      const m = res.module.kernels.get(kn).metadata;
      const threads = m.blockDim.reduce((a, b) => a * b) * m.gridDim.reduce((a, b) => a * b);
      console.log(`  ${name.padEnd(15)} ${kn.padEnd(8)} ${String(threads).padStart(8)} ${String(m.sharedMemBytes).padStart(7)}  ${m.launchDiagnosis ? m.launchDiagnosis.reason : '-'}`);
    }
  }
  console.log('');
}

console.log('  With the shipped defaults every kernel but the matmul launches one');
console.log('  thread. `CUDATarget` declares `{ gpuTiling: true }` and does not declare');
console.log('  `enabled`, and the scheduling pass tests `enabled` before it reaches the');
console.log('  general rules — so only the matmul and convolution templates, which are');
console.log('  applied ahead of that test, ever produce a thread binding. `WebGPUTarget`');
console.log('  declares `{ enabled: true }`, which is the whole difference between the');
console.log('  two GPU targets and is one key in one attribute table.');
