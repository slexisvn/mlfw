import {
  lowerToTir, toLIR, emit, Schedule, compileGraph, buildFunction,
  TensorType, ScalarType, FuncAttr, getCudaIntrin,
  CUDATarget, randn, manual_seed,
} from '../../_internals.mjs';

manual_seed(56);
const T = (s) => new TensorType(s, ScalarType.F32);

const sharedCase = async () => {
  const f = await lowerToTir((a) => a.mul(2.0).sum(1), [randn([8, 32])], CUDATarget());
  const s = new Schedule(f);
  s.bindThread(s.getLoops('mul_block_0')[1], 'threadIdx.x');
  s.bindThread(s.getLoops('reduce_init_1')[0], 'threadIdx.x');
  s.bindThread(s.getLoops('reduce_acc_2')[0], 'threadIdx.x');
  return toLIR(s.func, CUDATarget());
};

console.log('=== promote, or give up and run one thread ===\n');
console.log(`  ${'shared budget'.padStart(13)} ${'blockDim'.padStart(11)} ${'__shared__'.padStart(10)}  diagnosis`);
for (const limit of [49152, 1024, 512]) {
  const k = emit(await sharedCase(), CUDATarget({ sharedMemoryBytes: limit }));
  console.log(`  ${String(limit).padStart(13)} ${JSON.stringify(k.metadata.blockDim).padStart(11)} ${String((k.source.match(/__shared__/g) ?? []).length).padStart(10)}  ${k.metadata.launchDiagnosis ? k.metadata.launchDiagnosis.reason : '-'}`);
}
console.log('\n  The intermediate is written by one thread and read by another, so the');
console.log('  kernel is only correct if it lives in shared memory with a barrier');
console.log('  around it. 1024 bytes is exactly enough for a 256-element f32 array;');
console.log('  512 is not, so the backend abandons the parallel launch entirely and');
console.log('  emits the thread-bound loops as ordinary `for` loops on one thread.');
console.log('  The answers are the same. The kernel is 32 times narrower.');

console.log('\n=== the same decision, reached from a real graph ===\n');
const softmaxBig = buildFunction('smb', [T([64, 256])], [T([64, 256])],
  (b, a) => b.returnOp([b.softmax(a[0], 1).getResult(0)]));
const res = compileGraph(softmaxBig, CUDATarget(), { scheduling: { enabled: true } });
console.log(`  ${'kernel'.padEnd(8)} ${'threads'.padStart(8)}  diagnosis`);
for (const kn of res.module.kernels.names()) {
  const m = res.module.kernels.get(kn).metadata;
  const threads = m.blockDim.reduce((a, b) => a * b) * m.gridDim.reduce((a, b) => a * b);
  console.log(`  ${kn.padEnd(8)} ${String(threads).padStart(8)}  ${m.launchDiagnosis ? m.launchDiagnosis.reason : '-'}`);
}
console.log('\n  A 64x256 softmax splits into three kernels and the middle one is');
console.log('  serialized: it writes a buffer from one block and reads it from');
console.log('  another, and there is no barrier across blocks in a CUDA launch, so');
console.log('  no promotion can repair it. The reason is recorded on the kernel\'s');
console.log('  metadata as `launchDiagnosis`.');
console.log('');
console.log('  Nothing reads it. There is no trace event that carries it, so a');
console.log('  compilation in which one kernel silently became single-threaded looks');
console.log('  exactly like one in which nothing happened.');

console.log('\n=== a local array that will not fit in a thread ===\n');
const big = await lowerToTir((a) => a.mul(2.0).add(1.0), [randn([256, 256])], CUDATarget());
const bigK = emit(toLIR(big, CUDATarget()), CUDATarget());
console.log(`  kernel parameters: ${bigK.metadata.params.join(', ')}`);
console.log(`  offloaded to global scratch: ${JSON.stringify(bigK.metadata.scratch)}`);
console.log('\n  Without thread bindings a temporary would be a plain local array in a');
console.log('  __global__ function, which the hardware places in per-thread local');
console.log('  memory — and a thread has 512 KiB of it. A 65536-element f32 buffer is');
console.log('  256 KiB and the threshold is 32768 elements, so the backend turns it');
console.log('  into an extra kernel parameter the runtime allocates in global memory.');

console.log('\n=== the block reduction, and the path that cannot reach it ===\n');
const redTir = await lowerToTir((a) => a.sum(), [randn([64, 64])], CUDATarget());

const fromTir = emit(redTir, CUDATarget());
const fromLir = emit(toLIR(redTir, CUDATarget()), CUDATarget());

console.log(`  from the PrimFunc: blockDim ${JSON.stringify(fromTir.metadata.blockDim)}  __shared__ decls ${(fromTir.source.match(/__shared__/g) ?? []).length}  reported sharedMemBytes ${fromTir.metadata.sharedMemBytes}`);
console.log(`  from the LIRFunc : blockDim ${JSON.stringify(fromLir.metadata.blockDim)}  __shared__ decls ${(fromLir.source.match(/__shared__/g) ?? []).length}  reported sharedMemBytes ${fromLir.metadata.sharedMemBytes}`);
console.log('');
console.log('  from the PrimFunc:');
console.log(fromTir.source.split('\n').filter((l) => /_redsh|_racc|_rs|_rf/.test(l)).map((l) => `    ${l.trim()}`).join('\n'));
console.log('');
console.log('  from the LIRFunc:');
console.log(fromLir.source.split('\n').filter((l) => /_acc_|for \(/.test(l)).map((l) => `    ${l.trim()}`).join('\n'));
console.log('');
console.log('  `_matchFullReduction` pattern-matches a ForNode whose body is a');
console.log('  BlockNode whose body is a BufferStoreNode accumulating into a numel-1');
console.log('  buffer. Chapter 53 replaces exactly that shape with an');
console.log('  LIRAccumulatorNode — so the 256-thread tree reduction fires on the');
console.log('  pre-LIR path the unit tests exercise, and never on the path a');
console.log('  compilation takes. Note also that the pre-LIR kernel declares 1024');
console.log('  bytes of __shared__ and reports 0: `_emitParallelReduction` emits the');
console.log('  declaration directly instead of registering a shared buffer.');

console.log('\n=== an intrinsic replaces the body outright ===\n');
const intr = await lowerToTir((a, b) => a.matmul(b), [randn([64, 64]), randn([64, 64])], CUDATarget());
intr.setAttr(FuncAttr.TENSOR_INTRIN, {
  name: 'wmma_16x16x16_f16f16f32',
  info: { M: 64, N: 64, K: 64, a: 'buf_1', b: 'buf_3', c: 'buf_5' },
});
const intrK = emit(toLIR(intr, CUDATarget()), CUDATarget());
console.log(intrK.source);
console.log(`  blockDim ${JSON.stringify(intrK.metadata.blockDim)}  gridDim ${JSON.stringify(intrK.metadata.gridDim)}`);
console.log(`  registered intrinsics: ${['wmma_16x16x16_f16f16f32', 'gemm_pipelined_f32'].filter((n) => getCudaIntrin(n)).join(', ')}`);

const unknown = await lowerToTir((a, b) => a.matmul(b), [randn([16, 16]), randn([16, 16])], CUDATarget());
unknown.setAttr(FuncAttr.TENSOR_INTRIN, { name: 'wmma_32x32x32_tf32', info: { M: 16, N: 16, K: 16, a: 'x', b: 'y', c: 'z' } });
try {
  emit(toLIR(unknown, CUDATarget()), CUDATarget());
  console.log('  an unregistered intrinsic: accepted');
} catch (e) {
  console.log(`  an unregistered intrinsic: ${e.message}`);
}
console.log('\n  When a function carries a tensor-intrinsic attribute the backend does');
console.log('  not walk its body at all: it looks the name up in a registry and calls');
console.log('  an emitter that writes the whole kernel, including its own launch');
console.log('  geometry. The attribute is a promise made by whichever schedule');
console.log('  primitive set it, and the backend does not check it — but an unknown');
console.log('  name is a hard error rather than a silent fallback.');
