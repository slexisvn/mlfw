import { TensorType, ScalarType, DYNAMIC } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { CPUTarget, GPUTarget } from '../../../src/backend/target.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { TraceLevel } from '../../../src/compiler/pipeline/trace.js';
import { RuntimeTensor } from '../../../src/compiler/runtime/runtime.js';

const f32 = ScalarType.F32;
const T = s => new TensorType(s, f32);

function section(t) { console.log('\n' + '='.repeat(80)); console.log('  ' + t); console.log('='.repeat(80)); }
function countIn(src, pat) { return (src.match(pat) || []).length; }

section('1. FUSION — elementwise chain: add->mul->exp->neg');

const mk1 = () => buildFunction('chain', [T([256]), T([256])], [T([256])],
  (b, [x, y]) => {
    const a = b.add(x, y);
    const m = b.mul(a.getResult(0), x);
    const e = b.exp(m.getResult(0));
    b.returnOp([b.neg(e.getResult(0)).getResult(0)]);
  });

const r1a = compileGraph(mk1(), CPUTarget(), { fusion: { enabled: false } });
const r1b = compileGraph(mk1(), CPUTarget(), { fusion: { enabled: true } });
const s1a = r1a.getSource('chain'), s1b = r1b.getSource('chain');

console.log('Without fusion: ' + s1a.length + ' chars, ' + countIn(s1a, /for \(/g) + ' loops, ' + countIn(s1a, /new Float32Array/g) + ' temp allocs');
console.log('With fusion:    ' + s1b.length + ' chars, ' + countIn(s1b, /for \(/g) + ' loops, ' + countIn(s1b, /new Float32Array/g) + ' temp allocs');
console.log('Reduction: size ' + ((1 - s1b.length / s1a.length) * 100).toFixed(0) + '%, loops ' + countIn(s1a, /for \(/g) + '->' + countIn(s1b, /for \(/g) + ', allocs ' + countIn(s1a, /new Float32Array/g) + '->' + countIn(s1b, /new Float32Array/g));

const X = RuntimeTensor.fromArray(Float32Array.from({length: 256}, (_, i) => i * 0.01), [256]);
const Y = RuntimeTensor.fromArray(new Float32Array(256).fill(0.5), [256]);
const o1a = RuntimeTensor.zeros([256]), o1b = RuntimeTensor.zeros([256]);
r1a.run('chain', X, Y, o1a);
r1b.run('chain', X, Y, o1b);
let maxDiff = 0;
for (let i = 0; i < 256; i++) maxDiff = Math.max(maxDiff, Math.abs(o1a.data[i] - o1b.data[i]));
console.log('Correctness: maxdiff=' + maxDiff.toExponential(2) + (maxDiff < 1e-5 ? ' MATCH' : ' MISMATCH'));

console.log('\n-- Unfused kernel (excerpt) --');
console.log(s1a.substring(0, 500));
console.log('\n-- Fused kernel (excerpt) --');
console.log(s1b.substring(0, 500));

section('2. CSE — duplicate matmul elimination');

const ev2 = [];
const mk2 = () => buildFunction('cse', [T([16, 16]), T([16, 16])], [T([16, 16])],
  (b, [x, w]) => {
    const mm1 = b.matmul(x, w);
    const mm2 = b.matmul(x, w);
    const out = b.add(mm1.getResult(0), mm2.getResult(0));
    b.returnOp([out.getResult(0)]);
  });
const r2 = compileGraph(mk2(), CPUTarget(), {
  fusion: { enabled: false },
  trace: { level: TraceLevel.DEBUG, sink: e => ev2.push(e) },
});
const cseD = ev2.find(e => e.passName === 'cse' && e.type === 'pass_detail');
const s2 = r2.getSource('cse');
const matmulLoops = countIn(s2, /\+=/g);
console.log('CSE eliminated: ' + (cseD?.eliminated || 0) + ' ops');
console.log('Accumulate ops (+=) in kernel: ' + matmulLoops + ' (expect 1 matmul, reused)');

section('3. DCE — dead branch removal');

const ev3 = [];
const mk3 = () => buildFunction('dce', [T([64]), T([64])], [T([64])],
  (b, [x, y]) => {
    const live = b.add(x, y);
    b.mul(x, y);
    b.exp(b.mul(x, y).getResult(0));
    b.returnOp([live.getResult(0)]);
  });
const r3 = compileGraph(mk3(), CPUTarget(), {
  fusion: { enabled: false },
  trace: { level: TraceLevel.DEBUG, sink: e => ev3.push(e) },
});
const dceD = ev3.find(e => e.passName === 'dce' && e.type === 'pass_detail');
const s3 = r3.getSource('dce');
console.log('DCE erased: ' + (dceD?.erasedCount || 0) + ' ops');
console.log('Kernel has Math.exp? ' + s3.includes('Math.exp') + ' (expect false)');
console.log('Kernel has *? ' + (countIn(s3, /\*/g) > 2) + ' (expect false — only indexing muls)');
console.log('Kernel size: ' + s3.length + ' chars');

section('4. CONSTANT FOLDING — compile-time eval');

const ev4 = [];
const mk4 = () => buildFunction('cf', [T([8])], [T([8])],
  (b, [x]) => {
    const c2 = b.broadcast(b.scalarConstant(2, f32).getResult(0), [8], []);
    const c3 = b.broadcast(b.scalarConstant(3, f32).getResult(0), [8], []);
    const six = b.mul(c2.getResult(0), c3.getResult(0));
    b.returnOp([b.add(x, six.getResult(0)).getResult(0)]);
  });
const r4 = compileGraph(mk4(), CPUTarget(), {
  fusion: { enabled: false },
  trace: { level: TraceLevel.DEBUG, sink: e => ev4.push(e) },
});
const cfD = ev4.find(e => e.passName === 'constant_fold' && e.type === 'pass_detail');
console.log('Folded: ' + (cfD?.foldedCount || 0) + ' ops');
const xc = RuntimeTensor.fromArray(new Float32Array(8).fill(1), [8]);
const oc = RuntimeTensor.zeros([8]);
r4.run('cf', xc, oc);
console.log('Output[0]=' + oc.data[0] + ' (expect 7=1+2*3): ' + (Math.abs(oc.data[0] - 7) < 0.01 ? 'CORRECT' : 'WRONG'));

section('5. DECOMPOSITION — layernorm+softmax+gelu to primitives');

const ev5 = [];
const mk5 = () => buildFunction('decomp', [T([2, 4, 8]), T([8]), T([8])], [T([2, 4, 8])],
  (b, [x, g, beta]) => {
    const ln = b.layernorm(x, g, beta, -1, 1e-5);
    const sm = b.softmax(ln.getResult(0), -1);
    b.returnOp([b.gelu(sm.getResult(0)).getResult(0)]);
  });
const r5 = compileGraph(mk5(), CPUTarget(), {
  fusion: { enabled: false },
  trace: { level: TraceLevel.DEBUG, sink: e => ev5.push(e) },
});
const decD = ev5.find(e => e.passName === 'DecompositionPass' && e.type === 'pass_detail');
console.log('Decomposed: ' + JSON.stringify(decD?.decomposed));
const s5 = r5.getSource('decomp');
console.log('Has Math.exp (softmax+gelu): ' + s5.includes('Math.exp'));
console.log('Has rsqrt-like (layernorm): ' + (s5.includes('1 / Math.sqrt') || s5.includes('rsqrt') || s5.includes('1.0 / Math.sqrt')));
console.log('No high-level op names in kernel: ' + (!s5.includes('softmax(') && !s5.includes('layernorm(') && !s5.includes('gelu(')));

section('6. MEMORY REUSE — 10-step neg(exp(neg(x))) chain');

const ev6 = [];
const mk6 = () => buildFunction('mem', [T([64, 64])], [T([64, 64])],
  (b, [x]) => {
    let cur = x;
    for (let i = 0; i < 10; i++) cur = b.neg(b.exp(b.neg(cur).getResult(0)).getResult(0)).getResult(0);
    b.returnOp([cur]);
  });
const r6 = compileGraph(mk6(), CPUTarget(), {
  fusion: { enabled: false },
  trace: { level: TraceLevel.VERBOSE, sink: e => ev6.push(e) },
});
const memS = ev6.find(e => e.type === 'memory');
const s6 = r6.getSource('mem');
console.log('Temps: ' + memS?.totalTemporaries + ', Inplace: ' + memS?.totalInplace + ', Peak: ' + memS?.peakMemory);
console.log('Reuse ratio: ' + ((memS?.totalInplace || 0) / Math.max(1, memS?.totalTemporaries) * 100).toFixed(0) + '%');
console.log('Allocs in kernel: ' + countIn(s6, /new Float32Array/g) + ' (should be << 20)');

section('7. GPU CODEGEN — matmul+gelu CUDA kernel');

const mk7 = () => buildFunction('gpu', [T([32, 64]), T([64, 32])], [T([32, 32])],
  (b, [x, w]) => {
    b.returnOp([b.gelu(b.matmul(x, w).getResult(0)).getResult(0)]);
  });
const r7 = compileGraph(mk7(), GPUTarget(), { fusion: { enabled: true, epilogue: false } });
const s7 = r7.getSource('gpu');
console.log('Size: ' + s7.length + ' chars');
console.log('__global__: ' + s7.includes('__global__'));
console.log('threadIdx: ' + s7.includes('threadIdx'));
console.log('blockIdx: ' + s7.includes('blockIdx'));
console.log('expf: ' + s7.includes('expf'));
console.log('float*: ' + s7.includes('float*'));
console.log('\n-- CUDA excerpt --');
console.log(s7.substring(0, 600));

section('8. DYNAMIC SHAPES — batch-polymorphic kernel');

const mk8 = () => buildFunction('dyn', [new TensorType([DYNAMIC, 8], f32), T([8, 8])], [new TensorType([DYNAMIC, 8], f32)],
  (b, [x, w]) => { b.returnOp([b.exp(b.matmul(x, w).getResult(0)).getResult(0)]); });
const r8 = compileGraph(mk8(), CPUTarget(), { fusion: { enabled: false } });
const s8 = r8.getSource('dyn');
console.log('Shape param in signature: ' + s8.includes('_ds'));
for (const batch of [1, 4, 16, 64]) {
  const xb = RuntimeTensor.fromArray(new Float32Array(batch * 8).fill(0.01), [batch, 8]);
  const wb = RuntimeTensor.fromArray(new Float32Array(64).fill(0.1), [8, 8]);
  const ob = RuntimeTensor.zeros([batch, 8]);
  r8.run('dyn', xb, wb, ob);
  let ok = true;
  for (let i = 0; i < batch * 8; i++) if (!isFinite(ob.data[i])) ok = false;
  process.stdout.write('  batch=' + batch + ':' + (ok ? 'ok' : 'FAIL') + ' ');
}
console.log();

section('9. PARTITION — 3 sub-kernels from 1 graph');

const mk9 = () => buildFunction('part', [T([16, 16]), T([16, 16])], [T([16, 16])],
  (b, [x, y]) => {
    const a = b.add(x, y); a.setAttr('device', 'cpu_generic');
    const d = b.dot(a.getResult(0), y, [1], [0]); d.setAttr('device', 'gpu_generic');
    b.returnOp([b.neg(d.getResult(0)).getResult(0)]);
  });
const r9 = compileGraph(mk9(), CPUTarget(), {
  fusion: { enabled: false },
  partition: { enabled: true, targets: [CPUTarget(), GPUTarget()], defaultTarget: CPUTarget() },
});
const k9 = r9.listKernels();
console.log('Kernels: ' + k9.join(', ') + ' (expect main + 3 partitions)');
for (const k of k9) console.log('  ' + k + ': ' + r9.getSource(k).length + ' chars');

section('10. DOMINATOR FUSION — conv+relu blocks');

const mk10 = () => {
  const N = 10;
  const params = [];
  for (let i = 0; i < N; i++) params.push(T([4, 4, 3, 3]), T([4, 4, 3, 3]));
  return buildFunction('domfuse', [T([1, 4, 8, 8]), ...params], [T([1, 4, 8, 8])],
    (b, args) => {
      let cur = args[0]; let idx = 1;
      for (let i = 0; i < N; i++) {
        const w1 = args[idx++], w2 = args[idx++];
        cur = b.relu(b.add(cur, b.conv(b.relu(b.conv(cur, w1, [1,1], [[1,1],[1,1]]).getResult(0)).getResult(0), w2, [1,1], [[1,1],[1,1]]).getResult(0)).getResult(0)).getResult(0);
      }
      b.returnOp([cur]);
    });
};
const ev10 = [];
const r10 = compileGraph(mk10(), CPUTarget(), {
  fusion: { enabled: true, strategy: 'dominator' },
  trace: { level: TraceLevel.DEBUG, sink: e => ev10.push(e) },
});
const domD = ev10.find(e => e.passName === 'DominatorFusionPass' && e.type === 'pass_detail');
console.log('Groups built: ' + domD?.groupsBuilt + ', fused: ' + domD?.groupsFused);

console.log('\n' + '='.repeat(80));
console.log('  KERNEL OPTIMIZATION ANALYSIS COMPLETE');
console.log('='.repeat(80));
