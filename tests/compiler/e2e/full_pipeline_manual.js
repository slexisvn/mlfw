import { performance } from 'node:perf_hooks';
import { TensorType, ScalarType, DYNAMIC } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { GraphModule } from '../../../src/compiler/ir/graph/module.js';
import { CPUTarget, GPUTarget, WasmTarget } from '../../../src/backend/target.js';
import { Compiler, compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { TraceLevel } from '../../../src/compiler/pipeline/trace.js';
import { RuntimeTensor } from '../../../src/compiler/runtime/runtime.js';

const f32 = ScalarType.F32;
function T(s) { return new TensorType(s, f32); }
function Tdyn(s) { return new TensorType(s, f32); }
function rand(n) { const a = new Float32Array(n); for (let i = 0; i < n; i++) a[i] = (Math.random() - 0.5) * 0.1; return a; }
function ones(n) { return new Float32Array(n).fill(1); }
function zeros(n) { return new Float32Array(n).fill(0); }
function numel(s) { return s.reduce((a, b) => a * b, 1); }

function assertFinite(data, label) {
  for (let i = 0; i < Math.min(data.length, 50); i++) {
    if (!isFinite(data[i])) throw new Error(label + '[' + i + '] = ' + data[i]);
  }
}

function hr() { console.log('─'.repeat(80)); }

let totalPass = 0;
let totalFail = 0;

function report(name, events, result, extra) {
  const phases = events.filter(e => e.type === 'phase' && e.action === 'end');
  const passes = events.filter(e => e.type === 'pass');
  const details = events.filter(e => e.type === 'pass_detail');
  const mem = events.filter(e => e.type === 'memory');
  const cg = events.filter(e => e.type === 'codegen');
  const at = events.filter(e => e.type === 'autotune');
  const fd = events.filter(e => e.type === 'fusion_decision');
  const snap = events.filter(e => e.type === 'ir_snapshot');

  const compilePhase = phases.find(e => e.phase === 'compile');
  console.log('  compile: ' + (compilePhase ? compilePhase.durationMs.toFixed(1) + 'ms' : 'N/A'));
  console.log('  kernels: ' + result.listKernels().join(', '));
  console.log('  events: ' + events.length + ' total | ' + passes.length + ' passes | ' + details.length + ' details | ' + fd.length + ' fusion_decisions');

  if (passes.length > 0) {
    const changed = passes.filter(p => p.changed);
    console.log('  passes changed: ' + changed.map(p => p.passName).join(', '));
  }
  if (mem.length > 0) {
    for (const m of mem) console.log('  memory [' + m.funcName + ']: peak=' + m.peakMemory + ' temps=' + m.totalTemporaries + ' inplace=' + m.totalInplace);
  }
  if (cg.length > 0) {
    for (const c of cg) console.log('  codegen [' + c.funcName + ']: ' + c.sourceSize + ' chars, target=' + c.targetName);
  }
  if (at.length > 0) {
    for (const a of at) console.log('  autotune [' + a.funcName + ']: blocks=' + a.blockCount + ' applied=' + a.applied + ' cached=' + a.cacheHits + ' ' + a.durationMs.toFixed(0) + 'ms');
  }
  if (details.length > 0) {
    for (const d of details) {
      const { type, level, timestamp, passName, ...rest } = d;
      console.log('  detail [' + passName + ']: ' + JSON.stringify(rest));
    }
  }
  if (snap.length > 0) {
    console.log('  IR snapshots: ' + snap.map(s => s.label).join(', '));
  }
  if (extra) console.log('  ' + extra);
}

function runModel(name, buildFn, compileOpts, runFn) {
  hr();
  console.log('MODEL: ' + name);
  const events = [];
  const opts = {
    ...compileOpts,
    trace: {
      level: TraceLevel.DEBUG,
      sink: e => events.push(e),
      irSnapshot: { afterGraphPasses: true, afterLowering: true },
      ...(compileOpts.trace || {}),
    },
  };

  try {
    const t0 = performance.now();
    const func = buildFn();
    const buildMs = performance.now() - t0;

    let result;
    if (func instanceof GraphModule) {
      const compiler = new Compiler(opts);
      result = compiler.compile(func);
    } else {
      result = compileGraph(func, opts.target, opts);
    }

    let extra = 'build=' + buildMs.toFixed(1) + 'ms';
    if (runFn) {
      const runResult = runFn(result, func);
      extra += ' | ' + runResult;
    }

    report(name, events, result, extra);
    console.log('  STATUS: PASS ✅');
    totalPass++;
  } catch (e) {
    console.log('  STATUS: FAIL ❌ ' + e.message);
    totalFail++;
  }
}

console.log('╔══════════════════════════════════════════════════════════════════════════════╗');
console.log('║  FULL PIPELINE MANUAL COMPILATION — TRIGGER ALL COMPILER CONCEPTS          ║');
console.log('╚══════════════════════════════════════════════════════════════════════════════╝');

runModel('1. Transformer 12L — CPU, XLA fusion, autotune, memory planning',
  () => {
    const B = 1, S = 4, D = 8, Dff = 32, N = 12;
    const inputs = [T([B, S, D])];
    const params = [];
    for (let i = 0; i < N; i++) {
      params.push(T([D, D]), T([D, D]), T([D, D]), T([D, D]));
      params.push(T([D]), T([D]));
      params.push(T([D, Dff]), T([Dff, D]));
      params.push(T([D]), T([D]));
    }
    return buildFunction('transformer_12L', [...inputs, ...params], [T([B, S, D])],
      (b, args) => {
        let cur = args[0]; let idx = 1;
        for (let layer = 0; layer < N; layer++) {
          const wq = args[idx++], wk = args[idx++], wv = args[idx++], wo = args[idx++];
          const ln1g = args[idx++], ln1b = args[idx++];
          const w1 = args[idx++], w2 = args[idx++];
          const ln2g = args[idx++], ln2b = args[idx++];
          const flat = b.reshape(cur, [B * S, D]);
          const Q = b.reshape(b.matmul(flat.getResult(0), wq).getResult(0), [B, S, D]);
          const K = b.reshape(b.matmul(flat.getResult(0), wk).getResult(0), [B, S, D]);
          const V = b.reshape(b.matmul(flat.getResult(0), wv).getResult(0), [B, S, D]);
          const scores = b.dot(Q.getResult(0), K.getResult(0), [2], [2], [0], [0]);
          const scale = b.broadcast(b.scalarConstant(0.35, f32).getResult(0), [B, S, S], []);
          const attn = b.softmax(b.mul(scores.getResult(0), scale.getResult(0)).getResult(0), -1);
          const ctx = b.dot(attn.getResult(0), V.getResult(0), [2], [1], [0], [0]);
          const proj = b.reshape(b.matmul(b.reshape(ctx.getResult(0), [B * S, D]).getResult(0), wo).getResult(0), [B, S, D]);
          const res1 = b.add(cur, proj.getResult(0));
          const ln1 = b.layernorm(res1.getResult(0), ln1g, ln1b, -1, 1e-5);
          const ff1 = b.reshape(b.matmul(b.reshape(ln1.getResult(0), [B * S, D]).getResult(0), w1).getResult(0), [B, S, Dff]);
          const act = b.gelu(ff1.getResult(0));
          const ff2 = b.reshape(b.matmul(b.reshape(act.getResult(0), [B * S, Dff]).getResult(0), w2).getResult(0), [B, S, D]);
          const res2 = b.add(ln1.getResult(0), ff2.getResult(0));
          cur = b.layernorm(res2.getResult(0), ln2g, ln2b, -1, 1e-5).getResult(0);
        }
        b.returnOp([cur]);
      }
    );
  },
  { target: CPUTarget(), fusion: { enabled: true, strategy: 'xla' }, scheduling: { autotune: true }, autotuneConfig: { strategy: 'random', numTrials: 4, seed: 42 } },
  (result, func) => {
    const shapes = func.inputTypes.map(t => t.shape);
    const inputs = shapes.map(s => RuntimeTensor.fromArray(rand(numel(s)), s));
    const out = RuntimeTensor.zeros([1, 4, 8]);
    result.run('transformer_12L', ...inputs, out);
    assertFinite(out.data, 'transformer_12L');
    return 'output finite ✓';
  }
);

runModel('2. ResNet 50-block — CPU, dominator fusion, epilogue fusion',
  () => {
    const B = 1, C = 4, H = 8, W = 8, N = 50;
    const params = [];
    for (let i = 0; i < N; i++) params.push(T([C, C, 3, 3]), T([C, C, 3, 3]));
    return buildFunction('resnet50', [T([B, C, H, W]), ...params], [T([B, C, H, W])],
      (b, args) => {
        let cur = args[0]; let idx = 1;
        for (let i = 0; i < N; i++) {
          const w1 = args[idx++], w2 = args[idx++];
          const c1 = b.conv(cur, w1, [1, 1], [[1, 1], [1, 1]]);
          const r1 = b.relu(c1.getResult(0));
          const c2 = b.conv(r1.getResult(0), w2, [1, 1], [[1, 1], [1, 1]]);
          const skip = b.add(cur, c2.getResult(0));
          cur = b.relu(skip.getResult(0)).getResult(0);
        }
        b.returnOp([cur]);
      }
    );
  },
  { target: CPUTarget({ enableEpilogueFusion: true }), fusion: { enabled: true, strategy: 'dominator' } },
  (result, func) => {
    const shapes = func.inputTypes.map(t => t.shape);
    const inputs = shapes.map(s => RuntimeTensor.fromArray(rand(numel(s)), s));
    const out = RuntimeTensor.zeros([1, 4, 8, 8]);
    result.run('resnet50', ...inputs, out);
    assertFinite(out.data, 'resnet50');
    return 'output finite ✓';
  }
);

runModel('3. GPU Transformer — CUDA codegen, fusion, epilogue',
  () => {
    const B = 1, S = 4, D = 16;
    return buildFunction('gpu_transformer',
      [T([B, S, D]), T([D, D]), T([D, D]), T([D]), T([D]), T([D, D * 4]), T([D * 4, D]), T([D]), T([D])],
      [T([B, S, D])],
      (b, [x, wq, wk, lng, lnb, w1, w2, ln2g, ln2b]) => {
        const flat = b.reshape(x, [B * S, D]);
        const Q = b.reshape(b.matmul(flat.getResult(0), wq).getResult(0), [B, S, D]);
        const K = b.reshape(b.matmul(flat.getResult(0), wk).getResult(0), [B, S, D]);
        const scores = b.dot(Q.getResult(0), K.getResult(0), [2], [2], [0], [0]);
        const attn = b.softmax(scores.getResult(0), -1);
        const ctx = b.dot(attn.getResult(0), Q.getResult(0), [2], [1], [0], [0]);
        const ln1 = b.layernorm(ctx.getResult(0), lng, lnb, -1, 1e-5);
        const ff1 = b.matmul(b.reshape(ln1.getResult(0), [B * S, D]).getResult(0), w1);
        const act = b.gelu(b.reshape(ff1.getResult(0), [B, S, D * 4]).getResult(0));
        const ff2 = b.matmul(b.reshape(act.getResult(0), [B * S, D * 4]).getResult(0), w2);
        const res = b.add(x, b.reshape(ff2.getResult(0), [B, S, D]).getResult(0));
        const out = b.layernorm(res.getResult(0), ln2g, ln2b, -1, 1e-5);
        b.returnOp([out.getResult(0)]);
      }
    );
  },
  { target: GPUTarget(), fusion: { enabled: true, epilogue: false } },
  (result) => {
    const src = result.getSource('gpu_transformer');
    const hasCuda = src.includes('__global__');
    const hasExpf = src.includes('expf');
    return 'CUDA=' + hasCuda + ' expf=' + hasExpf + ' size=' + src.length;
  }
);

runModel('4. Dynamic shape MLP — shape params, runtime dispatch',
  () => {
    const D = 16;
    return buildFunction('dyn_mlp',
      [new TensorType([DYNAMIC, D], f32), T([D, D]), T([D, D]), T([D, D])],
      [new TensorType([DYNAMIC, D], f32)],
      (b, [x, w1, w2, w3]) => {
        let cur = b.gelu(b.matmul(x, w1).getResult(0)).getResult(0);
        cur = b.gelu(b.matmul(cur, w2).getResult(0)).getResult(0);
        cur = b.matmul(cur, w3).getResult(0);
        b.returnOp([cur]);
      }
    );
  },
  { target: CPUTarget(), fusion: { enabled: true } },
  (result) => {
    for (const batchSize of [2, 8, 16]) {
      const D = 16;
      const x = RuntimeTensor.fromArray(rand(batchSize * D), [batchSize, D]);
      const w1 = RuntimeTensor.fromArray(rand(D * D), [D, D]);
      const w2 = RuntimeTensor.fromArray(rand(D * D), [D, D]);
      const w3 = RuntimeTensor.fromArray(rand(D * D), [D, D]);
      const out = RuntimeTensor.zeros([batchSize, D]);
      result.run('dyn_mlp', x, w1, w2, w3, out);
      assertFinite(out.data, 'dyn_mlp_batch' + batchSize);
    }
    return 'batch=[2,8,16] all finite ✓';
  }
);

runModel('5. Multi-target partition — CPU+GPU, copy_to_device',
  () => {
    const f64 = new TensorType([64, 64], f32);
    return buildFunction('partitioned',
      [f64, f64], [f64],
      (b, [x, y]) => {
        const add = b.add(x, y);
        add.setAttr('device', 'cpu_generic');
        const dot = b.dot(add.getResult(0), y, [1], [0]);
        dot.setAttr('device', 'gpu_generic');
        const neg = b.neg(dot.getResult(0));
        neg.setAttr('device', 'cpu_generic');
        b.returnOp([neg.getResult(0)]);
      }
    );
  },
  {
    target: CPUTarget(),
    fusion: { enabled: false },
    partition: { enabled: true, targets: [CPUTarget(), GPUTarget()], defaultTarget: CPUTarget() },
  },
  (result) => {
    const kernels = result.listKernels();
    const hasSubs = kernels.some(k => k.includes('partition'));
    return 'kernels=' + kernels.length + ' has_partitions=' + hasSubs;
  }
);

runModel('6. WASM target — conv+BN+SiLU chain',
  () => {
    const B = 1, C = 4, H = 8, W = 8;
    return buildFunction('wasm_cnn',
      [T([B, C, H, W]), T([C, C, 3, 3]), T([C]), T([C]), T([C]), T([C])],
      [T([B, C, H, W])],
      (b, [x, w, g, beta, mean, variance]) => {
        const c = b.conv(x, w, [1, 1], [[1, 1], [1, 1]]);
        const bn = b.batchnorm(c.getResult(0), g, beta, mean, variance, 1, 1e-5);
        const act = b.silu(bn.getResult(0));
        b.returnOp([act.getResult(0)]);
      }
    );
  },
  { target: WasmTarget(), fusion: { enabled: false } },
  (result) => {
    const src = result.getSource('wasm_cnn');
    return 'wasm_src=' + src.length + ' chars';
  }
);

runModel('7. Epilogue fusion — dot+bias+relu, dot+bias+gelu',
  () => {
    const M = 8, K = 16, N = 8;
    return buildFunction('epilogue_chain',
      [T([M, K]), T([K, N]), T([N]), T([N, N]), T([N])],
      [T([M, N])],
      (b, [x, w1, b1, w2, b2]) => {
        const dot1 = b.matmul(x, w1);
        const bias1 = b.add(dot1.getResult(0), b.broadcast(b1, [M, N], [1]).getResult(0));
        const relu1 = b.relu(bias1.getResult(0));
        const dot2 = b.matmul(relu1.getResult(0), w2);
        const bias2 = b.add(dot2.getResult(0), b.broadcast(b2, [M, N], [1]).getResult(0));
        const out = b.gelu(bias2.getResult(0));
        b.returnOp([out.getResult(0)]);
      }
    );
  },
  { target: CPUTarget(), fusion: { enabled: true } },
  (result, func) => {
    const shapes = func.inputTypes.map(t => t.shape);
    const inputs = shapes.map(s => RuntimeTensor.fromArray(rand(numel(s)), s));
    const out = RuntimeTensor.zeros([8, 8]);
    result.run('epilogue_chain', ...inputs, out);
    assertFinite(out.data, 'epilogue_chain');
    return 'output finite ✓';
  }
);

runModel('8. Deep MLP 500L — stress codegen, memory, scheduling',
  () => {
    const B = 4, D = 16, N = 500;
    const params = [];
    for (let i = 0; i < N; i++) params.push(T([D, D]));
    return buildFunction('deep_mlp_500', [T([B, D]), ...params], [T([B, D])],
      (b, args) => {
        let cur = args[0];
        for (let i = 1; i <= N; i++) {
          cur = b.gelu(b.matmul(cur, args[i]).getResult(0)).getResult(0);
        }
        b.returnOp([cur]);
      }
    );
  },
  { target: CPUTarget(), fusion: { enabled: true }, scheduling: { autotune: true }, autotuneConfig: { strategy: 'random', numTrials: 4, seed: 99 } },
  (result, func) => {
    const shapes = func.inputTypes.map(t => t.shape);
    const inputs = shapes.map(s => RuntimeTensor.fromArray(rand(numel(s)), s));
    const out = RuntimeTensor.zeros([4, 16]);
    result.run('deep_mlp_500', ...inputs, out);
    assertFinite(out.data, 'deep_mlp_500');
    return 'output finite ✓';
  }
);

runModel('9. LSTM 100-step unrolled — slice, gates, tanh, sigmoid, elementwise',
  () => {
    const B = 2, D = 8, N = 100;
    return buildFunction('lstm_100',
      [T([B, D]), T([B, D]), T([B, D]), T([D, 4 * D]), T([D, 4 * D]), T([4 * D])],
      [T([B, D]), T([B, D])],
      (b, [x, hInit, cInit, wx, wh, bias]) => {
        let h = hInit, c = cInit;
        for (let t = 0; t < N; t++) {
          const xProj = b.matmul(t === 0 ? x : h, wx);
          const hProj = b.matmul(h, wh);
          const sum = b.add(xProj.getResult(0), hProj.getResult(0));
          const bcast = b.broadcast(bias, [B, 4 * D], [1]);
          const gates = b.add(sum.getResult(0), bcast.getResult(0));
          const i_gate = b.sigmoid(b.slice(gates.getResult(0), [0, 0], [B, D]).getResult(0));
          const f_gate = b.sigmoid(b.slice(gates.getResult(0), [0, D], [B, 2 * D]).getResult(0));
          const g_val = b.tanh(b.slice(gates.getResult(0), [0, 2 * D], [B, 3 * D]).getResult(0));
          const o_gate = b.sigmoid(b.slice(gates.getResult(0), [0, 3 * D], [B, 4 * D]).getResult(0));
          const fC = b.mul(f_gate.getResult(0), c);
          const iG = b.mul(i_gate.getResult(0), g_val.getResult(0));
          const cNew = b.add(fC.getResult(0), iG.getResult(0));
          const hNew = b.mul(o_gate.getResult(0), b.tanh(cNew.getResult(0)).getResult(0));
          h = hNew.getResult(0);
          c = cNew.getResult(0);
        }
        b.returnOp([h, c]);
      }
    );
  },
  { target: CPUTarget(), fusion: { enabled: true } },
  (result, func) => {
    const shapes = func.inputTypes.map(t => t.shape);
    const inputs = shapes.map(s => RuntimeTensor.fromArray(rand(numel(s)), s));
    const outH = RuntimeTensor.zeros([2, 8]);
    const outC = RuntimeTensor.zeros([2, 8]);
    result.run('lstm_100', ...inputs, outH, outC);
    assertFinite(outH.data, 'lstm_h');
    assertFinite(outC.data, 'lstm_c');
    return 'h finite ✓, c finite ✓';
  }
);

runModel('10. Mixed: attention + MoE gating + contrastive — diverse op mix',
  () => {
    const B = 2, S = 4, D = 8, E = 4;
    return buildFunction('mixed_mega',
      [T([B, S, D]), T([D, D]), T([D, D]), T([D, D]),
       T([D]), T([D]), T([D, E]), T([D, D])],
      [T([B, S, D])],
      (b, [x, wq, wk, wv, lng, lnb, wGate, wExpert]) => {
        const flat = b.reshape(x, [B * S, D]);
        const Q = b.reshape(b.matmul(flat.getResult(0), wq).getResult(0), [B, S, D]);
        const K = b.reshape(b.matmul(flat.getResult(0), wk).getResult(0), [B, S, D]);
        const V = b.reshape(b.matmul(flat.getResult(0), wv).getResult(0), [B, S, D]);
        const scores = b.dot(Q.getResult(0), K.getResult(0), [2], [2], [0], [0]);
        const attn = b.softmax(scores.getResult(0), -1);
        const ctx = b.dot(attn.getResult(0), V.getResult(0), [2], [1], [0], [0]);
        const res = b.add(x, ctx.getResult(0));
        const ln = b.layernorm(res.getResult(0), lng, lnb, -1, 1e-5);

        const lnFlat = b.reshape(ln.getResult(0), [B * S, D]);
        const gateLogits = b.matmul(lnFlat.getResult(0), wGate);
        const gateWeights = b.softmax(gateLogits.getResult(0), -1);
        const topGate = b.reduce(gateWeights.getResult(0), b.scalarConstant(-Infinity, f32).getResult(0), [1], 'max');
        const gateScale = b.broadcast(topGate.getResult(0), [B * S, D], [0]);

        const expertOut = b.matmul(lnFlat.getResult(0), wExpert);
        const gated = b.mul(expertOut.getResult(0), gateScale.getResult(0));
        const sig = b.sigmoid(b.reshape(gated.getResult(0), [B, S, D]).getResult(0));
        const out = b.mul(ln.getResult(0), sig.getResult(0));
        b.returnOp([out.getResult(0)]);
      }
    );
  },
  { target: CPUTarget(), fusion: { enabled: true, strategy: 'xla' } },
  (result, func) => {
    const shapes = func.inputTypes.map(t => t.shape);
    const inputs = shapes.map(s => RuntimeTensor.fromArray(rand(numel(s)), s));
    const out = RuntimeTensor.zeros([2, 4, 8]);
    result.run('mixed_mega', ...inputs, out);
    assertFinite(out.data, 'mixed_mega');
    return 'output finite ✓';
  }
);

runModel('11. Multi-function module — 3 functions in 1 module',
  () => {
    const mod = new GraphModule('multi');
    mod.addFunction(buildFunction('encoder', [T([4, 8]), T([8, 16])], [T([4, 16])],
      (b, [x, w]) => { b.returnOp([b.gelu(b.matmul(x, w).getResult(0)).getResult(0)]); }));
    mod.addFunction(buildFunction('decoder', [T([4, 16]), T([16, 8])], [T([4, 8])],
      (b, [x, w]) => { b.returnOp([b.tanh(b.matmul(x, w).getResult(0)).getResult(0)]); }));
    mod.addFunction(buildFunction('classifier', [T([4, 8]), T([8, 2])], [T([4, 2])],
      (b, [x, w]) => { b.returnOp([b.softmax(b.matmul(x, w).getResult(0), -1).getResult(0)]); }));
    return mod;
  },
  { target: CPUTarget(), fusion: { enabled: true } },
  (result) => {
    const x = RuntimeTensor.fromArray(rand(32), [4, 8]);
    const w1 = RuntimeTensor.fromArray(rand(128), [8, 16]);
    const h = RuntimeTensor.zeros([4, 16]);
    result.run('encoder', x, w1, h);
    assertFinite(h.data, 'encoder');

    const w2 = RuntimeTensor.fromArray(rand(128), [16, 8]);
    const dec = RuntimeTensor.zeros([4, 8]);
    result.run('decoder', h, w2, dec);
    assertFinite(dec.data, 'decoder');

    const w3 = RuntimeTensor.fromArray(rand(16), [8, 2]);
    const cls = RuntimeTensor.zeros([4, 2]);
    result.run('classifier', dec, w3, cls);
    assertFinite(cls.data, 'classifier');
    let rowSum = cls.data[0] + cls.data[1];
    return 'all 3 funcs finite ✓ | softmax sum=' + rowSum.toFixed(4);
  }
);

runModel('12. GPU + autotune — large matmul chain for CUDA',
  () => {
    const M = 64, K = 32, N = 64;
    return buildFunction('gpu_matchain',
      [T([M, K]), T([K, N]), T([N, N]), T([N, N])],
      [T([M, N])],
      (b, [x, w1, w2, w3]) => {
        const h1 = b.gelu(b.matmul(x, w1).getResult(0));
        const h2 = b.gelu(b.matmul(h1.getResult(0), w2).getResult(0));
        const out = b.matmul(h2.getResult(0), w3);
        b.returnOp([out.getResult(0)]);
      }
    );
  },
  { target: GPUTarget(), fusion: { enabled: true, epilogue: false }, scheduling: { autotune: true }, autotuneConfig: { strategy: 'random', numTrials: 4, seed: 77 } },
  (result) => {
    const src = result.getSource('gpu_matchain');
    return 'CUDA=' + src.includes('__global__') + ' size=' + src.length;
  }
);

hr();
console.log();
console.log('═══════════════════════════════════════════════════════════════');
console.log('  TOTAL: ' + (totalPass + totalFail) + ' models | PASS: ' + totalPass + ' | FAIL: ' + totalFail);
console.log('═══════════════════════════════════════════════════════════════');
if (totalFail > 0) process.exit(1);
