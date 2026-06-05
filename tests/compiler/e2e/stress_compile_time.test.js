import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { CPUTarget, GPUTarget } from '../../../src/backend/target.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { RuntimeTensor } from '../../../src/compiler/runtime/runtime.js';
import { performance } from 'node:perf_hooks';

const f32 = ScalarType.F32;
function T(s) { return new TensorType(s, f32); }
function numel(s) { return s.reduce((a, b) => a * b, 1); }
function rand(n) {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = (Math.random() - 0.5) * 0.02;
  return a;
}
function ones(n) { return new Float32Array(n).fill(1); }
function zeros(n) { return new Float32Array(n).fill(0); }
function assertFinite(data, label) {
  for (let i = 0; i < Math.min(data.length, 100); i++)
    assert.ok(isFinite(data[i]), `${label}[${i}] = ${data[i]}`);
}

function countOps(func) {
  let c = 0;
  for (const op of func.ops()) c++;
  return c;
}

function timedCompile(label, func, target, opts = {}) {
  const ops = countOps(func);
  const t0 = performance.now();
  const compiled = compileGraph(func, target, opts);
  const ms = performance.now() - t0;
  const opsPerMs = ops / ms;
  return { compiled, ms, ops, opsPerMs };
}

function buildTransformerStack(name, B, S, D, Dff, numLayers) {
  const inputs = [T([B, S, D])];
  const paramTypes = [];
  for (let i = 0; i < numLayers; i++) {
    paramTypes.push(T([D, D]), T([D, D]), T([D, D]), T([D, D]));
    paramTypes.push(T([D]), T([D]));
    paramTypes.push(T([D, Dff]), T([Dff, D]));
    paramTypes.push(T([D]), T([D]));
  }
  return buildFunction(name, [...inputs, ...paramTypes], [T([B, S, D])],
    (b, args) => {
      let cur = args[0];
      let idx = 1;
      for (let layer = 0; layer < numLayers; layer++) {
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
}

function buildResNetStack(name, B, C, H, W, numBlocks) {
  const paramTypes = [];
  for (let i = 0; i < numBlocks; i++) {
    paramTypes.push(T([C, C, 3, 3]), T([C, C, 3, 3]));
  }
  return buildFunction(name, [T([B, C, H, W]), ...paramTypes], [T([B, C, H, W])],
    (b, args) => {
      let cur = args[0];
      let idx = 1;
      for (let i = 0; i < numBlocks; i++) {
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
}

function buildDeepMLP(name, B, D, numLayers) {
  const paramTypes = [];
  for (let i = 0; i < numLayers; i++) {
    paramTypes.push(T([D, D]));
  }
  return buildFunction(name, [T([B, D]), ...paramTypes], [T([B, D])],
    (b, args) => {
      let cur = args[0];
      for (let i = 1; i <= numLayers; i++) {
        const mm = b.matmul(cur, args[i]);
        cur = b.gelu(mm.getResult(0)).getResult(0);
      }
      b.returnOp([cur]);
    }
  );
}

function buildMixedArchitecture(name, B, S, D) {
  return buildFunction(name,
    [T([B, S, D]), T([D, D]), T([D, D]), T([D]), T([D]),
     T([D, D]), T([D, D]), T([D]), T([D])],
    [T([B, S, D])],
    (b, [x, wq, wk, lng, lnb, w1, w2, lng2, lnb2]) => {
      const flat = b.reshape(x, [B * S, D]);
      const Q = b.reshape(b.matmul(flat.getResult(0), wq).getResult(0), [B, S, D]);
      const K = b.reshape(b.matmul(flat.getResult(0), wk).getResult(0), [B, S, D]);
      const scores = b.dot(Q.getResult(0), K.getResult(0), [2], [2], [0], [0]);
      const attn = b.softmax(scores.getResult(0), -1);
      const ctx = b.dot(attn.getResult(0), Q.getResult(0), [2], [1], [0], [0]);
      const proj = b.reshape(ctx.getResult(0), [B * S, D]);
      const ff = b.matmul(proj.getResult(0), w1);
      const act = b.silu(b.reshape(ff.getResult(0), [B, S, D]).getResult(0));
      const ff2 = b.matmul(b.reshape(act.getResult(0), [B * S, D]).getResult(0), w2);
      const res = b.add(x, b.reshape(ff2.getResult(0), [B, S, D]).getResult(0));
      const ln = b.layernorm(res.getResult(0), lng, lnb, -1, 1e-5);
      const sig = b.sigmoid(ln.getResult(0));
      const gated = b.mul(ln.getResult(0), sig.getResult(0));
      const out = b.layernorm(gated.getResult(0), lng2, lnb2, -1, 1e-5);
      b.returnOp([out.getResult(0)]);
    }
  );
}

describe('Compile Time Stress: Transformer', () => {
  it('6-layer transformer (~300 ops) compiles under 500ms', () => {
    const func = buildTransformerStack('t6', 1, 4, 8, 32, 6);
    const r = timedCompile('transformer-6L', func, CPUTarget(), { fusion: { enabled: false } });
    assert.ok(r.ms < 500, `took ${r.ms.toFixed(0)}ms for ${r.ops} ops`);
  });

  it('12-layer transformer (~600 ops) compiles under 1s', () => {
    const func = buildTransformerStack('t12', 1, 4, 8, 32, 12);
    const r = timedCompile('transformer-12L', func, CPUTarget(), { fusion: { enabled: false } });
    assert.ok(r.ms < 1000, `took ${r.ms.toFixed(0)}ms for ${r.ops} ops`);
  });

  it('24-layer transformer (~1200 ops) compiles under 3s', () => {
    const func = buildTransformerStack('t24', 1, 4, 8, 32, 24);
    const r = timedCompile('transformer-24L', func, CPUTarget(), { fusion: { enabled: false } });
    assert.ok(r.ms < 3000, `took ${r.ms.toFixed(0)}ms for ${r.ops} ops`);
  });

  it('48-layer transformer (~2400 ops) compiles under 10s', () => {
    const func = buildTransformerStack('t48', 1, 4, 8, 32, 48);
    const r = timedCompile('transformer-48L', func, CPUTarget(), { fusion: { enabled: false } });
    assert.ok(r.ms < 10000, `took ${r.ms.toFixed(0)}ms for ${r.ops} ops`);
  });

  it('200-layer transformer (~10000 ops) compiles under 30s', () => {
    const func = buildTransformerStack('t200', 1, 4, 8, 32, 200);
    const r = timedCompile('transformer-200L', func, CPUTarget(), { fusion: { enabled: false } });
    assert.ok(r.ms < 30000, `took ${r.ms.toFixed(0)}ms for ${r.ops} ops (${r.opsPerMs.toFixed(1)} ops/ms)`);
  });

  it('400-layer transformer (~20000 ops) compiles under 60s', () => {
    const func = buildTransformerStack('t400', 1, 4, 8, 32, 400);
    const r = timedCompile('transformer-400L', func, CPUTarget(), { fusion: { enabled: false } });
    assert.ok(r.ms < 60000, `took ${r.ms.toFixed(0)}ms for ${r.ops} ops (${r.opsPerMs.toFixed(1)} ops/ms)`);
  });
});

describe('Compile Time Stress: ResNet', () => {
  it('50-block ResNet (~500 ops) compiles under 2s', () => {
    const func = buildResNetStack('rn50', 1, 4, 8, 8, 50);
    const r = timedCompile('resnet-50', func, CPUTarget(), { fusion: { enabled: false } });
    assert.ok(r.ms < 2000, `took ${r.ms.toFixed(0)}ms for ${r.ops} ops`);
  });

  it('100-block ResNet (~1000 ops) compiles under 5s', () => {
    const func = buildResNetStack('rn100', 1, 4, 8, 8, 100);
    const r = timedCompile('resnet-100', func, CPUTarget(), { fusion: { enabled: false } });
    assert.ok(r.ms < 5000, `took ${r.ms.toFixed(0)}ms for ${r.ops} ops`);
  });

  it('1000-block ResNet (~5000 ops) compiles under 30s', () => {
    const func = buildResNetStack('rn1k', 1, 4, 4, 4, 1000);
    const r = timedCompile('resnet-1k', func, CPUTarget(), { fusion: { enabled: false } });
    assert.ok(r.ms < 30000, `took ${r.ms.toFixed(0)}ms for ${r.ops} ops (${r.opsPerMs.toFixed(1)} ops/ms)`);
  });

  it('2000-block ResNet (~10000 ops) compiles under 60s', () => {
    const func = buildResNetStack('rn2k', 1, 4, 4, 4, 2000);
    const r = timedCompile('resnet-2k', func, CPUTarget(), { fusion: { enabled: false } });
    assert.ok(r.ms < 60000, `took ${r.ms.toFixed(0)}ms for ${r.ops} ops (${r.opsPerMs.toFixed(1)} ops/ms)`);
  });
});

describe('Compile Time Stress: Deep MLP', () => {
  it('50-layer MLP (~200 ops) compiles under 500ms', () => {
    const func = buildDeepMLP('mlp50', 4, 16, 50);
    const r = timedCompile('mlp-50', func, CPUTarget(), { fusion: { enabled: false } });
    assert.ok(r.ms < 500, `took ${r.ms.toFixed(0)}ms for ${r.ops} ops`);
  });

  it('200-layer MLP (~800 ops) compiles under 2s', () => {
    const func = buildDeepMLP('mlp200', 4, 16, 200);
    const r = timedCompile('mlp-200', func, CPUTarget(), { fusion: { enabled: false } });
    assert.ok(r.ms < 2000, `took ${r.ms.toFixed(0)}ms for ${r.ops} ops`);
  });

  it('2000-layer MLP (~8000 ops) compiles under 30s', () => {
    const func = buildDeepMLP('mlp2k', 4, 16, 2000);
    const r = timedCompile('mlp-2k', func, CPUTarget(), { fusion: { enabled: false } });
    assert.ok(r.ms < 30000, `took ${r.ms.toFixed(0)}ms for ${r.ops} ops (${r.opsPerMs.toFixed(1)} ops/ms)`);
  });

  it('5000-layer MLP (~20000 ops) compiles under 60s', () => {
    const func = buildDeepMLP('mlp5k', 4, 16, 5000);
    const r = timedCompile('mlp-5k', func, CPUTarget(), { fusion: { enabled: false } });
    assert.ok(r.ms < 60000, `took ${r.ms.toFixed(0)}ms for ${r.ops} ops (${r.opsPerMs.toFixed(1)} ops/ms)`);
  });
});

describe('Compile Time Stress: With Fusion', () => {
  it('12-layer transformer with XLA fusion under 2s', () => {
    const func = buildTransformerStack('t12f', 1, 4, 8, 32, 12);
    const r = timedCompile('transformer-12L-fused', func, CPUTarget());
    assert.ok(r.ms < 2000, `took ${r.ms.toFixed(0)}ms for ${r.ops} ops`);
  });

  it('12-layer transformer with dominator fusion under 2s', () => {
    const func = buildTransformerStack('t12d', 1, 4, 8, 32, 12);
    const r = timedCompile('transformer-12L-dom', func, CPUTarget(), { fusion: { strategy: 'dominator' } });
    assert.ok(r.ms < 2000, `took ${r.ms.toFixed(0)}ms for ${r.ops} ops`);
  });

  it('50-block ResNet with fusion under 3s', () => {
    const func = buildResNetStack('rn50f', 1, 4, 8, 8, 50);
    const r = timedCompile('resnet-50-fused', func, CPUTarget());
    assert.ok(r.ms < 3000, `took ${r.ms.toFixed(0)}ms for ${r.ops} ops`);
  });
});

describe('Compile Time Stress: GPU Codegen', () => {
  it('12-layer transformer GPU codegen under 2s', () => {
    const func = buildTransformerStack('t12g', 1, 4, 8, 32, 12);
    const r = timedCompile('transformer-12L-gpu', func, GPUTarget(), { fusion: { enabled: false } });
    assert.ok(r.ms < 2000, `took ${r.ms.toFixed(0)}ms for ${r.ops} ops`);
    assert.ok(r.compiled.getSource('t12g').includes('__global__'));
  });
});

describe('Compile Time Stress: Mixed Architecture', () => {
  it('stacked mixed blocks (~500 ops) compile under 1s', () => {
    const B = 1, S = 4, D = 8;
    const func = buildFunction('mixed_stack',
      [T([B,S,D]), T([D,D]), T([D,D]), T([D]), T([D]), T([D,D]), T([D,D]), T([D]), T([D])],
      [T([B,S,D])],
      (b, args) => {
        let cur = args[0];
        for (let i = 0; i < 10; i++) {
          const flat = b.reshape(cur, [B*S, D]);
          const mm = b.matmul(flat.getResult(0), args[1]);
          const back = b.reshape(mm.getResult(0), [B,S,D]);
          const act = b.gelu(back.getResult(0));
          const res = b.add(cur, act.getResult(0));
          const ln = b.layernorm(res.getResult(0), args[3], args[4], -1, 1e-5);
          const sm = b.softmax(ln.getResult(0), -1);
          const mm2 = b.matmul(b.reshape(sm.getResult(0), [B*S,D]).getResult(0), args[5]);
          const sig = b.sigmoid(b.reshape(mm2.getResult(0), [B,S,D]).getResult(0));
          cur = b.mul(ln.getResult(0), sig.getResult(0)).getResult(0);
        }
        b.returnOp([cur]);
      }
    );
    const r = timedCompile('mixed-10', func, CPUTarget(), { fusion: { enabled: false } });
    assert.ok(r.ms < 1000, `took ${r.ms.toFixed(0)}ms for ${r.ops} ops`);
  });
});

describe('Compile Time Stress: With Fusion at Scale', () => {
  it('200-layer transformer with XLA fusion under 30s', () => {
    const func = buildTransformerStack('t200f', 1, 4, 8, 32, 200);
    const r = timedCompile('transformer-200L-fused', func, CPUTarget());
    assert.ok(r.ms < 30000, `took ${r.ms.toFixed(0)}ms for ${r.ops} ops (${r.opsPerMs.toFixed(1)} ops/ms)`);
  });

  it('200-layer transformer with dominator fusion under 30s', () => {
    const func = buildTransformerStack('t200d', 1, 4, 8, 32, 200);
    const r = timedCompile('transformer-200L-dom', func, CPUTarget(), { fusion: { strategy: 'dominator' } });
    assert.ok(r.ms < 30000, `took ${r.ms.toFixed(0)}ms for ${r.ops} ops (${r.opsPerMs.toFixed(1)} ops/ms)`);
  });
});

describe('Compile Time Scaling', () => {
  it('compile time scales sub-quadratically up to 10k ops', () => {
    const sizes = [12, 48, 200];
    const times = [];
    for (const n of sizes) {
      const func = buildTransformerStack(`scale_${n}`, 1, 4, 8, 32, n);
      const r = timedCompile(`scale-${n}L`, func, CPUTarget(), { fusion: { enabled: false } });
      times.push({ n, ops: r.ops, ms: r.ms });
    }

    const rMid = times[1].ms / times[0].ms;
    const rBig = times[2].ms / times[0].ms;
    const opsRMid = times[1].ops / times[0].ops;
    const opsRBig = times[2].ops / times[0].ops;

    assert.ok(rBig < opsRBig * opsRBig,
      `compile time grew super-quadratically: ${rBig.toFixed(1)}x time for ${opsRBig.toFixed(1)}x ops`);
  });
});

describe('Correctness at Scale', () => {
  it('24-layer transformer produces finite output', () => {
    const B = 1, S = 4, D = 8, Dff = 32, N = 24;
    const func = buildTransformerStack('t24_correct', B, S, D, Dff, N);
    const compiled = compileGraph(func, CPUTarget(), { fusion: { enabled: false } });

    const shapes = func.inputTypes.map(t => t.shape);
    const inputs = shapes.map(s => RuntimeTensor.fromArray(rand(numel(s)), s));
    const out = RuntimeTensor.zeros([B, S, D]);
    compiled.run('t24_correct', ...inputs, out);
    assertFinite(out.data, 't24_correct');
  });

  it('100-block ResNet produces finite output', () => {
    const B = 1, C = 4, H = 4, W = 4, N = 100;
    const func = buildResNetStack('rn100_correct', B, C, H, W, N);
    const compiled = compileGraph(func, CPUTarget(), { fusion: { enabled: false } });

    const shapes = func.inputTypes.map(t => t.shape);
    const inputs = shapes.map(s => RuntimeTensor.fromArray(rand(numel(s)), s));
    const out = RuntimeTensor.zeros([B, C, H, W]);
    compiled.run('rn100_correct', ...inputs, out);
    assertFinite(out.data, 'rn100_correct');
  });
});
