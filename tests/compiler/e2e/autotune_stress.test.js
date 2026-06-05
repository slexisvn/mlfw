import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { CPUTarget, GPUTarget } from '../../../src/backend/target.js';
import { Compiler, CompilerConfig, compileGraph } from '../../../src/compiler/pipeline/compiler.js';
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
function assertFinite(data, label) {
  for (let i = 0; i < Math.min(data.length, 100); i++)
    assert.ok(isFinite(data[i]), `${label}[${i}] = ${data[i]}`);
}
function countOps(func) {
  let c = 0;
  for (const op of func.ops()) c++;
  return c;
}

const AUTOTUNE_OPTS = {
  scheduling: { autotune: true },
  autotuneConfig: { strategy: 'random', numTrials: 8, seed: 42 },
};

const AUTOTUNE_EVO_OPTS = {
  scheduling: { autotune: true },
  autotuneConfig: { strategy: 'evolutionary', numTrials: 16, populationSize: 8, numGenerations: 2, seed: 42 },
};

function timedCompile(label, func, target, opts = {}) {
  const ops = countOps(func);
  const t0 = performance.now();
  const compiled = compileGraph(func, target, opts);
  const ms = performance.now() - t0;
  console.log(`  [${label}] ${ops} ops => ${ms.toFixed(0)}ms (${(ops / ms).toFixed(1)} ops/ms)`);
  return { compiled, ms, ops };
}

// ── Model builders ──────────────────────────────────────────────

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
  for (let i = 0; i < numLayers; i++) paramTypes.push(T([D, D]));
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

function buildMixedArch(name, B, S, D, repeats) {
  return buildFunction(name,
    [T([B, S, D]), T([D, D]), T([D, D]), T([D]), T([D]), T([D, D]), T([D, D]), T([D]), T([D])],
    [T([B, S, D])],
    (b, args) => {
      let cur = args[0];
      for (let i = 0; i < repeats; i++) {
        const flat = b.reshape(cur, [B * S, D]);
        const Q = b.reshape(b.matmul(flat.getResult(0), args[1]).getResult(0), [B, S, D]);
        const K = b.reshape(b.matmul(flat.getResult(0), args[2]).getResult(0), [B, S, D]);
        const scores = b.dot(Q.getResult(0), K.getResult(0), [2], [2], [0], [0]);
        const attn = b.softmax(scores.getResult(0), -1);
        const ctx = b.dot(attn.getResult(0), Q.getResult(0), [2], [1], [0], [0]);
        const proj = b.reshape(ctx.getResult(0), [B * S, D]);
        const ff = b.matmul(proj.getResult(0), args[5]);
        const act = b.silu(b.reshape(ff.getResult(0), [B, S, D]).getResult(0));
        const ff2 = b.matmul(b.reshape(act.getResult(0), [B * S, D]).getResult(0), args[6]);
        const res = b.add(cur, b.reshape(ff2.getResult(0), [B, S, D]).getResult(0));
        const ln = b.layernorm(res.getResult(0), args[3], args[4], -1, 1e-5);
        const sig = b.sigmoid(ln.getResult(0));
        cur = b.mul(ln.getResult(0), sig.getResult(0)).getResult(0);
      }
      const out = b.layernorm(cur, args[7], args[8], -1, 1e-5);
      b.returnOp([out.getResult(0)]);
    }
  );
}

function buildDeepConvBNChain(name, B, C, H, W, numBlocks) {
  const paramTypes = [];
  for (let i = 0; i < numBlocks; i++) {
    paramTypes.push(T([C, C, 3, 3]));  // conv weight
    paramTypes.push(T([C]), T([C]), T([C]), T([C]));  // BN: gamma, beta, mean, var
  }
  return buildFunction(name, [T([B, C, H, W]), ...paramTypes], [T([B, C, H, W])],
    (b, args) => {
      let cur = args[0];
      let idx = 1;
      for (let i = 0; i < numBlocks; i++) {
        const w = args[idx++];
        const g = args[idx++], beta = args[idx++], mean = args[idx++], variance = args[idx++];
        const c = b.conv(cur, w, [1, 1], [[1, 1], [1, 1]]);
        const bn = b.batchnorm(c.getResult(0), g, beta, mean, variance, 1, 1e-5);
        const act = b.silu(bn.getResult(0));
        const skip = b.add(cur, act.getResult(0));
        cur = skip.getResult(0);
      }
      b.returnOp([cur]);
    }
  );
}

function buildLSTMStack(name, B, D, numSteps) {
  return buildFunction(name,
    [T([B, D]), T([B, D]), T([B, D]), T([D, 4 * D]), T([D, 4 * D]), T([4 * D])],
    [T([B, D]), T([B, D])],
    (b, [x, hInit, cInit, wx, wh, bias]) => {
      let h = hInit, c = cInit;
      for (let t = 0; t < numSteps; t++) {
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
}

// ── Tests ───────────────────────────────────────────────────────

describe('Autotune: Transformer (random search)', () => {
  it('6-layer (~300 ops) compiles with autotune', () => {
    const func = buildTransformerStack('at_t6', 1, 4, 8, 32, 6);
    const r = timedCompile('transformer-6L-autotune', func, CPUTarget(), {
      fusion: { enabled: false }, ...AUTOTUNE_OPTS,
    });
    assert.ok(r.ms < 3000, `took ${r.ms.toFixed(0)}ms`);
  });

  it('12-layer (~600 ops) compiles with autotune', () => {
    const func = buildTransformerStack('at_t12', 1, 4, 8, 32, 12);
    const r = timedCompile('transformer-12L-autotune', func, CPUTarget(), {
      fusion: { enabled: false }, ...AUTOTUNE_OPTS,
    });
    assert.ok(r.ms < 5000, `took ${r.ms.toFixed(0)}ms`);
  });

  it('24-layer (~1200 ops) compiles with autotune', () => {
    const func = buildTransformerStack('at_t24', 1, 4, 8, 32, 24);
    const r = timedCompile('transformer-24L-autotune', func, CPUTarget(), {
      fusion: { enabled: false }, ...AUTOTUNE_OPTS,
    });
    assert.ok(r.ms < 15000, `took ${r.ms.toFixed(0)}ms`);
  });

  it('48-layer (~2400 ops) compiles with autotune', () => {
    const func = buildTransformerStack('at_t48', 1, 4, 8, 32, 48);
    const r = timedCompile('transformer-48L-autotune', func, CPUTarget(), {
      fusion: { enabled: false }, ...AUTOTUNE_OPTS,
    });
    assert.ok(r.ms < 30000, `took ${r.ms.toFixed(0)}ms`);
  });

  it('200-layer (~10k ops) compiles with autotune under 120s', () => {
    const func = buildTransformerStack('at_t200', 1, 4, 8, 32, 200);
    const r = timedCompile('transformer-200L-autotune', func, CPUTarget(), {
      fusion: { enabled: false }, ...AUTOTUNE_OPTS,
    });
    assert.ok(r.ms < 120000, `took ${r.ms.toFixed(0)}ms`);
  });
});

describe('Autotune: Transformer (evolutionary search)', () => {
  it('12-layer (~600 ops) with evolutionary search', () => {
    const func = buildTransformerStack('at_t12e', 1, 4, 8, 32, 12);
    const r = timedCompile('transformer-12L-evo', func, CPUTarget(), {
      fusion: { enabled: false }, ...AUTOTUNE_EVO_OPTS,
    });
    assert.ok(r.ms < 10000, `took ${r.ms.toFixed(0)}ms`);
  });

  it('48-layer (~2400 ops) with evolutionary search', () => {
    const func = buildTransformerStack('at_t48e', 1, 4, 8, 32, 48);
    const r = timedCompile('transformer-48L-evo', func, CPUTarget(), {
      fusion: { enabled: false }, ...AUTOTUNE_EVO_OPTS,
    });
    assert.ok(r.ms < 60000, `took ${r.ms.toFixed(0)}ms`);
  });
});

describe('Autotune: ResNet', () => {
  it('50-block (~500 ops) compiles with autotune', () => {
    const func = buildResNetStack('at_rn50', 1, 4, 8, 8, 50);
    const r = timedCompile('resnet-50-autotune', func, CPUTarget(), {
      fusion: { enabled: false }, ...AUTOTUNE_OPTS,
    });
    assert.ok(r.ms < 10000, `took ${r.ms.toFixed(0)}ms`);
  });

  it('200-block (~2000 ops) compiles with autotune', () => {
    const func = buildResNetStack('at_rn200', 1, 4, 8, 8, 200);
    const r = timedCompile('resnet-200-autotune', func, CPUTarget(), {
      fusion: { enabled: false }, ...AUTOTUNE_OPTS,
    });
    assert.ok(r.ms < 30000, `took ${r.ms.toFixed(0)}ms`);
  });

  it('1000-block (~5000 ops) compiles with autotune under 120s', () => {
    const func = buildResNetStack('at_rn1k', 1, 4, 4, 4, 1000);
    const r = timedCompile('resnet-1k-autotune', func, CPUTarget(), {
      fusion: { enabled: false }, ...AUTOTUNE_OPTS,
    });
    assert.ok(r.ms < 120000, `took ${r.ms.toFixed(0)}ms`);
  });
});

describe('Autotune: Deep MLP', () => {
  it('200-layer (~800 ops) compiles with autotune', () => {
    const func = buildDeepMLP('at_mlp200', 4, 16, 200);
    const r = timedCompile('mlp-200-autotune', func, CPUTarget(), {
      fusion: { enabled: false }, ...AUTOTUNE_OPTS,
    });
    assert.ok(r.ms < 10000, `took ${r.ms.toFixed(0)}ms`);
  });

  it('2000-layer (~8000 ops) compiles with autotune under 120s', () => {
    const func = buildDeepMLP('at_mlp2k', 4, 16, 2000);
    const r = timedCompile('mlp-2k-autotune', func, CPUTarget(), {
      fusion: { enabled: false }, ...AUTOTUNE_OPTS,
    });
    assert.ok(r.ms < 120000, `took ${r.ms.toFixed(0)}ms`);
  });

  it('5000-layer (~20k ops) compiles with autotune under 180s', () => {
    const func = buildDeepMLP('at_mlp5k', 4, 16, 5000);
    const r = timedCompile('mlp-5k-autotune', func, CPUTarget(), {
      fusion: { enabled: false }, ...AUTOTUNE_OPTS,
    });
    assert.ok(r.ms < 180000, `took ${r.ms.toFixed(0)}ms`);
  });
});

describe('Autotune: Mixed Architecture (attn + FFN + gating)', () => {
  it('20-block (~600 ops) compiles with autotune', () => {
    const func = buildMixedArch('at_mix20', 1, 4, 8, 20);
    const r = timedCompile('mixed-20-autotune', func, CPUTarget(), {
      fusion: { enabled: false }, ...AUTOTUNE_OPTS,
    });
    assert.ok(r.ms < 10000, `took ${r.ms.toFixed(0)}ms`);
  });

  it('100-block (~3000 ops) compiles with autotune', () => {
    const func = buildMixedArch('at_mix100', 1, 4, 8, 100);
    const r = timedCompile('mixed-100-autotune', func, CPUTarget(), {
      fusion: { enabled: false }, ...AUTOTUNE_OPTS,
    });
    assert.ok(r.ms < 60000, `took ${r.ms.toFixed(0)}ms`);
  });
});

describe('Autotune: Conv + BatchNorm chain', () => {
  it('100-block conv-BN-SiLU (~1000 ops) compiles with autotune', () => {
    const func = buildDeepConvBNChain('at_cbn100', 1, 4, 8, 8, 100);
    const r = timedCompile('conv-bn-100-autotune', func, CPUTarget(), {
      fusion: { enabled: false }, ...AUTOTUNE_OPTS,
    });
    assert.ok(r.ms < 30000, `took ${r.ms.toFixed(0)}ms`);
  });

  it('500-block conv-BN-SiLU (~5000 ops) compiles with autotune under 120s', () => {
    const func = buildDeepConvBNChain('at_cbn500', 1, 4, 4, 4, 500);
    const r = timedCompile('conv-bn-500-autotune', func, CPUTarget(), {
      fusion: { enabled: false }, ...AUTOTUNE_OPTS,
    });
    assert.ok(r.ms < 120000, `took ${r.ms.toFixed(0)}ms`);
  });
});

describe('Autotune: LSTM unrolled', () => {
  it('50-step LSTM (~1500 ops) compiles with autotune', () => {
    const func = buildLSTMStack('at_lstm50', 2, 8, 50);
    const r = timedCompile('lstm-50step-autotune', func, CPUTarget(), {
      fusion: { enabled: false }, ...AUTOTUNE_OPTS,
    });
    assert.ok(r.ms < 30000, `took ${r.ms.toFixed(0)}ms`);
  });

  it('200-step LSTM (~6000 ops) compiles with autotune under 120s', () => {
    const func = buildLSTMStack('at_lstm200', 2, 8, 200);
    const r = timedCompile('lstm-200step-autotune', func, CPUTarget(), {
      fusion: { enabled: false }, ...AUTOTUNE_OPTS,
    });
    assert.ok(r.ms < 120000, `took ${r.ms.toFixed(0)}ms`);
  });
});

describe('Autotune + Fusion combined', () => {
  it('12-layer transformer with autotune + XLA fusion', () => {
    const func = buildTransformerStack('at_t12f', 1, 4, 8, 32, 12);
    const r = timedCompile('transformer-12L-autotune+fusion', func, CPUTarget(), {
      fusion: { enabled: true, strategy: 'xla' }, ...AUTOTUNE_OPTS,
    });
    assert.ok(r.ms < 10000, `took ${r.ms.toFixed(0)}ms`);
  });

  it('48-layer transformer with autotune + dominator fusion', () => {
    const func = buildTransformerStack('at_t48d', 1, 4, 8, 32, 48);
    const r = timedCompile('transformer-48L-autotune+dom', func, CPUTarget(), {
      fusion: { enabled: true, strategy: 'dominator' }, ...AUTOTUNE_OPTS,
    });
    assert.ok(r.ms < 30000, `took ${r.ms.toFixed(0)}ms`);
  });

  it('200-layer transformer with autotune + fusion under 120s', () => {
    const func = buildTransformerStack('at_t200f', 1, 4, 8, 32, 200);
    const r = timedCompile('transformer-200L-autotune+fusion', func, CPUTarget(), {
      fusion: { enabled: true }, ...AUTOTUNE_OPTS,
    });
    assert.ok(r.ms < 120000, `took ${r.ms.toFixed(0)}ms`);
  });
});

describe('Autotune: GPU codegen', () => {
  it('12-layer transformer GPU with autotune', () => {
    const func = buildTransformerStack('at_t12g', 1, 4, 8, 32, 12);
    const r = timedCompile('transformer-12L-gpu-autotune', func, GPUTarget(), {
      fusion: { enabled: false }, ...AUTOTUNE_OPTS,
    });
    assert.ok(r.ms < 10000, `took ${r.ms.toFixed(0)}ms`);
    assert.ok(r.compiled.getSource('at_t12g').includes('__global__'));
  });
});

describe('Autotune: Correctness at scale', () => {
  it('24-layer transformer + autotune produces finite output', () => {
    const B = 1, S = 4, D = 8, Dff = 32, N = 24;
    const func = buildTransformerStack('at_t24_correct', B, S, D, Dff, N);
    const compiled = compileGraph(func, CPUTarget(), {
      fusion: { enabled: false }, ...AUTOTUNE_OPTS,
    });
    const shapes = func.inputTypes.map(t => t.shape);
    const inputs = shapes.map(s => RuntimeTensor.fromArray(rand(numel(s)), s));
    const out = RuntimeTensor.zeros([B, S, D]);
    compiled.run('at_t24_correct', ...inputs, out);
    assertFinite(out.data, 'at_t24_correct');
  });

  it('100-block ResNet + autotune produces finite output', () => {
    const B = 1, C = 4, H = 4, W = 4, N = 100;
    const func = buildResNetStack('at_rn100_correct', B, C, H, W, N);
    const compiled = compileGraph(func, CPUTarget(), {
      fusion: { enabled: false }, ...AUTOTUNE_OPTS,
    });
    const shapes = func.inputTypes.map(t => t.shape);
    const inputs = shapes.map(s => RuntimeTensor.fromArray(rand(numel(s)), s));
    const out = RuntimeTensor.zeros([B, C, H, W]);
    compiled.run('at_rn100_correct', ...inputs, out);
    assertFinite(out.data, 'at_rn100_correct');
  });
});

describe('Autotune vs No-autotune: compile time overhead', () => {
  it('measures autotune overhead for 12-layer transformer', () => {
    const buildFunc = () => buildTransformerStack('overhead', 1, 4, 8, 32, 12);

    const t0 = performance.now();
    compileGraph(buildFunc(), CPUTarget(), { fusion: { enabled: false } });
    const baseMs = performance.now() - t0;

    const t1 = performance.now();
    compileGraph(buildFunc(), CPUTarget(), { fusion: { enabled: false }, ...AUTOTUNE_OPTS });
    const autotuneMs = performance.now() - t1;

    const overhead = autotuneMs / baseMs;
    console.log(`  Baseline: ${baseMs.toFixed(0)}ms | Autotune: ${autotuneMs.toFixed(0)}ms | Overhead: ${overhead.toFixed(1)}x`);
    assert.ok(overhead < 50, `Autotune overhead too high: ${overhead.toFixed(1)}x`);
  });

  it('measures autotune overhead for 200-layer MLP', () => {
    const buildFunc = () => buildDeepMLP('overhead_mlp', 4, 16, 200);

    const t0 = performance.now();
    compileGraph(buildFunc(), CPUTarget(), { fusion: { enabled: false } });
    const baseMs = performance.now() - t0;

    const t1 = performance.now();
    compileGraph(buildFunc(), CPUTarget(), { fusion: { enabled: false }, ...AUTOTUNE_OPTS });
    const autotuneMs = performance.now() - t1;

    const overhead = autotuneMs / baseMs;
    console.log(`  Baseline: ${baseMs.toFixed(0)}ms | Autotune: ${autotuneMs.toFixed(0)}ms | Overhead: ${overhead.toFixed(1)}x`);
    assert.ok(overhead < 50, `Autotune overhead too high: ${overhead.toFixed(1)}x`);
  });
});
