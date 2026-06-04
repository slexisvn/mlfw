import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { CPUTarget, GPUTarget } from '../../../src/compiler/backend/target.js';
import { RuntimeTensor } from '../../../src/compiler/runtime/runtime.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';

const f32 = ScalarType.F32;

function rand(n) {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = (Math.random() - 0.5) * 0.1;
  return a;
}

function assertFinite(data, label) {
  for (let i = 0; i < data.length; i++)
    assert.ok(isFinite(data[i]), `${label}[${i}] = ${data[i]}`);
}

function T(shape) { return new TensorType(shape, f32); }

function buildTransformerBlock(b, x, B, S, D, Dff, layerIdx) {
  const prefix = `l${layerIdx}`;
  const flat = b.reshape(x, [B * S, D]);

  const wq = b._inferAndBuild('constant', [], { value: 0.01, tensor_type: T([D, D]) }, null, [T([D, D])]);
  const wk = b._inferAndBuild('constant', [], { value: 0.01, tensor_type: T([D, D]) }, null, [T([D, D])]);
  const wv = b._inferAndBuild('constant', [], { value: 0.01, tensor_type: T([D, D]) }, null, [T([D, D])]);
  const wo = b._inferAndBuild('constant', [], { value: 0.01, tensor_type: T([D, D]) }, null, [T([D, D])]);

  const Q = b.reshape(b.matmul(flat.getResult(0), wq.getResult(0)).getResult(0), [B, S, D]);
  const K = b.reshape(b.matmul(flat.getResult(0), wk.getResult(0)).getResult(0), [B, S, D]);
  const V = b.reshape(b.matmul(flat.getResult(0), wv.getResult(0)).getResult(0), [B, S, D]);

  const scores = b.dot(Q.getResult(0), K.getResult(0), [2], [2], [0], [0]);
  const scale = b.broadcast(b.scalarConstant(1.0 / Math.sqrt(D), f32).getResult(0), [B, S, S], []);
  const attn = b.softmax(b.mul(scores.getResult(0), scale.getResult(0)).getResult(0), -1);
  const ctx = b.dot(attn.getResult(0), V.getResult(0), [2], [1], [0], [0]);
  const proj = b.reshape(b.matmul(b.reshape(ctx.getResult(0), [B * S, D]).getResult(0), wo.getResult(0)).getResult(0), [B, S, D]);

  const res1 = b.add(x, proj.getResult(0));
  const lng = b._inferAndBuild('constant', [], { value: 1.0, tensor_type: T([D]) }, null, [T([D])]);
  const lnb = b._inferAndBuild('constant', [], { value: 0.0, tensor_type: T([D]) }, null, [T([D])]);
  const ln1 = b.layernorm(res1.getResult(0), lng.getResult(0), lnb.getResult(0), -1, 1e-5);

  const wff1 = b._inferAndBuild('constant', [], { value: 0.01, tensor_type: T([D, Dff]) }, null, [T([D, Dff])]);
  const bff1 = b._inferAndBuild('constant', [], { value: 0.0, tensor_type: T([Dff]) }, null, [T([Dff])]);
  const wff2 = b._inferAndBuild('constant', [], { value: 0.01, tensor_type: T([Dff, D]) }, null, [T([Dff, D])]);
  const bff2 = b._inferAndBuild('constant', [], { value: 0.0, tensor_type: T([D]) }, null, [T([D])]);

  const ff1 = b.reshape(ln1.getResult(0), [B * S, D]);
  const h1 = b.add(b.matmul(ff1.getResult(0), wff1.getResult(0)).getResult(0), b.broadcast(bff1.getResult(0), [B * S, Dff], [1]).getResult(0));
  const act = b.relu(h1.getResult(0));
  const h2 = b.add(b.matmul(act.getResult(0), wff2.getResult(0)).getResult(0), b.broadcast(bff2.getResult(0), [B * S, D], [1]).getResult(0));
  const ffOut = b.reshape(h2.getResult(0), [B, S, D]);
  const res2 = b.add(ln1.getResult(0), ffOut.getResult(0));

  const lng2 = b._inferAndBuild('constant', [], { value: 1.0, tensor_type: T([D]) }, null, [T([D])]);
  const lnb2 = b._inferAndBuild('constant', [], { value: 0.0, tensor_type: T([D]) }, null, [T([D])]);
  return b.layernorm(res2.getResult(0), lng2.getResult(0), lnb2.getResult(0), -1, 1e-5).getResult(0);
}

function buildResNetStage(b, x, inDim, outDim, blocks, batchSize) {
  let cur = x;
  for (let i = 0; i < blocks; i++) {
    const curDim = i === 0 ? inDim : outDim;
    const w = b._inferAndBuild('constant', [], { value: 0.01, tensor_type: T([curDim, outDim]) }, null, [T([curDim, outDim])]);
    const bias = b._inferAndBuild('constant', [], { value: 0.0, tensor_type: T([outDim]) }, null, [T([outDim])]);
    const h = b.matmul(cur, w.getResult(0));
    const biased = b.add(h.getResult(0), b.broadcast(bias.getResult(0), [batchSize, outDim], [1]).getResult(0));
    const act = b.relu(biased.getResult(0));
    if (curDim === outDim) {
      cur = b.add(cur, act.getResult(0)).getResult(0);
    } else {
      cur = act.getResult(0);
    }
  }
  return cur;
}

function buildMLPStack(b, x, dims, batchSize) {
  let cur = x;
  for (let i = 0; i < dims.length - 1; i++) {
    const w = b._inferAndBuild('constant', [], { value: 0.01, tensor_type: T([dims[i], dims[i + 1]]) }, null, [T([dims[i], dims[i + 1]])]);
    const bias = b._inferAndBuild('constant', [], { value: 0.0, tensor_type: T([dims[i + 1]]) }, null, [T([dims[i + 1]])]);
    const h = b.matmul(cur, w.getResult(0));
    const biased = b.add(h.getResult(0), b.broadcast(bias.getResult(0), [batchSize, dims[i + 1]], [1]).getResult(0));
    if (i < dims.length - 2) {
      cur = b.relu(biased.getResult(0)).getResult(0);
    } else {
      cur = biased.getResult(0);
    }
  }
  return cur;
}

function countOps(func) {
  let count = 0;
  for (const op of func.ops()) count++;
  return count;
}

describe('Scaled transformer: 4-layer', () => {
  it('compiles ~200+ ops', () => {
    const B = 1, S = 4, D = 8, Dff = 32;
    const numLayers = 4;
    const func = buildFunction('transformer_4L', [T([B, S, D])], [T([B, S, D])],
      (b, [x]) => {
        let cur = x;
        for (let i = 0; i < numLayers; i++) cur = buildTransformerBlock(b, cur, B, S, D, Dff, i);
        b.returnOp([cur]);
      }
    );
    const ops = countOps(func);
    assert.ok(ops >= 200, `expected 200+ ops, got ${ops}`);
    const compiled = compileGraph(func, CPUTarget({ enableEpilogueFusion: false }), { enableFusion: false, enableEpilogueFusion: false });
    const out = RuntimeTensor.zeros([B, S, D]);
    compiled.run('transformer_4L', RuntimeTensor.fromArray(rand(B * S * D), [B, S, D]), out);
    assertFinite(out.data, 'transformer_4L');
  });
});

describe('Scaled transformer: 12-layer (BERT-base scale)', () => {
  it('compiles ~600+ ops', () => {
    const B = 1, S = 4, D = 8, Dff = 32;
    const numLayers = 12;
    const func = buildFunction('transformer_12L', [T([B, S, D])], [T([B, S, D])],
      (b, [x]) => {
        let cur = x;
        for (let i = 0; i < numLayers; i++) cur = buildTransformerBlock(b, cur, B, S, D, Dff, i);
        b.returnOp([cur]);
      }
    );
    const ops = countOps(func);
    assert.ok(ops >= 600, `expected 600+ ops, got ${ops}`);
    const start = performance.now();
    const compiled = compileGraph(func, CPUTarget({ enableEpilogueFusion: false }), { enableFusion: false, enableEpilogueFusion: false });
    const compileMs = performance.now() - start;
    const out = RuntimeTensor.zeros([B, S, D]);
    compiled.run('transformer_12L', RuntimeTensor.fromArray(rand(B * S * D), [B, S, D]), out);
    assertFinite(out.data, 'transformer_12L');
    assert.ok(compileMs < 10000, `compile took ${compileMs}ms, should be < 10s`);
  });
});

describe('Scaled transformer: 24-layer (BERT-large scale)', () => {
  it('compiles ~1200+ ops', () => {
    const B = 1, S = 4, D = 8, Dff = 32;
    const numLayers = 24;
    const func = buildFunction('transformer_24L', [T([B, S, D])], [T([B, S, D])],
      (b, [x]) => {
        let cur = x;
        for (let i = 0; i < numLayers; i++) cur = buildTransformerBlock(b, cur, B, S, D, Dff, i);
        b.returnOp([cur]);
      }
    );
    const ops = countOps(func);
    assert.ok(ops >= 1200, `expected 1200+ ops, got ${ops}`);
    const start = performance.now();
    const compiled = compileGraph(func, CPUTarget({ enableEpilogueFusion: false }), { enableFusion: false, enableEpilogueFusion: false });
    const compileMs = performance.now() - start;
    const out = RuntimeTensor.zeros([B, S, D]);
    compiled.run('transformer_24L', RuntimeTensor.fromArray(rand(B * S * D), [B, S, D]), out);
    assertFinite(out.data, 'transformer_24L');
    assert.ok(compileMs < 30000, `compile took ${compileMs}ms, should be < 30s`);
  });
});

describe('Deep ResNet: 50 blocks', () => {
  it('compiles 200+ ops with residual connections', () => {
    const batch = 2, D = 16;
    const func = buildFunction('resnet50', [T([batch, D])], [T([batch, D])],
      (b, [x]) => {
        let cur = buildResNetStage(b, x, D, D, 16, batch);
        cur = buildResNetStage(b, cur, D, D, 16, batch);
        cur = buildResNetStage(b, cur, D, D, 18, batch);
        b.returnOp([cur]);
      }
    );
    const ops = countOps(func);
    assert.ok(ops >= 200, `expected 200+ ops, got ${ops}`);
    const compiled = compileGraph(func, CPUTarget({ enableEpilogueFusion: false }), { enableFusion: false, enableEpilogueFusion: false });
    const out = RuntimeTensor.zeros([batch, D]);
    compiled.run('resnet50', RuntimeTensor.fromArray(rand(batch * D), [batch, D]), out);
    assertFinite(out.data, 'resnet50');
  });
});

describe('Wide MLP: 20 layers', () => {
  it('compiles 100+ ops, 20 matmuls', () => {
    const batch = 4;
    const dims = [64, 128, 128, 128, 128, 64, 64, 64, 64, 32, 32, 32, 32, 16, 16, 16, 16, 8, 8, 8, 4];
    const func = buildFunction('mlp20', [T([batch, dims[0]])], [T([batch, dims[dims.length - 1]])],
      (b, [x]) => {
        const out = buildMLPStack(b, x, dims, batch);
        b.returnOp([out]);
      }
    );
    const ops = countOps(func);
    assert.ok(ops >= 100, `expected 100+ ops, got ${ops}`);
    const compiled = compileGraph(func, CPUTarget({ enableEpilogueFusion: false }), { enableFusion: false, enableEpilogueFusion: false });
    const out = RuntimeTensor.zeros([batch, dims[dims.length - 1]]);
    compiled.run('mlp20', RuntimeTensor.fromArray(rand(batch * dims[0]), [batch, dims[0]]), out);
    assertFinite(out.data, 'mlp20');
  });
});

describe('Elementwise chain: 100 ops', () => {
  it('compiles and fuses 100 chained ops', () => {
    const N = 32;
    const chainLen = 100;
    const func = buildFunction('chain100', [T([N])], [T([N])],
      (b, [x]) => {
        let cur = x;
        for (let i = 0; i < chainLen; i++) {
          switch (i % 5) {
            case 0: cur = b.add(cur, b.broadcast(b.scalarConstant(0.001, f32).getResult(0), [N], []).getResult(0)).getResult(0); break;
            case 1: cur = b.mul(cur, b.broadcast(b.scalarConstant(0.999, f32).getResult(0), [N], []).getResult(0)).getResult(0); break;
            case 2: cur = b.abs(cur).getResult(0); break;
            case 3: cur = b.neg(b.neg(cur).getResult(0)).getResult(0); break;
            case 4: cur = b.add(cur, b.broadcast(b.scalarConstant(0, f32).getResult(0), [N], []).getResult(0)).getResult(0); break;
          }
        }
        b.returnOp([cur]);
      }
    );
    const ops = countOps(func);
    assert.ok(ops >= 100, `expected 100+ ops, got ${ops}`);

    const start = performance.now();
    const compiled = compileGraph(func, CPUTarget({ enableEpilogueFusion: false }), { enableFusion: true });
    const compileMs = performance.now() - start;

    const out = RuntimeTensor.zeros([N]);
    compiled.run('chain100', RuntimeTensor.fromArray(rand(N), [N]), out);
    assertFinite(out.data, 'chain100');
    assert.ok(compileMs < 5000, `compile took ${compileMs}ms, should be < 5s`);
  });
});

describe('Stacked reduce: 10 sequential reductions', () => {
  it('mean -> variance -> normalize, repeated 5x', () => {
    const rows = 4, cols = 8;
    const func = buildFunction('reduce_stack', [T([rows, cols])], [T([rows, cols])],
      (b, [x]) => {
        let cur = x;
        for (let i = 0; i < 5; i++) {
          const init = b.scalarConstant(0, f32);
          const mean = b.reduce(cur, init.getResult(0), [1], 'mean');
          const bcast = b.broadcast(mean.getResult(0), [rows, cols], [0]);
          const centered = b.sub(cur, bcast.getResult(0));
          const sq = b.mul(centered.getResult(0), centered.getResult(0));
          const variance = b.reduce(sq.getResult(0), init.getResult(0), [1], 'mean');
          const eps = b.broadcast(b.scalarConstant(1e-5, f32).getResult(0), [rows], []);
          const rstd = b.rsqrt(b.add(variance.getResult(0), eps.getResult(0)).getResult(0));
          const bcastRstd = b.broadcast(rstd.getResult(0), [rows, cols], [0]);
          cur = b.mul(centered.getResult(0), bcastRstd.getResult(0)).getResult(0);
        }
        b.returnOp([cur]);
      }
    );
    const ops = countOps(func);
    assert.ok(ops >= 50, `expected 50+ ops, got ${ops}`);
    const compiled = compileGraph(func, CPUTarget({ enableEpilogueFusion: false }), { enableFusion: false, enableEpilogueFusion: false });
    const out = RuntimeTensor.zeros([rows, cols]);
    compiled.run('reduce_stack', RuntimeTensor.fromArray(rand(rows * cols), [rows, cols]), out);
    assertFinite(out.data, 'reduce_stack');
  });
});

describe('GPU codegen: 12-layer transformer', () => {
  it('produces CUDA for 600+ op model', () => {
    const B = 1, S = 4, D = 8, Dff = 32;
    const func = buildFunction('gpu_12L', [T([B, S, D])], [T([B, S, D])],
      (b, [x]) => {
        let cur = x;
        for (let i = 0; i < 12; i++) cur = buildTransformerBlock(b, cur, B, S, D, Dff, i);
        b.returnOp([cur]);
      }
    );
    const ops = countOps(func);
    assert.ok(ops >= 600, `expected 600+ ops, got ${ops}`);
    const compiled = compileGraph(func, GPUTarget({ enableEpilogueFusion: false }), { enableFusion: false, enableEpilogueFusion: false });
    const source = compiled.getSource('gpu_12L');
    assert.ok(source.includes('__global__'));
    assert.ok(source.length > 5000, `CUDA source should be substantial, got ${source.length} chars`);
  });
});

describe('Compile time scaling', () => {
  it('compile time grows sub-quadratically with op count', () => {
    const times = [];
    for (const numLayers of [2, 4, 8]) {
      const func = buildFunction(`scale_${numLayers}`, [T([1, 4, 8])], [T([1, 4, 8])],
        (b, [x]) => {
          let cur = x;
          for (let i = 0; i < numLayers; i++) cur = buildTransformerBlock(b, cur, 1, 4, 8, 32, i);
          b.returnOp([cur]);
        }
      );
      const ops = countOps(func);
      const start = performance.now();
      compileGraph(func, CPUTarget({ enableEpilogueFusion: false }), { enableFusion: false, enableEpilogueFusion: false, verify: false });
      const ms = performance.now() - start;
      times.push({ numLayers, ops, ms });
    }
    const ratio = times[2].ms / times[0].ms;
    const opsRatio = times[2].ops / times[0].ops;
    assert.ok(ratio < opsRatio * opsRatio,
      `compile time ratio ${ratio.toFixed(1)}x should be less than ops ratio² ${(opsRatio * opsRatio).toFixed(1)}x (times: ${times.map(t => `${t.numLayers}L=${t.ms.toFixed(0)}ms/${t.ops}ops`).join(', ')})`);
  });
});
