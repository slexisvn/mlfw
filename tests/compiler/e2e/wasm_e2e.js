import { TensorType, ScalarType, DYNAMIC } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { WasmTarget } from '../../../src/backend/target.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { RuntimeTensor } from '../../../src/compiler/runtime/runtime.js';

const f32 = ScalarType.F32;
const T = s => new TensorType(s, f32);
const rand = n => Float32Array.from({length: n}, () => (Math.random() - 0.5) * 0.2);
const ones = n => new Float32Array(n).fill(1);
const zeros = n => new Float32Array(n).fill(0);

let pass = 0, fail = 0;

function test(name, fn) {
  try {
    const ok = fn();
    console.log((ok ? 'PASS' : 'FAIL') + ' ' + name);
    ok ? pass++ : fail++;
  } catch (e) {
    console.log('FAIL ' + name + ': ' + e.message.substring(0, 100));
    fail++;
  }
}

test('add [8]', () => {
  const r = compileGraph(buildFunction('a', [T([8]), T([8])], [T([8])], (b, [x, y]) => {
    b.returnOp([b.add(x, y).getResult(0)]);
  }), WasmTarget(), { fusion: { enabled: false } });
  const o = RuntimeTensor.zeros([8]);
  r.run('a', RuntimeTensor.fromArray([1, 2, 3, 4, 5, 6, 7, 8], [8]), RuntimeTensor.fromArray([10, 20, 30, 40, 50, 60, 70, 80], [8]), o);
  return [...o.data].every((v, i) => v === (i + 1) + (i + 1) * 10);
});

test('mul [4]', () => {
  const r = compileGraph(buildFunction('m', [T([4]), T([4])], [T([4])], (b, [x, y]) => {
    b.returnOp([b.mul(x, y).getResult(0)]);
  }), WasmTarget(), { fusion: { enabled: false } });
  const o = RuntimeTensor.zeros([4]);
  r.run('m', RuntimeTensor.fromArray([2, 3, 4, 5], [4]), RuntimeTensor.fromArray([10, 10, 10, 10], [4]), o);
  return o.data[0] === 20 && o.data[1] === 30 && o.data[2] === 40 && o.data[3] === 50;
});

test('neg [4]', () => {
  const r = compileGraph(buildFunction('n', [T([4])], [T([4])], (b, [x]) => {
    b.returnOp([b.neg(x).getResult(0)]);
  }), WasmTarget(), { fusion: { enabled: false } });
  const o = RuntimeTensor.zeros([4]);
  r.run('n', RuntimeTensor.fromArray([1, -2, 3, -4], [4]), o);
  return o.data[0] === -1 && o.data[1] === 2 && o.data[2] === -3 && o.data[3] === 4;
});

test('exp [4]', () => {
  const r = compileGraph(buildFunction('e', [T([4])], [T([4])], (b, [x]) => {
    b.returnOp([b.exp(x).getResult(0)]);
  }), WasmTarget(), { fusion: { enabled: false } });
  const o = RuntimeTensor.zeros([4]);
  r.run('e', RuntimeTensor.fromArray([0, 1, -1, 0.5], [4]), o);
  return Math.abs(o.data[0] - 1) < 0.01 && Math.abs(o.data[1] - Math.exp(1)) < 0.01;
});

test('relu [4]', () => {
  const r = compileGraph(buildFunction('rl', [T([4])], [T([4])], (b, [x]) => {
    b.returnOp([b.relu(x).getResult(0)]);
  }), WasmTarget(), { fusion: { enabled: false } });
  const o = RuntimeTensor.zeros([4]);
  r.run('rl', RuntimeTensor.fromArray([-2, -1, 1, 2], [4]), o);
  return o.data[0] === 0 && o.data[1] === 0 && o.data[2] === 1 && o.data[3] === 2;
});

test('sigmoid [4]', () => {
  const r = compileGraph(buildFunction('sig', [T([4])], [T([4])], (b, [x]) => {
    b.returnOp([b.sigmoid(x).getResult(0)]);
  }), WasmTarget(), { fusion: { enabled: false } });
  const o = RuntimeTensor.zeros([4]);
  r.run('sig', RuntimeTensor.fromArray([0, -10, 10, 1], [4]), o);
  return Math.abs(o.data[0] - 0.5) < 0.01 && o.data[1] < 0.01 && o.data[2] > 0.99;
});

test('gelu [8]', () => {
  const r = compileGraph(buildFunction('gl', [T([8])], [T([8])], (b, [x]) => {
    b.returnOp([b.gelu(x).getResult(0)]);
  }), WasmTarget(), { fusion: { enabled: false } });
  const o = RuntimeTensor.zeros([8]);
  r.run('gl', RuntimeTensor.fromArray([-2, -1, 0, 0.5, 1, 2, 3, 4], [8]), o);
  return Math.abs(o.data[2]) < 0.01 && o.data[7] > 3.9 && o.data.every(v => isFinite(v));
});

test('matmul [2,3]@[3,2]', () => {
  const r = compileGraph(buildFunction('mm', [T([2, 3]), T([3, 2])], [T([2, 2])], (b, [x, w]) => {
    b.returnOp([b.matmul(x, w).getResult(0)]);
  }), WasmTarget(), { fusion: { enabled: false } });
  const o = RuntimeTensor.zeros([2, 2]);
  r.run('mm', RuntimeTensor.fromArray([1, 0, 0, 0, 1, 0], [2, 3]), RuntimeTensor.fromArray([1, 2, 3, 4, 5, 6], [3, 2]), o);
  return o.data[0] === 1 && o.data[1] === 2 && o.data[2] === 3 && o.data[3] === 4;
});

test('matmul [4,8]@[8,4] value check', () => {
  const r = compileGraph(buildFunction('mm2', [T([4, 8]), T([8, 4])], [T([4, 4])], (b, [x, w]) => {
    b.returnOp([b.matmul(x, w).getResult(0)]);
  }), WasmTarget(), { fusion: { enabled: false } });
  const o = RuntimeTensor.zeros([4, 4]);
  r.run('mm2', RuntimeTensor.fromArray(new Float32Array(32).fill(1), [4, 8]), RuntimeTensor.fromArray(new Float32Array(32).fill(0.5), [8, 4]), o);
  return Math.abs(o.data[0] - 4.0) < 0.01;
});

test('softmax [2,4] sums to 1', () => {
  const r = compileGraph(buildFunction('sm', [T([2, 4])], [T([2, 4])], (b, [x]) => {
    b.returnOp([b.softmax(x, -1).getResult(0)]);
  }), WasmTarget(), { fusion: { enabled: false } });
  const o = RuntimeTensor.zeros([2, 4]);
  r.run('sm', RuntimeTensor.fromArray([1, 2, 3, 4, 0, 0, 0, 0], [2, 4]), o);
  let s0 = 0, s1 = 0;
  for (let i = 0; i < 4; i++) { s0 += o.data[i]; s1 += o.data[4 + i]; }
  return Math.abs(s0 - 1) < 0.001 && Math.abs(s1 - 1) < 0.001 && o.data[3] > o.data[0];
});

test('layernorm [2,4,8]', () => {
  const r = compileGraph(buildFunction('ln', [T([2, 4, 8]), T([8]), T([8])], [T([2, 4, 8])], (b, [x, g, bt]) => {
    b.returnOp([b.layernorm(x, g, bt, -1, 1e-5).getResult(0)]);
  }), WasmTarget(), { fusion: { enabled: false } });
  const o = RuntimeTensor.zeros([2, 4, 8]);
  r.run('ln', RuntimeTensor.fromArray(rand(64), [2, 4, 8]), RuntimeTensor.fromArray(ones(8), [8]), RuntimeTensor.fromArray(zeros(8), [8]), o);
  return o.data.every(v => isFinite(v));
});

test('conv [1,2,4,4]', () => {
  const r = compileGraph(buildFunction('cv', [T([1, 2, 4, 4]), T([2, 2, 3, 3])], [T([1, 2, 4, 4])], (b, [x, w]) => {
    b.returnOp([b.conv(x, w, [1, 1], [[1, 1], [1, 1]]).getResult(0)]);
  }), WasmTarget(), { fusion: { enabled: false } });
  const o = RuntimeTensor.zeros([1, 2, 4, 4]);
  r.run('cv', RuntimeTensor.fromArray(ones(32), [1, 2, 4, 4]), RuntimeTensor.fromArray(new Float32Array(36).fill(0.1), [2, 2, 3, 3]), o);
  return o.data.every(v => isFinite(v)) && o.data[0] > 0;
});

test('conv + batchnorm + silu', () => {
  const r = compileGraph(buildFunction('cbs', [T([1, 4, 8, 8]), T([4, 4, 3, 3]), T([4]), T([4]), T([4]), T([4])], [T([1, 4, 8, 8])], (b, [x, w, g, bt, m, v]) => {
    const c = b.conv(x, w, [1, 1], [[1, 1], [1, 1]]);
    const bn = b.batchnorm(c.getResult(0), g, bt, m, v, 1, 1e-5);
    b.returnOp([b.silu(bn.getResult(0)).getResult(0)]);
  }), WasmTarget(), { fusion: { enabled: false } });
  const o = RuntimeTensor.zeros([1, 4, 8, 8]);
  r.run('cbs', RuntimeTensor.fromArray(rand(256), [1, 4, 8, 8]), RuntimeTensor.fromArray(rand(144), [4, 4, 3, 3]), RuntimeTensor.fromArray(ones(4), [4]), RuntimeTensor.fromArray(zeros(4), [4]), RuntimeTensor.fromArray(zeros(4), [4]), RuntimeTensor.fromArray(ones(4), [4]), o);
  return o.data.every(v => isFinite(v));
});

test('attention (Q*K^T -> softmax -> *V)', () => {
  const r = compileGraph(buildFunction('at', [T([1, 4, 8]), T([1, 4, 8]), T([1, 4, 8])], [T([1, 4, 8])], (b, [Q, K, V]) => {
    const s = b.dot(Q, K, [2], [2], [0], [0]);
    const sm = b.softmax(s.getResult(0), -1);
    b.returnOp([b.dot(sm.getResult(0), V, [2], [1], [0], [0]).getResult(0)]);
  }), WasmTarget(), { fusion: { enabled: false } });
  const o = RuntimeTensor.zeros([1, 4, 8]);
  r.run('at', RuntimeTensor.fromArray(rand(32), [1, 4, 8]), RuntimeTensor.fromArray(rand(32), [1, 4, 8]), RuntimeTensor.fromArray(rand(32), [1, 4, 8]), o);
  return o.data.every(v => isFinite(v));
});

test('MLP 3-layer (matmul+gelu chain)', () => {
  const r = compileGraph(buildFunction('mlp', [T([4, 8]), T([8, 16]), T([16, 8]), T([8, 4])], [T([4, 4])], (b, [x, w1, w2, w3]) => {
    let c = b.gelu(b.matmul(x, w1).getResult(0)).getResult(0);
    c = b.gelu(b.matmul(c, w2).getResult(0)).getResult(0);
    b.returnOp([b.matmul(c, w3).getResult(0)]);
  }), WasmTarget(), { fusion: { enabled: false } });
  const o = RuntimeTensor.zeros([4, 4]);
  r.run('mlp', RuntimeTensor.fromArray(rand(32), [4, 8]), RuntimeTensor.fromArray(rand(128), [8, 16]), RuntimeTensor.fromArray(rand(128), [16, 8]), RuntimeTensor.fromArray(rand(32), [8, 4]), o);
  return o.data.every(v => isFinite(v));
});

test('reduce sum [3,4]->[3]', () => {
  const r = compileGraph(buildFunction('rs', [T([3, 4])], [T([3])], (b, [x]) => {
    b.returnOp([b.reduce(x, b.scalarConstant(0, f32).getResult(0), [1], 'sum').getResult(0)]);
  }), WasmTarget(), { fusion: { enabled: false } });
  const o = RuntimeTensor.zeros([3]);
  r.run('rs', RuntimeTensor.fromArray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [3, 4]), o);
  return o.data[0] === 10 && o.data[1] === 26 && o.data[2] === 42;
});

test('transpose [2,3,4]->[2,4,3]', () => {
  const r = compileGraph(buildFunction('tr', [T([2, 3, 4])], [T([2, 4, 3])], (b, [x]) => {
    b.returnOp([b.transpose(x, [0, 2, 1]).getResult(0)]);
  }), WasmTarget(), { fusion: { enabled: false } });
  const o = RuntimeTensor.zeros([2, 4, 3]);
  r.run('tr', RuntimeTensor.fromArray(Float32Array.from({ length: 24 }, (_, i) => i), [2, 3, 4]), o);
  return o.data[0] === 0 && o.data[1] === 4 && o.data[2] === 8;
});

test('quantized matmul', () => {
  const r = compileGraph(buildFunction('qm', [T([4, 8]), T([8, 4])], [T([4, 4])], (b, [x, w]) => {
    b.returnOp([b.matmul(x, w).getResult(0)]);
  }), WasmTarget({ supportsInt8: true }), { quantization: { enabled: true, weightOnly: false }, fusion: { enabled: false } });
  const o = RuntimeTensor.zeros([4, 4]);
  r.run('qm', RuntimeTensor.fromArray(new Float32Array(32).fill(0.5), [4, 8]), RuntimeTensor.fromArray(new Float32Array(32).fill(0.1), [8, 4]), o);
  return o.data.every(v => isFinite(v));
});

test('vectorized add [256]', () => {
  const r = compileGraph(buildFunction('va', [T([256]), T([256])], [T([256])], (b, [x, y]) => {
    b.returnOp([b.add(x, y).getResult(0)]);
  }), WasmTarget({ simd: true }), { scheduling: { enabled: true }, fusion: { enabled: false } });
  const o = RuntimeTensor.zeros([256]);
  r.run('va', RuntimeTensor.fromArray(Float32Array.from({ length: 256 }, (_, i) => i), [256]), RuntimeTensor.fromArray(new Float32Array(256).fill(1), [256]), o);
  return o.data[0] === 1 && o.data[255] === 256;
});

test('dynamic shape [DYNAMIC,4]', () => {
  const r = compileGraph(buildFunction('ds', [new TensorType([DYNAMIC, 4], f32), new TensorType([DYNAMIC, 4], f32)], [new TensorType([DYNAMIC, 4], f32)], (b, [x, y]) => {
    b.returnOp([b.add(x, y).getResult(0)]);
  }), WasmTarget(), { fusion: { enabled: false } });
  for (const batch of [1, 4, 8]) {
    const o = RuntimeTensor.zeros([batch, 4]);
    r.run('ds', RuntimeTensor.fromArray(new Float32Array(batch * 4).fill(1), [batch, 4]), RuntimeTensor.fromArray(new Float32Array(batch * 4).fill(2), [batch, 4]), o);
    if (!o.data.every(v => v === 3)) return false;
  }
  return true;
});

test('full transformer encoder block', () => {
  const B = 1, S = 4, D = 8, Dff = 16;
  const r = compileGraph(buildFunction('enc',
    [T([B, S, D]), T([D, D]), T([D, D]), T([D, D]), T([D, D]), T([D]), T([D]), T([D, Dff]), T([Dff, D]), T([D]), T([D])],
    [T([B, S, D])],
    (b, [x, wq, wk, wv, wo, ln1g, ln1b, w1, w2, ln2g, ln2b]) => {
      const flat = b.reshape(x, [B * S, D]);
      const Q = b.reshape(b.matmul(flat.getResult(0), wq).getResult(0), [B, S, D]);
      const K = b.reshape(b.matmul(flat.getResult(0), wk).getResult(0), [B, S, D]);
      const V = b.reshape(b.matmul(flat.getResult(0), wv).getResult(0), [B, S, D]);
      const scores = b.dot(Q.getResult(0), K.getResult(0), [2], [2], [0], [0]);
      const attn = b.softmax(scores.getResult(0), -1);
      const ctx = b.dot(attn.getResult(0), V.getResult(0), [2], [1], [0], [0]);
      const proj = b.reshape(b.matmul(b.reshape(ctx.getResult(0), [B * S, D]).getResult(0), wo).getResult(0), [B, S, D]);
      const res1 = b.add(x, proj.getResult(0));
      const ln1 = b.layernorm(res1.getResult(0), ln1g, ln1b, -1, 1e-5);
      const ff1 = b.reshape(b.matmul(b.reshape(ln1.getResult(0), [B * S, D]).getResult(0), w1).getResult(0), [B, S, Dff]);
      const act = b.gelu(ff1.getResult(0));
      const ff2 = b.reshape(b.matmul(b.reshape(act.getResult(0), [B * S, Dff]).getResult(0), w2).getResult(0), [B, S, D]);
      const res2 = b.add(ln1.getResult(0), ff2.getResult(0));
      b.returnOp([b.layernorm(res2.getResult(0), ln2g, ln2b, -1, 1e-5).getResult(0)]);
    }), WasmTarget(), { fusion: { enabled: false } });
  const inputs = [RuntimeTensor.fromArray(rand(B * S * D), [B, S, D]),
    ...Array(4).fill(null).map(() => RuntimeTensor.fromArray(rand(D * D), [D, D])),
    RuntimeTensor.fromArray(ones(D), [D]), RuntimeTensor.fromArray(zeros(D), [D]),
    RuntimeTensor.fromArray(rand(D * Dff), [D, Dff]), RuntimeTensor.fromArray(rand(Dff * D), [Dff, D]),
    RuntimeTensor.fromArray(ones(D), [D]), RuntimeTensor.fromArray(zeros(D), [D])];
  const o = RuntimeTensor.zeros([B, S, D]);
  r.run('enc', ...inputs, o);
  return o.data.every(v => isFinite(v));
});

console.log();
console.log(pass + '/' + (pass + fail) + ' WASM end-to-end');
if (fail > 0) process.exit(1);
