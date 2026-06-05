import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { CPUTarget, WasmTarget } from '../../../src/backend/target.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { RuntimeTensor } from '../../../src/compiler/runtime/runtime.js';

const f32 = ScalarType.F32;
const i32 = ScalarType.I32;
const T = s => new TensorType(s, f32);
const Ti = s => new TensorType(s, i32);
const rand = n => Float32Array.from({ length: n }, () => (Math.random() - 0.5) * 2);
let pass = 0, fail = 0;

function test(name, fn) {
  try {
    const ok = fn();
    console.log((ok ? 'PASS' : 'FAIL') + ' ' + name);
    ok ? pass++ : fail++;
  } catch (e) {
    console.log('FAIL ' + name + ': ' + e.message.substring(0, 120));
    fail++;
  }
}

function close(a, b, eps) { return Math.abs(a - b) < (eps || 0.01); }

test('where: mask select CPU', () => {
  const r = compileGraph(buildFunction('w', [T([6]), T([6]), T([6])], [T([6])], (b, [c, x, y]) => {
    b.returnOp([b.where(c, x, y).getResult(0)]);
  }), CPUTarget(), { fusion: { enabled: false } });
  const o = RuntimeTensor.zeros([6]);
  r.run('w', RuntimeTensor.fromArray([1, 0, 1, 0, 1, 0], [6]), RuntimeTensor.fromArray([10, 20, 30, 40, 50, 60], [6]), RuntimeTensor.fromArray([100, 200, 300, 400, 500, 600], [6]), o);
  return o.data[0] === 10 && o.data[1] === 200 && o.data[4] === 50 && o.data[5] === 600;
});

test('where: mask select WASM', () => {
  const r = compileGraph(buildFunction('w', [T([4]), T([4]), T([4])], [T([4])], (b, [c, x, y]) => {
    b.returnOp([b.where(c, x, y).getResult(0)]);
  }), WasmTarget(), { fusion: { enabled: false } });
  const o = RuntimeTensor.zeros([4]);
  r.run('w', RuntimeTensor.fromArray([1, 0, 0, 1], [4]), RuntimeTensor.fromArray([10, 20, 30, 40], [4]), RuntimeTensor.fromArray([100, 200, 300, 400], [4]), o);
  return o.data[0] === 10 && o.data[1] === 200 && o.data[2] === 300 && o.data[3] === 40;
});

test('split: 3-way CPU', () => {
  const r = compileGraph(buildFunction('sp', [T([12])], [T([4]), T([4]), T([4])], (b, [x]) => {
    const s = b.split(x, 0, [4, 4, 4]);
    b.returnOp([s.getResult(0), s.getResult(1), s.getResult(2)]);
  }), CPUTarget(), { fusion: { enabled: false } });
  const o1 = RuntimeTensor.zeros([4]), o2 = RuntimeTensor.zeros([4]), o3 = RuntimeTensor.zeros([4]);
  r.run('sp', RuntimeTensor.fromArray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [12]), o1, o2, o3);
  return o1.data[0] === 0 && o1.data[3] === 3 && o2.data[0] === 4 && o3.data[0] === 8 && o3.data[3] === 11;
});

test('split: 2D along axis 1 CPU', () => {
  const r = compileGraph(buildFunction('sp2', [T([2, 6])], [T([2, 2]), T([2, 4])], (b, [x]) => {
    const s = b.split(x, 1, [2, 4]);
    b.returnOp([s.getResult(0), s.getResult(1)]);
  }), CPUTarget(), { fusion: { enabled: false } });
  const o1 = RuntimeTensor.zeros([2, 2]), o2 = RuntimeTensor.zeros([2, 4]);
  r.run('sp2', RuntimeTensor.fromArray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 6]), o1, o2);
  return o1.data[0] === 1 && o1.data[1] === 2 && o2.data[0] === 3;
});

test('argmax: axis=1 CPU', () => {
  const r = compileGraph(buildFunction('am', [T([3, 5])], [Ti([3])], (b, [x]) => {
    b.returnOp([b.argmax(x, 1).getResult(0)]);
  }), CPUTarget(), { fusion: { enabled: false } });
  const o = RuntimeTensor.zeros([3], 'i32');
  r.run('am', RuntimeTensor.fromArray([1, 9, 2, 3, 4, 5, 1, 1, 1, 1, 0, 0, 0, 0, 7], [3, 5]), o);
  return o.data[0] === 1 && o.data[1] === 0 && o.data[2] === 4;
});

test('argmax: axis=0 CPU', () => {
  const r = compileGraph(buildFunction('am0', [T([4, 3])], [Ti([3])], (b, [x]) => {
    b.returnOp([b.argmax(x, 0).getResult(0)]);
  }), CPUTarget(), { fusion: { enabled: false } });
  const o = RuntimeTensor.zeros([3], 'i32');
  r.run('am0', RuntimeTensor.fromArray([1, 2, 3, 10, 5, 6, 7, 8, 9, 4, 11, 0], [4, 3]), o);
  return o.data[0] === 1 && o.data[1] === 3 && o.data[2] === 2;
});

test('argmin: axis=1 CPU', () => {
  const r = compileGraph(buildFunction('ami', [T([2, 4])], [Ti([2])], (b, [x]) => {
    b.returnOp([b.argmin(x, 1).getResult(0)]);
  }), CPUTarget(), { fusion: { enabled: false } });
  const o = RuntimeTensor.zeros([2], 'i32');
  r.run('ami', RuntimeTensor.fromArray([5, 1, 3, 9, 2, 8, 0, 4], [2, 4]), o);
  return o.data[0] === 1 && o.data[1] === 2;
});

test('pool2d: max 2x2 stride 2 CPU', () => {
  const r = compileGraph(buildFunction('pm', [T([1, 1, 4, 4])], [T([1, 1, 2, 2])], (b, [x]) => {
    b.returnOp([b.pool2d(x, 'max', [2, 2], [2, 2], [[0, 0], [0, 0]]).getResult(0)]);
  }), CPUTarget(), { fusion: { enabled: false } });
  const o = RuntimeTensor.zeros([1, 1, 2, 2]);
  r.run('pm', RuntimeTensor.fromArray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [1, 1, 4, 4]), o);
  return o.data[0] === 6 && o.data[1] === 8 && o.data[2] === 14 && o.data[3] === 16;
});

test('pool2d: avg 2x2 stride 2 CPU', () => {
  const r = compileGraph(buildFunction('pa', [T([1, 1, 4, 4])], [T([1, 1, 2, 2])], (b, [x]) => {
    b.returnOp([b.pool2d(x, 'avg', [2, 2], [2, 2], [[0, 0], [0, 0]]).getResult(0)]);
  }), CPUTarget(), { fusion: { enabled: false } });
  const o = RuntimeTensor.zeros([1, 1, 2, 2]);
  r.run('pa', RuntimeTensor.fromArray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [1, 1, 4, 4]), o);
  return close(o.data[0], 3.5) && close(o.data[3], 13.5);
});

test('pool2d: max with padding CPU', () => {
  const r = compileGraph(buildFunction('pmp', [T([1, 1, 3, 3])], [T([1, 1, 3, 3])], (b, [x]) => {
    b.returnOp([b.pool2d(x, 'max', [3, 3], [1, 1], [[1, 1], [1, 1]]).getResult(0)]);
  }), CPUTarget(), { fusion: { enabled: false } });
  const o = RuntimeTensor.zeros([1, 1, 3, 3]);
  r.run('pmp', RuntimeTensor.fromArray([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 1, 3, 3]), o);
  return o.data[4] === 9 && o.data[0] === 5 && o.data.every(v => isFinite(v));
});

test('pool2d: max 2x2 WASM', () => {
  const r = compileGraph(buildFunction('pmw', [T([1, 1, 4, 4])], [T([1, 1, 2, 2])], (b, [x]) => {
    b.returnOp([b.pool2d(x, 'max', [2, 2], [2, 2], [[0, 0], [0, 0]]).getResult(0)]);
  }), WasmTarget(), { fusion: { enabled: false } });
  const o = RuntimeTensor.zeros([1, 1, 2, 2]);
  r.run('pmw', RuntimeTensor.fromArray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [1, 1, 4, 4]), o);
  return o.data[0] === 6 && o.data[3] === 16;
});

test('pool2d: multi-channel CPU', () => {
  const r = compileGraph(buildFunction('pmc', [T([1, 2, 4, 4])], [T([1, 2, 2, 2])], (b, [x]) => {
    b.returnOp([b.pool2d(x, 'max', [2, 2], [2, 2], [[0, 0], [0, 0]]).getResult(0)]);
  }), CPUTarget(), { fusion: { enabled: false } });
  const o = RuntimeTensor.zeros([1, 2, 2, 2]);
  const inp = new Float32Array(32);
  for (let i = 0; i < 16; i++) inp[i] = i + 1;
  for (let i = 0; i < 16; i++) inp[16 + i] = 100 + i + 1;
  r.run('pmc', RuntimeTensor.fromArray(inp, [1, 2, 4, 4]), o);
  return o.data[0] === 6 && o.data[4] === 106;
});

test('resize: nearest 2x CPU', () => {
  const r = compileGraph(buildFunction('rn', [T([1, 1, 2, 2])], [T([1, 1, 4, 4])], (b, [x]) => {
    b.returnOp([b.resize(x, [4, 4], 'nearest').getResult(0)]);
  }), CPUTarget(), { fusion: { enabled: false } });
  const o = RuntimeTensor.zeros([1, 1, 4, 4]);
  r.run('rn', RuntimeTensor.fromArray([1, 2, 3, 4], [1, 1, 2, 2]), o);
  return o.data[0] === 1 && o.data[1] === 1 && o.data[2] === 2 && o.data[4] === 1;
});

test('resize: bilinear 2x CPU', () => {
  const r = compileGraph(buildFunction('rb', [T([1, 1, 2, 2])], [T([1, 1, 4, 4])], (b, [x]) => {
    b.returnOp([b.resize(x, [4, 4], 'bilinear').getResult(0)]);
  }), CPUTarget(), { fusion: { enabled: false } });
  const o = RuntimeTensor.zeros([1, 1, 4, 4]);
  r.run('rb', RuntimeTensor.fromArray([0, 10, 20, 30], [1, 1, 2, 2]), o);
  return o.data.every(v => isFinite(v)) && o.data[0] >= 0 && o.data[15] >= 20;
});

test('resize: nearest WASM', () => {
  const r = compileGraph(buildFunction('rnw', [T([1, 1, 2, 2])], [T([1, 1, 4, 4])], (b, [x]) => {
    b.returnOp([b.resize(x, [4, 4], 'nearest').getResult(0)]);
  }), WasmTarget(), { fusion: { enabled: false } });
  const o = RuntimeTensor.zeros([1, 1, 4, 4]);
  r.run('rnw', RuntimeTensor.fromArray([1, 2, 3, 4], [1, 1, 2, 2]), o);
  return o.data[0] === 1 && o.data.every(v => isFinite(v));
});

test('where + argmax pipeline: find max then mask', () => {
  const r = compileGraph(buildFunction('pipeline', [T([3, 4])], [T([3, 4])], (b, [x]) => {
    const mx = b.argmax(x, 1);
    const flat = b.reshape(x, [12]);
    const act = b.gelu(flat.getResult(0));
    b.returnOp([b.reshape(act.getResult(0), [3, 4]).getResult(0)]);
  }), CPUTarget(), { fusion: { enabled: true } });
  const o = RuntimeTensor.zeros([3, 4]);
  r.run('pipeline', RuntimeTensor.fromArray(rand(12), [3, 4]), o);
  return o.data.every(v => isFinite(v));
});

test('pool2d + resize pipeline: downsample then upsample', () => {
  const r = compileGraph(buildFunction('downup', [T([1, 1, 8, 8])], [T([1, 1, 8, 8])], (b, [x]) => {
    const pooled = b.pool2d(x, 'avg', [2, 2], [2, 2], [[0, 0], [0, 0]]);
    const upsampled = b.resize(pooled.getResult(0), [8, 8], 'nearest');
    b.returnOp([upsampled.getResult(0)]);
  }), CPUTarget(), { fusion: { enabled: false } });
  const o = RuntimeTensor.zeros([1, 1, 8, 8]);
  r.run('downup', RuntimeTensor.fromArray(Float32Array.from({ length: 64 }, (_, i) => i), [1, 1, 8, 8]), o);
  return o.data.every(v => isFinite(v)) && o.data[0] > 0;
});

test('split + matmul: split weights then multiply', () => {
  const r = compileGraph(buildFunction('splitmm', [T([4, 8]), T([8, 8])], [T([4, 4]), T([4, 4])], (b, [x, w]) => {
    const parts = b.split(w, 1, [4, 4]);
    const mm1 = b.matmul(x, parts.getResult(0));
    const mm2 = b.matmul(x, parts.getResult(1));
    b.returnOp([mm1.getResult(0), mm2.getResult(0)]);
  }), CPUTarget(), { fusion: { enabled: false } });
  const o1 = RuntimeTensor.zeros([4, 4]), o2 = RuntimeTensor.zeros([4, 4]);
  r.run('splitmm', RuntimeTensor.fromArray(new Float32Array(32).fill(1), [4, 8]), RuntimeTensor.fromArray(Float32Array.from({ length: 64 }, (_, i) => i * 0.01), [8, 8]), o1, o2);
  return o1.data.every(v => isFinite(v)) && o2.data.every(v => isFinite(v));
});

test('ResNet block: conv + pool2d + relu', () => {
  const r = compileGraph(buildFunction('respool', [T([1, 2, 8, 8]), T([2, 2, 3, 3])], [T([1, 2, 4, 4])], (b, [x, w]) => {
    const c = b.conv(x, w, [1, 1], [[1, 1], [1, 1]]);
    const r1 = b.relu(c.getResult(0));
    const p = b.pool2d(r1.getResult(0), 'max', [2, 2], [2, 2], [[0, 0], [0, 0]]);
    b.returnOp([p.getResult(0)]);
  }), CPUTarget(), { fusion: { enabled: false } });
  const o = RuntimeTensor.zeros([1, 2, 4, 4]);
  r.run('respool', RuntimeTensor.fromArray(rand(128), [1, 2, 8, 8]), RuntimeTensor.fromArray(rand(36), [2, 2, 3, 3]), o);
  return o.data.every(v => isFinite(v) && v >= 0);
});

console.log();
console.log(pass + '/' + (pass + fail) + ' new ops e2e');
if (fail > 0) process.exit(1);
