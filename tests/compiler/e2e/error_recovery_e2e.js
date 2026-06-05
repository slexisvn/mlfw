import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { GraphModule } from '../../../src/compiler/ir/graph/module.js';
import { CPUTarget, WasmTarget } from '../../../src/backend/target.js';
import { Compiler, compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { TraceLevel } from '../../../src/compiler/pipeline/trace.js';
import { RuntimeTensor } from '../../../src/compiler/runtime/runtime.js';

const f32 = ScalarType.F32;
const T = s => new TensorType(s, f32);
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

test('resilient: 3 good + 1 bad function → 3 compiled, 1 error', () => {
  const mod = new GraphModule('mixed');
  mod.addFunction(buildFunction('add_fn', [T([8]), T([8])], [T([8])], (b, [x, y]) => {
    b.returnOp([b.add(x, y).getResult(0)]);
  }));
  mod.addFunction(buildFunction('exp_fn', [T([8])], [T([8])], (b, [x]) => {
    b.returnOp([b.exp(x).getResult(0)]);
  }));
  mod.addFunction(buildFunction('bad_fn', [T([4])], [T([4])], (b, [x]) => {
    b._buildOp('totally_fake_op', [x], [x.type]);
    b.returnOp([x]);
  }));
  mod.addFunction(buildFunction('matmul_fn', [T([4, 8]), T([8, 4])], [T([4, 4])], (b, [x, w]) => {
    b.returnOp([b.matmul(x, w).getResult(0)]);
  }));

  const r = new Compiler({ target: CPUTarget(), errorMode: 'resilient', fusion: { enabled: false } }).compile(mod);

  if (r.succeeded) return false;
  if (r.errors.length === 0) return false;
  if (!r.failedFunctions.has('bad_fn')) return false;
  if (r.failedFunctions.has('add_fn')) return false;
  if (!r.listKernels().includes('add_fn')) return false;
  if (!r.listKernels().includes('exp_fn')) return false;
  if (!r.listKernels().includes('matmul_fn')) return false;
  if (r.listKernels().includes('bad_fn')) return false;

  const o1 = RuntimeTensor.zeros([8]);
  r.run('add_fn', RuntimeTensor.fromArray([1, 2, 3, 4, 5, 6, 7, 8], [8]), RuntimeTensor.fromArray([10, 20, 30, 40, 50, 60, 70, 80], [8]), o1);
  if (o1.data[0] !== 11 || o1.data[7] !== 88) return false;

  const o2 = RuntimeTensor.zeros([8]);
  r.run('exp_fn', RuntimeTensor.fromArray(new Float32Array(8).fill(0), [8]), o2);
  if (Math.abs(o2.data[0] - 1) > 0.01) return false;

  const o3 = RuntimeTensor.zeros([4, 4]);
  r.run('matmul_fn', RuntimeTensor.fromArray(new Float32Array(32).fill(1), [4, 8]), RuntimeTensor.fromArray(new Float32Array(32).fill(0.5), [8, 4]), o3);
  if (Math.abs(o3.data[0] - 4.0) > 0.01) return false;

  return true;
});

test('resilient: all bad → 0 compiled, multiple errors', () => {
  const mod = new GraphModule('allbad');
  mod.addFunction(buildFunction('bad1', [T([4])], [T([4])], (b, [x]) => {
    b._buildOp('fake_op_1', [x], [x.type]); b.returnOp([x]);
  }));
  mod.addFunction(buildFunction('bad2', [T([4])], [T([4])], (b, [x]) => {
    b._buildOp('fake_op_2', [x], [x.type]); b.returnOp([x]);
  }));
  const r = new Compiler({ target: CPUTarget(), errorMode: 'resilient', fusion: { enabled: false } }).compile(mod);
  return !r.succeeded && r.errors.length >= 2 && r.listKernels().length === 0;
});

test('resilient: all good → 0 errors, succeeded=true', () => {
  const mod = new GraphModule('allgood');
  mod.addFunction(buildFunction('f1', [T([4])], [T([4])], (b, [x]) => { b.returnOp([b.neg(x).getResult(0)]); }));
  mod.addFunction(buildFunction('f2', [T([4])], [T([4])], (b, [x]) => { b.returnOp([b.exp(x).getResult(0)]); }));
  const r = new Compiler({ target: CPUTarget(), errorMode: 'resilient', fusion: { enabled: false } }).compile(mod);
  return r.succeeded && r.errors.length === 0 && r.listKernels().length === 2;
});

test('strict mode: throws on first error', () => {
  const mod = new GraphModule('strict');
  mod.addFunction(buildFunction('good', [T([4])], [T([4])], (b, [x]) => { b.returnOp([b.neg(x).getResult(0)]); }));
  mod.addFunction(buildFunction('bad', [T([4])], [T([4])], (b, [x]) => { b._buildOp('nope', [x], [x.type]); b.returnOp([x]); }));
  try {
    new Compiler({ target: CPUTarget(), errorMode: 'strict', fusion: { enabled: false } }).compile(mod);
    return false;
  } catch (e) {
    return e.message.length > 0;
  }
});

test('resilient: error objects have correct structure', () => {
  const mod = new GraphModule('struct');
  mod.addFunction(buildFunction('fail_fn', [T([4])], [T([4])], (b, [x]) => {
    b._buildOp('unknown_op', [x], [x.type]); b.returnOp([x]);
  }));
  const ev = [];
  const r = new Compiler({
    target: CPUTarget(), errorMode: 'resilient', fusion: { enabled: false },
    trace: { level: TraceLevel.INFO, sink: e => ev.push(e) },
  }).compile(mod);
  if (r.errors.length === 0) return false;
  const err = r.errors[0];
  if (!err.phase) return false;
  if (!err.funcName) return false;
  if (!err.message) return false;
  if (typeof err.toString() !== 'string') return false;
  const errEvents = ev.filter(e => e.type === 'error');
  return errEvents.length > 0 && errEvents[0].funcName === 'fail_fn';
});

test('resilient: WASM target, good+bad → good compiles and runs', () => {
  const mod = new GraphModule('wasm_mix');
  mod.addFunction(buildFunction('wasm_good', [T([4]), T([4])], [T([4])], (b, [x, y]) => {
    b.returnOp([b.add(x, y).getResult(0)]);
  }));
  mod.addFunction(buildFunction('wasm_bad', [T([4])], [T([4])], (b, [x]) => {
    b._buildOp('fake_wasm_op', [x], [x.type]); b.returnOp([x]);
  }));
  const r = new Compiler({ target: WasmTarget(), errorMode: 'resilient', fusion: { enabled: false } }).compile(mod);
  if (r.succeeded) return false;
  if (!r.listKernels().includes('wasm_good')) return false;
  const o = RuntimeTensor.zeros([4]);
  r.run('wasm_good', RuntimeTensor.fromArray([1, 2, 3, 4], [4]), RuntimeTensor.fromArray([10, 20, 30, 40], [4]), o);
  return o.data[0] === 11 && o.data[3] === 44;
});

test('resilient: fusion + decomposition + bad function', () => {
  const mod = new GraphModule('complex');
  mod.addFunction(buildFunction('transformer_like', [T([2, 4, 8]), T([8]), T([8])], [T([2, 4, 8])], (b, [x, g, beta]) => {
    const ln = b.layernorm(x, g, beta, -1, 1e-5);
    const sm = b.softmax(ln.getResult(0), -1);
    b.returnOp([b.gelu(sm.getResult(0)).getResult(0)]);
  }));
  mod.addFunction(buildFunction('broken', [T([4])], [T([4])], (b, [x]) => {
    b._buildOp('invalid_activation', [x], [x.type]); b.returnOp([x]);
  }));
  const r = new Compiler({ target: CPUTarget(), errorMode: 'resilient', fusion: { enabled: true } }).compile(mod);
  if (!r.failedFunctions.has('broken')) return false;
  if (!r.listKernels().includes('transformer_like')) return false;
  const o = RuntimeTensor.zeros([2, 4, 8]);
  r.run('transformer_like',
    RuntimeTensor.fromArray(Float32Array.from({ length: 64 }, () => Math.random() - 0.5), [2, 4, 8]),
    RuntimeTensor.fromArray(new Float32Array(8).fill(1), [8]),
    RuntimeTensor.fromArray(new Float32Array(8).fill(0), [8]),
    o);
  return o.data.every(v => isFinite(v));
});

console.log();
console.log(pass + '/' + (pass + fail) + ' error recovery e2e');
if (fail > 0) process.exit(1);
