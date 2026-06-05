import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { WasmTarget, CPUTarget } from '../../../src/backend/target.js';
import { Compiler, compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { GraphModule } from '../../../src/compiler/ir/graph/module.js';
import { RuntimeTensor } from '../../../src/compiler/runtime/runtime.js';
import { encodeWat } from '../../../src/backend/wasm/wat_encoder.js';

const f32 = ScalarType.F32;
const T = s => new TensorType(s, f32);

console.log('=== CPU quantized accuracy ===');
const r1 = compileGraph(buildFunction('q', [T([4, 8]), T([8, 4])], [T([4, 4])], (b, [x, w]) => {
  b.returnOp([b.matmul(x, w).getResult(0)]);
}), CPUTarget({ supportsInt8: true }), { quantization: { enabled: true, weightOnly: false }, fusion: { enabled: false } });
const o1 = RuntimeTensor.zeros([4, 4]);
r1.run('q', RuntimeTensor.fromArray(new Float32Array(32).fill(0.5), [4, 8]),
  RuntimeTensor.fromArray(new Float32Array(32).fill(0.1), [8, 4]), o1);
console.log('Output[0]:', o1.data[0].toFixed(4), '| expect ~0.4 | error:', Math.abs(o1.data[0] - 0.4).toFixed(4));

console.log();
console.log('=== WASM quantized ===');
const mod = new GraphModule('m');
mod.addFunction(buildFunction('wq', [T([2, 4]), T([4, 2])], [T([2, 2])], (b, [x, w]) => {
  b.returnOp([b.matmul(x, w).getResult(0)]);
}));
const r2 = new Compiler({
  target: WasmTarget({ supportsInt8: true }),
  quantization: { enabled: true, weightOnly: false },
  fusion: { enabled: false },
  errorMode: 'resilient',
}).compile(mod);

if (r2.errors.length > 0) {
  console.log('Compile errors:', r2.errors.length);
  for (const e of r2.errors) console.log('  ', e.toString());

  const wat = r2.module.kernels.get('wq')?.source;
  if (wat) {
    const lines = wat.split('\n');
    const localLine = lines.find(l => l.includes('(local'));
    const localNames = [];
    const re = /\$([a-zA-Z_][a-zA-Z0-9_]*)/g;
    let m;
    while ((m = re.exec(localLine)) !== null) localNames.push(m[1]);
    console.log('Local names:', localNames.length);

    const bodyRefs = new Set();
    for (let i = 4; i < lines.length; i++) {
      const refs = lines[i].match(/\$([a-zA-Z_][a-zA-Z0-9_]*)/g);
      if (refs) for (const r of refs) {
        const n = r.replace('$', '');
        if (!n.startsWith('break_') && !n.startsWith('loop_') && !n.startsWith('math_')) bodyRefs.add(n);
      }
    }
    const missing = [...bodyRefs].filter(n => !localNames.includes(n));
    if (missing.length > 0) console.log('Missing locals:', missing);
    else console.log('All body refs found in locals');

    const bodyLines = wat.split('\n').slice(4);
    const bareSetGet = bodyLines.filter(l => {
      const t = l.trim();
      return (t === 'local.set' || t === 'local.get') || (t.startsWith('local.set ') && !t.includes('$')) || (t.startsWith('local.get ') && !t.includes('$'));
    });
    if (bareSetGet.length > 0) console.log('BARE local.set/get:', bareSetGet);
    else console.log('All local.set/get have $name');

    try {
      const binary = encodeWat(wat);
      console.log('Binary size:', binary.length);
      const wasmMod = new WebAssembly.Module(binary);
      console.log('WebAssembly.Module OK');
    } catch (e) {
      console.log('Encode/Module error:', e.message.substring(0, 120));
    }
  }
} else {
  console.log('Compiled OK');
  const o2 = RuntimeTensor.zeros([2, 2]);
  r2.run('wq', RuntimeTensor.fromArray(new Float32Array(8).fill(0.5), [2, 4]),
    RuntimeTensor.fromArray(new Float32Array(8).fill(0.1), [4, 2]), o2);
  console.log('Output:', [...o2.data].map(v => v.toFixed(4)));
}

console.log();
console.log('=== WASM where ===');
try {
  const r3 = compileGraph(buildFunction('ww', [T([4]), T([4]), T([4])], [T([4])], (b, [c, x, y]) => {
    b.returnOp([b.where(c, x, y).getResult(0)]);
  }), WasmTarget(), { fusion: { enabled: false } });
  const o3 = RuntimeTensor.zeros([4]);
  r3.run('ww', RuntimeTensor.fromArray([1, 0, 0, 1], [4]),
    RuntimeTensor.fromArray([10, 20, 30, 40], [4]),
    RuntimeTensor.fromArray([100, 200, 300, 400], [4]), o3);
  console.log('Output:', [...o3.data], '| expect [10,200,300,40]');
} catch (e) {
  console.log('ERROR:', e.message.substring(0, 120));
}
