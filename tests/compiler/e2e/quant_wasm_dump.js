import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { WasmTarget } from '../../../src/backend/target.js';
import { Compiler } from '../../../src/compiler/pipeline/compiler.js';
import { GraphModule } from '../../../src/compiler/ir/graph/module.js';
import { encodeWat } from '../../../src/backend/wasm/wat_encoder.js';

const f32 = ScalarType.F32;
const T = s => new TensorType(s, f32);
const mod = new GraphModule('m');
mod.addFunction(buildFunction('wq', [T([2, 4]), T([4, 2])], [T([2, 2])], (b, [x, w]) => {
  b.returnOp([b.matmul(x, w).getResult(0)]);
}));
const r = new Compiler({
  target: WasmTarget({ supportsInt8: true }),
  quantization: { enabled: true, weightOnly: false },
  fusion: { enabled: false },
  errorMode: 'resilient',
}).compile(mod);

const wat = r.module.kernels.get('wq')?.source;
const binary = encodeWat(wat);

const offset = 445;
const context = 20;
const start = Math.max(0, offset - context);
const end = Math.min(binary.length, offset + context);
const opcodeNames = {
  0x20: 'local.get', 0x21: 'local.set', 0x28: 'i32.load', 0x2a: 'f32.load',
  0x36: 'i32.store', 0x38: 'f32.store', 0x41: 'i32.const', 0x43: 'f32.const',
  0x45: 'i32.eqz', 0x46: 'i32.eq', 0x48: 'i32.lt_s', 0x4a: 'i32.gt_s',
  0x4e: 'i32.ge_s', 0x6a: 'i32.add', 0x6b: 'i32.sub', 0x6c: 'i32.mul',
  0x92: 'f32.add', 0x93: 'f32.sub', 0x94: 'f32.mul', 0x95: 'f32.div',
  0xa8: 'i32.trunc_f32_s', 0xb2: 'f32.convert_i32_s',
  0x0b: 'end', 0x0c: 'br', 0x0d: 'br_if', 0x02: 'block', 0x03: 'loop',
  0x10: 'call',
};

console.log('Binary around offset', offset, ':');
for (let i = start; i < end; i++) {
  const hex = binary[i].toString(16).padStart(2, '0');
  const name = opcodeNames[binary[i]] || '';
  const marker = i === offset ? ' <<<' : '';
  console.log('  +' + i + ': 0x' + hex + ' ' + name + marker);
}
