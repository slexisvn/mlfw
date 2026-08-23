import {
  lowerToTir, toLIR, emit, encodeWat, compile, tensor,
  PrimFunc, ForNode, SeqNode, BufferStoreNode, BufferLoadNode,
  VariableNode, IntImmNode, MathOpNode, Buffer, ForKind,
  CPUTarget, CUDATarget, WasmTarget, WebGPUTarget, randn, manual_seed,
} from '../../_internals.mjs';

manual_seed(53);

const TARGETS = [['cpu', CPUTarget()], ['wasm', WasmTarget()], ['cuda', CUDATarget()], ['webgpu', WebGPUTarget()]];

console.log('=== one LIRFunc, four texts ===\n');
const tir = await lowerToTir((t) => t.mul(t), [randn([4, 6])], CPUTarget());
const lir = toLIR(tir, CPUTarget());

console.log(`  ${'target'.padEnd(8)} ${'lines'.padStart(5)}  where the row stride 6 shows up`);
for (const [label, target] of TARGETS) {
  const src = emit(lir, target).source;
  const lines = src.split('\n').map((l) => l.trim());
  const hit = label === 'wasm'
    ? `${lines.filter((l) => l === 'i32.mul').length} x i32.mul, ${lines.filter((l) => l === 'i32.add').length} x i32.add, over ${lines.filter((l) => l === '(i32.const 6)').length} x (i32.const 6)`
    : (lines.find((l) => l.includes('* 6')) ?? '(not found)');
  console.log(`  ${label.padEnd(8)} ${String(src.split('\n').length).padStart(5)}  ${hit.slice(0, 72)}`);
}
console.log('\n  No backend sees `buf[i, j]`. Each of them received `i * 6 + j` already');
console.log('  built, from one function, and renders it in its own syntax.');

const i = new VariableNode('i', 'index');
const A = new Buffer('a', [8], 'i32', 'global');
const B = new Buffer('b', [8], 'i32', 'global');
const OUT = new Buffer('out', [8], 'i32', 'global');
const pa = new VariableNode('pa', 'handle'), pb = new VariableNode('pb', 'handle'), po = new VariableNode('po', 'handle');

const divProgram = (op) => new PrimFunc(
  'divmod', [pa, pb, po],
  new ForNode(i, new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL,
    new SeqNode([new BufferStoreNode(OUT, [i],
      new MathOpNode(op, new BufferLoadNode(A, [i]), new BufferLoadNode(B, [i])))])),
  new Map([[pa, A], [pb, B], [po, OUT]]),
);

const OPS = ['//', '%', 'tdiv', 'tmod'];
const rhs = (src) => (src.split('\n').find((s) => s.includes('out[')) ?? '').trim()
  .replace(/^out\[i\] *= */, '').replace(/;$/, '');

console.log('\n=== the four integer operators, as each backend writes them ===\n');
console.log(`  ${'op'.padEnd(6)} ${'cpu'.padEnd(32)} ${'cuda'.padEnd(34)} wasm`);
for (const op of OPS) {
  const cpu = rhs(emit(toLIR(divProgram(op), CPUTarget()), CPUTarget()).source);
  const cuda = rhs(emit(toLIR(divProgram(op), CUDATarget()), CUDATarget()).source);
  const wasm = emit(toLIR(divProgram(op), WasmTarget()), WasmTarget()).source
    .split('\n').map((l) => l.trim()).filter((l) => /^i32\.|^select$/.test(l)).length;
  console.log(`  ${op.padEnd(6)} ${cpu.padEnd(32)} ${(cuda.slice(0, 32) + (cuda.length > 32 ? '..' : '')).padEnd(34)} ${wasm} integer instructions`);
}

const run = async (op, as, bs) => {
  const js = emit(toLIR(divProgram(op), CPUTarget()), CPUTarget()).source;
  const fn = new Function(`return ${js}`)();
  const o = new Int32Array(8);
  fn(Int32Array.from(as), Int32Array.from(bs), o);

  const wat = emit(toLIR(divProgram(op), WasmTarget()), WasmTarget());
  const inst = await WebAssembly.instantiate(await WebAssembly.compile(encodeWat(wat.source)), { math: {} });
  const mem = new Int32Array(inst.exports.memory.buffer);
  const off = wat.metadata.bufferOffsets;
  mem.set(as, off.get('a') / 4);
  mem.set(bs, off.get('b') / 4);
  inst.exports.divmod(0, 0, 0);
  return { js: [...o], wasm: [...mem.slice(off.get('out') / 4, off.get('out') / 4 + 8)] };
};

const as = [7, -7, 7, -7, 7, -7, 0, 5];
const bs = [3, 3, -3, -3, 0, 0, 0, 5];
const results = {};
for (const op of OPS) results[op] = await run(op, as, bs);

console.log('\n=== and what they compute, on the operands that separate the conventions ===\n');
console.log(`  ${'a'.padStart(3)} ${'b'.padStart(3)}  ` + OPS.map((o) => `${o} js`.padStart(8) + `${o} wasm`.padStart(10)).join(''));
for (let k = 0; k < 8; k++) {
  console.log(`  ${String(as[k]).padStart(3)} ${String(bs[k]).padStart(3)}  ` +
    OPS.map((o) => String(results[o].js[k]).padStart(8) + String(results[o].wasm[k]).padStart(10)).join(''));
}
const disagree = OPS.filter((o) => results[o].js.some((v, k) => v !== results[o].wasm[k]));
console.log(`\n  operators on which the two disagree anywhere above: ${disagree.length ? disagree.join(' ') : '(none)'}`);
console.log('  The last four rows have a zero divisor and both columns read 0 —');
console.log('  but for different reasons. JavaScript evaluates Infinity and NaN and');
console.log('  the i32 store flattens both to 0; the WASM emitter cannot let that');
console.log('  happen, because i32.div_s traps on a zero divisor and would abort the');
console.log('  whole module, so it emits a guard that selects 0 explicitly.');

console.log('\n=== the one operator on which the four backends do not agree ===\n');
const x = tensor([[-7, 7, -7.5, 7.5]]);
const y = tensor([[3, 3, 2, 2]]);
const row = (label, v) => console.log(`  ${label.padEnd(8)} ` + v.map((n) => String(n).padStart(9)).join(''));
console.log(`  ${'source'.padEnd(8)} ` + ['-7 % 3', '7 % 3', '-7.5 % 2', '7.5 % 2'].map((h) => h.padStart(9)).join(''));
row('eager', (await x.remainder(y).toArray()).flat(9));
for (const [label, target] of [['cpu', CPUTarget()], ['wasm', WasmTarget()]]) {
  const k = compile({ forward: (p, q) => p.remainder(q) }, [x, y], { target });
  row(label, (await (await k(x, y)).toArray()).flat(9));
}
for (const [label, target] of [['cuda', CUDATarget()], ['webgpu', WebGPUTarget()]]) {
  const k = compile({ forward: (p, q) => p.remainder(q) }, [x, y], { target });
  const line = ((k.source() ?? '').split('\n').find((s) => /fmod|%/.test(s) && s.includes('buf_')) ?? '').trim();
  console.log(`  ${label.padEnd(8)} emits ${line.replace(/^buf_\d+\[[^\]]*\] *= */, '').replace(/;$/, '')}`);
}
console.log('\n  `remainder` is floor-mod in eager and on CPU, truncating on the other');
console.log('  three. The integer operators above were reconciled to one definition;');
console.log('  this one travels a different road — a CallExternNode named `fmod` —');
console.log('  and each backend maps that name onto its own language\'s primitive.');
