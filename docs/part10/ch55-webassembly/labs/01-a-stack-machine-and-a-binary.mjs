import {
  lowerToTir, toLIR, emit, encodeWat, compile,
  WasmTarget, randn, manual_seed,
} from '../../_internals.mjs';

manual_seed(55);

const tir = await lowerToTir((a, b) => a.mul(b), [randn([2, 3]), randn([2, 3])], WasmTarget());
const kernel = emit(toLIR(tir, WasmTarget()), WasmTarget());

console.log('=== a two-by-three elementwise multiply, as WAT ===\n');
console.log(kernel.source);

console.log('\n=== the flat memory it declares ===\n');
console.log(`  pages: ${kernel.metadata.memoryPages} x 65536 bytes`);
for (const [name, off] of kernel.metadata.bufferOffsets) {
  console.log(`  ${name.padEnd(8)} at byte ${String(off).padStart(4)}`);
}
console.log('\n  There are no named arrays in WebAssembly. Every buffer is a base');
console.log('  offset into one linear memory, and every access is that base plus the');
console.log('  flat index of Chapter 53 times the element size.');

console.log('\n=== an expression tree, flattened onto a stack ===\n');
const u = randn([4]);
const v = randn([4]);
const fused = compile({ forward: (a, b) => a.mul(b).add(a) }, [u, v], { target: WasmTarget() });
const body = (fused.source() ?? '').split('\n').map((l) => l.trim()).filter((l) => l && !l.startsWith('//'));
const guardAt = body.findIndex((l) => l.startsWith('br_if'));
const storeAt = body.findIndex((l) => l === 'f32.store');
console.log(body.slice(guardAt + 1, storeAt + 1).map((l) => `  ${l}`).join('\n'));
console.log('\n  Reading downward: the store address, then the multiply\'s two operand');
console.log('  addresses and loads, then `f32.mul`, then the third load, then');
console.log('  `f32.add`, then `f32.store`. It is a post-order walk of the same tree');
console.log('  the CPU backend renders as `x[i] * y[i] + x[i]`. Nothing is bracketed');
console.log('  because nothing needs to be: a stack machine has no precedence.');

console.log('\n=== a loop, without a goto ===\n');
const blockAt = body.findIndex((l) => l.startsWith('(block'));
console.log(body.slice(blockAt, blockAt + 6).map((l) => `  ${l}`).join('\n'));
console.log('  ...');
console.log(body.slice(storeAt + 1, storeAt + 6).map((l) => `  ${l}`).join('\n'));
console.log('\n  WebAssembly has no jump to an address. It has `block` and `loop`, which');
console.log('  push labels, and `br`/`br_if`, which name a label by its *depth* in that');
console.log('  stack. A `for` loop is therefore a `block` (whose end is the exit) around');
console.log('  a `loop` (whose start is the back edge), with the exit test inverted.');

const bytes = encodeWat(kernel.source);
console.log('\n=== and then the assembler, because a browser takes bytes ===\n');

const SECTION_NAMES = { 1: 'type', 2: 'import', 3: 'function', 5: 'memory', 7: 'export', 10: 'code' };
let p = 8;
console.log(`  magic + version              ${String(8).padStart(5)} bytes`);
while (p < bytes.length) {
  const start = p;
  const id = bytes[p++];
  let size = 0, shift = 0, b;
  do { b = bytes[p++]; size |= (b & 0x7f) << shift; shift += 7; } while (b & 0x80);
  p += size;
  console.log(`  section ${String(id).padStart(2)} ${(SECTION_NAMES[id] ?? '?').padEnd(20)} ${String(p - start).padStart(5)} bytes`);
}
console.log(`  ${'total'.padEnd(29)} ${String(bytes.length).padStart(5)} bytes`);

const uleb = (v) => { const r = []; do { let b = v & 0x7f; v >>>= 7; if (v) b |= 0x80; r.push(b); } while (v); return r; };
console.log('\n  Sizes and indices are LEB128, a variable-length encoding:');
for (const v of [0, 63, 64, 127, 128, 65536]) {
  console.log(`    ${String(v).padStart(6)} -> ${uleb(v).map((b) => b.toString(16).padStart(2, '0')).join(' ')}`);
}

const x = randn([2, 3]);
const y = randn([2, 3]);
const k = compile({ forward: (a, b) => a.mul(b) }, [x, y], { target: WasmTarget() });
const got = (await (await k(x, y)).toArray()).flat(9);
const want = (await x.mul(y).toArray()).flat(9);
console.log(`\n  the assembled module against eager: max err ${Math.max(...got.map((v, i) => Math.abs(v - want[i])))}`);

console.log('\n=== two things the assembler accepts that it should not ===\n');

const unknownInstr = `(module
  (memory (export "memory") 1 1)
  (func (export "k") (param i32)
    (local $t i32)
    (i32.const 7)
    i32.frobnicate
    local.set $t
  )
)`;
try {
  const b1 = encodeWat(unknownInstr);
  new WebAssembly.Module(b1);
  console.log(`  an unknown instruction: encoded to ${b1.length} bytes and validated.`);
  console.log('    `i32.frobnicate` is not in the instruction table, and the encoder\'s');
  console.log('    dispatch chain ends in `else if (INSTR.has(t))` with no else, so the');
  console.log('    token is dropped. The module is well formed and computes something');
  console.log('    other than what the text says.');
} catch (e) {
  console.log(`  an unknown instruction: rejected (${e.message.slice(0, 70)})`);
}

const unknownLabel = `(module
  (memory (export "memory") 1 1)
  (func (export "k") (param i32)
    (local $i i32)
    (block $outer
      (loop $inner
        (local.get $i)
        (i32.const 4)
        i32.ge_s
        br_if $nowhere
        (local.get $i)
        (i32.const 1)
        i32.add
        local.set $i
        br $inner
      )
    )
  )
)`;
try {
  const b2 = encodeWat(unknownLabel);
  new WebAssembly.Module(b2);
  console.log('\n  a branch to a label that was never opened: encoded and validated.');
  console.log('    `resolveBr` scans the label stack and returns 0 when it finds nothing,');
  console.log('    and depth 0 is the innermost enclosing block — so the branch goes');
  console.log('    somewhere legal and wrong instead of being an error.');
} catch (e) {
  console.log(`\n  a branch to an unopened label: rejected (${e.message.slice(0, 70)})`);
}

console.log('\n=== the flat layout has to be decided before the extents are known ===\n');
console.log('  DYNAMIC_BUFFER_SLAB_BYTES = 65536, so a dynamically-sized f32 buffer');
console.log('  is reserved 16384 elements of room in the linear memory.\n');
console.log(`  ${'elements'.padStart(9)}  ${'max err vs a hand-computed reference'}`);
for (const n of [4096, 16384, 16385, 20000]) {
  const t = randn([n]);
  const raw = await t.toArray();
  const ref = raw.map((v) => Math.fround(Math.fround(v * 2) + 1));
  try {
    const kd = compile({ forward: (a) => a.mul(2.0).add(1.0) }, [t], { target: WasmTarget(), dynamic_shapes: [true] });
    const out = await (await kd(t)).toArray();
    let e = 0;
    for (let j = 0; j < n; j++) e = Math.max(e, Math.abs(out[j] - ref[j]));
    console.log(`  ${String(n).padStart(9)}  ${e === 0 ? '0' : e.toExponential(3)}`);
  } catch (err) {
    console.log(`  ${String(n).padStart(9)}  threw: ${err.message.slice(0, 50)}`);
  }
}
console.log('\n  Correct to exactly 16384 elements and silently wrong past it. WASM is');
console.log('  the only backend that reads the LIR memory layout, so it is the only');
console.log('  one that pays. Run each size in a fresh process: the numbers above are');
console.log('  independent compilations.');
