import {
  lowerToTir, toLIR, printTensorIR, verifyLIR, detectAccumulator,
  CPUTarget, randn, manual_seed,
} from '../../_internals.mjs';

manual_seed(53);

const tir = await lowerToTir((a, b) => a.matmul(b), [randn([4, 6]), randn([6, 5])], CPUTarget());
const lir = toLIR(tir, CPUTarget());

const census = (root) => {
  const seen = new Map();
  const slots = ['body', 'stmts', 'value', 'thenBody', 'elseBody', 'initBody', 'condBody', 'loopBody',
    'a', 'b', 'expr', 'offsetExpr', 'initLoad', 'flushStore', 'args', 'indices', 'extent', 'condition'];
  const walk = (n) => {
    if (!n || typeof n !== 'object' || !n.type) return;
    seen.set(n.type, (seen.get(n.type) ?? 0) + 1);
    if (n.type === 'BlockNode') for (const iv of n.iterVars ?? []) walk(iv.binding);
    for (const s of slots) { const v = n[s]; Array.isArray(v) ? v.forEach(walk) : walk(v); }
  };
  walk(root);
  return seen;
};

const before = census(tir.body);
const after = census(lir.body);
const kinds = [...new Set([...before.keys(), ...after.keys()])].sort();

console.log('=== node kinds, before and after lowering to LIR ===\n');
console.log(`  ${'node kind'.padEnd(22)} ${'TIR'.padStart(4)} ${'LIR'.padStart(4)}`);
for (const k of kinds) {
  const b = before.get(k) ?? 0, a = after.get(k) ?? 0;
  if (b === a) continue;
  console.log(`  ${k.padEnd(22)} ${String(b).padStart(4)} ${String(a).padStart(4)}`);
}
console.log('\n  unchanged:', kinds.filter((k) => (before.get(k) ?? 0) === (after.get(k) ?? 0)).join(' '));

console.log('\n=== the reduction loop became a register ===\n');
const acc = [...census(lir.body).keys()].includes('LIRAccumulatorNode');
console.log(`  the matmul's k loop is an LIRAccumulatorNode: ${acc}`);

const forNodes = [];
(function collect(n) {
  if (!n || typeof n !== 'object' || !n.type) return;
  if (n.type === 'ForNode') forNodes.push(n);
  for (const s of ['body', 'stmts']) { const v = n[s]; Array.isArray(v) ? v.forEach(collect) : collect(v); }
})(tir.body);

for (const f of forNodes) {
  const info = detectAccumulator(f);
  console.log(`  for ${f.loopVar.name.padEnd(9)} extent ${String(f.extent.value ?? '?').padStart(2)}  ` +
    (info ? `accumulator over '${info.op}'` : 'not an accumulator'));
}

console.log('\n=== the metadata every backend reads instead of re-deriving it ===\n');
const m = lir.metadata;
console.log(`  locals            ${m.locals.size} (${[...m.locals].filter(([, d]) => d !== 'i32').map(([n, d]) => `${n}:${d}`).join(', ') || 'all i32'})`);
console.log(`  buffer offsets    ${[...m.memoryLayout.bufferOffsets].map(([n, o]) => `${n}@${o}`).join('  ')}`);
console.log(`  total bytes       ${m.memoryLayout.totalBytes} (alignment ${m.memoryLayout.alignment})`);
console.log(`  extern calls      ${m.externCalls.size === 0 ? '(none)' : [...m.externCalls.keys()].join(' ')}`);
console.log(`  zero buffers      ${m.zeroBuffers.size === 0 ? '(none)' : [...m.zeroBuffers].join(' ')}`);
console.log(`  param buffers     ${[...m.paramBuffers].join(' ')}`);
console.log(`  allocated buffers ${m.allocatedBuffers.size === 0 ? '(none)' : [...m.allocatedBuffers].join(' ')}`);

console.log('\n=== the subscript is gone, and it does not come back ===\n');
const tirLine = printTensorIR(tir).split('\n').find((l) => l.includes('buf_5[vls0_9, vrs0_10] ='));
console.log(`  TIR:  ${tirLine.trim()}`);
let flat = null;
(function find(n) {
  if (!n || typeof n !== 'object' || !n.type || flat) return;
  if (n.type === 'LIRAccumulatorNode') { flat = n; return; }
  for (const s of ['body', 'stmts']) { const v = n[s]; Array.isArray(v) ? v.forEach(find) : find(v); }
})(lir.body);
const render = (n) => {
  if (!n) return '?';
  if (n.type === 'IntImmNode' || n.type === 'FloatImmNode') return String(n.value);
  if (n.type === 'VariableNode') return n.name;
  if (n.type === 'MathOpNode') return n.b ? `(${render(n.a)} ${n.op} ${render(n.b)})` : `${n.op}${render(n.a)}`;
  if (n.type === 'LIRFlatLoadNode') return `${n.buffer.name}[${render(n.offsetExpr)}]`;
  return n.type;
};
console.log(`  LIR:  ${flat.flushStore.buffer.name}[${render(flat.flushStore.offsetExpr)}] = ${flat.localName}`);
console.log(`        ${flat.localName} = (${flat.localName} ${flat.op} ${render(flat.body)})   over ${flat.loopVar.name} < ${render(flat.extent)}`);
console.log('\n  Two subscripts became one integer. Nothing downstream can ask which');
console.log('  loop variable indexed which axis, so every analysis that needs to know');
console.log('  — dependence, tiling, buffer lifetimes — has already run.');

console.log('\n=== and the result verifies ===\n');
const errors = verifyLIR(lir);
console.log(`  verifyLIR: ${errors.length === 0 ? 'no errors' : errors.map((e) => e.toString()).join('\n')}`);
