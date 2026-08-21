import {
  lowerToTir, Schedule, randn,
} from '../../_internals.mjs';

// A block scope is the second index the schedule keeps: a producer-consumer
// graph over the sibling blocks of one scope, with a dependence kind and a
// buffer on every edge. This lab builds one and reads it.

const sch = new Schedule(await lowerToTir((x) => x.mul(x).sum(1), [randn([4, 6])]));

// `state.scopes` is a lazy getter; touching it is what runs buildBlockScopes.
const scopes = sch.state.scopes;

console.log('=== the scopes of one function ===\n');
for (const [root, scope] of scopes) {
  const name = root ? `block ${root.node.name}` : 'the function body';
  console.log(`  scope rooted at ${name}`);
  console.log(`    members        : ${scope.children.map((s) => s.node.name).join(', ') || '(none)'}`);
  console.log(`    opaque accesses: ${scope.opaqueAccesses.length}`);
  console.log(`    stagePipeline  : ${scope.stagePipeline}`);
}

const scope = scopes.get(null);

console.log('\n=== the dependence edges between siblings ===\n');
for (const d of scope.deps) {
  console.log(`  ${d.src.node.name.padEnd(14)} -> ${d.dst.node.name.padEnd(14)} ${d.kind.padEnd(4)} on ${d.buffer.name}`);
}

console.log('\n=== the same edges, read as a producer-consumer graph ===\n');
for (const m of scope.children) {
  const p = scope.producersOf(m).map((s) => s.node.name);
  const c = scope.consumersOf(m).map((s) => s.node.name);
  console.log(`  ${m.node.name.padEnd(14)} producers: ${(p.join(', ') || '—').padEnd(28)} consumers: ${c.join(', ') || '—'}`);
}

console.log('\n=== what the scope records about each member ===\n');
for (const m of scope.children) {
  const info = scope.blockInfo(m);
  console.log(`  ${m.node.name.padEnd(14)} affineBinding=${info.affineBinding}  regionCover=${info.regionCover}`);
}

console.log('\n  `stagePipeline` is false, and the reason is the one opaque access:');
console.log('  `buf_4[] = 0`, the scalar constant the reduction initialises from,');
console.log('  is a store at the top of the function body and');
console.log("  not inside any block. buildBlockScopes files it under the null");
console.log('  scope as an opaque access (block_scope.ts:205), and one opaque');
console.log('  access sets `pipeline = false` for the whole scope (block_scope.ts:241).');

// ---------------------------------------------------------------- reachability

console.log('\n=== who asks for any of this ===\n');
const consumers = {
  'scope.deps / depsBySrc / depsByDst': '_checkRelocationDependences (schedule.ts:905)',
  'scope.memberOf': '_checkRelocationDependences (schedule.ts:905)',
  'scope.producersOf': '— nothing in src/',
  'scope.consumersOf': '— nothing in src/',
  'scope.writersOf': '— nothing in src/',
  'scope.stagePipeline': '— nothing in src/',
  'BlockInfo.regionCover': '— nothing in src/',
  'BlockInfo.affineBinding': '— nothing in src/ (the same field is read off BlockAccessInfo instead)',
};
for (const [k, v] of Object.entries(consumers)) console.log(`  ${k.padEnd(36)} ${v}`);

console.log('\n  And `_checkRelocationDependences` is called only by computeAt and');
console.log('  reverseComputeAt, neither of which has a caller in src/. So');
console.log('  buildBlockScopes runs exactly when a lab such as this one, or a');
console.log('  test, asks for it. Chapter 41 comes back to the two primitives.');
console.log('\n  One export escapes that verdict. `linkAccessUnits` — the thirteen');
console.log('  lines that turn hulls into RAW/WAW/WAR edges (block_scope.ts:158) —');
console.log('  is imported directly by MemorySchedulePass (memory_scheduler.ts:6),');
console.log('  which runs on every compilation. The reusable piece was factored');
console.log('  out and reused; the 244 lines around it are dead.');
