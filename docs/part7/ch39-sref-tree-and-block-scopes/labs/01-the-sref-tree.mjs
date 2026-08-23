import {
  lowerToTir, printTensorIR, Schedule, randn,
} from '../../_internals.mjs';

const pf = await lowerToTir((x) => x.mul(x).sum(1), [randn([4, 6])]);
const sch = new Schedule(pf);

console.log(printTensorIR(pf));

const loopName = (s) => s.node.loopVar.name;
const label = (s) => (s.isLoop
  ? `for ${loopName(s)} in 0..${s.node.extent.value}`
  : `block ${s.node.name}`);

console.log('=== every sref, with the chain that reaches it ===\n');
for (const b of sch.state.tree.allBlocks()) {
  const chain = b.ancestors().reverse().map(label);
  console.log(`  ${[...chain, label(b)].join('  >  ')}`);
}

console.log(`\n  loops registered: ${sch.state.tree.allLoops().length}`);
console.log(`  blocks registered: ${sch.state.tree.allBlocks().length}`);
console.log(`  tree.root: ${sch.state.tree.root === null ? 'null' : sch.state.tree.root.node.type}`);
console.log('\n  `root` is null because the function body is a SeqNode, and only');
console.log('  ForNode and BlockNode get an sref (sref.ts:117). Three statement');
console.log('  chains hang off nothing. Every query in the file goes through the');
console.log('  name and node maps instead, which is why nothing notices.');

console.log('\n=== the two queries a primitive actually uses ===\n');
console.log(`  getLoops('reduce_acc_2')  -> ${sch.getLoops('reduce_acc_2').map((l) => l.loopVar.name).join(', ')}`);
console.log(`  getLoops('mul_block_0')   -> ${sch.getLoops('mul_block_0').map((l) => l.loopVar.name).join(', ')}`);
console.log('\n  `loopsOf` walks parent pointers from the block upward and reverses');
console.log('  (sref.ts:172), so it is O(depth) and needs no search. Without the');
console.log('  tree it would be a walk of the whole function per question.');

console.log('\n=== which srefs survive a split? ===\n');

const before = new Map(sch.state.tree.allBlocks().map((s) => [s.node.name, s]));
const beforeLoops = sch.state.tree.allLoops().length;

sch.split(sch.getLoops('reduce_acc_2')[1], 3);

const after = new Map(sch.state.tree.allBlocks().map((s) => [s.node.name, s]));
for (const [name, sref] of before) {
  const now = after.get(name);
  console.log(`  ${name.padEnd(16)} same sref object: ${now === sref}`);
}
console.log(`\n  loops before: ${beforeLoops}   loops after: ${sch.state.tree.allLoops().length}`);
console.log('\n  `split` calls state.replaceNode(oldLoop, newOuterLoop), which');
console.log('  unregisters the replaced subtree and rebuilds it (sref.ts:186).');
console.log('  Blocks under the split loop get fresh sref objects; blocks in the');
console.log('  other two statement chains are untouched. The rebuild is O(subtree),');
console.log('  not O(function) — and a name, not an object, is what survives it.');

console.log('\n' + printTensorIR(sch.func).split('\n').slice(-14).join('\n'));
