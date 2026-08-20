import {
  randn, Module, scan, trace, printModule, manual_seed,
} from '../../../../dist/index.node.js';

manual_seed(0);

class Recurrent extends Module {
  forward(xs, h0) {
    const [, ys] = scan((carry, x_t) => {
      const next = carry.mul(0.9).add(x_t).tanh();
      return [next, next];
    }, h0, xs);
    return ys;
  }
}

const graph = await trace((a, b) => new Recurrent().forward(a, b), [randn([4, 3]), randn([3])]);
console.log(printModule(graph));

const pad = (d) => '  '.repeat(d);

function describeOp(op, depth) {
  console.log(`${pad(depth)}Operation '${op.opName}'  ${op.numOperands} operands, ${op.numResults} results, ` +
              `${op.attributes.size} attributes, ${op.numRegions} regions`);
  for (const region of op.regions) describeRegion(region, depth + 1);
}

function describeRegion(region, depth) {
  console.log(`${pad(depth)}Region  ${region.blocks.length} block(s)`);
  for (const block of region.blocks) describeBlock(block, depth + 1);
}

function describeBlock(block, depth) {
  console.log(`${pad(depth)}Block  ${block.arguments.length} argument(s), ${block.size} operation(s)`);
  for (const op of block.ops()) describeOp(op, depth + 1);
}

console.log('\n=== the containment tree ===');
console.log(`Module '${graph.name}'  ${graph.functionCount} function(s)`);
for (const func of graph.functions()) {
  console.log(`${pad(1)}Function '${func.name}'  ${func.inputTypes.length} in, ${func.outputTypes.length} out`);
  describeRegion(func.body, 2);
}

const func = [...graph.functions()][0];
console.log('\n=== two ways to count ===');
console.log(`  ops() walks the top-level block only : ${[...func.ops()].length}`);
console.log(`  opsRecursive() descends into regions : ${[...func.opsRecursive()].length}`);
console.log(`  blocksRecursive()                    : ${[...func.blocksRecursive()].length}`);

console.log('\n=== who owns whom ===');
const inner = [...func.opsRecursive()].find((op) => op.opName === 'tanh');
let owner = inner;
const chain = [];
while (owner) {
  chain.push(`op '${owner.opName}'`);
  const block = owner.parentBlock;
  if (!block) break;
  chain.push(`block(${block.arguments.length} args)`);
  owner = block.parentOp;
}
console.log(`  ${chain.join('  in  ')}  in  function '${func.name}'`);
