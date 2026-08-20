import {
  tensor, Linear, ReLU, Sequential, compile, CPUTarget, TraceLevel, parseModule, manual_seed,
} from '../../../../dist/index.node.js';

manual_seed(0);

const model = new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
const x = tensor([[0.5, -1.5], [1.0, 2.0]]);

let optimized = null;
const compiled = compile(model, [x], {
  target: CPUTarget(),
  trace: {
    level: TraceLevel.DEBUG,
    irSnapshot: { afterGraphPasses: true },
    sink: (event) => { if (event.type === 'ir_snapshot') optimized = event.text; },
  },
});
await compiled._ready;

console.log(optimized);

const func = [...parseModule(optimized).functions()][0];
const fusion = func.findOp((op) => op.numRegions > 0);

const definedInside = new Set();
const usedInside = [];
for (const region of fusion.regions) {
  for (const block of region.blocks) {
    for (const arg of block.arguments) definedInside.add(arg);
    for (const op of block.ops()) {
      for (const r of op.results) definedInside.add(r);
      for (const operand of op.operands) usedInside.push({ op: op.opName, operand });
    }
  }
}

const captured = usedInside.filter(({ operand }) => !definedInside.has(operand));

console.log(`\nthe '${fusion.opName}' operation`);
console.log(`  operands passed in       : ${fusion.numOperands}`);
console.log(`  block arguments inside   : ${fusion.getRegion(0).entryBlock.arguments.length}`);
console.log(`  values defined inside    : ${definedInside.size}`);
console.log(`  values captured from out : ${captured.length}` +
            `${captured.length === 0 ? '  (nothing crosses the boundary implicitly)' : ''}`);

const operandTypes = fusion.operands.map((v) => v.type);
const argTypes = fusion.getRegion(0).entryBlock.arguments.map((a) => a.type);
console.log('\n  operand i  ->  block argument i');
for (let i = 0; i < operandTypes.length; i++) {
  const same = operandTypes[i].equals(argTypes[i]);
  console.log(`    ${i}: ${operandTypes[i].shape.join('x')}x${operandTypes[i].dtype}` +
              `  ->  ${argTypes[i].shape.join('x')}x${argTypes[i].dtype}   ${same ? 'same type' : 'DIFFERENT'}`);
}

const terminator = fusion.getRegion(0).entryBlock.lastOp;
console.log(`\n  the region ends with '${terminator.opName}', yielding ${terminator.numOperands} value(s)` +
            ` for the ${fusion.numResults} result(s) of '${fusion.opName}'`);
