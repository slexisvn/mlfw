import {
  randn, Module, Linear, trace, printModule, manual_seed,
} from '../../../../dist/index.node.js';

manual_seed(0);

class Net extends Module {
  constructor() { super(); this.l = new Linear(3, 2); }
  forward(t) { return this.l.forward(t).relu(); }
}

const model = new Net();
const example = randn([4, 3]);

const variants = [
  ['every dimension known', undefined],
  ['dimension 0 dynamic', { dynamic_shapes: [new Set([0])] }],
  ['the whole input dynamic', { dynamic_shapes: [true] }],
];

for (const [label, opts] of variants) {
  const graph = await trace((t) => model.forward(t), [example], opts);
  console.log(`=== ${label} ===`);
  console.log(printModule(graph));
  const func = [...graph.functions()][0];
  const input = func.args[0].type;
  const output = func.outputTypes[0];
  console.log(`  input  isFullyStatic ${String(input.isFullyStatic).padEnd(5)} hasDynamic ${String(input.hasDynamic).padEnd(5)} numel ${input.numel()}  sizeInBytes ${input.sizeInBytes()}`);
  console.log(`  output isFullyStatic ${String(output.isFullyStatic).padEnd(5)} hasDynamic ${String(output.hasDynamic).padEnd(5)} numel ${output.numel()}  sizeInBytes ${output.sizeInBytes()}`);
  console.log();
}

const statics = [...(await trace((t) => model.forward(t), [example])).functions()][0];
const dynamics = [...(await trace((t) => model.forward(t), [example], { dynamic_shapes: [new Set([0])] })).functions()][0];

console.log('=== a static type and a dynamic one, compared ===');
const s = statics.args[0].type;
const d = dynamics.args[0].type;
console.log(`  static  [${s.shape.join(', ')}]`);
console.log(`  dynamic [${d.shape.join(', ')}]   (-1 is what '?' prints as)`);
console.log(`  s.equals(d)          = ${s.equals(d)}`);
console.log(`  s.shapeCompatible(d) = ${s.shapeCompatible(d)}   <- a dynamic dimension is compatible with anything`);
console.log(`  d.shapeCompatible(s) = ${d.shapeCompatible(s)}`);

const other = statics.args[0].type.withShape([9, 3]);
console.log(`\n  a [9, 3] input`);
console.log(`  s.shapeCompatible(other) = ${s.shapeCompatible(other)}   <- two known, different sizes are not`);
console.log(`  d.shapeCompatible(other) = ${d.shapeCompatible(other)}`);
