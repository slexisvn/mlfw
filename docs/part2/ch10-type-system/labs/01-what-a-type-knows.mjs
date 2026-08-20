import {
  tensor, Linear, ReLU, Sequential, trace, manual_seed,
} from '../../../../dist/index.node.js';

manual_seed(0);

const model = new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
const graph = await trace((t) => model.forward(t), [tensor([[0.5, -1.5], [1.0, 2.0]])]);
const func = [...graph.functions()][0];

const rows = [];
func.args.forEach((arg, i) => rows.push([`%${i}`, 'argument', arg.type]));
let next = func.args.length;
for (const op of func.ops()) for (const r of op.results) rows.push([`%${next++}`, op.opName, r.type]);

console.log('value  produced by        shape      dtype  rank  numel  bytes  strides');
for (const [label, producer, type] of rows) {
  console.log(
    `${label.padEnd(7)}${producer.padEnd(18)}` +
    `[${type.shape.join(', ')}]`.padEnd(11) +
    `${type.dtype.padEnd(7)}${String(type.rank).padEnd(6)}` +
    `${String(type.numel()).padEnd(7)}${String(type.sizeInBytes()).padEnd(7)}` +
    `[${type.strides().join(', ')}]`
  );
}

console.log('\nthe scalar constant:');
const scalar = func.findOp((op) => op.opName === 'constant').getResult(0).type;
console.log(`  rank ${scalar.rank}, isScalar ${scalar.isScalar}, numel ${scalar.numel()}, printed as tensor<${scalar.dtype}>`);

console.log('\nequality is not compatibility:');
const a = func.findOp((op) => op.opName === 'add').getResult(0).type;
const b = func.findOp((op) => op.opName === 'maximum').getResult(0).type;
const out = func.findOps((op) => op.opName === 'add')[1].getResult(0).type;
console.log(`  add result   [${a.shape.join(', ')}]`);
console.log(`  maximum      [${b.shape.join(', ')}]`);
console.log(`  second add   [${out.shape.join(', ')}]`);
console.log(`  a.equals(b)            = ${a.equals(b)}`);
console.log(`  a.shapeEquals(b)       = ${a.shapeEquals(b)}`);
console.log(`  a.shapeCompatible(b)   = ${a.shapeCompatible(b)}`);
console.log(`  a.equals(out)          = ${a.equals(out)}`);
console.log(`  a.shapeCompatible(out) = ${a.shapeCompatible(out)}`);

console.log('\nlayout:');
for (const [label, , type] of rows.slice(0, 3)) {
  console.log(`  ${label}  order [${type.layout.order.join(', ')}]  identity ${type.layout.isIdentity()}  strides [${type.strides().join(', ')}]`);
}
