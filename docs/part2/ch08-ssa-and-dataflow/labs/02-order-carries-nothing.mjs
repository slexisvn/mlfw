import {
  tensor, Linear, ReLU, Sequential, trace, printModule, parseModule, manual_seed,
} from '../../../../dist/index.node.js';

manual_seed(0);

const model = new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
const graph = await trace((t) => model.forward(t), [tensor([[0.5, -1.5], [1.0, 2.0]])]);
const text = printModule(graph);

const lines = text.split('\n');
const head = lines.slice(0, 2);
const body = lines.slice(2, lines.length - 2);
const foot = lines.slice(lines.length - 2);
const reversed = [...head, ...body.reverse(), ...foot].join('\n');

console.log('=== the same function, printed bottom to top ===');
console.log(reversed);

const back = parseModule(reversed);
console.log('\nthe parser accepted it.');

function returnValues(func) {
  const ret = func.findOp((op) => op.opName === 'return');
  return ret ? [...ret.operands] : [];
}

function dataflowOrder(func) {
  const order = [];
  const done = new Set();
  const visit = (value) => {
    const op = value.definingOp;
    if (!op || done.has(op)) return;
    done.add(op);
    for (const operand of op.operands) visit(operand);
    order.push(op.opName);
  };
  for (const v of returnValues(func)) visit(v);
  return order;
}

const before = dataflowOrder([...graph.functions()][0]);
const after = dataflowOrder([...back.functions()][0]);

console.log('\ndataflow order, original : ' + before.join(' -> '));
console.log('dataflow order, reversed : ' + after.join(' -> '));
console.log('identical                : ' + (before.join() === after.join()));

console.log('\nprinted back, the reversed module reads:');
console.log(printModule(back));
console.log('same text as the original? ' + (printModule(back) === text));

console.log('\none accessor does care about textual order:');
for (const [label, f] of [['original', [...graph.functions()][0]], ['reversed', [...back.functions()][0]]]) {
  console.log(`  ${label.padEnd(9)} getReturnOp() -> ${f.getReturnOp() ? 'found' : 'null'}` +
              `   findOp(name === 'return') -> ${f.findOp((op) => op.opName === 'return') ? 'found' : 'null'}`);
}
