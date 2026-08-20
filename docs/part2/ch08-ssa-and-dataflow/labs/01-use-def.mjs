import {
  tensor, Module, Linear, ReLU, Sequential, trace, printModule, manual_seed,
} from '../../../../dist/index.node.js';

manual_seed(0);

function nameValues(func) {
  const name = new Map();
  func.args.forEach((arg, i) => name.set(arg, `%${i}`));
  let next = func.args.length;
  for (const op of func.ops()) for (const r of op.results) name.set(r, `%${next++}`);
  return name;
}

function reachableFromReturn(func) {
  const live = new Set();
  const stack = [...func.getReturnValues()];
  while (stack.length > 0) {
    const v = stack.pop();
    if (live.has(v)) continue;
    live.add(v);
    if (v.definingOp) for (const operand of v.definingOp.operands) stack.push(operand);
  }
  return live;
}

const model = new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
const graph = await trace((t) => model.forward(t), [tensor([[0.5, -1.5], [1.0, 2.0]])]);
const func = [...graph.functions()][0];
const name = nameValues(func);

console.log(printModule(graph));

console.log('\nvalue      defined by            used by');
for (const [value, label] of name) {
  const definedBy = value.definingOp ? value.definingOp.opName : 'function argument';
  const users = value.getUsers().map((u) => u.opName);
  console.log(`${label.padEnd(11)}${definedBy.padEnd(22)}${users.join(', ') || '(nobody)'}`);
}

const producedTwice = [...name.keys()].filter((v) => v.definingOp && v.definingOp.results.indexOf(v) < 0);
console.log(`\nvalues claimed by more than one producer: ${producedTwice.length}`);

class DeadBranch extends Module {
  constructor() {
    super();
    this.used = new Linear(2, 2);
    this.unused = new Linear(2, 8);
  }
  forward(t) {
    this.unused.forward(t).relu().tanh();
    return this.used.forward(t);
  }
}

const dead = await trace((t) => new DeadBranch().forward(t), [tensor([[1, 2], [3, 4]])]);
const deadFunc = [...dead.functions()][0];
const deadName = nameValues(deadFunc);
const live = reachableFromReturn(deadFunc);

console.log('\nthe same walk on a model with a branch nobody reads:');
console.log(`  reachable from the return : ${[...deadName].filter(([v]) => live.has(v)).map(([, l]) => l).join(' ')}`);
console.log(`  not reachable             : ${[...deadName].filter(([v]) => !live.has(v)).map(([, l]) => l).join(' ')}`);
console.log(`  operations whose every result is unreachable: ` +
            `${deadFunc.opsArray().filter((op) => op.numResults > 0 && op.results.every((r) => !live.has(r))).map((op) => op.opName).join(', ')}`);
