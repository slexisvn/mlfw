import {
  tensor, Module, Linear, trace, printModule, compile, CPUTarget, manual_seed,
} from '../../../../dist/index.node.js';

manual_seed(0);

class DeadBranch extends Module {
  constructor() {
    super();
    this.used = new Linear(2, 2);
    this.unused = new Linear(2, 8);
  }
  forward(t) {
    const wasted = this.unused.forward(t).relu().tanh();
    return this.used.forward(t);
  }
}

const model = new DeadBranch();
const x = tensor([[1, 2], [3, 4]]);

const graph = await trace((t) => model.forward(t), [x]);
const traced = printModule(graph);
console.log('=== what tracing recorded ===');
console.log(traced);
console.log(`operations recorded: ${traced.split('\n').filter(l => l.includes(' = ')).length}`);

const compiled = compile(model, [x], { target: CPUTarget() });
await compiled._ready;
const source = compiled.source();
console.log('\n=== what survived compilation ===');
console.log(source);
console.log(`unused weight buffers still in the signature, never read: ` +
            `${['buf_3', 'buf_5'].filter(b => source.split('\n').slice(2).join('\n').includes(b)).length === 0 ? 'yes' : 'no'}`);
