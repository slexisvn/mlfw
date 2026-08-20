import {
  tensor, randn, Module, Linear, ReLU, Sequential, LayerNorm, scan,
  trace, printModule, parseModule, manual_seed,
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

class Normed extends Module {
  constructor() { super(); this.l = new Linear(4, 4); this.n = new LayerNorm(4); }
  forward(t) { return this.n.forward(this.l.forward(t).relu()); }
}

async function roundTrip(label, graph) {
  const once = printModule(graph);
  const twice = printModule(parseModule(once));
  const ok = once === twice;
  console.log(`${label.padEnd(28)} ${String(once.split('\n').length).padStart(3)} lines   round-trips: ${ok}`);
  if (!ok) {
    const a = once.split('\n'), b = twice.split('\n');
    for (let i = 0; i < Math.max(a.length, b.length); i++) {
      if (a[i] !== b[i]) console.log(`   line ${i + 1}\n     printed once : ${a[i]}\n     printed twice: ${b[i]}`);
    }
  }
  return ok;
}

const mlp = new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
const rec = new Recurrent();
const norm = new Normed();

const results = [];
results.push(await roundTrip('two-layer MLP', await trace((t) => mlp.forward(t), [tensor([[0.5, -1.5], [1.0, 2.0]])])));
results.push(await roundTrip('a scan region', await trace((a, b) => rec.forward(a, b), [randn([4, 3]), randn([3])])));
results.push(await roundTrip('linear + relu + layernorm', await trace((t) => norm.forward(t), [randn([2, 4])])));
results.push(await roundTrip('dynamic batch dimension', await trace((t) => mlp.forward(t), [randn([4, 2])], { dynamic_shapes: [new Set([0])] })));

console.log(`\nall round-trip: ${results.every(Boolean)}`);
