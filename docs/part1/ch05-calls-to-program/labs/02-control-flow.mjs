import {
  tensor, randn, Module, scan, trace, printModule, compile, CPUTarget, manual_seed,
} from '../../../../dist/index.node.js';

manual_seed(0);

class DataDependent extends Module {
  forward(t) {
    if (t.sum().item() > 0) return t.mul(10);
    return t.mul(-1);
  }
}

console.log('=== a branch that depends on tensor values ===');
try {
  await trace((t) => new DataDependent().forward(t), [tensor([[1, 2], [3, 4]])]);
  console.log('traced without complaint');
} catch (e) {
  console.log('trace failed:\n');
  console.log(e.message);
}

class Recurrent extends Module {
  forward(xs, h0) {
    const [last, ys] = scan((carry, x_t) => {
      const next = carry.mul(0.9).add(x_t).tanh();
      return [next, next];
    }, h0, xs);
    return ys;
  }
}

console.log('\n=== the same shape of computation, expressed as a scan ===');
const xs = randn([4, 3]);
const h0 = randn([3]);
const graph = await trace((a, b) => new Recurrent().forward(a, b), [xs, h0]);
console.log(printModule(graph));

class ModeDependent extends Module {
  constructor() { super(); this.flag = true; }
  forward(t) { return this.flag ? t.mul(10) : t.mul(-1); }
}

console.log('\n=== a branch on host state: no error, and no protection ===');
const mode = new ModeDependent();
const x = tensor([[1, 2], [3, 4]]);
const compiled = compile(mode, [x], { target: CPUTarget() });
await compiled._ready;

console.log(`flag=true    eager ${mode.forward(x).data.join(',')}   compiled ${(await compiled(x)).data.join(',')}`);
mode.flag = false;
console.log(`flag=false   eager ${mode.forward(x).data.join(',')}   compiled ${(await compiled(x)).data.join(',')}   <-- compiled is stale`);

class Scaled extends Module {
  constructor() { super(); this.scale = 2; }
  forward(t) { return t.mul(this.scale); }
}

console.log('\n=== host state read rather than branched on: same staleness, no branch at all ===');
const scaled = new Scaled();
const compiledScaled = compile(scaled, [x], { target: CPUTarget() });
await compiledScaled._ready;

console.log(`scale=2      eager ${scaled.forward(x).data.join(',')}   compiled ${(await compiledScaled(x)).data.join(',')}`);
scaled.scale = 5;
console.log(`scale=5      eager ${scaled.forward(x).data.join(',')}   compiled ${(await compiledScaled(x)).data.join(',')}   <-- compiled is stale`);
