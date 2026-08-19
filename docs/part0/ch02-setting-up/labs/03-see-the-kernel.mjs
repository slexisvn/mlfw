import {
  tensor, Linear, ReLU, Sequential, compile, CPUTarget, manual_seed,
} from '../../../../dist/index.node.js';

manual_seed(0);

const model = new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
const x = tensor([[0.5, -1.5], [1.0, 2.0]]);

const compiled = compile(model, [x], { target: CPUTarget() });
await compiled._ready;

console.log('kernels:', compiled.kernels().join(', '));
console.log();
console.log(compiled.source());
