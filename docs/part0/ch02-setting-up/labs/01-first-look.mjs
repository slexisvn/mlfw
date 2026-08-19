import {
  tensor, Linear, ReLU, Sequential, trace, printModule, manual_seed,
} from '../../../../dist/index.node.js';

manual_seed(0);

const model = new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
const x = tensor([[0.5, -1.5], [1.0, 2.0]]);

const graph = await trace((t) => model.forward(t), [x]);
console.log(printModule(graph));
