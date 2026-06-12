import './src/index.js';
import * as ops from './src/tensor/ops/ops.js';
import { tensor, zeros, ones } from './src/tensor/factory/creation_ops.js';
const x = tensor([[1,2],[3,4],[5,6]], {dtype:'f32'});
const out = ops.batch_norm(x, ones([2]), zeros([2]), zeros([2]), ones([2]), 1, 1e-5);
console.log('out:', JSON.stringify(out.toArray()));
