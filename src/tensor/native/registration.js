import { Library } from '../../dispatcher/library.js';
import { DispatchKey } from '../../dispatcher/dispatch_key.js';
import { registerJITKernels } from '../../dispatcher/jit_dispatch.js';

import {
  metaAdd, metaSub, metaMul, metaDiv, metaPow, metaRem,
  metaMaximum, metaMinimum, metaNeg,
  metaExp, metaLog, metaSqrt, metaRsqrt, metaAbs,
  metaSin, metaCos, metaTanh, metaSigmoid,
  metaRelu, metaGelu, metaSilu, metaSign, metaFloor, metaCeil,
  metaSum, metaMean, metaMax, metaMin, metaProd,
  metaMatmul, metaClone,
} from './meta/meta_ops.js';

const _OP_DEFS = [
  'add(Tensor self, Tensor other) -> Tensor',
  'sub(Tensor self, Tensor other) -> Tensor',
  'mul(Tensor self, Tensor other) -> Tensor',
  'div(Tensor self, Tensor other) -> Tensor',
  'neg(Tensor self) -> Tensor',
  'pow(Tensor self, Tensor exponent) -> Tensor',
  'rem(Tensor self, Tensor other) -> Tensor',
  'maximum(Tensor self, Tensor other) -> Tensor',
  'minimum(Tensor self, Tensor other) -> Tensor',

  'exp(Tensor self) -> Tensor',
  'log(Tensor self) -> Tensor',
  'sqrt(Tensor self) -> Tensor',
  'rsqrt(Tensor self) -> Tensor',
  'abs(Tensor self) -> Tensor',
  'sin(Tensor self) -> Tensor',
  'cos(Tensor self) -> Tensor',
  'tanh(Tensor self) -> Tensor',
  'sigmoid(Tensor self) -> Tensor',
  'relu(Tensor self) -> Tensor',
  'gelu(Tensor self) -> Tensor',
  'silu(Tensor self) -> Tensor',
  'sign(Tensor self) -> Tensor',
  'floor(Tensor self) -> Tensor',
  'ceil(Tensor self) -> Tensor',

  'eq(Tensor self, Tensor other) -> Tensor',
  'ne(Tensor self, Tensor other) -> Tensor',
  'lt(Tensor self, Tensor other) -> Tensor',
  'le(Tensor self, Tensor other) -> Tensor',
  'gt(Tensor self, Tensor other) -> Tensor',
  'ge(Tensor self, Tensor other) -> Tensor',
  'where(Tensor condition, Tensor self, Tensor other) -> Tensor',
  'clamp(Tensor self, Tensor min, Tensor max) -> Tensor',
  'pad(Tensor self, Tensor value, int[] low, int[] high) -> Tensor',
  'one_hot(Tensor indices, int depth) -> Tensor',
  'index_select(Tensor self, Tensor index, int dim) -> Tensor',
  'gather(Tensor self, Tensor index, int dim) -> Tensor',
  'scatter_add(Tensor self, Tensor index, Tensor src, int dim) -> Tensor',

  'sum(Tensor self, int[] dim, bool keepdim) -> Tensor',
  'mean(Tensor self, int[] dim, bool keepdim) -> Tensor',
  'max(Tensor self, int[] dim, bool keepdim) -> Tensor',
  'min(Tensor self, int[] dim, bool keepdim) -> Tensor',
  'prod(Tensor self, int[] dim, bool keepdim) -> Tensor',
  'argmax(Tensor self, int dim, bool keepdim) -> Tensor',
  'argmin(Tensor self, int dim, bool keepdim) -> Tensor',

  'matmul(Tensor self, Tensor other) -> Tensor',
  'dot(Tensor self, Tensor other) -> Tensor',

  'cat(Tensor[] tensors, int dim) -> Tensor',
  'stack(Tensor[] tensors, int dim) -> Tensor',

  'clone(Tensor self) -> Tensor',
  'fill(Tensor self, Scalar value) -> Tensor',

  'transpose(Tensor self, int dim0, int dim1) -> Tensor',
  'softmax(Tensor self, int dim) -> Tensor',
  'log_softmax(Tensor self, int dim) -> Tensor',
  'layer_norm(Tensor input, Tensor weight, Tensor bias, int axis, float eps) -> Tensor',
  'batch_norm(Tensor input, Tensor weight, Tensor bias, Tensor mean, Tensor var, int axis, float eps) -> Tensor',
  'conv2d(Tensor input, Tensor weight, int[] strides, int[] padding, int[] dilation, int groups) -> Tensor',
  'pool2d(Tensor input, str pool_type, int[] kernel_size, int[] strides, int[] padding) -> Tensor',
  'embedding(Tensor weight, Tensor indices) -> Tensor',
];

const META = DispatchKey.META;

const _META_KERNELS = {
  add: metaAdd, sub: metaSub, mul: metaMul, div: metaDiv,
  pow: metaPow, rem: metaRem, maximum: metaMaximum, minimum: metaMinimum,
  neg: metaNeg,
  exp: metaExp, log: metaLog, sqrt: metaSqrt, rsqrt: metaRsqrt,
  abs: metaAbs, sin: metaSin, cos: metaCos, tanh: metaTanh,
  sigmoid: metaSigmoid, relu: metaRelu, gelu: metaGelu, silu: metaSilu,
  sign: metaSign, floor: metaFloor, ceil: metaCeil,
  sum: metaSum, mean: metaMean, max: metaMax, min: metaMin, prod: metaProd,
  matmul: metaMatmul, clone: metaClone,
};

let _registered = false;

export function registerNativeOps() {
  if (_registered) return;
  _registered = true;

  const defLib = new Library('mlc', 'DEF');
  for (const schema of _OP_DEFS) {
    defLib.def(schema);
  }

  const implLib = new Library('mlc', 'IMPL');
  for (const [name, fn] of Object.entries(_META_KERNELS)) {
    implLib.impl(name, META, fn);
  }

  registerJITKernels();
}
