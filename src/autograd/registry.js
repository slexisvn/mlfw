import { AddBackward, SubBackward, MulBackward, DivBackward, NegBackward, PowBackward } from './function/basic.js';

import { ExpBackward, LogBackward, SqrtBackward, TanhBackward, SigmoidBackward, ReluBackward, GeluBackward, SiluBackward } from './function/unary.js';

import { SumBackward, MeanBackward } from './function/reduction.js';
import { MatmulBackward, DotBackward } from './function/linalg.js';


import { CatBackward, ClampBackward, PadBackward, IndexSelectBackward, WhereBackward } from './function/indexing.js';

const _registry = new Map();

function _register(name, factory) {
  _registry.set(name, factory);
}

_register('add', () => new AddBackward());
_register('sub', () => new SubBackward());
_register('mul', () => new MulBackward());
_register('div', () => new DivBackward());
_register('neg', () => new NegBackward());
_register('pow', () => new PowBackward());

_register('exp', () => new ExpBackward());
_register('log', () => new LogBackward());
_register('sqrt', () => new SqrtBackward());
_register('tanh', () => new TanhBackward());
_register('sigmoid', () => new SigmoidBackward());
_register('relu', () => new ReluBackward());
_register('gelu', () => new GeluBackward());
_register('silu', () => new SiluBackward());

_register('sum', () => new SumBackward());
_register('mean', () => new MeanBackward());

_register('matmul', () => new MatmulBackward());
_register('dot', () => new DotBackward());

_register('cat', () => new CatBackward());
_register('clamp', () => new ClampBackward());
_register('pad', () => new PadBackward());
_register('index_select', () => new IndexSelectBackward());
_register('where', () => new WhereBackward());

export function getGradFn(opName) {
  const factory = _registry.get(opName);
  return factory ? factory() : null;
}

export function hasGradFn(opName) {
  return _registry.has(opName);
}
