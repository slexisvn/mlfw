import { AddBackward, SubBackward, MulBackward, DivBackward, NegBackward, PowBackward } from './function/basic.js';

import { ExpBackward, LogBackward, SqrtBackward, TanhBackward, SigmoidBackward, ReluBackward, GeluBackward, SiluBackward, SoftmaxBackward, LogSoftmaxBackward, ErfBackward, ErfcBackward, LgammaBackward, GammaBackward } from './function/unary.js';

import { SumBackward, MeanBackward, MaxBackward, MinBackward, ProdBackward } from './function/reduction.js';
import { AbsBackward, SinBackward, CosBackward, RsqrtBackward, IdentityBackward } from './function/unary.js';
import { MaximumBackward, MinimumBackward, GatherBackward, ScatterAddBackward, RemBackward } from './function/indexing.js';
import { Conv2dBackward, Pool2dBackward, LayerNormBackward, BatchNormBackward, EmbeddingBackward } from './function/nn.js';
import { MatmulBackward, DotBackward } from './function/linalg.js';


import { CatBackward, StackBackward, ClampBackward, PadBackward, IndexSelectBackward, WhereBackward } from './function/indexing.js';
import { ReshapeBackward, TransposeBackward, PermuteBackward, SliceBackward, ExpandBackward, SelectBackward } from './function/view.js';
import type { AutogradNode } from './node.js';
import { COMPOSITE_KERNELS } from '../tensor/native/composite/composite_ops.js';
import type { OpArgs } from './types.js';

type GradFactory = (args: OpArgs) => AutogradNode;

const _registry = new Map<string, GradFactory>();

function _register(name: string, factory: GradFactory) {
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
_register('erf', () => new ErfBackward());
_register('erfc', () => new ErfcBackward());
_register('lgamma', () => new LgammaBackward());
_register('gamma', () => new GammaBackward());
_register('sigmoid', () => new SigmoidBackward());
_register('relu', () => new ReluBackward());
_register('softmax', () => new SoftmaxBackward());
_register('log_softmax', () => new LogSoftmaxBackward());
_register('gelu', () => new GeluBackward());
_register('silu', () => new SiluBackward());

_register('sum', () => new SumBackward());
_register('mean', () => new MeanBackward());

_register('matmul', () => new MatmulBackward());
_register('dot', () => new DotBackward());

_register('cat', () => new CatBackward());
_register('stack', () => new StackBackward());
_register('clamp', () => new ClampBackward());
_register('pad', () => new PadBackward());
_register('index_select', () => new IndexSelectBackward());
_register('where', () => new WhereBackward());
_register('reshape', () => new ReshapeBackward());
_register('transpose', (args) => new TransposeBackward(args![1] as number, args![2] as number));
_register('permute', (args) => new PermuteBackward(args![1] as readonly number[]));
_register('broadcast_in_dim', () => new ExpandBackward());
_register('expand', () => new ExpandBackward());
_register('slice', (args) => new SliceBackward(args![1] as number, args![2] as number, args![3] as number, args![4] as number));
_register('narrow', (args) => new SliceBackward(args![1] as number, args![2] as number, (args![2] as number) + (args![3] as number), 1));
_register('select', (args) => new SelectBackward(args![1] as number, args![2] as number));
_register('unsqueeze', () => new ReshapeBackward());
_register('squeeze', () => new ReshapeBackward());

_register('abs', () => new AbsBackward());
_register('sin', () => new SinBackward());
_register('cos', () => new CosBackward());
_register('rsqrt', () => new RsqrtBackward());
_register('clone', () => new IdentityBackward());
_register('contiguous', () => new IdentityBackward());

_register('max', () => new MaxBackward());
_register('min', () => new MinBackward());
_register('prod', () => new ProdBackward());

_register('maximum', () => new MaximumBackward());
_register('minimum', () => new MinimumBackward());
_register('rem', () => new RemBackward());
_register('gather', () => new GatherBackward());
_register('scatter_add', () => new ScatterAddBackward());


_register('conv2d', () => new Conv2dBackward());
_register('pool2d', () => new Pool2dBackward());
_register('layer_norm', () => new LayerNormBackward());
_register('batch_norm', () => new BatchNormBackward());
_register('embedding', () => new EmbeddingBackward());

const _decomposed = new Set<string>(Object.keys(COMPOSITE_KERNELS));

export function isDecomposedOp(opName: string): boolean {
  return _decomposed.has(opName);
}

const _barriers = new Set<string>();

export function registerGradientBarrier(...opNames: readonly string[]): void {
  for (const name of opNames) _barriers.add(name);
}

export function isGradientBarrier(opName: string): boolean {
  return _barriers.has(opName);
}

export function getGradFn(opName: string, args: OpArgs = null): AutogradNode | null {
  const factory = _registry.get(opName);
  if (factory) return factory(args);
  if (_barriers.has(opName) || _decomposed.has(opName)) return null;
  throw new Error(`autograd: op '${opName}' is on the gradient path but has no backward rule and is not a registered gradient barrier — the gradient would be silently dropped. Register one in src/autograd/registry.ts.`);
}

export function hasGradFn(opName: string): boolean {
  return _registry.has(opName);
}

export function listOpsWithoutGrad(opNames: Iterable<string>): string[] {
  const missing: string[] = [];
  for (const name of opNames) {
    if (!_registry.has(name) && !_barriers.has(name) && !_decomposed.has(name)) missing.push(name);
  }
  return missing;
}

registerGradientBarrier(
  'eq', 'ne', 'lt', 'le', 'gt', 'ge',
  'argmax', 'argmin', 'argsort',
  'floor', 'ceil', 'sign', 'one_hot', 'fill',
  'cholesky', 'cov', 'det', 'eigh', 'inv', 'lstsq', 'pinv', 'qr', 'solve', 'svd',
  'fft', 'ifft',
  'decision_tree_fit', 'decision_tree_predict', 'elastic_net',
  'gaussian_nb_fit', 'gaussian_nb_predict',
  'kmeans', 'kmeans_predict', 'knn_predict',
);
