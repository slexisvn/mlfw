import { AddBackward, SubBackward, MulBackward, DivBackward, NegBackward, PowBackward } from './function/basic.js';

import { ExpBackward, LogBackward, SqrtBackward, TanhBackward, SigmoidBackward, ReluBackward, GeluBackward, SiluBackward, SoftmaxBackward, LogSoftmaxBackward, ErfBackward, ErfcBackward, LgammaBackward, GammaBackward } from './function/unary.js';

import { SumBackward, MeanBackward } from './function/reduction.js';
import { MatmulBackward, DotBackward } from './function/linalg.js';


import { CatBackward, StackBackward, ClampBackward, PadBackward, IndexSelectBackward, WhereBackward } from './function/indexing.js';
import { ReshapeBackward, TransposeBackward, PermuteBackward, SliceBackward, ExpandBackward, SelectBackward } from './function/view.js';
import type { AutogradNode } from './node.js';
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

export function getGradFn(opName: string, args: OpArgs = null): AutogradNode | null {
  const factory = _registry.get(opName);
  return factory ? factory(args) : null;
}

export function hasGradFn(opName: string): boolean {
  return _registry.has(opName);
}
