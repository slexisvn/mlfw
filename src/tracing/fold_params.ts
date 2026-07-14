import { Operation } from '../compiler/ir/graph/operation.js';
import type { Tensor } from '../tensor/core/tensor.js';
import type { AttrMap, GraphFunctionLike, IRValueLike, IROperationLike, TracedCore } from './types.js';

const FLOAT_DTYPES = new Set(['f16', 'f32', 'f64']);

type FoldPredicate = (param: Tensor, index?: number, arg?: IRValueLike) => boolean;
type GetData = (param: Tensor) => unknown;
type OperationCtor = new (opName: string, operands: IRValueLike[], resultTypes: readonly unknown[], attributes?: AttrMap | Map<string, unknown> | null, regions?: unknown) => IROperationLike;

export function defaultWeightPredicate(param: Tensor): boolean {
  return param && param.shape && param.shape.length >= 2 && FLOAT_DTYPES.has(param.dtype);
}

export function foldWeightParams(traced: TracedCore, getData: GetData, shouldFold: FoldPredicate = defaultWeightPredicate): TracedCore {
  const func = traced.graph.functions().next().value as GraphFunctionLike | undefined;
  if (!func) return traced;
  const entry = func.entryBlock;
  const numUser = traced.numUserInputs;
  const params = traced.capturedParams;

  const folds = [];
  for (let j = 0; j < params.length; j++) {
    const argIndex = numUser + j;
    const arg = entry.getArgument(argIndex);
    if (!arg) continue;
    if (!shouldFold(params[j], j, arg)) continue;
    folds.push({ j, argIndex, arg, param: params[j] });
  }
  if (folds.length === 0) return traced;

  for (const f of folds) {
    const data = getData(f.param);
    const tt = f.arg.type;
    const OperationClass = Operation as unknown as OperationCtor;
    const c = new OperationClass('constant', [], [tt], { value: data, tensor_type: tt });
    const first = entry.firstOp;
    if (first) entry.insertBefore(c, first);
    else entry.pushOp(c);
    f.arg.replaceAllUsesWith!(c.getResult(0));
  }

  const removeSet = new Set(folds.map(f => f.argIndex));
  entry.removeArguments(removeSet);
  func.inputTypes = Object.freeze(func.inputTypes.filter((_type, i) => !removeSet.has(i)));

  const foldedJ = new Set(folds.map(f => f.j));
  const newParams = params.filter((_param, j) => !foldedJ.has(j));

  return { ...traced, capturedParams: newParams };
}
