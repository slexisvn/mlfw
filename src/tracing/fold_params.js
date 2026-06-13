import { Operation } from '../compiler/ir/graph/operation.js';

const FLOAT_DTYPES = new Set(['f16', 'f32', 'f64']);

export function defaultWeightPredicate(param) {
  return param && param.shape && param.shape.length >= 2 && FLOAT_DTYPES.has(param.dtype);
}

export function foldWeightParams(traced, getData, shouldFold = defaultWeightPredicate) {
  const func = traced.graph.functions().next().value;
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
    const c = new Operation('constant', [], [tt], { value: data, tensor_type: tt });
    const first = entry.firstOp;
    if (first) entry.insertBefore(c, first);
    else entry.pushOp(c);
    f.arg.replaceAllUsesWith(c.getResult(0));
  }

  const removeSet = new Set(folds.map(f => f.argIndex));
  entry.removeArguments(removeSet);
  func.inputTypes = Object.freeze(func.inputTypes.filter((_, i) => !removeSet.has(i)));

  const foldedJ = new Set(folds.map(f => f.j));
  const newParams = params.filter((_, j) => !foldedJ.has(j));

  return { ...traced, capturedParams: newParams };
}
