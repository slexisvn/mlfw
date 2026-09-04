import { TensorType } from './types.js';
import { sharesOperandAndResultShape } from './op_traits.js';
import { propagateSymbolicShapes } from './builder.js';
import { carriesSymbol, learnSymbolNames, symbolicDimsOf } from './symbolic_shape.js';
import type { GraphFunction } from './function.js';
import type { Operation } from './operation.js';
import type { Value } from './value.js';
import type { Dim } from './types.js';

function familySymbols(family: readonly Value[]): Dim[] | null {
  for (const value of family) {
    const symbols = symbolicDimsOf(value);
    if (carriesSymbol(symbols)) return symbols;
  }
  return null;
}

export function unifyShapeSymbols(func: GraphFunction): void {
  const ops: Operation[] = [];
  let anyDynamic = false;
  for (const op of func.opsRecursive()) {
    ops.push(op);
    if (!anyDynamic) {
      anyDynamic = op.results.some((r) => r.type instanceof TensorType && r.type.hasDynamic);
    }
  }
  if (!anyDynamic) return;

  const worklist = [...ops];
  const queued = new Set<Operation>(ops);
  const enqueue = (op: Operation | null): void => {
    if (op === null || queued.has(op)) return;
    queued.add(op);
    worklist.push(op);
  };
  const spread = (value: Value): void => {
    enqueue(value.definingOp);
    for (const user of value.getUsers()) enqueue(user);
  };

  while (worklist.length > 0) {
    const op = worklist.pop() as Operation;
    queued.delete(op);

    if (propagateSymbolicShapes(op)) {
      for (const result of op.results) spread(result);
    }
    if (!sharesOperandAndResultShape(op.opName)) continue;

    const family = [...op.operands, ...op.results];
    const known = familySymbols(family);
    if (known === null) continue;
    for (const value of family) {
      if (learnSymbolNames(value, known)) spread(value);
    }
  }
}
