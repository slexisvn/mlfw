import { TensorType } from '../ir/graph/types.js';
import type { Value } from '../ir/graph/value.js';
import type { IRBuilder } from '../ir/graph/builder.js';

export function gradOrZero(builder: IRBuilder, input: Value, accumulator: GradAccumulator): Value {
  const grad = accumulator.get(input.id);
  if (grad) return grad;
  const zeroConst = builder.scalarConstant(0, (input.type as TensorType).dtype).getResult(0);
  return builder.broadcast(zeroConst, (input.type as TensorType).shape, [], [], input).getResult(0);
}

export class GradAccumulator {
  private _builder: IRBuilder;
  private _pending: Map<number, Value[]>;
  private _reduced: Map<number, Value>;

  constructor(builder: IRBuilder) {
    this._builder = builder;
    this._pending = new Map();
    this._reduced = new Map();
  }

  accumulate(valueId: number, gradValue: Value | null | undefined): void {
    if (!gradValue) return;
    let arr = this._pending.get(valueId);
    if (!arr) { arr = []; this._pending.set(valueId, arr); }
    arr.push(gradValue);
    this._reduced.delete(valueId);
  }

  _treeReduce(values: readonly Value[]): Value {
    let level: readonly Value[] = values;
    while (level.length > 1) {
      const next: Value[] = [];
      for (let i = 0; i < level.length; i += 2) {
        if (i + 1 < level.length) next.push(this._builder.add(level[i], level[i + 1]).getResult(0));
        else next.push(level[i]);
      }
      level = next;
    }
    return level[0];
  }

  get(valueId: number): Value | null {
    if (this._reduced.has(valueId)) return this._reduced.get(valueId) as Value;
    const arr = this._pending.get(valueId);
    if (!arr || arr.length === 0) return null;
    const result = this._treeReduce(arr);
    this._reduced.set(valueId, result);
    return result;
  }

  has(valueId: number): boolean {
    const arr = this._pending.get(valueId);
    return !!arr && arr.length > 0;
  }
}
