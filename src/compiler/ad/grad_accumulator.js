export function gradOrZero(builder, input, accumulator) {
  const grad = accumulator.get(input.id);
  if (grad) return grad;
  const zeroConst = builder.scalarConstant(0, input.type.dtype).getResult(0);
  return builder.broadcast(zeroConst, input.type.shape, []).getResult(0);
}

export class GradAccumulator {
  constructor(builder) {
    this._builder = builder;
    this._pending = new Map();
    this._reduced = new Map();
  }

  accumulate(valueId, gradValue) {
    if (!gradValue) return;
    let arr = this._pending.get(valueId);
    if (!arr) { arr = []; this._pending.set(valueId, arr); }
    arr.push(gradValue);
    this._reduced.delete(valueId);
  }

  _treeReduce(values) {
    let level = values;
    while (level.length > 1) {
      const next = [];
      for (let i = 0; i < level.length; i += 2) {
        if (i + 1 < level.length) next.push(this._builder.add(level[i], level[i + 1]).getResult(0));
        else next.push(level[i]);
      }
      level = next;
    }
    return level[0];
  }

  get(valueId) {
    if (this._reduced.has(valueId)) return this._reduced.get(valueId);
    const arr = this._pending.get(valueId);
    if (!arr || arr.length === 0) return null;
    const result = this._treeReduce(arr);
    this._reduced.set(valueId, result);
    return result;
  }

  has(valueId) {
    const arr = this._pending.get(valueId);
    return !!arr && arr.length > 0;
  }
}
