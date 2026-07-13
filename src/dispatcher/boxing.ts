enum IValueTag {
  TENSOR = 0,
  INT = 1,
  FLOAT = 2,
  BOOL = 3,
  INT_LIST = 4,
  TENSOR_LIST = 5,
  STRING = 6,
  NONE = 7,
  DEVICE = 8,
  DTYPE = 9,
}

type IValueTagValue = IValueTag;
type BoxedFn = (keySet: unknown, stack: IValue[]) => IValue | IValue[] | unknown;
type UnboxedFn = (keySet: unknown, ...args: unknown[]) => unknown;

export class IValue {
  readonly tag: IValueTagValue;
  readonly value: unknown;

  constructor(tag: IValueTagValue, value: unknown) {
    this.tag = tag;
    this.value = value;
  }

  static tensor(t: unknown) { return new IValue(IValueTag.TENSOR, t); }
  static int(n: number) { return new IValue(IValueTag.INT, n); }
  static float(n: number) { return new IValue(IValueTag.FLOAT, n); }
  static bool(b: boolean) { return new IValue(IValueTag.BOOL, b); }
  static intList(arr: unknown[]) { return new IValue(IValueTag.INT_LIST, arr); }
  static tensorList(arr: unknown[]) { return new IValue(IValueTag.TENSOR_LIST, arr); }
  static string(s: string) { return new IValue(IValueTag.STRING, s); }
  static none() { return new IValue(IValueTag.NONE, null); }
  static device(d: unknown) { return new IValue(IValueTag.DEVICE, d); }
  static dtype(d: unknown) { return new IValue(IValueTag.DTYPE, d); }

  isTensor() { return this.tag === IValueTag.TENSOR; }
  isInt() { return this.tag === IValueTag.INT; }
  isFloat() { return this.tag === IValueTag.FLOAT; }
  isBool() { return this.tag === IValueTag.BOOL; }
  isIntList() { return this.tag === IValueTag.INT_LIST; }
  isTensorList() { return this.tag === IValueTag.TENSOR_LIST; }
  isString() { return this.tag === IValueTag.STRING; }
  isNone() { return this.tag === IValueTag.NONE; }

  toTensor() { return this.value; }
  toInt() { return this.value; }
  toFloat() { return this.value; }
  toBool() { return this.value; }
  toIntList() { return this.value; }
  toTensorList() { return this.value; }
  toString() { return this.value; }
  toDevice() { return this.value; }
  toDtype() { return this.value; }
}

export { IValueTag };

export class KernelFunction {
  private readonly _boxed: BoxedFn | null;
  private readonly _unboxed: UnboxedFn | null;

  constructor(boxed: BoxedFn | null | undefined, unboxed: UnboxedFn | null | undefined) {
    this._boxed = boxed || null;
    this._unboxed = unboxed || null;
  }

  static fromBoxed(fn: BoxedFn) {
    return new KernelFunction(fn, null);
  }

  static fromUnboxed(fn: UnboxedFn) {
    return new KernelFunction(null, fn);
  }

  static fromBoth(boxed: BoxedFn, unboxed: UnboxedFn) {
    return new KernelFunction(boxed, unboxed);
  }

  get isBoxed() { return this._boxed !== null; }
  get isUnboxed() { return this._unboxed !== null; }

  callUnboxed(keySet: unknown, ...args: unknown[]): unknown {
    if (this._unboxed) return this._unboxed(keySet, ...args);
    return this._callBoxedAsUnboxed(keySet, args);
  }

  callBoxed(keySet: unknown, stack: IValue[]): unknown {
    if (this._boxed) return this._boxed(keySet, stack);
    return this._callUnboxedAsBoxed(keySet, stack);
  }

  _callBoxedAsUnboxed(keySet: unknown, args: unknown[]): unknown {
    const stack = args.map(a => _toIValue(a));
    const boxed = this._boxed;
    if (!boxed) return undefined;
    const result = boxed(keySet, stack);
    if (Array.isArray(result)) {
      return result.length === 1 ? result[0].value : result.map(iv => iv.value);
    }
    return result instanceof IValue ? result.value : result;
  }

  _callUnboxedAsBoxed(keySet: unknown, stack: IValue[]): IValue[] {
    const args = stack.map(iv => iv.value);
    const unboxed = this._unboxed;
    if (!unboxed) return [];
    const result = unboxed(keySet, ...args);
    if (result === undefined || result === null) return [];
    return [_toIValue(result)];
  }
}

function hasImpl(val: unknown): boolean {
  return typeof val === 'object' && val !== null && '_impl' in val;
}

function _toIValue(val: unknown): IValue {
  if (val instanceof IValue) return val;
  if (hasImpl(val)) return IValue.tensor(val);
  if (typeof val === 'number') {
    return Number.isInteger(val) ? IValue.int(val) : IValue.float(val);
  }
  if (typeof val === 'boolean') return IValue.bool(val);
  if (typeof val === 'string') return IValue.string(val);
  if (Array.isArray(val)) {
    if (val.length > 0 && hasImpl(val[0])) return IValue.tensorList(val);
    return IValue.intList(val);
  }
  if (val === null || val === undefined) return IValue.none();
  return IValue.none();
}
