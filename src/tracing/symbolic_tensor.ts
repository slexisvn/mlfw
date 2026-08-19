import { Tensor } from '../tensor/core/tensor.js';
import { TensorImpl } from '../tensor/core/tensor_impl.js';
import { Storage } from '../tensor/core/storage.js';
import { DispatchKey, DispatchKeySet } from '../dispatcher/dispatch_key.js';
import { META_DEVICE } from '../tensor/types/device.js';
import { computeStrides } from '../tensor/utils/shape_utils.js';
import { DYNAMIC } from '../compiler/ir/graph/types.js';
import type { DType, NumericTypedArray } from '../tensor/types/dtype.js';
import type { Tracer } from './tracer.js';
import type { IRValueLike, SymbolicShape } from './types.js';

const TRACING_KEY = DispatchKeySet.fromKey(DispatchKey.TRACING);

function _formatShape(shape: readonly number[], symbolicShape: SymbolicShape): string {
  const dims = shape.map((d, i) => {
    if (d !== DYNAMIC) return String(d);
    const sym = symbolicShape && i < symbolicShape.length ? symbolicShape[i] : null;
    return typeof sym === 'string' ? sym : '?';
  });
  return `[${dims.join(', ')}]`;
}

const _NO_VALUE_CAUSE = 'The usual cause is data-dependent control flow: a branch or loop condition computed from tensor contents. A trace cannot capture that, because it records whichever path ran at trace time and silently reuses it for every later input.';

const _NO_VALUE_REMEDY = 'Rewrite the decision as tensor math (where), or express the loop with a region-carrying op: scan (src/tracing/scan.ts) records the loop body as a graph region instead of reading a value. If what you need is a dimension rather than an element, trace with dynamic_shapes so the size stays symbolic.';

function _valueUnavailable(op: string, shape: string, dtype: DType): never {
  const what = `${op} is not available on a symbolic tensor: tracing records operations instead of computing them, so this tensor (shape ${shape}, dtype ${dtype}) carries no value to read.`;
  throw new Error(`${what}\n${_NO_VALUE_CAUSE}\n${_NO_VALUE_REMEDY}`);
}

export class SymbolicTensor extends Tensor {
  private _irValue: IRValueLike;
  private _tracer: Tracer;
  private _symbolicShape: SymbolicShape;

  constructor(irValue: IRValueLike, shape: readonly number[], dtype: DType, tracer: Tracer, symbolicShape: SymbolicShape) {
    const strides = computeStrides(shape);
    const storage = Storage.allocate(0, dtype, META_DEVICE);
    const impl = new TensorImpl(storage, 0, shape, strides, dtype, META_DEVICE);
    super(impl);
    this._irValue = irValue;
    this._tracer = tracer;
    this._symbolicShape = symbolicShape;
  }

  get irValue(): IRValueLike {
    return this._irValue;
  }

  get tracer(): Tracer {
    return this._tracer;
  }

  get symbolicShape(): SymbolicShape {
    return this._symbolicShape;
  }

  get shape(): readonly number[] {
    const raw = super.shape;
    const sym = this._symbolicShape;
    if (!sym) return raw;
    let specialized: number[] | null = null;
    for (let i = 0; i < raw.length; i++) {
      if (raw[i] !== DYNAMIC || i >= sym.length) continue;
      const hint = this._tracer.shapeEnv.specialize(sym[i]);
      if (hint === null) continue;
      if (specialized === null) specialized = [...raw];
      specialized[i] = hint;
    }
    return specialized || raw;
  }

  get dispatchKeySet(): DispatchKeySet {
    return super.dispatchKeySet.union(TRACING_KEY);
  }

  get isSymbolic(): boolean {
    return true;
  }

  get data(): NumericTypedArray | null {
    return this._noValue('.data');
  }

  get storage(): Storage {
    return this._noValue('.storage');
  }

  item(): number | bigint {
    return this._noValue('item()');
  }

  toArray(): ReturnType<Tensor['toArray']> {
    return this._noValue('toArray()');
  }

  _select(dim: number, index: number): Tensor {
    return this._noValue(`_select(${dim}, ${index})`);
  }

  [Symbol.iterator](): Generator<Tensor> {
    return this._noValue('iterating a tensor');
  }

  private _noValue(op: string): never {
    return _valueUnavailable(op, _formatShape(super.shape, this._symbolicShape), this.dtype);
  }
}
