import { Tensor } from '../tensor/core/tensor.js';
import { TensorImpl } from '../tensor/core/tensor_impl.js';
import { Storage } from '../tensor/core/storage.js';
import { DispatchKey, DispatchKeySet } from '../dispatcher/dispatch_key.js';
import { META_DEVICE } from '../tensor/types/device.js';
import { computeStrides } from '../tensor/utils/shape_utils.js';
import { DYNAMIC } from '../compiler/ir/graph/types.js';
import type { DType } from '../tensor/types/dtype.js';
import type { Tracer } from './tracer.js';
import type { IRValueLike, SymbolicShape } from './types.js';

const TRACING_KEY = DispatchKeySet.fromKey(DispatchKey.TRACING);

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
}
