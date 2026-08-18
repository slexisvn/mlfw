import { cat } from '../tensor/ops/ops.js';
import type { NNTensor } from './types.js';

const SEQ_AXIS = 2;

export type KVCacheSlot = {
  append(key: NNTensor, value: NNTensor): [NNTensor, NNTensor];
};

class LayerSlot implements KVCacheSlot {
  private _k: NNTensor | null = null;
  private _v: NNTensor | null = null;

  append(key: NNTensor, value: NNTensor): [NNTensor, NNTensor] {
    this._k = this._k === null ? key : cat([this._k, key], SEQ_AXIS) as NNTensor;
    this._v = this._v === null ? value : cat([this._v, value], SEQ_AXIS) as NNTensor;
    return [this._k, this._v];
  }

  get length(): number {
    return this._k === null ? 0 : this._k.shape[SEQ_AXIS];
  }

  clear(): void {
    this._k = null;
    this._v = null;
  }
}

export class KVCache {
  private _slots: Map<number, LayerSlot> = new Map();

  slot(layerIndex: number): KVCacheSlot {
    let s = this._slots.get(layerIndex);
    if (!s) { s = new LayerSlot(); this._slots.set(layerIndex, s); }
    return s;
  }

  get length(): number {
    let n = 0;
    for (const s of this._slots.values()) n = Math.max(n, s.length);
    return n;
  }

  get layerCount(): number {
    return this._slots.size;
  }

  reset(): void {
    for (const s of this._slots.values()) s.clear();
    this._slots.clear();
  }
}
