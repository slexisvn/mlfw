import type { Tensor } from '../tensor/core/tensor.js';
import type { Optimizer } from './optimizer.js';
import type { NumberTypedArray } from './types.js';

export class GradScaler {
  enabled: boolean;
  private _scale: number;
  private readonly _growthFactor: number;
  private readonly _backoffFactor: number;
  private readonly _growthInterval: number;
  private _growthTracker: number;
  _foundInf: boolean;
  private _unscaled: WeakSet<Optimizer>;

  constructor(opts: {
    enabled?: boolean;
    initScale?: number;
    growthFactor?: number;
    backoffFactor?: number;
    growthInterval?: number;
  } = {}) {
    this.enabled = opts.enabled !== false;
    this._scale = opts.initScale ?? 65536;
    this._growthFactor = opts.growthFactor ?? 2.0;
    this._backoffFactor = opts.backoffFactor ?? 0.5;
    this._growthInterval = opts.growthInterval ?? 2000;
    this._growthTracker = 0;
    this._foundInf = false;
    this._unscaled = new WeakSet();
  }

  getScale(): number {
    return this.enabled ? this._scale : 1.0;
  }

  get growthTracker(): number {
    return this._growthTracker;
  }

  scale(loss: Tensor): Tensor {
    if (!this.enabled) return loss;
    const d = loss._impl.storage.data! as NumberTypedArray;
    for (let i = 0; i < d.length; i++) d[i] *= this._scale;
    if (loss._impl.bumpVersion) loss._impl.bumpVersion();
    return loss;
  }

  unscale_(optimizer: Optimizer): boolean {
    if (!this.enabled) return false;
    const inv = 1 / this._scale;
    let foundInf = false;
    for (const group of optimizer.paramGroups) {
      for (const p of group.params) {
        if (p.grad === null || p.grad === undefined) continue;
        const g = p.grad._impl.storage.data! as NumberTypedArray;
        for (let i = 0; i < g.length; i++) {
          const v = g[i] * inv;
          if (!Number.isFinite(v)) foundInf = true;
          g[i] = v;
        }
      }
    }
    this._unscaled.add(optimizer);
    if (foundInf) this._foundInf = true;
    return foundInf;
  }

  step(optimizer: Optimizer): boolean {
    if (!this.enabled) {
      optimizer.step();
      return true;
    }
    if (!this._unscaled.has(optimizer)) this.unscale_(optimizer);
    if (this._foundInf) return false;
    optimizer.step();
    return true;
  }

  update(newScale?: number): void {
    if (!this.enabled) return;
    if (newScale !== undefined) {
      this._scale = newScale;
      this._growthTracker = 0;
    } else if (this._foundInf) {
      this._scale *= this._backoffFactor;
      this._growthTracker = 0;
    } else {
      this._growthTracker++;
      if (this._growthTracker >= this._growthInterval) {
        this._scale *= this._growthFactor;
        this._growthTracker = 0;
      }
    }
    this._foundInf = false;
    this._unscaled = new WeakSet();
  }
}
