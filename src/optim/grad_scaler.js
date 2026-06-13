export class GradScaler {
  constructor(opts = {}) {
    this.enabled = opts.enabled !== false;
    this._scale = opts.initScale ?? 65536;
    this._growthFactor = opts.growthFactor ?? 2.0;
    this._backoffFactor = opts.backoffFactor ?? 0.5;
    this._growthInterval = opts.growthInterval ?? 2000;
    this._growthTracker = 0;
    this._foundInf = false;
    this._unscaled = new WeakSet();
  }

  getScale() {
    return this.enabled ? this._scale : 1.0;
  }

  get growthTracker() {
    return this._growthTracker;
  }

  scale(loss) {
    if (!this.enabled) return loss;
    const d = loss._impl.storage.data;
    for (let i = 0; i < d.length; i++) d[i] *= this._scale;
    if (loss._impl.bumpVersion) loss._impl.bumpVersion();
    return loss;
  }

  unscale_(optimizer) {
    if (!this.enabled) return false;
    const inv = 1 / this._scale;
    let foundInf = false;
    for (const group of optimizer.paramGroups) {
      for (const p of group.params) {
        if (p.grad === null || p.grad === undefined) continue;
        const g = p.grad._impl.storage.data;
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

  step(optimizer) {
    if (!this.enabled) {
      optimizer.step();
      return true;
    }
    if (!this._unscaled.has(optimizer)) this.unscale_(optimizer);
    if (this._foundInf) return false;
    optimizer.step();
    return true;
  }

  update(newScale) {
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
