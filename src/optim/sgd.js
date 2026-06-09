import { Optimizer } from './optimizer.js';

export class SGD extends Optimizer {
  constructor(params, { lr = 0.01, momentum = 0, dampening = 0, weightDecay = 0, nesterov = false } = {}) {
    if (nesterov && (momentum === 0 || dampening !== 0)) {
      throw new Error('Nesterov momentum requires momentum > 0 and dampening = 0');
    }
    super(params, { lr, momentum, dampening, weightDecay, nesterov });
  }

  step() {
    for (const group of this._paramGroups) {
      const { lr, momentum, dampening, weightDecay, nesterov } = group;
      for (const p of group.params) {
        if (p.grad === null) continue;
        const w = p._impl.storage.data;
        const g = p.grad._impl.storage.data;
        const n = w.length;
        const state = this._getState(p);

        if (momentum === 0) {
          if (weightDecay === 0) {
            for (let i = 0; i < n; i++) w[i] -= lr * g[i];
          } else {
            for (let i = 0; i < n; i++) w[i] -= lr * (g[i] + weightDecay * w[i]);
          }
        } else {
          let buf = state.momentumBuffer;
          if (!buf) {
            buf = new w.constructor(n);
            for (let i = 0; i < n; i++) {
              buf[i] = weightDecay !== 0 ? g[i] + weightDecay * w[i] : g[i];
            }
            state.momentumBuffer = buf;
          } else {
            for (let i = 0; i < n; i++) {
              const gi = weightDecay !== 0 ? g[i] + weightDecay * w[i] : g[i];
              buf[i] = momentum * buf[i] + (1 - dampening) * gi;
            }
          }
          if (nesterov) {
            for (let i = 0; i < n; i++) {
              const gi = weightDecay !== 0 ? g[i] + weightDecay * w[i] : g[i];
              w[i] -= lr * (gi + momentum * buf[i]);
            }
          } else {
            for (let i = 0; i < n; i++) w[i] -= lr * buf[i];
          }
        }
        p._impl.bumpVersion();
      }
    }
  }
}
