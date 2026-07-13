import { Optimizer } from './optimizer.js';
import type { NumberTypedArray, NumberTypedArrayConstructor } from './types.js';

export class SGD extends Optimizer {
  constructor(params: ConstructorParameters<typeof Optimizer>[0], { lr = 0.01, momentum = 0, dampening = 0, weightDecay = 0, nesterov = false } = {}) {
    if (nesterov && (momentum === 0 || dampening !== 0)) {
      throw new Error('Nesterov momentum requires momentum > 0 and dampening = 0');
    }
    super(params, { lr, momentum, dampening, weightDecay, nesterov });
  }

  step(): void {
    for (const group of this._paramGroups) {
      const lr = group.lr as number;
      const momentum = group.momentum as number;
      const dampening = group.dampening as number;
      const weightDecay = group.weightDecay as number;
      const nesterov = group.nesterov as boolean;
      for (const p of group.params) {
        if (p.grad === null) continue;
        const w = p._impl.storage.data! as NumberTypedArray;
        const g = p.grad._impl.storage.data! as NumberTypedArray;
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
            buf = new (w.constructor as NumberTypedArrayConstructor)(n);
            for (let i = 0; i < n; i++) {
              buf[i] = weightDecay !== 0 ? g[i] + weightDecay * w[i] : g[i];
            }
            state.momentumBuffer = buf;
          } else {
            buf = buf as NumberTypedArray;
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
