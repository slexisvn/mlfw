import { Optimizer } from './optimizer.js';
import type { NumberTypedArray, NumberTypedArrayConstructor } from './types.js';

export class AdamW extends Optimizer {
  constructor(params: ConstructorParameters<typeof Optimizer>[0], { lr = 0.001, betas = [0.9, 0.999], eps = 1e-8, weightDecay = 0.01, amsgrad = false } = {}) {
    super(params, { lr, betas: [...betas], eps, weightDecay, amsgrad });
  }

  step(): void {
    for (const group of this._paramGroups) {
      const lr = group.lr as number;
      const betas = group.betas as readonly [number, number];
      const eps = group.eps as number;
      const weightDecay = group.weightDecay as number;
      const amsgrad = group.amsgrad as boolean;
      const [beta1, beta2] = betas;

      for (const p of group.params) {
        if (p.grad === null) continue;
        const w = p._impl.storage.data! as NumberTypedArray;
        const g = p.grad._impl.storage.data! as NumberTypedArray;
        const n = w.length;
        const state = this._getState(p);
        const Ctor = w.constructor as NumberTypedArrayConstructor;

        if (state.step === undefined) {
          state.step = 0;
          state.expAvg = new Ctor(n);
          state.expAvgSq = new Ctor(n);
          if (amsgrad) state.maxExpAvgSq = new Ctor(n);
        }

        state.step = (state.step as number) + 1;
        const t = state.step as number;
        const m = state.expAvg as NumberTypedArray;
        const v = state.expAvgSq as NumberTypedArray;
        const bc1 = 1 - Math.pow(beta1, t);
        const bc2 = 1 - Math.pow(beta2, t);
        const stepSize = lr / bc1;
        const biasCorrection2Sqrt = Math.sqrt(bc2);

        if (weightDecay !== 0) {
          const decay = 1 - lr * weightDecay;
          for (let i = 0; i < n; i++) w[i] *= decay;
        }

        if (amsgrad) {
          const vmax = state.maxExpAvgSq as NumberTypedArray;
          for (let i = 0; i < n; i++) {
            m[i] = beta1 * m[i] + (1 - beta1) * g[i];
            v[i] = beta2 * v[i] + (1 - beta2) * g[i] * g[i];
            if (v[i] > vmax[i]) vmax[i] = v[i];
            w[i] -= stepSize * m[i] / (Math.sqrt(vmax[i]) / biasCorrection2Sqrt + eps);
          }
        } else {
          for (let i = 0; i < n; i++) {
            m[i] = beta1 * m[i] + (1 - beta1) * g[i];
            v[i] = beta2 * v[i] + (1 - beta2) * g[i] * g[i];
            w[i] -= stepSize * m[i] / (Math.sqrt(v[i]) / biasCorrection2Sqrt + eps);
          }
        }
        p._impl.bumpVersion();
      }
    }
  }
}
