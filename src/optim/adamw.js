import { Optimizer } from './optimizer.js';

export class AdamW extends Optimizer {
  constructor(params, { lr = 0.001, betas = [0.9, 0.999], eps = 1e-8, weightDecay = 0.01, amsgrad = false } = {}) {
    super(params, { lr, betas: [...betas], eps, weightDecay, amsgrad });
  }

  step() {
    for (const group of this._paramGroups) {
      const { lr, betas, eps, weightDecay, amsgrad } = group;
      const [beta1, beta2] = betas;

      for (const p of group.params) {
        if (p.grad === null) continue;
        const w = p._impl.storage.data;
        const g = p.grad._impl.storage.data;
        const n = w.length;
        const state = this._getState(p);

        if (state.step === undefined) {
          state.step = 0;
          state.expAvg = new w.constructor(n);
          state.expAvgSq = new w.constructor(n);
          if (amsgrad) state.maxExpAvgSq = new w.constructor(n);
        }

        state.step++;
        const t = state.step;
        const m = state.expAvg;
        const v = state.expAvgSq;
        const bc1 = 1 - Math.pow(beta1, t);
        const bc2 = 1 - Math.pow(beta2, t);
        const stepSize = lr / bc1;
        const biasCorrection2Sqrt = Math.sqrt(bc2);

        if (weightDecay !== 0) {
          const decay = 1 - lr * weightDecay;
          for (let i = 0; i < n; i++) w[i] *= decay;
        }

        if (amsgrad) {
          const vmax = state.maxExpAvgSq;
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
