import { Optimizer } from './optimizer.js';
import { getGpuAdamFn } from '../dispatcher/jit_dispatch.js';
import { DeviceType } from '../tensor/types/device.js';

export class Adam extends Optimizer {
  constructor(params, { lr = 0.001, betas = [0.9, 0.999], eps = 1e-8, weightDecay = 0, amsgrad = false } = {}) {
    super(params, { lr, betas: [...betas], eps, weightDecay, amsgrad });
  }

  step() {
    const gpuAdam = getGpuAdamFn();
    for (const group of this._paramGroups) {
      const { lr, betas, eps, weightDecay, amsgrad } = group;
      const [beta1, beta2] = betas;

      for (const p of group.params) {
        if (p.grad === null) continue;
        const state = this._getState(p);

        if (gpuAdam && !amsgrad && p.device && p.device.type === DeviceType.GPU) {
          state.step = (state.step || 0) + 1;
          const t = state.step;
          const bc1 = 1 - Math.pow(beta1, t);
          const bc2 = 1 - Math.pow(beta2, t);
          if (gpuAdam(p, state, { beta1, beta2, omb1: 1 - beta1, omb2: 1 - beta2, eps, stepSize: lr / bc1, bc2sqrt: Math.sqrt(bc2), wd: weightDecay })) {
            p._impl.bumpVersion();
            continue;
          }
        }

        const w = p._impl.storage.data;
        const g = p.grad._impl.storage.data;
        const n = w.length;

        if (state.expAvg === undefined) {
          if (state.step === undefined) state.step = 0;
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

        if (amsgrad) {
          const vmax = state.maxExpAvgSq;
          for (let i = 0; i < n; i++) {
            const gi = weightDecay !== 0 ? g[i] + weightDecay * w[i] : g[i];
            m[i] = beta1 * m[i] + (1 - beta1) * gi;
            v[i] = beta2 * v[i] + (1 - beta2) * gi * gi;
            if (v[i] > vmax[i]) vmax[i] = v[i];
            w[i] -= stepSize * m[i] / (Math.sqrt(vmax[i]) / biasCorrection2Sqrt + eps);
          }
        } else {
          for (let i = 0; i < n; i++) {
            const gi = weightDecay !== 0 ? g[i] + weightDecay * w[i] : g[i];
            m[i] = beta1 * m[i] + (1 - beta1) * gi;
            v[i] = beta2 * v[i] + (1 - beta2) * gi * gi;
            w[i] -= stepSize * m[i] / (Math.sqrt(v[i]) / biasCorrection2Sqrt + eps);
          }
        }
        p._impl.bumpVersion();
      }
    }
  }
}
