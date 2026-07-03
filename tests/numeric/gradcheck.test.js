import { describe, it, expect } from 'vitest';
import { tensor, sum, erf, erfc, lgamma, gamma, numeric } from '../../src/index.js';

const FD_STEP = 1e-4;

function gradcheck(op, points, precision) {
  const t = tensor(new Float64Array(points), { shape: [points.length], dtype: 'f64', requiresGrad: true });
  const y = sum(op(t));
  y.backward();
  const analytic = t.grad.toArray();
  for (let i = 0; i < points.length; i++) {
    const plus = points.slice();
    const minus = points.slice();
    plus[i] += FD_STEP;
    minus[i] -= FD_STEP;
    const fp = sum(op(tensor(new Float64Array(plus), { shape: [points.length], dtype: 'f64' }))).item();
    const fm = sum(op(tensor(new Float64Array(minus), { shape: [points.length], dtype: 'f64' }))).item();
    const numeric_grad = (fp - fm) / (2 * FD_STEP);
    expect(analytic[i]).toBeCloseTo(numeric_grad, precision);
  }
}

describe('gradcheck for special-function autograd', () => {
  it('erf gradient matches finite differences', () => {
    gradcheck(erf, [-1.5, -0.3, 0, 0.7, 2.1], 4);
  });

  it('erfc gradient matches finite differences', () => {
    gradcheck(erfc, [-1.1, 0.2, 1.4], 4);
  });

  it('lgamma gradient matches finite differences', () => {
    gradcheck(lgamma, [0.4, 1.3, 3.7, 8.2], 5);
  });

  it('gamma gradient matches finite differences', () => {
    gradcheck(gamma, [0.6, 1.5, 3.2], 5);
  });

  it('lgamma gradient equals digamma', () => {
    const pts = [0.7, 2.4, 6.1];
    const t = tensor(new Float64Array(pts), { shape: [3], dtype: 'f64', requiresGrad: true });
    sum(lgamma(t)).backward();
    const g = t.grad.toArray();
    for (let i = 0; i < pts.length; i++) {
      expect(g[i]).toBeCloseTo(numeric.digammaScalar(pts[i]), 8);
    }
  });
});
