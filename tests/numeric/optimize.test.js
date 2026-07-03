import { describe, it, expect } from 'vitest';
import { numeric } from '../../src/index.js';

const {
  nelderMead, differentialEvolution, lbfgs, lbfgsB, levenbergMarquardt, constrainedMinimize,
} = numeric;

const rosenbrock = (p) => (1 - p[0]) ** 2 + 100 * (p[1] - p[0] * p[0]) ** 2;
const sphere = (p) => p.reduce((s, v) => s + v * v, 0);

describe('optimizers', () => {
  it('nelderMead reaches the Rosenbrock minimum', () => {
    const r = nelderMead(rosenbrock, [-1.2, 1]);
    expect(r.converged).toBe(true);
    expect(r.point[0]).toBeCloseTo(1, 4);
    expect(r.point[1]).toBeCloseTo(1, 4);
  });

  it('nelderMead minimizes sphere in 5 dims', () => {
    const r = nelderMead(sphere, [3, -2, 1, 4, -5]);
    expect(r.value).toBeLessThan(1e-8);
  });

  it('lbfgs reaches the Rosenbrock minimum', () => {
    const r = lbfgs(rosenbrock, [-1.2, 1]);
    expect(r.converged).toBe(true);
    expect(r.point[0]).toBeCloseTo(1, 5);
    expect(r.point[1]).toBeCloseTo(1, 5);
  });

  it('lbfgs uses an analytic gradient when provided', () => {
    const grad = (p) => [
      -2 * (1 - p[0]) - 400 * p[0] * (p[1] - p[0] * p[0]),
      200 * (p[1] - p[0] * p[0]),
    ];
    const r = lbfgs(rosenbrock, [-1.2, 1], { gradient: grad });
    expect(r.point[0]).toBeCloseTo(1, 6);
    expect(r.point[1]).toBeCloseTo(1, 6);
  });

  it('lbfgsB respects box bounds', () => {
    const r = lbfgsB(rosenbrock, [0, 0], [[-2, 0.5], [-2, 2]]);
    expect(r.point[0]).toBeLessThanOrEqual(0.5 + 1e-12);
    expect(r.point[0]).toBeCloseTo(0.5, 6);
    expect(r.point[1]).toBeCloseTo(0.25, 4);
  });

  it('differentialEvolution finds global minimum and is seed-deterministic', () => {
    const rastrigin = (p) => 20 + p[0] ** 2 - 10 * Math.cos(2 * Math.PI * p[0])
      + p[1] ** 2 - 10 * Math.cos(2 * Math.PI * p[1]);
    const bounds = [[-5.12, 5.12], [-5.12, 5.12]];
    const a = differentialEvolution(rastrigin, bounds, { seed: 7 });
    const b = differentialEvolution(rastrigin, bounds, { seed: 7 });
    expect(a.value).toBeLessThan(1e-6);
    expect(a.point).toEqual(b.point);
  });

  it('levenbergMarquardt recovers planted parameters', () => {
    const xs = Array.from({ length: 25 }, (_, i) => i / 4);
    const planted = [2.5, -1.3];
    const ys = xs.map((x) => planted[0] * Math.exp(planted[1] * x));
    const residual = (p) => xs.map((x, i) => p[0] * Math.exp(p[1] * x) - ys[i]);
    const r = levenbergMarquardt(residual, [1, -0.5]);
    expect(r.converged).toBe(true);
    expect(r.point[0]).toBeCloseTo(planted[0], 6);
    expect(r.point[1]).toBeCloseTo(planted[1], 6);
  });

  it('levenbergMarquardt accepts a user Jacobian', () => {
    const xs = [0, 1, 2, 3];
    const ys = [1, 3, 5, 7];
    const residual = (p) => xs.map((x, i) => p[0] * x + p[1] - ys[i]);
    const jacobian = () => xs.map((x) => [x, 1]);
    const r = levenbergMarquardt(residual, [0, 0], { jacobian });
    expect(r.point[0]).toBeCloseTo(2, 8);
    expect(r.point[1]).toBeCloseTo(1, 8);
  });

  it('constrainedMinimize honors inequality constraints', () => {
    const r = constrainedMinimize(sphere, [2, 2], {
      inequalities: [(p) => 1 - p[0] - p[1]],
    });
    expect(r.converged).toBe(true);
    expect(r.point[0] + r.point[1]).toBeGreaterThanOrEqual(1 - 1e-4);
    expect(r.point[0]).toBeCloseTo(0.5, 3);
    expect(r.point[1]).toBeCloseTo(0.5, 3);
  });
});
