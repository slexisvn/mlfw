import { describe, it, expect } from 'vitest';
import {
  tensor, zeros, ones, randn,
  Linear, Sequential, ReLU, GELU, SiLU, Sigmoid, Tanh,
  LayerNorm, Softmax,
  compile, CPUTarget, WasmTarget,
} from '../../../src/index.js';
import { Tensor } from '../../../src/tensor/core/tensor.js';

const TARGETS = [
  { name: 'cpu', factory: () => CPUTarget() },
  { name: 'wasm', factory: () => WasmTarget() },
];

function compileFor(model, inputs, target) {
  return compile(model, inputs, { target: target.factory() });
}

function expectFinite(t) {
  const data = t.data;
  for (let i = 0; i < data.length; i++) {
    expect(Number.isFinite(data[i])).toBe(true);
  }
}

function expectClose(a, b, tol = 4) {
  expect(a.shape).toEqual(b.shape);
  const ad = a.data;
  const bd = b.data;
  for (let i = 0; i < ad.length; i++) {
    expect(ad[i]).toBeCloseTo(bd[i], tol);
  }
}

for (const target of TARGETS) {
  describe(`compile+exec [${target.name}]`, () => {

    describe('single Linear', () => {
      it('forward [1,4] -> [1,2]', () => {
        const model = new Sequential(new Linear(4, 2));
        const x = tensor([[1, 2, 3, 4]]);
        const compiled = compileFor(model, [x], target);
        const out = compiled(x);

        expect(out).toBeInstanceOf(Tensor);
        expect(out.shape).toEqual([1, 2]);
        expectFinite(out);
      });

      it('batch [3,4] -> [3,2]', () => {
        const model = new Sequential(new Linear(4, 2));
        const x = tensor([[1, 2, 3, 4], [5, 6, 7, 8], [0.1, 0.2, 0.3, 0.4]]);
        const compiled = compileFor(model, [x], target);
        const out = compiled(x);

        expect(out.shape).toEqual([3, 2]);
        expectFinite(out);
      });

      it('different inputs produce different outputs', () => {
        const model = new Sequential(new Linear(4, 2));
        const x1 = tensor([[1, 0, 0, 0]]);
        const x2 = tensor([[0, 0, 0, 1]]);
        const compiled = compileFor(model, [x1], target);

        const o1 = compiled(x1);
        const o2 = compiled(x2);
        const diff = Math.abs(o1.data[0] - o2.data[0]) + Math.abs(o1.data[1] - o2.data[1]);
        expect(diff).toBeGreaterThan(0);
      });
    });

    describe('Linear + activation', () => {
      it('Linear + ReLU', () => {
        const model = new Sequential(new Linear(4, 4), new ReLU());
        const x = tensor([[1, -1, 2, -2]]);
        const compiled = compileFor(model, [x], target);
        const out = compiled(x);

        expect(out.shape).toEqual([1, 4]);
        expectFinite(out);
        for (let i = 0; i < out.data.length; i++) {
          expect(out.data[i]).toBeGreaterThanOrEqual(0);
        }
      });

      it('Linear + Sigmoid', () => {
        const model = new Sequential(new Linear(3, 3), new Sigmoid());
        const x = tensor([[10, -10, 0]]);
        const compiled = compileFor(model, [x], target);
        const out = compiled(x);

        expect(out.shape).toEqual([1, 3]);
        expectFinite(out);
        for (let i = 0; i < out.data.length; i++) {
          expect(out.data[i]).toBeGreaterThanOrEqual(0);
          expect(out.data[i]).toBeLessThanOrEqual(1);
        }
      });

      it('Linear + Tanh', () => {
        const model = new Sequential(new Linear(3, 3), new Tanh());
        const x = tensor([[5, -5, 0]]);
        const compiled = compileFor(model, [x], target);
        const out = compiled(x);

        expect(out.shape).toEqual([1, 3]);
        expectFinite(out);
        for (let i = 0; i < out.data.length; i++) {
          expect(out.data[i]).toBeGreaterThanOrEqual(-1);
          expect(out.data[i]).toBeLessThanOrEqual(1);
        }
      });
    });

    describe('deep models', () => {
      it('3-layer MLP', () => {
        const model = new Sequential(
          new Linear(8, 16),
          new ReLU(),
          new Linear(16, 8),
          new ReLU(),
          new Linear(8, 4),
        );
        const x = tensor([[1, 2, 3, 4, 5, 6, 7, 8]]);
        const compiled = compileFor(model, [x], target);
        const out = compiled(x);

        expect(out.shape).toEqual([1, 4]);
        expectFinite(out);
      });

      it('4-layer with mixed activations', () => {
        const model = new Sequential(
          new Linear(4, 8),
          new ReLU(),
          new Linear(8, 8),
          new Sigmoid(),
          new Linear(8, 4),
          new Tanh(),
          new Linear(4, 2),
        );
        const x = tensor([[1, 2, 3, 4]]);
        const compiled = compileFor(model, [x], target);
        const out = compiled(x);

        expect(out.shape).toEqual([1, 2]);
        expectFinite(out);
      });

      it('wide layer (64 hidden)', () => {
        const model = new Sequential(
          new Linear(4, 64),
          new ReLU(),
          new Linear(64, 2),
        );
        const x = tensor([[0.5, -0.5, 1.0, -1.0]]);
        const compiled = compileFor(model, [x], target);
        const out = compiled(x);

        expect(out.shape).toEqual([1, 2]);
        expectFinite(out);
      });
    });

    describe('batch sizes', () => {
      it('single sample [1,4]', () => {
        const model = new Sequential(new Linear(4, 2));
        const x = tensor([[1, 2, 3, 4]]);
        const compiled = compileFor(model, [x], target);
        expect(compiled(x).shape).toEqual([1, 2]);
      });

      it('recompiles for different batch size', () => {
        const model = new Sequential(new Linear(4, 2));
        const x1 = tensor([[1, 2, 3, 4]]);
        const compiled = compileFor(model, [x1], target);
        const o1 = compiled(x1);
        expect(o1.shape).toEqual([1, 2]);

        const x4 = tensor([
          [1, 2, 3, 4],
          [5, 6, 7, 8],
          [9, 10, 11, 12],
          [13, 14, 15, 16],
        ]);
        const o4 = compiled(x4);
        expect(o4.shape).toEqual([4, 2]);
        expectFinite(o4);
      });
    });

    describe('numerical stability', () => {
      it('zero input', () => {
        const model = new Sequential(new Linear(4, 2));
        const x = tensor([[0, 0, 0, 0]]);
        const compiled = compileFor(model, [x], target);
        const out = compiled(x);

        expect(out.shape).toEqual([1, 2]);
        expectFinite(out);
      });

      it('large values', () => {
        const model = new Sequential(new Linear(4, 2), new Tanh());
        const x = tensor([[100, -100, 50, -50]]);
        const compiled = compileFor(model, [x], target);
        const out = compiled(x);

        expect(out.shape).toEqual([1, 2]);
        expectFinite(out);
        for (let i = 0; i < out.data.length; i++) {
          expect(out.data[i]).toBeGreaterThanOrEqual(-1);
          expect(out.data[i]).toBeLessThanOrEqual(1);
        }
      });

      it('small values', () => {
        const model = new Sequential(new Linear(4, 2));
        const x = tensor([[1e-7, -1e-7, 1e-7, -1e-7]]);
        const compiled = compileFor(model, [x], target);
        const out = compiled(x);

        expect(out.shape).toEqual([1, 2]);
        expectFinite(out);
      });

      it('repeated execution is deterministic', () => {
        const model = new Sequential(new Linear(4, 2));
        const x = tensor([[1, 2, 3, 4]]);
        const compiled = compileFor(model, [x], target);

        const r1 = compiled(x);
        const r2 = compiled(x);
        const r3 = compiled(x);

        expectClose(r1, r2);
        expectClose(r2, r3);
      });
    });

    if (target.name === 'cpu') {
      describe('cpu-eager vs compiled agreement', () => {
        it('single Linear matches eager', () => {
          const model = new Sequential(new Linear(4, 2));
          const x = tensor([[1, 2, 3, 4]]);

          const eager = model.forward(x);
          const compiled = compileFor(model, [x], target);
          const comp = compiled(x);

          expectClose(eager, comp, 3);
        });

        it('Linear+ReLU matches eager', () => {
          const model = new Sequential(new Linear(3, 3), new ReLU());
          const x = tensor([[1, 2, 3]]);

          const eager = model.forward(x);
          const compiled = compileFor(model, [x], target);
          const comp = compiled(x);

          expectClose(eager, comp, 3);
        });
      });
    }

    describe('lazy compile', () => {
      it('compiles on first call', () => {
        const model = new Sequential(new Linear(4, 2));
        const compiled = compile(model, undefined, { target: target.factory() });
        expect(compiled.kernels()).toEqual([]);

        const x = tensor([[1, 2, 3, 4]]);
        const out = compiled(x);

        expect(out).toBeInstanceOf(Tensor);
        expect(out.shape).toEqual([1, 2]);
        expect(compiled.kernels().length).toBeGreaterThan(0);
      });
    });

    describe('accessors', () => {
      it('.source() returns code', () => {
        const model = new Sequential(new Linear(4, 2));
        const x = tensor([[1, 2, 3, 4]]);
        const compiled = compileFor(model, [x], target);
        const src = compiled.source();
        expect(typeof src).toBe('string');
        expect(src.length).toBeGreaterThan(10);
      });

      it('.kernels() returns names', () => {
        const model = new Sequential(new Linear(4, 2));
        const x = tensor([[1, 2, 3, 4]]);
        const compiled = compileFor(model, [x], target);
        const names = compiled.kernels();
        expect(names.length).toBeGreaterThan(0);
        expect(typeof names[0]).toBe('string');
      });
    });
  });
}
