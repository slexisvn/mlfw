import { describe, it, expect } from 'vitest';
import { tensor, one_hot, scatter_add, index_select, gather, log_softmax, mul, sum, mean, neg, reshape, sort, argsort, topk, exp, pad, add, silu } from '../../src/index.js';
import * as nn from '../../src/nn/index.js';
import { compile } from '../../src/tracing/compile.js';
import { WasmTarget, CPUTarget } from '../../src/backend/target.js';
import { mulberry32 } from '../_utils/rng.js';
import { randomTensor, flat } from '../_utils/tensor_data.js';

const SCHEDULED = { fusion: { enabled: true }, scheduling: { enabled: true } };

const deepFlat = (v) => (Array.isArray(v) ? v.flatMap((t) => flat(t)) : flat(v));

async function wasmMatchesEager(label, fwd, inputs, opts = SCHEDULED) {
  const eager = deepFlat(fwd(...inputs));
  const compiled = compile({ forward: (...a) => fwd(...a) }, inputs, { target: WasmTarget(), ...opts });
  if (compiled._ready) await compiled._ready;
  const out = deepFlat(await compiled(...inputs));
  expect(out.length, `${label} output length`).toBe(eager.length);
  for (let i = 0; i < eager.length; i++) {
    const relErr = Math.abs(eager[i] - out[i]) / (1 + Math.abs(eager[i]));
    expect(relErr, `${label} idx ${i}: eager=${eager[i]} wasm=${out[i]}`).toBeLessThan(3e-3);
  }
}

describe('scheduled WASM matches eager when a loop body mixes value dtypes', () => {
  const rng = mulberry32(4242);
  const x = randomTensor(rng, [8, 6]);
  const idx = tensor([0, 1, 2, 3, 4, 5, 0, 1]);

  it('one_hot masked reduction matches eager under fusion + scheduling', async () => {
    await wasmMatchesEager(
      'one_hot masked reduce',
      (a, t) => neg(mean(sum(mul(log_softmax(a, -1), one_hot(t, 6)), -1))),
      [x, idx],
    );
  });

  it('CrossEntropyLoss matches eager under fusion + scheduling', async () => {
    const fc = new nn.Linear(16, 6), ce = new nn.CrossEntropyLoss();
    await wasmMatchesEager(
      'cross entropy',
      (a, t) => ce.forward(fc.forward(a), t),
      [randomTensor(rng, [8, 16]), idx],
    );
  });

  it('scatter_add does not trap under scheduling', async () => {
    const src = randomTensor(rng, [4, 6]);
    const rows = tensor([0, 1, 2, 3]);
    await wasmMatchesEager(
      'scatter_add',
      (a, i) => scatter_add(mul(a, tensor([0.5])), 0, reshape(i, [4, 1]).repeat([1, 6]), a),
      [src, rows],
    );
  });

  it('index_select and gather stay correct under scheduling', async () => {
    const rows = tensor([0, 2, 1, 3]);
    await wasmMatchesEager('index_select', (a, i) => index_select(a, 0, i), [randomTensor(rng, [4, 6]), rows]);
    await wasmMatchesEager('gather', (a, i) => gather(a, 1, reshape(i, [4, 1])), [randomTensor(rng, [4, 6]), rows]);
  });

  it('embedding lookup matches eager under fusion + scheduling', async () => {
    const emb = new nn.Embedding(32, 8), fc = new nn.Linear(8, 4);
    await wasmMatchesEager(
      'embedding bag',
      (ids) => fc.forward(mean(emb.forward(ids), 1)),
      [tensor([[1, 5, 9, 13], [2, 6, 10, 14]])],
    );
  });

  it('uniform-dtype elementwise loops still vectorize', () => {
    const t = randomTensor(rng, [256]);
    const compiled = compile({ forward: (a, b) => mul(a, b) }, [t, t], { target: WasmTarget(), ...SCHEDULED });
    const src = compiled.source();
    expect(src, 'f32-only body must still emit SIMD').toContain('f32x4.mul');
  });

  it('CPU and WASM agree on a one_hot masked reduction', async () => {
    const fwd = (a, t) => neg(mean(sum(mul(log_softmax(a, -1), one_hot(t, 6)), -1)));
    const cpu = compile({ forward: fwd }, [x, idx], { target: CPUTarget(), ...SCHEDULED });
    if (cpu._ready) await cpu._ready;
    const wasmC = compile({ forward: fwd }, [x, idx], { target: WasmTarget(), ...SCHEDULED });
    if (wasmC._ready) await wasmC._ready;
    const a = flat(await cpu(x, idx)), b = flat(await wasmC(x, idx));
    for (let i = 0; i < a.length; i++) {
      expect(Math.abs(a[i] - b[i]) / (1 + Math.abs(a[i])), `idx ${i}`).toBeLessThan(3e-3);
    }
  });
});

describe('scheduled WASM matches eager when loads are indirectly indexed', () => {
  it('sort, argsort and topk hold across reduction widths', async () => {
    for (const width of [3, 5, 6, 8, 9, 12, 17]) {
      const data = randomTensor(mulberry32(200 + width), [4, width]);
      await wasmMatchesEager(`sort C=${width}`, (a) => sort(a, 1), [data]);
      await wasmMatchesEager(`argsort C=${width}`, (a) => argsort(a, 1), [data]);
      await wasmMatchesEager(`topk C=${width}`, (a) => {
        const [values, indices] = topk(a, Math.min(3, width), 1);
        return [values, indices];
      }, [data]);
    }
  });

});

describe('scheduled WASM matches eager when a reduction has a partial final tile', () => {
  it('log_softmax and sum(exp) hold for widths that are not a lane multiple', async () => {
    for (const width of [5, 6, 7, 9, 10, 11, 13, 17, 21]) {
      const data = randomTensor(mulberry32(300 + width), [8, width]);
      await wasmMatchesEager(`log_softmax C=${width}`, (a) => log_softmax(a, -1), [data]);
      await wasmMatchesEager(`sum(exp) C=${width}`, (a) => sum(exp(a), -1), [data]);
    }
  });

});

describe('scheduled WASM matches eager when a guarded load sits under a select', () => {
  it('pad never addresses outside the source buffer', async () => {
    for (const [rows, cols] of [[4, 6], [2, 8], [1, 5], [4, 4], [3, 7], [8, 16]]) {
      const data = randomTensor(mulberry32(31 + cols), [rows, cols]);
      await wasmMatchesEager(`pad cols [${rows},${cols}]`, (a) => pad(a, [0, 1], [0, 1]), [data]);
      await wasmMatchesEager(`pad leading cols [${rows},${cols}]`, (a) => pad(a, [0, 2], [0, 0]), [data]);
      await wasmMatchesEager(`pad rows [${rows},${cols}]`, (a) => pad(a, [1, 0], [1, 0]), [data]);
      await wasmMatchesEager(`pad trailing [${rows},${cols}]`, (a) => pad(a, [0, 0], [0, 2]), [data]);
    }
  });

});

describe('int8 quantization of a deep residual stack runs on WASM', () => {
  function residualStack(depth, width) {
    const lins = Array.from({ length: depth }, () => new nn.Linear(width, width));
    const norms = Array.from({ length: depth }, () => new nn.LayerNorm(width));
    return (x) => {
      let h = x;
      for (let i = 0; i < depth; i++) h = add(h, norms[i].forward(silu(lins[i].forward(h))));
      return h;
    };
  }

  it('completes without a runtime trap', async () => {
    const fwd = residualStack(6, 32);
    const inputs = [randomTensor(mulberry32(2), [8, 32])];
    const compiled = compile({ forward: fwd }, inputs, {
      target: WasmTarget(), quantization: { enabled: true }, fusion: { enabled: true },
    });
    if (compiled._ready) await compiled._ready;
    const out = flat(await compiled(...inputs));
    expect(out.length).toBe(8 * 32);
    expect(out.every(Number.isFinite), 'quantized output must stay finite').toBe(true);
  });
});
