import { describe, it, expect } from 'vitest';
import { tensor } from '../../src/index.js';
import * as nn from '../../src/nn/index.js';
import * as T from '../../src/tensor/ops/ops.js';
import { mulberry32 } from '../_utils/rng.js';
import { randomNested, flat } from '../_utils/tensor_data.js';

const maxRelErr = (a, b) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i]) / (1 + Math.abs(v))), 0);

function buildDecoder({ V = 16, D = 8, heads = 2, ff = 16, layers = 2, maxLen = 12, normFirst = true } = {}) {
  const tok = new nn.Embedding(V, D);
  const pos = new nn.Embedding(maxLen, D);
  const enc = new nn.TransformerEncoder(
    new nn.TransformerEncoderLayer(D, heads, ff, 0.0, 'gelu', 1e-5, true, normFirst),
    layers,
  );
  enc.eval();
  const lnf = new nn.LayerNorm(D, 1e-5);
  const head = new nn.Linear(D, V);

  const embed = (ids, startPos) => {
    const n = ids.shape[1];
    const positions = tensor(Array.from({ length: n }, (_, i) => startPos + i), { dtype: 'i32' });
    return T.add(tok.forward(ids), pos.forward(positions));
  };

  return {
    V,
    full: (ids) => head.forward(lnf.forward(enc.forward(embed(ids, 0), null, null, true))),
    step: (ids, startPos, cache) => head.forward(lnf.forward(enc.forward(embed(ids, startPos), null, null, true, cache))),
    layers,
  };
}

function lastRow(logits, V) {
  const f = flat(logits);
  return f.slice(f.length - V);
}

describe('KV cache produces the same logits as recomputing the whole prefix', () => {
  for (const normFirst of [true, false]) {
    it(`${normFirst ? 'pre-norm' : 'post-norm'} decoder, token by token`, () => {
      const rng = mulberry32(normFirst ? 11 : 22);
      const m = buildDecoder({ normFirst });
      const promptIds = [3, 7, 1, 5];
      const cache = new nn.KVCache();

      for (let t = 0; t < promptIds.length; t++) {
        const prefix = promptIds.slice(0, t + 1);
        const viaFull = lastRow(m.full(tensor([prefix], { dtype: 'i32' })), m.V);
        const viaCache = lastRow(m.step(tensor([[promptIds[t]]], { dtype: 'i32' }), t, cache), m.V);
        expect(viaCache.length).toBe(m.V);
        expect(maxRelErr(viaFull, viaCache), `step ${t}`).toBeLessThan(2e-4);
      }
      expect(rng()).toBeGreaterThanOrEqual(0);
    });
  }

  it('a multi-token prefill followed by single-token steps matches full recompute', () => {
    const m = buildDecoder();
    const ids = [2, 4, 6, 8, 10];
    const cache = new nn.KVCache();

    m.step(tensor([ids.slice(0, 3)], { dtype: 'i32' }), 0, cache);
    for (let t = 3; t < ids.length; t++) {
      const viaCache = lastRow(m.step(tensor([[ids[t]]], { dtype: 'i32' }), t, cache), m.V);
      const viaFull = lastRow(m.full(tensor([ids.slice(0, t + 1)], { dtype: 'i32' })), m.V);
      expect(maxRelErr(viaFull, viaCache), `step ${t}`).toBeLessThan(2e-4);
    }
  });

  it('the cache grows one position per decoded token, one slot per layer', () => {
    const m = buildDecoder({ layers: 3 });
    const cache = new nn.KVCache();
    expect(cache.length).toBe(0);

    m.step(tensor([[1, 2]], { dtype: 'i32' }), 0, cache);
    expect(cache.length).toBe(2);
    expect(cache.layerCount).toBe(3);

    m.step(tensor([[3]], { dtype: 'i32' }), 2, cache);
    expect(cache.length).toBe(3);

    cache.reset();
    expect(cache.length).toBe(0);
    expect(cache.layerCount).toBe(0);
  });

  it('two independent caches do not leak into each other', () => {
    const m = buildDecoder();
    const a = new nn.KVCache(), b = new nn.KVCache();
    m.step(tensor([[1, 2, 3]], { dtype: 'i32' }), 0, a);
    const bOnly = lastRow(m.step(tensor([[9]], { dtype: 'i32' }), 0, b), m.V);
    const fresh = lastRow(m.full(tensor([[9]], { dtype: 'i32' })), m.V);
    expect(maxRelErr(fresh, bOnly)).toBeLessThan(2e-4);
    expect(a.length).toBe(3);
    expect(b.length).toBe(1);
  });

  it('greedy decoding with the cache picks the same tokens as without it', () => {
    const m = buildDecoder({ maxLen: 16 });
    const argmaxOf = (row) => row.indexOf(Math.max(...row));

    let prefix = [1, 4];
    const cache = new nn.KVCache();
    m.step(tensor([prefix], { dtype: 'i32' }), 0, cache);

    const cached = [];
    let cursor = prefix.length;
    let next = argmaxOf(lastRow(m.full(tensor([prefix], { dtype: 'i32' })), m.V));
    for (let s = 0; s < 5; s++) {
      cached.push(next);
      next = argmaxOf(lastRow(m.step(tensor([[next]], { dtype: 'i32' }), cursor, cache), m.V));
      cursor++;
    }

    const recomputed = [];
    let ctx = [...prefix];
    for (let s = 0; s < 5; s++) {
      const tok = argmaxOf(lastRow(m.full(tensor([ctx], { dtype: 'i32' })), m.V));
      recomputed.push(tok);
      ctx.push(tok);
    }

    expect(cached).toEqual(recomputed);
  });

  it('decoding with the cache does asymptotically less work than recomputing', () => {
    const m = buildDecoder({ maxLen: 40 });
    const STEPS = 24;

    const cache = new nn.KVCache();
    m.step(tensor([[1]], { dtype: 'i32' }), 0, cache);
    let t0 = performance.now();
    for (let s = 1; s <= STEPS; s++) m.step(tensor([[(s % 15) + 1]], { dtype: 'i32' }), s, cache);
    const cachedMs = performance.now() - t0;

    const ctx = [1];
    m.full(tensor([ctx], { dtype: 'i32' }));
    t0 = performance.now();
    for (let s = 1; s <= STEPS; s++) {
      ctx.push((s % 15) + 1);
      m.full(tensor([ctx], { dtype: 'i32' }));
    }
    const recomputeMs = performance.now() - t0;

    expect(cachedMs, `cached=${cachedMs.toFixed(1)}ms recompute=${recomputeMs.toFixed(1)}ms`).toBeLessThan(recomputeMs);
  });
});

describe('KV cache leaves the non-incremental path untouched', () => {
  it('a forward without a cache is identical before and after cached decoding', () => {
    const rng = mulberry32(404);
    const layer = new nn.TransformerEncoderLayer(8, 2, 16, 0.0, 'gelu', 1e-5, true, false);
    layer.eval();
    const x = tensor(randomNested(rng, [1, 4, 8]));

    const before = flat(layer.forward(x));
    const cache = new nn.KVCache();
    layer.forward(x, null, null, true, cache.slot(0));
    const after = flat(layer.forward(x));

    expect(after).toEqual(before);
  });
});
