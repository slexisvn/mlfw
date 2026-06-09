import { describe, it, expect } from 'vitest';
import {
  tensor, mean, Parameter, LayerNorm,
  MultiheadAttention,
  TransformerEncoderLayer, TransformerDecoderLayer,
  TransformerEncoder, TransformerDecoder,
  Transformer, PositionalEncoding,
} from '../../src/index.js';
import { scaled_dot_product_attention } from '../../src/nn/functional/attention.js';
import { Adam } from '../../src/optim/adam.js';

function randn(shape) {
  const n = shape.reduce((a, b) => a * b, 1);
  const data = Array.from({ length: n }, () => {
    const u1 = Math.random() || 1e-10;
    const u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  });
  return tensor(data, { shape });
}

describe('scaled_dot_product_attention', () => {
  it('produces correct output shape', () => {
    const q = randn([2, 4, 3, 8]);
    const k = randn([2, 4, 5, 8]);
    const v = randn([2, 4, 5, 8]);
    const out = scaled_dot_product_attention(q, k, v);
    expect(out.shape).toEqual([2, 4, 3, 8]);
  });

  it('with causal mask blocks future positions', () => {
    const q = tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], { shape: [1, 1, 3, 3] });
    const k = tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], { shape: [1, 1, 3, 3] });
    const v = tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], { shape: [1, 1, 3, 3] });
    const out = scaled_dot_product_attention(q, k, v, null, 0, true);
    const data = out._impl.storage.data;
    expect(data[2]).toBeCloseTo(0, 1);
    expect(data[5]).toBeCloseTo(0, 1);
  });

  it('with additive mask zeros out masked positions', () => {
    const q = randn([1, 1, 2, 4]);
    const k = randn([1, 1, 3, 4]);
    const v = tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], { shape: [1, 1, 3, 4] });
    const mask = tensor([0, -Infinity, -Infinity, 0, 0, -Infinity], { shape: [1, 1, 2, 3] });
    const out = scaled_dot_product_attention(q, k, v, mask);
    const data = out._impl.storage.data;
    expect(data[0]).toBeCloseTo(1, 4);
    expect(data[1]).toBeCloseTo(0, 4);
    expect(data[2]).toBeCloseTo(0, 4);
    expect(data[3]).toBeCloseTo(0, 4);
  });
});

describe('MultiheadAttention', () => {
  it('self-attention output shape [B, L, E]', () => {
    const mha = new MultiheadAttention(16, 4, 0);
    mha.eval();
    const x = randn([2, 5, 16]);
    const out = mha.forward(x, x, x);
    expect(out.shape).toEqual([2, 5, 16]);
  });

  it('cross-attention with different S', () => {
    const mha = new MultiheadAttention(8, 2, 0);
    mha.eval();
    const q = randn([2, 3, 8]);
    const kv = randn([2, 7, 8]);
    const out = mha.forward(q, kv, kv);
    expect(out.shape).toEqual([2, 3, 8]);
  });

  it('has correct number of parameters (4 Linear modules)', () => {
    const mha = new MultiheadAttention(8, 2, 0, true);
    const params = [...mha.parameters()];
    expect(params.length).toBe(8);
  });

  it('no bias mode', () => {
    const mha = new MultiheadAttention(8, 2, 0, false);
    const params = [...mha.parameters()];
    expect(params.length).toBe(4);
  });

  it('batchFirst=false transposes input/output', () => {
    const mha = new MultiheadAttention(8, 2, 0, true, null, null, false);
    mha.eval();
    const x = randn([5, 2, 8]);
    const out = mha.forward(x, x, x);
    expect(out.shape).toEqual([5, 2, 8]);
  });

  it('causal attention produces finite output', () => {
    const mha = new MultiheadAttention(8, 2, 0);
    mha.eval();
    const x = randn([1, 4, 8]);
    const out = mha.forward(x, x, x, null, null, true);
    const data = out._impl.storage.data;
    for (let i = 0; i < data.length; i++) {
      expect(isFinite(data[i])).toBe(true);
    }
  });
});

describe('TransformerEncoderLayer', () => {
  it('preserves output shape', () => {
    const enc = new TransformerEncoderLayer(8, 2, 16, 0);
    enc.eval();
    const x = randn([2, 5, 8]);
    const out = enc.forward(x);
    expect(out.shape).toEqual([2, 5, 8]);
  });

  it('pre-norm produces finite output', () => {
    const enc = new TransformerEncoderLayer(8, 2, 16, 0, 'relu', 1e-5, true, true);
    enc.eval();
    const x = randn([2, 3, 8]);
    const out = enc.forward(x);
    const data = out._impl.storage.data;
    for (let i = 0; i < data.length; i++) {
      expect(isFinite(data[i])).toBe(true);
    }
  });

  it('gelu activation works', () => {
    const enc = new TransformerEncoderLayer(8, 2, 16, 0, 'gelu');
    enc.eval();
    const x = randn([1, 3, 8]);
    const out = enc.forward(x);
    expect(out.shape).toEqual([1, 3, 8]);
  });

  it('has correct parameter count', () => {
    const enc = new TransformerEncoderLayer(8, 2, 16, 0);
    const params = [...enc.namedParameters()];
    expect(params.length).toBe(16);
  });
});

describe('TransformerDecoderLayer', () => {
  it('output shape with different src/tgt lengths', () => {
    const dec = new TransformerDecoderLayer(8, 2, 16, 0);
    dec.eval();
    const tgt = randn([2, 3, 8]);
    const memory = randn([2, 7, 8]);
    const out = dec.forward(tgt, memory);
    expect(out.shape).toEqual([2, 3, 8]);
  });

  it('pre-norm produces finite output', () => {
    const dec = new TransformerDecoderLayer(8, 2, 16, 0, 'relu', 1e-5, true, true);
    dec.eval();
    const tgt = randn([1, 3, 8]);
    const memory = randn([1, 5, 8]);
    const out = dec.forward(tgt, memory);
    const data = out._impl.storage.data;
    for (let i = 0; i < data.length; i++) {
      expect(isFinite(data[i])).toBe(true);
    }
  });

  it('with causal self-attention', () => {
    const dec = new TransformerDecoderLayer(8, 2, 16, 0);
    dec.eval();
    const tgt = randn([1, 4, 8]);
    const memory = randn([1, 6, 8]);
    const out = dec.forward(tgt, memory, null, null, null, null, true);
    expect(out.shape).toEqual([1, 4, 8]);
  });

  it('has correct parameter count', () => {
    const dec = new TransformerDecoderLayer(8, 2, 16, 0);
    const params = [...dec.namedParameters()];
    expect(params.length).toBe(26);
  });
});

describe('TransformerEncoder', () => {
  it('stacks N layers', () => {
    const layer = new TransformerEncoderLayer(8, 2, 16, 0);
    const encoder = new TransformerEncoder(layer, 3);
    encoder.eval();
    const x = randn([2, 5, 8]);
    const out = encoder.forward(x);
    expect(out.shape).toEqual([2, 5, 8]);
  });

  it('applies optional final norm', () => {
    const layer = new TransformerEncoderLayer(8, 2, 16, 0);
    const norm = new LayerNorm(8);
    const encoder = new TransformerEncoder(layer, 2, norm);
    encoder.eval();
    const x = randn([1, 3, 8]);
    const out = encoder.forward(x);
    expect(out.shape).toEqual([1, 3, 8]);
  });

  it('correct total parameter count for N layers', () => {
    const layer = new TransformerEncoderLayer(8, 2, 16, 0);
    const encoder = new TransformerEncoder(layer, 3);
    const params = [...encoder.parameters()];
    const singleLayerParams = [...new TransformerEncoderLayer(8, 2, 16, 0).parameters()].length;
    expect(params.length).toBe(singleLayerParams * 3);
  });
});

describe('TransformerDecoder', () => {
  it('stacks N layers', () => {
    const layer = new TransformerDecoderLayer(8, 2, 16, 0);
    const decoder = new TransformerDecoder(layer, 2);
    decoder.eval();
    const tgt = randn([2, 3, 8]);
    const memory = randn([2, 5, 8]);
    const out = decoder.forward(tgt, memory);
    expect(out.shape).toEqual([2, 3, 8]);
  });
});

describe('Transformer', () => {
  it('end-to-end forward', () => {
    const model = new Transformer({
      dModel: 8, nhead: 2,
      numEncoderLayers: 2, numDecoderLayers: 2,
      dimFeedforward: 16, dropout: 0,
    });
    model.eval();
    const src = randn([2, 5, 8]);
    const tgt = randn([2, 3, 8]);
    const out = model.forward(src, tgt);
    expect(out.shape).toEqual([2, 3, 8]);
  });

  it('generateSquareSubsequentMask', () => {
    const mask = Transformer.generateSquareSubsequentMask(4);
    expect(mask.shape).toEqual([4, 4]);
    const data = mask._impl.storage.data;
    expect(data[0]).toBe(0);
    expect(data[1]).toBe(-Infinity);
    expect(data[4]).toBe(0);
    expect(data[5]).toBe(0);
    expect(data[6]).toBe(-Infinity);
    expect(data[15]).toBe(0);
  });

  it('with causal decoder mask', () => {
    const model = new Transformer({
      dModel: 8, nhead: 2,
      numEncoderLayers: 1, numDecoderLayers: 1,
      dimFeedforward: 16, dropout: 0,
    });
    model.eval();
    const src = randn([1, 4, 8]);
    const tgt = randn([1, 3, 8]);
    const tgtMask = Transformer.generateSquareSubsequentMask(3);
    const out = model.forward(src, tgt, null, tgtMask);
    expect(out.shape).toEqual([1, 3, 8]);
  });
});

describe('PositionalEncoding', () => {
  it('output shape matches input', () => {
    const pe = new PositionalEncoding(8, 100, 0);
    pe.eval();
    const x = randn([2, 10, 8]);
    const out = pe.forward(x);
    expect(out.shape).toEqual([2, 10, 8]);
  });

  it('sinusoidal values are deterministic', () => {
    const pe1 = new PositionalEncoding(4, 10, 0);
    const pe2 = new PositionalEncoding(4, 10, 0);
    const d1 = pe1.pe._impl.storage.data;
    const d2 = pe2.pe._impl.storage.data;
    for (let i = 0; i < d1.length; i++) {
      expect(d1[i]).toBeCloseTo(d2[i], 6);
    }
  });

  it('PE at position 0 starts with sin(0)=0 cos(0)=1', () => {
    const pe = new PositionalEncoding(4, 10, 0);
    const data = pe.pe._impl.storage.data;
    expect(data[0]).toBeCloseTo(0, 5);
    expect(data[1]).toBeCloseTo(1, 5);
  });

  it('different sequence lengths work', () => {
    const pe = new PositionalEncoding(8, 100, 0);
    pe.eval();
    expect(pe.forward(randn([1, 5, 8])).shape).toEqual([1, 5, 8]);
    expect(pe.forward(randn([1, 50, 8])).shape).toEqual([1, 50, 8]);
  });
});

describe('Transformer autograd', () => {
  it('backward through MultiheadAttention', () => {
    const mha = new MultiheadAttention(8, 2, 0);
    mha.eval();
    const x = randn([2, 3, 8]);
    const out = mha.forward(x, x, x);
    const loss = mean(out);
    loss.backward();
    for (const [, p] of mha.namedParameters()) {
      expect(p.grad).not.toBeNull();
    }
  });

  it('backward through TransformerEncoderLayer', () => {
    const enc = new TransformerEncoderLayer(8, 2, 16, 0);
    enc.eval();
    const x = randn([2, 3, 8]);
    const out = enc.forward(x);
    const loss = mean(out);
    loss.backward();
    for (const [, p] of enc.namedParameters()) {
      expect(p.grad).not.toBeNull();
    }
  });

  it('backward through full Transformer', () => {
    const model = new Transformer({
      dModel: 8, nhead: 2,
      numEncoderLayers: 1, numDecoderLayers: 1,
      dimFeedforward: 16, dropout: 0,
    });
    model.eval();
    const src = randn([1, 4, 8]);
    const tgt = randn([1, 3, 8]);
    const out = model.forward(src, tgt);
    const loss = mean(out);
    loss.backward();
    let total = 0, grads = 0;
    for (const [, p] of model.namedParameters()) {
      total++;
      if (p.grad) grads++;
    }
    expect(grads).toBe(total);
  });

  it('training loop does not produce NaN gradients', () => {
    const enc = new TransformerEncoderLayer(8, 2, 16, 0);
    const x = randn([1, 3, 8]);
    const out = enc.forward(x);
    const loss = mean(out);
    loss.backward();
    for (const [name, p] of enc.namedParameters()) {
      expect(p.grad).not.toBeNull();
      const data = p.grad._impl.storage.data;
      for (let i = 0; i < data.length; i++) {
        expect(isFinite(data[i])).toBe(true);
      }
    }
  });
});
