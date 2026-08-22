import { describe, it, expect } from 'vitest';
import { tensor, compile, compileWithBackward, CPUTarget, WasmTarget } from '../../src/index.js';
import * as nn from '../../src/nn/index.js';
import * as T from '../../src/tensor/ops/ops.js';
import { ones } from '../../src/tensor/factory/creation_ops.js';
import { flat } from '../_utils/tensor_data.js';

const V = 8, D = 4;
const inRange = () => tensor([0, 3, 7], { dtype: 'i32' });
const overflow = () => tensor([0, 3, V], { dtype: 'i32' });
const negative = () => tensor([0, -1, 2], { dtype: 'i32' });

const OOB = /out of range for a table of 8 row\(s\)/;
const POS = /index 99 at position 2/;

describe('embedding rejects out-of-range indices instead of producing NaN', () => {
  it('eager forward throws on an index equal to the table size', () => {
    const emb = new nn.Embedding(V, D);
    expect(() => emb.forward(overflow())).toThrow(OOB);
  });

  it('eager forward throws on a negative index', () => {
    const emb = new nn.Embedding(V, D);
    expect(() => emb.forward(negative())).toThrow(OOB);
  });

  it('eager forward still accepts every valid index', () => {
    const emb = new nn.Embedding(V, D);
    const out = emb.forward(tensor([0, V - 1], { dtype: 'i32' }));
    expect(out.shape).toEqual([2, D]);
    expect(flat(out).every(Number.isFinite)).toBe(true);
  });

  for (const [name, makeTarget] of [['cpu', CPUTarget], ['wasm', WasmTarget]]) {
    it(`compiled model on ${name} throws when a later call passes an out-of-range index`, async () => {
      const emb = new nn.Embedding(V, D);
      const head = new nn.Linear(D, 2);
      const fwd = (x) => head.forward(emb.forward(x));

      const compiled = compile({ forward: fwd }, [inRange()], { target: makeTarget() });
      const good = await compiled(inRange());
      expect(flat(good).every(Number.isFinite)).toBe(true);

      await expect(async () => await compiled(overflow())).rejects.toThrow(OOB);
    });
  }

  it('compiled backward throws on an out-of-range index too', () => {
    const emb = new nn.Embedding(V, D);
    const head = new nn.Linear(D, 2);
    const cf = compileWithBackward({ forward: (x) => head.forward(emb.forward(x)) }, [inRange()], { target: CPUTarget() });

    const out = cf(inRange());
    expect(flat(cf.backward(ones(out.shape))[1]).every(Number.isFinite)).toBe(true);

    expect(() => cf(overflow())).toThrow(OOB);
  });

  it('the check names the offending index and its position', () => {
    const emb = new nn.Embedding(V, D);
    expect(() => emb.forward(tensor([1, 1, 99], { dtype: 'i32' })))
      .toThrow(POS);
  });

  it('an index tensor that views a larger padded buffer is judged on its own elements', () => {
    const x = tensor([[5, 3, 1, 4, 2], [-1, -5, 0, 9, 6]]);
    const idx = T.argsort(x);
    expect(idx._impl.storage.data.length).toBeGreaterThan(idx.shape[0] * idx.shape[1]);
    expect(() => T.gather(x, 1, idx)).not.toThrow();
    expect(flat(T.gather(x, 1, idx))).toEqual(flat(T.sort(x)));
  });

  it('a transposed (non-contiguous) index tensor is walked by strides', () => {
    const emb = new nn.Embedding(V, D);
    const idx = T.transpose(tensor([[0, 1], [2, 3]], { dtype: 'i32' }), 0, 1);
    expect(() => emb.forward(idx)).not.toThrow();
    const bad = T.transpose(tensor([[0, 1], [2, 99]], { dtype: 'i32' }), 0, 1);
    expect(() => emb.forward(bad)).toThrow(/index 99/);
  });

  it('a computed index is guarded on the argument it was derived from', async () => {
    const emb = new nn.Embedding(V, D);
    const compiled = compile({ forward: (x) => emb.forward(T.add(x, 1)) }, [tensor([0, 1, 2], { dtype: 'i32' })], { target: CPUTarget() });

    const good = await compiled(tensor([0, 1, 2], { dtype: 'i32' }));
    expect(flat(good).every(Number.isFinite)).toBe(true);

    await expect(async () => await compiled(tensor([0, 1, V - 1], { dtype: 'i32' }))).rejects.toThrow(/out of range/);
  });

  it('a computed index shifted the other way admits the value the shift makes valid', async () => {
    const emb = new nn.Embedding(V, D);
    const compiled = compile({ forward: (x) => emb.forward(T.sub(x, 1)) }, [tensor([1, 2, 3], { dtype: 'i32' })], { target: CPUTarget() });

    const good = await compiled(tensor([1, 2, V], { dtype: 'i32' }));
    expect(flat(good).every(Number.isFinite)).toBe(true);

    await expect(async () => await compiled(tensor([0, 2, 3], { dtype: 'i32' }))).rejects.toThrow(/out of range/);
  });

  it('an index tensor that is only reachable through a captured weight is not falsely rejected', () => {
    const emb = new nn.Embedding(V, D);
    const fixed = tensor([1, 2], { dtype: 'i32' });
    const compiled = compile({ forward: (x) => emb.forward(fixed).add(x) }, [tensor([[0, 0, 0, 0], [0, 0, 0, 0]])], { target: CPUTarget() });
    expect(flat(compiled(tensor([[0, 0, 0, 0], [0, 0, 0, 0]]))).every(Number.isFinite)).toBe(true);
  });
});
