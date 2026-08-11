import { describe, it, expect, afterEach } from 'vitest';
import { manual_seed, seed, unseed, randn, randperm, Linear, Dropout, TensorDataset, DataLoader, tensor } from '../../src/index.js';
import { flat } from '../_utils/tensor_data.js';

afterEach(() => unseed());

describe('manual_seed makes framework randomness reproducible', () => {
  it('randn repeats exactly for the same seed and differs for another', () => {
    manual_seed(1234);
    const a = flat(randn([64]));
    manual_seed(1234);
    const b = flat(randn([64]));
    manual_seed(1235);
    const c = flat(randn([64]));

    expect(b).toEqual(a);
    expect(c).not.toEqual(a);
  });

  it('randperm repeats for the same seed and stays a permutation', () => {
    manual_seed(7);
    const a = flat(randperm(64));
    manual_seed(7);
    const b = flat(randperm(64));

    expect(b).toEqual(a);
    expect(a.slice().sort((p, q) => p - q)).toEqual(Array.from({ length: 64 }, (_, i) => i));
  });

  it('layer init is reproducible, which is what training tests depend on', () => {
    manual_seed(99);
    const first = new Linear(8, 4);
    manual_seed(99);
    const second = new Linear(8, 4);

    expect(flat(second.weight)).toEqual(flat(first.weight));
    expect(flat(second.bias)).toEqual(flat(first.bias));
  });

  it('dropout masks repeat for the same seed in training mode', () => {
    const x = tensor(Array.from({ length: 512 }, () => 1));
    const drop = new Dropout(0.5);
    drop.train();

    manual_seed(5);
    const a = flat(drop.forward(x));
    manual_seed(5);
    const b = flat(drop.forward(x));

    expect(b).toEqual(a);
    expect(a.some((v) => v === 0)).toBe(true);
  });

  it('DataLoader shuffling repeats for the same seed', () => {
    const n = 32;
    const ds = () => new TensorDataset(
      tensor(Array.from({ length: n }, (_, i) => i), { shape: [n, 1] }),
      tensor(Array.from({ length: n }, (_, i) => i), { shape: [n, 1] }),
    );
    const order = async () => {
      const out = [];
      for await (const [x] of new DataLoader(ds(), { batchSize: 4, shuffle: true })) out.push(...flat(x));
      return out;
    };

    manual_seed(21);
    const a = order();
    manual_seed(21);
    const b = order();

    return Promise.all([a, b]).then(([first, second]) => {
      expect(second).toEqual(first);
      expect(first.slice().sort((p, q) => p - q)).toEqual(Array.from({ length: n }, (_, i) => i));
    });
  });

  it('seed() reports the active seed and unseed() releases it', () => {
    expect(seed()).toBe(null);
    manual_seed(42);
    expect(seed()).toBe(42);
    unseed();
    expect(seed()).toBe(null);
  });

  it('without a seed two draws differ', () => {
    unseed();
    expect(flat(randn([128]))).not.toEqual(flat(randn([128])));
  });
});
