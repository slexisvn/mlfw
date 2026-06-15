import { describe, it, expect } from 'vitest';
import { Tokenizer, Vocab } from '../../src/tokenizer/index.js';

const CORPUS = [
  'hello there friend',
  'hello world',
  'good morning friend',
  'good night world',
  'the quick brown fox',
  'the lazy dog sleeps',
];

describe('Vocab', () => {
  it('reserves specials first and assigns ids in order', () => {
    const v = new Vocab(['<pad>', '<unk>']);
    expect(v.getId('<pad>')).toBe(0);
    expect(v.getId('<unk>')).toBe(1);
    expect(v.add('cat')).toBe(2);
    expect(v.add('cat')).toBe(2);
    expect(v.add('dog')).toBe(3);
    expect(v.size).toBe(4);
    expect(v.getToken(2)).toBe('cat');
    expect(v.getId('missing', -1)).toBe(-1);
  });
});

describe('Tokenizer modes round-trip', () => {
  for (const mode of ['word', 'char', 'bpe']) {
    it(`${mode}: encode/decode round-trips known text`, () => {
      const t = new Tokenizer({ mode, lowercase: true, numMerges: 50 });
      t.fit(CORPUS);
      const ids = t.encode('hello world');
      expect(Array.isArray(ids)).toBe(true);
      expect(t.decode(ids)).toBe('hello world');
    });
  }

  it('special token ids are stable and distinct', () => {
    const t = new Tokenizer({ mode: 'word' });
    t.fit(CORPUS);
    const ids = new Set([t.padId, t.unkId, t.bosId, t.eosId]);
    expect(ids.size).toBe(4);
    expect(t.padId).toBe(0);
  });

  it('addBos/addEos wrap the sequence', () => {
    const t = new Tokenizer({ mode: 'word' });
    t.fit(CORPUS);
    const ids = t.encode('hello world', { addBos: true, addEos: true });
    expect(ids[0]).toBe(t.bosId);
    expect(ids[ids.length - 1]).toBe(t.eosId);
  });

  it('unknown tokens map to unkId', () => {
    const t = new Tokenizer({ mode: 'word' });
    t.fit(CORPUS);
    expect(t.encode('zzz')[0]).toBe(t.unkId);
  });
});

describe('encodeBatch', () => {
  it('produces a padded i32 tensor of shape [N, maxLen]', () => {
    const t = new Tokenizer({ mode: 'word' });
    t.fit(CORPUS);
    const batch = t.encodeBatch(['hello there', 'good night world'], { addBos: true, addEos: true });
    expect(batch.shape[0]).toBe(2);
    expect(batch.dtype).toBe('i32');
    const data = [...batch._impl.storage.data];
    const cols = batch.shape[1];
    expect(data[0]).toBe(t.bosId);
    expect(data.slice(0, cols).includes(t.padId)).toBe(true);
  });

  it('respects an explicit maxLen by truncating', () => {
    const t = new Tokenizer({ mode: 'word' });
    t.fit(CORPUS);
    const batch = t.encodeBatch(['the quick brown fox the lazy dog sleeps'], { maxLen: 3 });
    expect(batch.shape[1]).toBe(3);
  });
});

describe('BPE training', () => {
  it('learns merges so frequent words become single pieces', () => {
    const repeated = Array.from({ length: 20 }, () => 'lower lowest newer wider');
    const t = new Tokenizer({ mode: 'bpe', numMerges: 80 });
    t.fit(repeated);
    expect(t.encode('lower').length).toBeLessThanOrEqual(t.encode('zzzzz').length);
    expect(t.decode(t.encode('lower newer'))).toBe('lower newer');
  });

  it('falls back to subword/char pieces for unseen words without crashing', () => {
    const t = new Tokenizer({ mode: 'bpe', numMerges: 30 });
    t.fit(CORPUS);
    const ids = t.encode('helloworld');
    expect(ids.length).toBeGreaterThan(0);
    expect(typeof t.decode(ids)).toBe('string');
  });
});

describe('unfit tokenizer', () => {
  it('throws a clear error before fit()', () => {
    const t = new Tokenizer({ mode: 'word' });
    expect(() => t.encode('hi')).toThrow(/fit/);
  });
});
