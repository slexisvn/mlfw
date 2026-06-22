import { describe, it, expect } from 'vitest';
import { mkdtempSync, rmSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
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

describe('Tokenizer serialization', () => {
  for (const mode of ['word', 'char', 'bpe']) {
    it(`${mode}: saves and loads a fitted tokenizer artifact`, () => {
      const dir = mkdtempSync(join(tmpdir(), 'mlfw-tokenizer-'));
      try {
        const path = join(dir, 'tokenizer.json');
        const t = new Tokenizer({ mode, lowercase: true, numMerges: 50 });
        t.fit(CORPUS);
        t.save(path);
        const raw = readFileSync(path, 'utf8');
        expect(raw).not.toContain('\n  ');
        expect(raw.startsWith('mlfw-tokenizer-v1\n')).toBe(true);
        expect(raw).not.toContain('"format"');
        const loaded = Tokenizer.load(path);
        expect(loaded.vocabSize).toBe(t.vocabSize);
        expect(loaded.padId).toBe(t.padId);
        expect(loaded.unkId).toBe(t.unkId);
        expect(loaded.bosId).toBe(t.bosId);
        expect(loaded.eosId).toBe(t.eosId);
        expect(loaded.encode('hello world', { addBos: true, addEos: true })).toEqual(t.encode('hello world', { addBos: true, addEos: true }));
        expect(loaded.decode(t.encode('hello world'))).toBe('hello world');
        const batch = loaded.encodeBatch(['hello there', 'good night world'], { maxLen: 5 });
        expect(batch.shape).toEqual([2, 5]);
        expect(batch.dtype).toBe('i32');
      } finally {
        rmSync(dir, { recursive: true, force: true });
      }
    });
  }

  it('preserves BPE merge ranks after load', () => {
    const repeated = Array.from({ length: 20 }, () => 'lower lowest newer wider');
    const t = new Tokenizer({ mode: 'bpe', numMerges: 80 });
    t.fit(repeated);
    const loaded = Tokenizer.fromJSON(t.toJSON());
    expect(loaded.encode('lower newer')).toEqual(t.encode('lower newer'));
    expect(loaded.encode('zzzzz')).toEqual(t.encode('zzzzz'));
  });

  it('throws when saving before fit', () => {
    const t = new Tokenizer({ mode: 'word' });
    expect(() => t.save('missing.json')).toThrow(/fit/);
  });

  it('validates malformed tokenizer artifacts', () => {
    const dir = mkdtempSync(join(tmpdir(), 'mlfw-tokenizer-'));
    try {
      const path = join(dir, 'bad.json');
      writeFileSync(path, 'wrong', 'utf8');
      expect(() => Tokenizer.load(path)).toThrow(/format/);
      expect(() => Tokenizer.fromJSON({ format: 'mlfw-tokenizer', version: 999 })).toThrow(/version/);
      expect(() => Tokenizer.fromJSON({
        format: 'mlfw-tokenizer',
        version: 1,
        mode: 'bpe',
        config: { vocabSize: null },
        specialTokens: { pad: '<pad>', unk: '<unk>', bos: '<bos>', eos: '<eos>' },
        vocab: ['<pad>', '<unk>', '<bos>', '<eos>'],
        strategy: { merges: [['a']] },
      })).toThrow(/merges/);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('rejects a vocab size smaller than special tokens', () => {
    expect(() => new Tokenizer({ mode: 'word', vocabSize: 2 })).toThrow(/vocabSize/);
  });
});

describe('unfit tokenizer', () => {
  it('throws a clear error before fit()', () => {
    const t = new Tokenizer({ mode: 'word' });
    expect(() => t.encode('hi')).toThrow(/fit/);
  });
});
