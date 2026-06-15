import { Vocab } from './vocab.js';
import { WordStrategy } from './strategies/word.js';
import { CharStrategy } from './strategies/char.js';
import { BpeStrategy } from './strategies/bpe.js';
import { tensor } from '../tensor/factory/from_ops.js';

const STRATEGIES = {
  word: WordStrategy,
  char: CharStrategy,
  bpe: BpeStrategy,
};

const DEFAULT_SPECIALS = { pad: '<pad>', unk: '<unk>', bos: '<bos>', eos: '<eos>' };

export class Tokenizer {
  constructor(options = {}) {
    const mode = options.mode ?? 'word';
    const Strategy = STRATEGIES[mode];
    if (!Strategy) throw new Error(`Unknown tokenizer mode '${mode}'. Available: ${Object.keys(STRATEGIES).join(', ')}`);
    this._mode = mode;
    this._specials = { ...DEFAULT_SPECIALS, ...(options.specialTokens ?? {}) };
    this._maxVocab = options.vocabSize ?? null;
    this._strategy = new Strategy(options);
    this._vocab = null;
  }

  get mode() { return this._mode; }
  get vocabSize() { this._ensureFit(); return this._vocab.size; }
  get padId() { return this._specialId('pad'); }
  get unkId() { return this._specialId('unk'); }
  get bosId() { return this._specialId('bos'); }
  get eosId() { return this._specialId('eos'); }

  _ensureFit() {
    if (!this._vocab) throw new Error('Tokenizer must be tokenize()d on a corpus before use');
  }

  _specialId(name) {
    this._ensureFit();
    return this._vocab.getId(this._specials[name]);
  }

  tokenize(texts) {
    const corpus = Array.isArray(texts) ? texts : [texts];
    this._strategy.fit(corpus, { vocabSize: this._maxVocab });
    const vocab = new Vocab(Object.values(this._specials));
    if (this._maxVocab == null) {
      for (const text of corpus) for (const piece of this._strategy.segment(text)) vocab.add(piece);
    } else {
      const freq = new Map();
      for (const text of corpus) for (const piece of this._strategy.segment(text)) freq.set(piece, (freq.get(piece) || 0) + 1);
      const ranked = [...freq.entries()].sort((a, b) => b[1] - a[1]);
      const room = this._maxVocab - vocab.size;
      for (let i = 0; i < ranked.length && i < room; i++) vocab.add(ranked[i][0]);
    }
    this._vocab = vocab;
    return this;
  }

  encode(text, options = {}) {
    this._ensureFit();
    const ids = [];
    if (options.addBos) ids.push(this.bosId);
    const unk = this.unkId;
    for (const piece of this._strategy.segment(text)) ids.push(this._vocab.getId(piece, unk));
    if (options.addEos) ids.push(this.eosId);
    return ids;
  }

  decode(ids, options = {}) {
    this._ensureFit();
    const skipSpecial = options.skipSpecial ?? true;
    const specialIds = new Set(Object.values(this._specials).map((token) => this._vocab.getId(token)));
    const tokens = [];
    for (const id of ids) {
      if (skipSpecial && specialIds.has(id)) continue;
      const token = this._vocab.getToken(id);
      if (token !== undefined) tokens.push(token);
    }
    return this._strategy.detokenize(tokens);
  }

  encodeBatch(texts, options = {}) {
    this._ensureFit();
    const corpus = Array.isArray(texts) ? texts : [texts];
    const rows = corpus.map((text) => this.encode(text, options));
    const maxLen = options.maxLen ?? rows.reduce((longest, row) => Math.max(longest, row.length), 0);
    const padId = options.padId ?? this.padId;
    const count = rows.length;
    const data = new Int32Array(count * maxLen).fill(padId);
    for (let i = 0; i < count; i++) {
      const row = rows[i];
      const limit = Math.min(row.length, maxLen);
      for (let j = 0; j < limit; j++) data[i * maxLen + j] = row[j];
    }
    return tensor(data, { shape: [count, maxLen], dtype: 'i32' });
  }
}
