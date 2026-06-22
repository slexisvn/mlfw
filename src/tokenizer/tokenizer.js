import { Vocab } from './vocab.js';
import { WordStrategy } from './strategies/word.js';
import { CharStrategy } from './strategies/char.js';
import { BpeStrategy } from './strategies/bpe.js';
import { tensor } from '../tensor/factory/from_ops.js';
import { fs } from '#io/fs';

export const TOKENIZER_FORMAT = 'mlfw-tokenizer';
export const TOKENIZER_VERSION = 1;
export const DEFAULT_SPECIALS = Object.freeze({ pad: '<pad>', unk: '<unk>', bos: '<bos>', eos: '<eos>' });

const STRATEGIES = {
  word: WordStrategy,
  char: CharStrategy,
  bpe: BpeStrategy,
};

const SPECIAL_KEYS = Object.freeze(['pad', 'unk', 'bos', 'eos']);

export class Tokenizer {
  constructor(options = {}) {
    const mode = options.mode ?? 'word';
    const Strategy = STRATEGIES[mode];
    if (!Strategy) throw new Error(`Unknown tokenizer mode '${mode}'. Available: ${Object.keys(STRATEGIES).join(', ')}`);
    this._mode = mode;
    this._specials = normalizeSpecials(options.specialTokens);
    validateSpecials(this._specials);
    this._maxVocab = options.vocabSize ?? null;
    validateVocabSize(this._maxVocab, this._specials);
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
    if (!this._vocab) throw new Error('Tokenizer must be fit() on a corpus before use');
  }

  _specialId(name) {
    this._ensureFit();
    return this._vocab.getId(this._specials[name]);
  }

  fit(texts) {
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

  toJSON() {
    this._ensureFit();
    const strategy = this._strategy.toJSON();
    return {
      format: TOKENIZER_FORMAT,
      version: TOKENIZER_VERSION,
      mode: this._mode,
      config: {
        vocabSize: this._maxVocab,
        ...strategy,
      },
      specialTokens: { ...this._specials },
      vocab: this._vocab.tokens(),
      strategy,
    };
  }

  save(path) {
    if (typeof path !== 'string') throw new Error('mlfw tokenizer: save(path) requires a file path string');
    const tmp = path + '.tmp';
    fs.writeFile(tmp, JSON.stringify(this.toJSON(), null, 2) + '\n');
    fs.rename(tmp, path);
  }

  static load(path) {
    if (typeof path !== 'string') throw new Error('mlfw tokenizer: load(path) requires a file path string');
    const data = fs.readFile(path);
    const text = typeof data === 'string' ? data : new TextDecoder().decode(data);
    return Tokenizer.fromJSON(JSON.parse(text));
  }

  static fromJSON(data) {
    validateTokenizerData(data);
    const Strategy = STRATEGIES[data.mode];
    const tokenizer = new Tokenizer({
      mode: data.mode,
      vocabSize: data.config.vocabSize,
      specialTokens: data.specialTokens,
    });
    tokenizer._strategy = Strategy.fromJSON(data.strategy);
    tokenizer._vocab = Vocab.fromTokens(data.vocab);
    for (const token of Object.values(tokenizer._specials)) {
      if (!tokenizer._vocab.has(token)) throw new Error(`mlfw tokenizer: special token '${token}' is missing from vocab`);
    }
    return tokenizer;
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

function normalizeSpecials(value) {
  if (Array.isArray(value)) {
    const specials = { ...DEFAULT_SPECIALS };
    for (let i = 0; i < value.length && i < SPECIAL_KEYS.length; i++) specials[SPECIAL_KEYS[i]] = value[i];
    return specials;
  }
  return { ...DEFAULT_SPECIALS, ...(value ?? {}) };
}

function validateSpecials(specials) {
  const seen = new Set();
  for (const key of SPECIAL_KEYS) {
    const token = specials[key];
    if (typeof token !== 'string' || token.length === 0) throw new Error(`mlfw tokenizer: special token '${key}' must be a non-empty string`);
    if (seen.has(token)) throw new Error(`mlfw tokenizer: duplicate special token '${token}'`);
    seen.add(token);
  }
}

function validateVocabSize(vocabSize, specials) {
  if (vocabSize == null) return;
  if (!Number.isInteger(vocabSize) || vocabSize < Object.keys(specials).length) {
    throw new Error(`mlfw tokenizer: vocabSize must be an integer >= ${Object.keys(specials).length}`);
  }
}

function validateTokenizerData(data) {
  if (!data || typeof data !== 'object') throw new Error('mlfw tokenizer: artifact must be an object');
  if (data.format !== TOKENIZER_FORMAT) throw new Error('mlfw tokenizer: unrecognized tokenizer format');
  if (data.version !== TOKENIZER_VERSION) throw new Error(`mlfw tokenizer: unsupported tokenizer version ${data.version}`);
  if (!STRATEGIES[data.mode]) throw new Error(`mlfw tokenizer: unknown tokenizer mode '${data.mode}'`);
  if (!data.config || typeof data.config !== 'object') throw new Error('mlfw tokenizer: config must be an object');
  validateSpecials(normalizeSpecials(data.specialTokens));
  validateVocabSize(data.config.vocabSize, normalizeSpecials(data.specialTokens));
  if (!Array.isArray(data.vocab)) throw new Error('mlfw tokenizer: vocab must be an array');
  if (!data.strategy || typeof data.strategy !== 'object') throw new Error('mlfw tokenizer: strategy must be an object');
}
