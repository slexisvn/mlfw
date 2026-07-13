import { Vocab } from './vocab.js';
import { WordStrategy } from './strategies/word.js';
import { CharStrategy } from './strategies/char.js';
import { BpeStrategy } from './strategies/bpe.js';
import { tensor } from '../tensor/factory/from_ops.js';
import { fs } from '#io/fs';
import type { Tensor } from '../tensor/core/tensor.js';

export const TOKENIZER_FORMAT = 'mlfw-tokenizer';
export const TOKENIZER_VERSION = 1;
export const DEFAULT_SPECIALS = Object.freeze({ pad: '<pad>', unk: '<unk>', bos: '<bos>', eos: '<eos>' });

type TokenizerMode = 'word' | 'char' | 'bpe';
type SpecialKey = 'pad' | 'unk' | 'bos' | 'eos';
type SpecialTokens = Record<SpecialKey, string>;
type PartialSpecialTokens = Partial<SpecialTokens> | readonly string[];
type Strategy = {
  fit(texts: readonly string[], options: { vocabSize: number | null }): void;
  segment(text: string): string[];
  detokenize(tokens: readonly string[]): string;
  toJSON(): StrategyData;
};
type StrategyConstructor = {
  new(options?: TokenizerOptions): Strategy;
  fromJSON(data?: StrategyData): Strategy;
};
type StrategyData = {
  lowercase?: boolean;
  numMerges?: number;
  endOfWord?: string;
  merges?: [string, string][];
};
type TokenizerOptions = {
  mode?: TokenizerMode;
  specialTokens?: PartialSpecialTokens;
  vocabSize?: number | null;
  lowercase?: boolean;
  numMerges?: number;
  endOfWord?: string;
};
type EncodeOptions = {
  addBos?: boolean;
  addEos?: boolean;
};
type DecodeOptions = {
  skipSpecial?: boolean;
};
type EncodeBatchOptions = EncodeOptions & {
  maxLen?: number;
  padId?: number;
};
type TokenizerJSON = {
  format: typeof TOKENIZER_FORMAT;
  version: typeof TOKENIZER_VERSION;
  mode: TokenizerMode;
  config: StrategyData & { vocabSize: number | null };
  specialTokens: SpecialTokens;
  vocab: string[];
  strategy: StrategyData;
};

const TOKENIZER_ARTIFACT_HEADER = `${TOKENIZER_FORMAT}-v${TOKENIZER_VERSION}`;
const STRATEGIES: Record<TokenizerMode, StrategyConstructor> = {
  word: WordStrategy,
  char: CharStrategy,
  bpe: BpeStrategy,
};

const SPECIAL_KEYS = Object.freeze(['pad', 'unk', 'bos', 'eos'] as const);

export class Tokenizer {
  private _mode: TokenizerMode;
  private _specials: SpecialTokens;
  private _maxVocab: number | null;
  private _strategy: Strategy;
  private _vocab: Vocab | null;

  constructor(options: TokenizerOptions = {}) {
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

  get mode(): TokenizerMode { return this._mode; }
  get vocabSize(): number { return this._fitVocab().size; }
  get padId(): number { return this._specialId('pad'); }
  get unkId(): number { return this._specialId('unk'); }
  get bosId(): number { return this._specialId('bos'); }
  get eosId(): number { return this._specialId('eos'); }

  _ensureFit(): void {
    if (!this._vocab) throw new Error('Tokenizer must be fit() on a corpus before use');
  }

  _fitVocab(): Vocab {
    this._ensureFit();
    return this._vocab!;
  }

  _specialId(name: SpecialKey): number {
    const vocab = this._fitVocab();
    return vocab.getId(this._specials[name]);
  }

  fit(texts: string | readonly string[]): this {
    const corpus = Array.isArray(texts) ? texts : [texts];
    this._strategy.fit(corpus, { vocabSize: this._maxVocab });
    const vocab = new Vocab(Object.values(this._specials));
    if (this._maxVocab == null) {
      for (const text of corpus) for (const piece of this._strategy.segment(text)) vocab.add(piece);
    } else {
      const freq = new Map<string, number>();
      for (const text of corpus) for (const piece of this._strategy.segment(text)) freq.set(piece, (freq.get(piece) || 0) + 1);
      const ranked = [...freq.entries()].sort((a, b) => b[1] - a[1]);
      const room = this._maxVocab - vocab.size;
      for (let i = 0; i < ranked.length && i < room; i++) vocab.add(ranked[i][0]);
    }
    this._vocab = vocab;
    return this;
  }

  toJSON(): TokenizerJSON {
    const vocab = this._fitVocab();
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
      vocab: vocab.tokens(),
      strategy,
    };
  }

  save(path: string): void {
    if (typeof path !== 'string') throw new Error('mlfw tokenizer: save(path) requires a file path string');
    const tmp = path + '.tmp';
    fs.writeFile(tmp, serializeTokenizer(this.toJSON()));
    fs.rename(tmp, path);
  }

  static load(path: string): Tokenizer {
    if (typeof path !== 'string') throw new Error('mlfw tokenizer: load(path) requires a file path string');
    const data = fs.readFile(path);
    const text = typeof data === 'string' ? data : new TextDecoder().decode(data);
    return Tokenizer.fromJSON(parseTokenizer(text));
  }

  static fromJSON(data: unknown): Tokenizer {
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

  encode(text: string, options: EncodeOptions = {}): number[] {
    const vocab = this._fitVocab();
    const ids: number[] = [];
    if (options.addBos) ids.push(this.bosId);
    const unk = this.unkId;
    for (const piece of this._strategy.segment(text)) ids.push(vocab.getId(piece, unk));
    if (options.addEos) ids.push(this.eosId);
    return ids;
  }

  decode(ids: Iterable<number>, options: DecodeOptions = {}): string {
    const vocab = this._fitVocab();
    const skipSpecial = options.skipSpecial ?? true;
    const specialIds = new Set(Object.values(this._specials).map((token) => vocab.getId(token)));
    const tokens: string[] = [];
    for (const id of ids) {
      if (skipSpecial && specialIds.has(id)) continue;
      const token = vocab.getToken(id);
      if (token !== undefined) tokens.push(token);
    }
    return this._strategy.detokenize(tokens);
  }

  encodeBatch(texts: string | readonly string[], options: EncodeBatchOptions = {}): Tensor {
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

function normalizeSpecials(value?: PartialSpecialTokens): SpecialTokens {
  if (Array.isArray(value)) {
    const specials: SpecialTokens = { ...DEFAULT_SPECIALS };
    for (let i = 0; i < value.length && i < SPECIAL_KEYS.length; i++) specials[SPECIAL_KEYS[i]] = value[i]!;
    return specials;
  }
  return { ...DEFAULT_SPECIALS, ...(value ?? {}) };
}

function validateSpecials(specials: SpecialTokens): void {
  const seen = new Set();
  for (const key of SPECIAL_KEYS) {
    const token = specials[key];
    if (typeof token !== 'string' || token.length === 0) throw new Error(`mlfw tokenizer: special token '${key}' must be a non-empty string`);
    if (seen.has(token)) throw new Error(`mlfw tokenizer: duplicate special token '${token}'`);
    seen.add(token);
  }
}

function validateVocabSize(vocabSize: number | null | undefined, specials: SpecialTokens): void {
  if (vocabSize == null) return;
  if (!Number.isInteger(vocabSize) || vocabSize < Object.keys(specials).length) {
    throw new Error(`mlfw tokenizer: vocabSize must be an integer >= ${Object.keys(specials).length}`);
  }
}

function validateTokenizerData(data: unknown): asserts data is TokenizerJSON {
  if (!data || typeof data !== 'object') throw new Error('mlfw tokenizer: artifact must be an object');
  const artifact = data as Record<string, unknown>;
  if (artifact.format !== TOKENIZER_FORMAT) throw new Error('mlfw tokenizer: unrecognized tokenizer format');
  if (artifact.version !== TOKENIZER_VERSION) throw new Error(`mlfw tokenizer: unsupported tokenizer version ${artifact.version}`);
  if (typeof artifact.mode !== 'string' || !(artifact.mode in STRATEGIES)) throw new Error(`mlfw tokenizer: unknown tokenizer mode '${artifact.mode}'`);
  if (!artifact.config || typeof artifact.config !== 'object') throw new Error('mlfw tokenizer: config must be an object');
  const config = artifact.config as Record<string, unknown>;
  const specials = normalizeSpecials(asSpecialTokensInput(artifact.specialTokens));
  validateSpecials(specials);
  validateVocabSize(asNullableNumber(config.vocabSize), specials);
  if (!Array.isArray(artifact.vocab)) throw new Error('mlfw tokenizer: vocab must be an array');
  if (!artifact.strategy || typeof artifact.strategy !== 'object') throw new Error('mlfw tokenizer: strategy must be an object');
}

function serializeTokenizer(data: TokenizerJSON): string {
  const out = [
    TOKENIZER_ARTIFACT_HEADER,
    record('m', [data.mode]),
    record('z', [data.config.vocabSize == null ? '' : String(data.config.vocabSize)]),
    record('s', SPECIAL_KEYS.map(key => data.specialTokens[key])),
  ];
  if (data.mode === 'bpe') {
    const merges: string[] = [];
    for (const pair of data.strategy.merges!) merges.push(pair[0], pair[1]);
    out.push(record('g', [String(data.strategy.lowercase ? 1 : 0), String(data.strategy.numMerges), data.strategy.endOfWord!]));
    out.push(record('r', merges));
  } else {
    out.push(record('g', [String(data.strategy.lowercase ? 1 : 0)]));
  }
  out.push(record('v', data.vocab));
  return out.join('\n');
}

function parseTokenizer(text: string): TokenizerJSON {
  if (!text.startsWith(TOKENIZER_ARTIFACT_HEADER)) throw new Error('mlfw tokenizer: unrecognized tokenizer format');
  const records = new Map<string, string[]>();
  for (const line of text.split(/\r?\n/).slice(1)) {
    if (!line) continue;
    const tab = line.indexOf('\t');
    if (tab < 0) throw new Error('mlfw tokenizer: malformed tokenizer artifact');
    records.set(line.slice(0, tab), parseRecord(line.slice(tab + 1)));
  }
  const mode = requiredField(records, 'm')[0];
  const rawVocabSize = requiredField(records, 'z')[0];
  const specials = requiredField(records, 's');
  const strategyFields = requiredField(records, 'g');
  const vocab = requiredField(records, 'v');
  const specialTokens = Object.fromEntries(SPECIAL_KEYS.map((key, i) => [key, specials[i]]));
  const strategy: StrategyData = { lowercase: strategyFields[0] === '1' };
  if (mode === 'bpe') {
    const rawMerges = requiredField(records, 'r');
    if (rawMerges.length % 2 !== 0) throw new Error('mlfw tokenizer: bpe merges must be string pairs');
    strategy.numMerges = Number(strategyFields[1]);
    strategy.endOfWord = strategyFields[2];
    strategy.merges = [];
    for (let i = 0; i < rawMerges.length; i += 2) strategy.merges.push([rawMerges[i], rawMerges[i + 1]]);
  }
  return {
    format: TOKENIZER_FORMAT,
    version: TOKENIZER_VERSION,
    mode: mode as TokenizerMode,
    config: { vocabSize: rawVocabSize === '' ? null : Number(rawVocabSize), ...strategy },
    specialTokens: specialTokens as SpecialTokens,
    vocab,
    strategy,
  };
}

function record(name: string, fields: readonly string[]): string {
  return `${name}\t${fields.map(escapeField).join('\t')}`;
}

function parseRecord(text: string): string[] {
  return text.split('\t').map(unescapeField);
}

function escapeField(value: string): string {
  return String(value)
    .replace(/\\/g, '\\\\')
    .replace(/\t/g, '\\t')
    .replace(/\n/g, '\\n')
    .replace(/\r/g, '\\r');
}

function unescapeField(value: string): string {
  let out = '';
  for (let i = 0; i < value.length; i++) {
    const ch = value[i];
    if (ch !== '\\') {
      out += ch;
      continue;
    }
    const next = value[++i];
    if (next === 't') out += '\t';
    else if (next === 'n') out += '\n';
    else if (next === 'r') out += '\r';
    else if (next === '\\') out += '\\';
    else out += next ?? '';
  }
  return out;
}

function requiredField(records: Map<string, string[]>, name: string): string[] {
  const value = records.get(name);
  if (!value) throw new Error(`mlfw tokenizer: missing '${name}' record`);
  return value;
}

function asSpecialTokensInput(value: unknown): PartialSpecialTokens | undefined {
  if (value === undefined || value === null) return undefined;
  if (Array.isArray(value)) return value.map(String);
  if (typeof value !== 'object') return undefined;
  const input = value as Record<string, unknown>;
  const out: Partial<SpecialTokens> = {};
  for (const key of SPECIAL_KEYS) {
    const token = input[key];
    if (typeof token === 'string') out[key] = token;
  }
  return out;
}

function asNullableNumber(value: unknown): number | null | undefined {
  if (value === null || value === undefined) return value;
  return typeof value === 'number' ? value : Number(value);
}
