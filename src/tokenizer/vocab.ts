export class Vocab {
  private readonly _tokenToId: Map<string, number>;
  private readonly _idToToken: string[];

  constructor(specials: readonly string[] = []) {
    this._tokenToId = new Map();
    this._idToToken = [];
    for (const token of specials) this.add(token);
  }

  add(token: string): number {
    let id = this._tokenToId.get(token);
    if (id === undefined) {
      id = this._idToToken.length;
      this._tokenToId.set(token, id);
      this._idToToken.push(token);
    }
    return id;
  }

  getId(token: string, fallback = -1): number {
    const id = this._tokenToId.get(token);
    return id === undefined ? fallback : id;
  }

  getToken(id: number): string | undefined {
    return id >= 0 && id < this._idToToken.length ? this._idToToken[id] : undefined;
  }

  has(token: string): boolean {
    return this._tokenToId.has(token);
  }

  get size(): number {
    return this._idToToken.length;
  }

  tokens(): string[] {
    return this._idToToken.slice();
  }

  static fromTokens(tokens: unknown): Vocab {
    if (!Array.isArray(tokens)) throw new Error('mlfw tokenizer: vocab must be an array');
    const vocab = new Vocab();
    const seen = new Set();
    for (const token of tokens) {
      if (typeof token !== 'string') throw new Error('mlfw tokenizer: vocab entries must be strings');
      if (seen.has(token)) throw new Error(`mlfw tokenizer: duplicate vocab token '${token}'`);
      seen.add(token);
      vocab.add(token);
    }
    return vocab;
  }
}
