export class CharStrategy {
  constructor({ lowercase = false } = {}) {
    this._lowercase = lowercase;
  }

  fit() {}

  segment(text) {
    const normalized = this._lowercase ? String(text).toLowerCase() : String(text);
    return Array.from(normalized);
  }

  detokenize(tokens) {
    return tokens.join('');
  }

  toJSON() {
    return { lowercase: this._lowercase };
  }

  static fromJSON(data = {}) {
    return new CharStrategy({ lowercase: data.lowercase ?? false });
  }
}
