export class WordStrategy {
  constructor({ lowercase = false } = {}) {
    this._lowercase = lowercase;
  }

  fit() {}

  segment(text) {
    const normalized = this._lowercase ? String(text).toLowerCase() : String(text);
    return normalized.split(/\s+/).filter(Boolean);
  }

  detokenize(tokens) {
    return tokens.join(' ');
  }
}
