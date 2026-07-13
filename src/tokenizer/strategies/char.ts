export class CharStrategy {
  private readonly _lowercase: boolean;

  constructor({ lowercase = false }: { lowercase?: boolean } = {}) {
    this._lowercase = lowercase;
  }

  fit(_texts?: readonly string[], _options?: { vocabSize?: number | null }): void {}

  segment(text: string): string[] {
    const normalized = this._lowercase ? String(text).toLowerCase() : String(text);
    return Array.from(normalized);
  }

  detokenize(tokens: readonly string[]): string {
    return tokens.join('');
  }

  toJSON(): { lowercase: boolean } {
    return { lowercase: this._lowercase };
  }

  static fromJSON(data: { lowercase?: boolean } = {}): CharStrategy {
    return new CharStrategy({ lowercase: data.lowercase ?? false });
  }
}
