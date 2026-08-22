export type SegmentStrategyOptions = { lowercase?: boolean };

export abstract class SegmentStrategy {
  protected readonly _lowercase: boolean;

  constructor({ lowercase = false }: SegmentStrategyOptions = {}) {
    this._lowercase = lowercase;
  }

  fit(_texts?: readonly string[], _options?: { vocabSize?: number | null }): void {}

  protected _normalize(text: string): string {
    return this._lowercase ? String(text).toLowerCase() : String(text);
  }

  abstract segment(text: string): string[];

  abstract detokenize(tokens: readonly string[]): string;

  toJSON(): { lowercase: boolean } {
    return { lowercase: this._lowercase };
  }
}
