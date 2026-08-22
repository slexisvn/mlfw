import { SegmentStrategy } from './segment_strategy.js';

export class CharStrategy extends SegmentStrategy {
  segment(text: string): string[] {
    return Array.from(this._normalize(text));
  }

  detokenize(tokens: readonly string[]): string {
    return tokens.join('');
  }

  static fromJSON(data: { lowercase?: boolean } = {}): CharStrategy {
    return new CharStrategy({ lowercase: data.lowercase ?? false });
  }
}
