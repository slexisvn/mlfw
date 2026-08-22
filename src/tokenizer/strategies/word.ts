import { SegmentStrategy } from './segment_strategy.js';

export class WordStrategy extends SegmentStrategy {
  segment(text: string): string[] {
    return this._normalize(text).split(/\s+/).filter(Boolean);
  }

  detokenize(tokens: readonly string[]): string {
    return tokens.join(' ');
  }

  static fromJSON(data: { lowercase?: boolean } = {}): WordStrategy {
    return new WordStrategy({ lowercase: data.lowercase ?? false });
  }
}
