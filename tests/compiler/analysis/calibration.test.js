import { describe, it, expect } from 'vitest';
import { HistogramCollector } from '../../../src/compiler/analysis/calibration.js';

describe('HistogramCollector out-of-range batch handling', () => {
  it('does not double-count a batch that triggers range expansion', () => {
    const hist = new HistogramCollector(16);
    hist.update(new Float64Array([0, 1, 2, 3]));
    expect(hist.totalCount).toBe(4);

    hist.update(new Float64Array([0, 1, 100, 2]));
    expect(hist.totalCount).toBe(8);

    let binSum = 0;
    for (let i = 0; i < hist.numBins; i++) binSum += hist.bins[i];
    expect(binSum).toBe(8);
  });

  it('in-range batches accumulate exactly once', () => {
    const hist = new HistogramCollector(8);
    hist.update(new Float64Array([0, 10]));
    hist.update(new Float64Array([1, 2, 3]));
    expect(hist.totalCount).toBe(5);

    let binSum = 0;
    for (let i = 0; i < hist.numBins; i++) binSum += hist.bins[i];
    expect(binSum).toBe(5);
  });
});
