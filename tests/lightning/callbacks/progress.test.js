import { describe, it, expect, afterEach } from 'vitest';
import { Trainer } from '../../../src/index.js';
import { progress } from '#io/progress';
import { SimpleModel, makeData } from '../_fixtures.js';

const originalWrite = process.stdout.write.bind(process.stdout);

function captureStdout() {
  const chunks = [];
  process.stdout.write = (chunk) => { chunks.push(String(chunk)); return true; };
  return chunks;
}

afterEach(() => { process.stdout.write = originalWrite; });

describe('ProgressCallback', () => {
  it('renders a bar per batch and terminates the line when training ends', async () => {
    const chunks = captureStdout();
    await new Trainer({ maxEpochs: 2, enableCheckpointing: false, logger: false })
      .fit(new SimpleModel(), makeData(40, 10));
    process.stdout.write = originalWrite;

    const bars = chunks.filter((c) => c.startsWith('\r'));
    expect(bars.length).toBeGreaterThan(0);
    expect(chunks[chunks.length - 1]).toBe('\n');

    for (const bar of bars) {
      expect(bar).toMatch(/^\r\S.*: {1,3}\d{1,3}%\|.{24}\| \d+\/\d+ \[\d\d:\d\d<\d\d:\d\d, \d+\.\d\dit\/s/);
    }
    expect(bars.some((b) => b.includes('Epoch 1/2'))).toBe(true);
    expect(bars.some((b) => b.includes('Epoch 2/2'))).toBe(true);
    expect(bars.some((b) => b.includes('100%'))).toBe(true);
    expect(bars.some((b) => b.includes('train_loss='))).toBe(true);
  });

  it('is absent when the trainer disables it, leaving stdout untouched', async () => {
    const chunks = captureStdout();
    await new Trainer({ maxEpochs: 1, enableProgress: false, enableCheckpointing: false, logger: false })
      .fit(new SimpleModel(), makeData(20, 10));
    process.stdout.write = originalWrite;
    expect(chunks).toEqual([]);
  });
});

describe('node progress sink', () => {
  it('pads a shrinking line so no glyph of the previous render survives', () => {
    const chunks = captureStdout();
    progress.update('a'.repeat(20));
    progress.update('b'.repeat(5));
    progress.finish();
    process.stdout.write = originalWrite;

    expect(chunks).toEqual(['\r' + 'a'.repeat(20), '\r' + 'b'.repeat(5) + ' '.repeat(15), '\n']);
  });

  it('writes no trailing newline when nothing was rendered', () => {
    const chunks = captureStdout();
    progress.finish();
    process.stdout.write = originalWrite;
    expect(chunks).toEqual([]);
  });
});
