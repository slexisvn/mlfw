import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtempSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import {
  tensor, Tokenizer, Embedding, Linear, CrossEntropyLoss, Adam,
  TensorDataset, DataLoader, argmax, manual_seed, unseed,
} from '../../src/index.js';
import { LightningModule, Trainer } from '../../src/lightning/index.js';
import { flat } from '../_utils/tensor_data.js';

const CORPUS = [
  'the cat sat on the mat',
  'the dog sat on the log',
  'the cat ate the fish',
  'the dog ate the bone',
  'a cat sat on a log',
  'a dog sat on a mat',
];

let dir;
beforeEach(() => {
  manual_seed(17);
  dir = mkdtempSync(join(tmpdir(), 'mlfw-lm-'));
});
afterEach(() => {
  unseed();
  rmSync(dir, { recursive: true, force: true });
});

const CTX = 2;

function windows(tok) {
  const ctx = [];
  const next = [];
  for (const line of CORPUS) {
    const ids = tok.encode(line);
    for (let i = 0; i + CTX < ids.length; i++) {
      ctx.push(...ids.slice(i, i + CTX));
      next.push(ids[i + CTX]);
    }
  }
  return { ctx, next, rows: next.length };
}

class Bigram extends LightningModule {
  constructor(vocab, dim) {
    super();
    this.embed = new Embedding(vocab, dim);
    this.head = new Linear(CTX * dim, vocab);
    this.loss = new CrossEntropyLoss();
    this.dim = dim;
  }
  forward(ids) {
    const e = this.embed.forward(ids);
    return this.head.forward(e.reshape([ids.shape[0], CTX * this.dim]));
  }
  trainingStep(batch) {
    const [x, y] = batch;
    return this.loss.forward(this.forward(x), y);
  }
  configureOptimizers() { return new Adam(this.parameters(), { lr: 0.05 }); }
}

describe('journey: tokenizer -> dataset -> train -> generate', () => {
  it('a tokenizer trained on a corpus round-trips through save/load', () => {
    const tok = new Tokenizer({ mode: 'word' }).fit(CORPUS);
    const path = join(dir, 'tok.json');
    tok.save(path);

    const reloaded = Tokenizer.load(path);
    expect(reloaded.vocabSize).toBe(tok.vocabSize);
    for (const line of CORPUS) {
      expect(reloaded.encode(line)).toEqual(tok.encode(line));
      expect(reloaded.decode(reloaded.encode(line))).toBe(tok.decode(tok.encode(line)));
    }
  });

  it('training on tokenized text lowers loss and produces in-vocab greedy continuations', async () => {
    const tok = new Tokenizer({ mode: 'word' }).fit(CORPUS);
    const { ctx, next, rows } = windows(tok);
    expect(rows, 'corpus produced no training windows').toBeGreaterThan(10);

    const X = tensor(ctx, { shape: [rows, CTX], dtype: 'i32' });
    const Y = tensor(next, { shape: [rows], dtype: 'i32' });

    const model = new Bigram(tok.vocabSize, 12);
    const lossOf = (m) => m.loss.forward(m.forward(X), Y).item();
    const before = lossOf(model);

    await new Trainer({ maxEpochs: 60, logger: false, enableProgress: false, enableCheckpointing: false })
      .fit(model, new DataLoader(new TensorDataset(X, Y), { batchSize: 8 }));

    const after = lossOf(model);
    expect(after, `loss did not fall: ${before} -> ${after}`).toBeLessThan(before * 0.6);

    const prompt = tok.encode('the cat').slice(0, CTX);
    expect(prompt).toHaveLength(CTX);

    const generated = [...prompt];
    for (let step = 0; step < 4; step++) {
      const window = generated.slice(-CTX);
      const logits = model.forward(tensor(window, { shape: [1, CTX], dtype: 'i32' }));
      generated.push(flat(argmax(logits, 1))[0]);
    }

    expect(generated).toHaveLength(CTX + 4);
    for (const id of generated) {
      expect(id, `generated id ${id} is outside the vocab`).toBeGreaterThanOrEqual(0);
      expect(id).toBeLessThan(tok.vocabSize);
    }

    const text = tok.decode(generated);
    expect(typeof text).toBe('string');
    expect(text.split(/\s+/).length).toBeGreaterThanOrEqual(CTX + 4);
  });

  it('greedy generation is deterministic under a fixed seed', async () => {
    const tok = new Tokenizer({ mode: 'word' }).fit(CORPUS);
    const { ctx, next, rows } = windows(tok);
    const X = tensor(ctx, { shape: [rows, CTX], dtype: 'i32' });
    const Y = tensor(next, { shape: [rows], dtype: 'i32' });

    const run = async () => {
      manual_seed(5);
      const model = new Bigram(tok.vocabSize, 8);
      await new Trainer({ maxEpochs: 20, logger: false, enableProgress: false, enableCheckpointing: false })
        .fit(model, new DataLoader(new TensorDataset(X, Y), { batchSize: 8 }));
      const logits = model.forward(tensor(tok.encode('the cat').slice(0, CTX), { shape: [1, CTX], dtype: 'i32' }));
      return flat(argmax(logits, 1))[0];
    };

    expect(await run()).toBe(await run());
  });
});
