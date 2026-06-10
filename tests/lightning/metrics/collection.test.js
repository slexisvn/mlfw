import { describe, it, expect } from 'vitest';
import { tensor } from '../../../src/index.js';
import {
  MetricCollection, Accuracy, SumMetric, MeanMetric, ConfusionMatrix,
} from '../../../src/lightning/index.js';

describe('MetricCollection', () => {
  it('broadcasts update to every member and computes a keyed result', () => {
    const collection = new MetricCollection({
      acc: new Accuracy({ task: 'multiclass', numClasses: 2 }),
      cm: new ConfusionMatrix({ numClasses: 2 }),
    });
    collection.update(tensor([0.9, 0.1, 0.2, 0.8], { shape: [2, 2] }), tensor([0, 1]));
    const out = collection.compute();
    expect(out.acc).toBe(1.0);
    expect(out.cm[0][0]).toBe(1);
    expect(out.cm[1][1]).toBe(1);
  });

  it('add registers a metric and is chainable', () => {
    const c = new MetricCollection();
    const ret = c.add('s', new SumMetric());
    expect(ret).toBe(c);
    expect(c.has('s')).toBe(true);
    expect(c.size).toBe(1);
  });

  it('reset clears every member', () => {
    const c = new MetricCollection({ s: new SumMetric() });
    c.get('s').update(5);
    c.reset();
    expect(c.compute().s).toBe(0);
  });

  it('forward updates then computes in one call', () => {
    const c = new MetricCollection({ acc: new Accuracy({ task: 'binary' }) });
    const out = c.forward(tensor([0.9, 0.1]), tensor([1, 0]));
    expect(out.acc).toBe(1.0);
  });

  it('is iterable over [name, metric] pairs', () => {
    const c = new MetricCollection({ a: new SumMetric(), b: new MeanMetric() });
    const names = [...c].map(([name]) => name);
    expect(names).toEqual(['a', 'b']);
  });
});
