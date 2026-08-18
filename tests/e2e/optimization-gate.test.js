import { describe, it, expect, beforeEach } from 'vitest';
import { tensor, compile, CPUTarget, WasmTarget, manual_seed, unseed } from '../../src/index.js';
import * as nn from '../../src/nn/index.js';
import { clearOptimizationGateCache } from '../../src/tracing/compile.js';
import { BASELINE } from '../../src/compiler/pipeline/opt_gate.js';
import { mulberry32 } from '../_utils/rng.js';
import { randomNested, flat } from '../_utils/tensor_data.js';

const maxRelErr = (a, b) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i]) / (1 + Math.abs(v))), 0);

function convModel(rng) {
  const c1 = new nn.Conv2d(3, 8, 3, { padding: 1 });
  const c2 = new nn.Conv2d(8, 8, 3, { padding: 1 });
  const bn = new nn.BatchNorm2d(8);
  bn.eval();
  const pool = new nn.MaxPool2d(2), fl = new nn.Flatten(), fc = new nn.Linear(8 * 8 * 8, 10);
  return {
    fwd: (x) => fc.forward(fl.forward(pool.forward(nn.F.relu(bn.forward(c2.forward(nn.F.relu(c1.forward(x)))))))),
    input: tensor(randomNested(rng, [2, 3, 16, 16])),
  };
}

beforeEach(() => clearOptimizationGateCache());

describe('the optimization gate compiles, measures, and keeps the winner', () => {
  it('is off by default, and then reports nothing', async () => {
    const { fwd, input } = convModel(mulberry32(1));
    const compiled = compile({ forward: fwd }, [input], { target: CPUTarget() });
    await compiled(input);
    expect(compiled.tuningReport()).toEqual([]);
  });

  it('records one decision per compiled graph when enabled', async () => {
    const { fwd, input } = convModel(mulberry32(2));
    const compiled = compile({ forward: fwd }, [input], { target: CPUTarget(), tuneOptimizations: true });
    await compiled(input);

    const report = compiled.tuningReport();
    expect(report.length).toBe(1);
    const d = report[0];
    expect(d.measurements.map(m => m.name)).toContain(BASELINE);
    expect(d.measurements.map(m => m.name)).toContain('layout');
    expect(d.baselineMs).toBeGreaterThan(0);
  });

  it('the chosen configuration still produces the baseline numbers', async () => {
    manual_seed(1234);
    const plain = convModel(mulberry32(3));
    manual_seed(1234);
    const tuned = convModel(mulberry32(3));
    unseed();

    const base = compile({ forward: plain.fwd }, [plain.input], { target: CPUTarget() });
    const gated = compile({ forward: tuned.fwd }, [tuned.input], { target: CPUTarget(), tuneOptimizations: true });

    const expected = flat(await base(plain.input));
    const actual = flat(await gated(tuned.input));
    expect(actual.length).toBe(expected.length);
    expect(maxRelErr(expected, actual)).toBeLessThan(2e-3);
  });

  it('never reports a winner that lost to the baseline', async () => {
    const { fwd, input } = convModel(mulberry32(4));
    const compiled = compile({ forward: fwd }, [input], { target: CPUTarget(), tuneOptimizations: true });
    await compiled(input);

    for (const d of compiled.tuningReport()) {
      if (d.winner === BASELINE) {
        expect(d.gain).toBe(1);
      } else {
        expect(d.winnerMs).toBeLessThan(d.baselineMs);
        expect(d.gain).toBeGreaterThanOrEqual(1.05);
        expect(d.measurements.find(m => m.name === d.winner).correct).toBe(true);
      }
    }
  });

  it('every candidate it accepted was verified against the baseline output', async () => {
    const { fwd, input } = convModel(mulberry32(5));
    const compiled = compile({ forward: fwd }, [input], { target: CPUTarget(), tuneOptimizations: true });
    await compiled(input);

    const d = compiled.tuningReport()[0];
    const baseline = d.measurements.find(m => m.name === BASELINE);
    expect(baseline.correct).toBe(true);
    for (const m of d.measurements) {
      expect(typeof m.correct).toBe('boolean');
      if (!m.correct) expect(m.name).not.toBe(BASELINE);
    }
  });

  it('skips quietly on a target that offers no candidates', async () => {
    const { fwd, input } = convModel(mulberry32(6));
    const compiled = compile({ forward: fwd }, [input], { target: WasmTarget(), tuneOptimizations: true });
    const out = flat(await compiled(input));
    expect(out.every(Number.isFinite)).toBe(true);
    expect(compiled.tuningReport()).toEqual([]);
  });

  it('reuses a cached decision instead of re-measuring an identical graph', async () => {
    const first = convModel(mulberry32(7));
    const c1 = compile({ forward: first.fwd }, [first.input], { target: CPUTarget(), tuneOptimizations: true });
    await c1(first.input);
    expect(c1.tuningReport().length).toBe(1);

    const second = convModel(mulberry32(8));
    const c2 = compile({ forward: second.fwd }, [second.input], { target: CPUTarget(), tuneOptimizations: true });
    await c2(second.input);
    expect(c2.tuningReport().length, 'the second identical graph must reuse the decision').toBe(0);
  });
});
