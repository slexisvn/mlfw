import { describe, it, expect } from 'vitest';
import { tensor } from '../../src/index.js';
import * as nn from '../../src/nn/index.js';
import * as ops from '../../src/tensor/ops/ops.js';
import { compile, _traceCore } from '../../src/tracing/compile.js';
import { CPUTarget, WasmTarget } from '../../src/backend/target.js';
import { mulberry32 } from '../_utils/rng.js';
import { randomTensor, flat } from '../_utils/tensor_data.js';

const TOL = 3e-3;

function maxRelErr(expected, actual) {
  expect(actual.length).toBe(expected.length);
  let worst = 0, at = -1;
  for (let i = 0; i < expected.length; i++) {
    const e = Math.abs(expected[i] - actual[i]) / (1 + Math.abs(expected[i]));
    if (!(e <= worst)) { worst = e; at = i; }
  }
  return { worst, at };
}

async function expectDynamicMatchesEager(fwd, inputs, makeTarget = CPUTarget) {
  const eager = flat(await fwd(...inputs));
  const compiled = compile({ forward: fwd }, inputs, {
    target: makeTarget(),
    dynamic_shapes: inputs.map(() => new Set([0])),
  });
  if (compiled._ready) await compiled._ready;
  const got = flat(await compiled(...inputs));
  const { worst, at } = maxRelErr(eager, got);
  expect(worst, `idx ${at}: eager=${eager[at]} compiled=${got[at]}`).toBeLessThan(TOL);
}

const rng = (seed) => mulberry32(seed);

describe('dynamic leading dimension matches eager', () => {
  it('conv + batchnorm + maxpool + linear head', async () => {
    const c1 = new nn.Conv2d(3, 8, 3, { stride: 1, padding: 1 });
    const bn = new nn.BatchNorm2d(8); bn.eval();
    const pool = new nn.MaxPool2d(2), fl = new nn.Flatten(), fc = new nn.Linear(8 * 4 * 4, 5);
    const fwd = (x) => fc.forward(fl.forward(pool.forward(nn.F.relu(bn.forward(c1.forward(x))))));
    await expectDynamicMatchesEager(fwd, [randomTensor(rng(1), [2, 3, 8, 8])]);
  });

  it('strided + dilated conv with adaptive average pooling', async () => {
    const c1 = new nn.Conv2d(3, 8, 3, { stride: 2, padding: 1 });
    const c2 = new nn.Conv2d(8, 8, 3, { stride: 1, padding: 2, dilation: 2 });
    const gap = new nn.AdaptiveAvgPool2d(1);
    const fwd = (x) => gap.forward(nn.F.relu(c2.forward(nn.F.relu(c1.forward(x)))));
    await expectDynamicMatchesEager(fwd, [randomTensor(rng(2), [2, 3, 16, 16])]);
  });

  it('transformer encoder stack', async () => {
    const layers = Array.from({ length: 2 }, () => new nn.TransformerEncoderLayer(16, 2, 32, 0.0, 'gelu', 1e-5, true, false));
    layers.forEach((m) => m.eval());
    const fwd = (x) => { let h = x; for (const l of layers) h = l.forward(h); return h; };
    await expectDynamicMatchesEager(fwd, [randomTensor(rng(3), [2, 5, 16])]);
  });

  it('causal multi-head attention', async () => {
    const mha = new nn.MultiheadAttention(16, 2, 0.0, true); mha.eval();
    const fwd = (x) => mha.forward(x, x, x, null, null, true);
    await expectDynamicMatchesEager(fwd, [randomTensor(rng(4), [2, 6, 16])]);
  });

  it('LSTM and GRU stacks', async () => {
    const lstm = new nn.LSTM(8, 12, 2, true);
    await expectDynamicMatchesEager((x) => lstm.forward(x)[0], [randomTensor(rng(5), [2, 4, 8])]);
    const gru = new nn.GRU(8, 12, 2, true);
    await expectDynamicMatchesEager((x) => gru.forward(x)[0], [randomTensor(rng(6), [2, 4, 8])]);
  });

  it('recurrent cell driven by an explicit time loop', async () => {
    const cell = new nn.LSTMCell(6, 8), fc = new nn.Linear(8, 4);
    const fwd = (x) => {
      let h = null;
      for (let t = 0; t < 4; t++) h = cell.forward(x.select(1, t), h);
      return fc.forward(h[0]);
    };
    await expectDynamicMatchesEager(fwd, [randomTensor(rng(7), [3, 4, 6])]);
  });

  it('argmax, argmin, sort, topk and padded bitonic widths', async () => {
    const x = [randomTensor(rng(8), [4, 12], 0.5, 1.5)];
    await expectDynamicMatchesEager((a) => a.argmax(-1), x);
    await expectDynamicMatchesEager((a) => a.argmin(-1), x);
    await expectDynamicMatchesEager((a) => ops.sort(a, -1), x);
    await expectDynamicMatchesEager((a) => ops.topk(a, 4, -1)[0], x);
    await expectDynamicMatchesEager((a) => ops.pad(a, [0, 0], [0, 4], 7), x);
  });

  it('runs on the wasm backend too', async () => {
    const fc1 = new nn.Linear(12, 16), fc2 = new nn.Linear(16, 3);
    const fwd = (x) => fc2.forward(nn.F.gelu(fc1.forward(x)));
    await expectDynamicMatchesEager(fwd, [randomTensor(rng(9), [4, 12])], WasmTarget);
  });
});

describe('dynamic dimensions stay symbolic unless the model reads them', () => {
  const dynSpec = [new Set([0])];

  function traceDynamic(fwd, inputs) {
    return _traceCore(fwd, inputs, { dynamicShapes: dynSpec });
  }

  const eqGuardsOnSymbols = (shapeEnv) =>
    shapeEnv.guards.filter((g) => g.op === 'eq' && typeof g.lhs === 'string' && shapeEnv.symbols.get(g.lhs)?.inputIdx === 0
      && shapeEnv.symbols.get(g.lhs)?.dimIdx === 0);

  it('a model that never reads .shape keeps the dimension dynamic', () => {
    const fc = new nn.Linear(8, 4);
    const core = traceDynamic((x) => nn.F.relu(fc.forward(x)), [randomTensor(rng(10), [3, 8])]);
    expect(core.outputTypes[0].shape[0]).toBe(-1);
    expect(core.outputSymShapes[0][0]).toBe('s0');
    expect(eqGuardsOnSymbols(core.shapeEnv).length).toBe(0);
  });

  it('reading a dynamic dimension specializes it behind a guard', () => {
    const core = traceDynamic((x) => ops.reshape(x, [x.shape[0], 2, 4]), [randomTensor(rng(11), [3, 8])]);
    const guards = eqGuardsOnSymbols(core.shapeEnv);
    expect(guards.length).toBe(1);
    expect(guards[0].rhs).toBe(3);
  });

  it('the same symbol is only guarded once no matter how often it is read', () => {
    const core = traceDynamic((x) => {
      let h = x;
      for (let i = 0; i < 5; i++) h = ops.reshape(h, [h.shape[0], 8]);
      return h;
    }, [randomTensor(rng(12), [3, 8])]);
    expect(eqGuardsOnSymbols(core.shapeEnv).length).toBe(1);
  });

  it('symbolic shapes survive multi-op decompositions', () => {
    const core = traceDynamic((x) => x.select(1, 2), [randomTensor(rng(13), [3, 5, 4])]);
    expect(core.outputSymShapes[0][0]).toBe('s0');
    expect(core.outputTypes[0].shape[0]).toBe(-1);
  });

  it('an inferred reshape dimension stays symbolic instead of multiplying the dynamic sentinel', () => {
    const core = traceDynamic((x) => nn.F.relu(ops.reshape(x, [-1])), [randomTensor(rng(18), [4, 3])]);
    expect(core.outputTypes[0].shape).toEqual([-1]);
    expect(core.shapeEnv.hintOf(core.outputSymShapes[0][0])).toBe(12);
    expect(eqGuardsOnSymbols(core.shapeEnv).length).toBe(0);

    const kept = traceDynamic((x) => ops.reshape(x, [-1, 3]), [randomTensor(rng(19), [4, 3])]);
    expect(kept.outputTypes[0].shape).toEqual([-1, 3]);
    expect(kept.outputSymShapes[0][0]).toBe('s0');
  });
});

describe('a specialized model recompiles for a new batch size', () => {
  it('LSTM stays correct across batch sizes', async () => {
    const lstm = new nn.LSTM(6, 8, 1, true);
    const fwd = (x) => lstm.forward(x)[0];
    const first = [randomTensor(rng(14), [2, 3, 6])];
    const compiled = compile({ forward: fwd }, first, { target: CPUTarget(), dynamic_shapes: [new Set([0])] });
    if (compiled._ready) await compiled._ready;

    for (const batch of [2, 5, 1]) {
      const inputs = [randomTensor(rng(100 + batch), [batch, 3, 6])];
      const eager = flat(await fwd(...inputs));
      const got = flat(await compiled(...inputs));
      const { worst, at } = maxRelErr(eager, got);
      expect(worst, `batch ${batch} idx ${at}`).toBeLessThan(TOL);
    }
  });

  it('a reshape with an inferred dimension follows the batch size', async () => {
    const fwd = (x) => nn.F.relu(ops.reshape(x, [-1]));
    const compiled = compile({ forward: fwd }, [randomTensor(rng(20), [4, 3])], {
      target: CPUTarget(), dynamic_shapes: [new Set([0])],
    });
    if (compiled._ready) await compiled._ready;

    for (const batch of [4, 6, 1]) {
      const inputs = [randomTensor(rng(300 + batch), [batch, 3])];
      const got = await compiled(...inputs);
      expect(got.shape).toEqual([batch * 3]);
      const { worst, at } = maxRelErr(flat(await fwd(...inputs)), flat(got));
      expect(worst, `batch ${batch} idx ${at}`).toBeLessThan(TOL);
    }
  });

  it('a fully dynamic MLP serves several batch sizes', async () => {
    const fc1 = new nn.Linear(8, 12), fc2 = new nn.Linear(12, 3);
    const fwd = (x) => fc2.forward(nn.F.gelu(fc1.forward(x)));
    const compiled = compile({ forward: fwd }, [randomTensor(rng(15), [4, 8])], {
      target: CPUTarget(), dynamic_shapes: [new Set([0])],
    });
    if (compiled._ready) await compiled._ready;

    for (const batch of [1, 4, 9]) {
      const inputs = [randomTensor(rng(200 + batch), [batch, 8])];
      const eager = flat(await fwd(...inputs));
      const got = flat(await compiled(...inputs));
      const { worst, at } = maxRelErr(eager, got);
      expect(worst, `batch ${batch} idx ${at}`).toBeLessThan(TOL);
    }
  });
});

describe('pooling honours a dynamic batch extent', () => {
  it('max pooling does not leave the -Infinity initial value in the output', async () => {
    const pool = new nn.MaxPool2d(2);
    const inputs = [randomTensor(rng(16), [2, 3, 8, 8])];
    const compiled = compile({ forward: (x) => pool.forward(x) }, inputs, {
      target: CPUTarget(), dynamic_shapes: [new Set([0])],
    });
    if (compiled._ready) await compiled._ready;
    const got = flat(await compiled(...inputs));
    expect(got.every(Number.isFinite)).toBe(true);
  });

  it('average pooling does not leave zeros in the output', async () => {
    const pool = new nn.AvgPool2d(2);
    const inputs = [randomTensor(rng(17), [2, 3, 8, 8], 0.5, 1.5)];
    const compiled = compile({ forward: (x) => pool.forward(x) }, inputs, {
      target: CPUTarget(), dynamic_shapes: [new Set([0])],
    });
    if (compiled._ready) await compiled._ready;
    const got = flat(await compiled(...inputs));
    expect(got.some((v) => v !== 0)).toBe(true);
  });
});

describe('size-1 axes are specialized, not symbolized', () => {
  const fwd3 = (a, b, c) => ops.relu(ops.sub(ops.mul(ops.add(a, b), ops.sub(a, c)), ops.tanh(ops.mul(b, c))));

  it('broadcasting operands with a dynamic size-1 axis match eager', async () => {
    const r = rng(101);
    const inputs = [randomTensor(r, [16, 1, 32]), randomTensor(r, [1, 12, 32]), randomTensor(r, [16, 12, 1])];
    const eager = flat(fwd3(...inputs));

    for (const spec of [inputs.map(() => new Set([0])), inputs.map(() => true)]) {
      const compiled = compile({ forward: fwd3 }, inputs, { target: CPUTarget(), dynamic_shapes: spec });
      if (compiled._ready) await compiled._ready;
      const got = flat(await compiled(...inputs));
      const { worst, at } = maxRelErr(eager, got);
      expect(worst, `idx ${at}: eager=${eager[at]} compiled=${got[at]}`).toBeLessThan(TOL);
    }
  });

  it('a batch-1 example still recompiles correctly for a larger batch', async () => {
    const fc = new nn.Linear(32, 16);
    const fwd = (x) => ops.sum(nn.F.relu(fc.forward(x)), -1);
    const one = randomTensor(rng(102), [1, 32]);
    const compiled = compile({ forward: fwd }, [one], { target: CPUTarget(), dynamic_shapes: [new Set([0])] });
    if (compiled._ready) await compiled._ready;

    const first = maxRelErr(flat(fwd(one)), flat(await compiled(one)));
    expect(first.worst).toBeLessThan(TOL);

    const many = randomTensor(rng(103), [6, 32]);
    const second = maxRelErr(flat(fwd(many)), flat(await compiled(many)));
    expect(second.worst, `idx ${second.at}`).toBeLessThan(TOL);
  });
});

describe('compiled scratch buffers are declared once', () => {
  it('argmax emits a single allocation for its running-best scratch', () => {
    const compiled = compile({ forward: (a) => a.argmax(0, false) }, [tensor([3, 1, 4, 1, 5])], { target: CPUTarget() });
    const source = compiled.source();
    const declarations = source.match(/const _argval_\d+ = /g) || [];
    expect(declarations.length).toBe(1);
  });
});
