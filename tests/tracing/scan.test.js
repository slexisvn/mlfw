import { describe, it, expect } from 'vitest';
import { tensor, compile, scan, CPUTarget, WasmTarget } from '../../src/index.js';
import { add } from '../../src/tensor/ops/ops.js';
import { _traceCore } from '../../src/tracing/compile.js';

function cumsum(xs, c0) {
  const [, ys] = scan((carry, x_t) => {
    const next = add(carry, x_t);
    return [next, next];
  }, c0, xs);
  return ys;
}

describe('scan tracing', () => {
  it('traces a timestep loop to a SINGLE scan op (not unrolled)', () => {
    const xs = tensor([[1, 2], [3, 4], [5, 6]]);
    const c0 = tensor([0, 0]);
    const traced = _traceCore((a, b) => cumsum(a, b), [xs, c0], { name: 'cs' });
    const fn = traced.graph.functions().next().value;
    const ops = [...fn.ops()];
    expect(ops.filter(o => o.opName === 'scan').length).toBe(1);
    expect(ops.filter(o => o.opName === 'add').length).toBe(0);
  });

  for (const [tname, makeTarget] of [['cpu', CPUTarget], ['wasm', WasmTarget]]) {
    it(`compiles + runs scan cumulative sum on ${tname}`, async () => {
      const xs = tensor([[1, 2], [3, 4], [5, 6]]);
      const c0 = tensor([0, 0]);
      const cc = compile({ forward: cumsum }, [xs, c0], { target: makeTarget() });
      let out = cc(xs, c0);
      if (out && out.then) out = await out;
      const data = Array.from(out.contiguous ? out.contiguous().data : out.data);
      expect(data).toEqual([1, 2, 4, 6, 9, 12]);
    });
  }
});
