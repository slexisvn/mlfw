import { describe, it, expect } from 'vitest';
import { tensor, cond, compile, trace, CPUTarget } from '../../src/index.js';
import { verifyModule } from '../../src/compiler/ir/graph/verifier.js';
import { getActiveTracer } from '../../src/tracing/tracer.js';

const predicate = (value) => tensor(value, { dtype: 'bool' });

describe('cond', () => {
  it('evaluates only the selected branch in eager mode', () => {
    const x = tensor([-2, 4]);
    const fail = () => { throw new Error('unselected branch ran'); };
    expect([...cond(predicate(1), () => x.mul(x), fail).data]).toEqual([4, 16]);
    expect([...cond(predicate(0), fail, () => x.neg()).data]).toEqual([2, -4]);
  });

  it('shares a captured scalar across both branches and their enclosing block', () => {
    const offset = tensor(3);
    const forward = (p, x) => cond(p, () => x.mul(offset), () => x.sub(offset)).add(offset);
    const x = tensor([-2, 4]);
    const inputs = [predicate(1), x];
    expect(verifyModule(trace(forward, inputs)).map((e) => e.message)).toEqual([]);
    const compiled = compile({ forward }, inputs, { target: CPUTarget() });
    expect([...compiled(predicate(1), x).data]).toEqual([-3, 15]);
    expect([...compiled(predicate(0), x).data]).toEqual([-2, 4]);
  });

  it('keeps independent results and selects the requested branch at runtime', () => {
    const forward = (p, x) => cond(p,
      () => [x.mul(x), x.neg()],
      () => [x.neg(), x.add(x)]);
    const x = tensor([-2, 4]);
    const compiled = compile({ forward }, [predicate(1), x], { target: CPUTarget() });
    expect(compiled(predicate(1), x).map((t) => [...t.data])).toEqual([[4, 16], [2, -4]]);
    expect(compiled(predicate(0), x).map((t) => [...t.data])).toEqual([[2, -4], [-4, 8]]);
  });

  it('rejects inconsistent branch return structure and restores the tracer after failure', () => {
    const x = tensor([1, 2]);
    expect(() => trace((p, v) => cond(p, () => [v], () => v), [predicate(1), x]))
      .toThrow(/cond.*same.*structure/);
    expect(getActiveTracer()).toBeNull();
    expect([...cond(predicate(0), () => x, () => x.neg()).data]).toEqual([-1, -2]);
  });

  it('rejects branches with different numbers of returned tensors', () => {
    expect(() => trace((p, x) => cond(p, () => [x, x.neg()], () => [x]), [predicate(1), tensor([1, 2])]))
      .toThrow(/cond.*same.*structure/);
    expect(getActiveTracer()).toBeNull();
  });

  it('leaves eager execution usable when a traced branch throws', () => {
    const x = tensor([1, 2]);
    expect(() => trace((p, v) => cond(p, () => v, () => { throw new Error('branch failed'); }), [predicate(1), x]))
      .toThrow('branch failed');
    expect(getActiveTracer()).toBeNull();
    expect([...x.add(x).data]).toEqual([2, 4]);
  });
});
