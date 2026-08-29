import { describe, it, expect } from 'vitest';
import { ddmin } from '../../../tools/visualizer/src/worker/ddmin.js';

const PASSES = [
  'CallInlinerPass', 'DecompositionPass', 'canonicalize', 'algebraic_simplify',
  'constant_fold', 'cse', 'dce', 'PriorityFusionPass', 'MultiOutputFusionPass',
  'InlineReindexPass', 'SchedulePass', 'SimplifyPass', 'MemoryPlanPass', 'FlatIndexSimplifyPass',
];

function needsAll(...required) {
  const calls = [];
  const test = async (subset) => {
    calls.push([...subset]);
    const off = new Set(subset);
    return required.every(name => off.has(name));
  };
  return { test, calls };
}

describe('ddmin over a pass list', () => {
  it('names the one pass that has to be off', async () => {
    const { test } = needsAll('cse');
    expect(await ddmin(PASSES, test)).toEqual(['cse']);
  });

  it('names every pass of a conjunction and nothing else', async () => {
    const { test } = needsAll('canonicalize', 'SchedulePass');
    const found = await ddmin(PASSES, test);
    expect([...found].sort()).toEqual(['SchedulePass', 'canonicalize']);
  });

  it('returns a result no subset of which still passes the test', async () => {
    const { test } = needsAll('dce', 'MemoryPlanPass', 'algebraic_simplify');
    const found = await ddmin(PASSES, test);

    expect(await test(found)).toBe(true);
    for (const name of found) {
      expect(await test(found.filter(other => other !== name))).toBe(false);
    }
  });

  it('costs far fewer probes than turning passes off one at a time', async () => {
    const { test, calls } = needsAll('SimplifyPass');
    await ddmin(PASSES, test);
    expect(calls.length).toBeLessThan(PASSES.length);
  });

  it('never probes with the whole candidate set, which the caller already knows', async () => {
    const { test, calls } = needsAll('cse');
    await ddmin(PASSES, test);
    expect(calls.some(subset => subset.length === PASSES.length)).toBe(false);
  });

  it('keeps a single candidate without probing at all', async () => {
    const { test, calls } = needsAll('cse');
    expect(await ddmin(['cse'], test)).toEqual(['cse']);
    expect(calls).toHaveLength(0);
  });

  it('hands back everything when no subset is enough', async () => {
    const test = async (subset) => subset.length === PASSES.length;
    expect(await ddmin(PASSES, test)).toEqual(PASSES);
  });
});
