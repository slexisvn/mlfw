import { describe, it, expect } from 'vitest';
import { TeraRuntime } from '../../src/cli/runtime.js';

function run(source) {
  return new TeraRuntime({ output: () => {} }).execute(source);
}

describe('Tera dict / map', () => {
  it('literal, get, set, missing-key, .size', async () => {
    const r = await run(`
d = {"a": 1, "b": 2}
d["c"] = 3
d["a"] = 10
[d["a"], d["b"], d["c"], d["missing"], d.size]
`);
    expect(r).toEqual([10, 2, 3, null, 3]);
  });

  it('empty dict literal and dynamic keys', async () => {
    const r = await run(`
d = {}
k = "x"
d[k] = 5
[d.size, d["x"]]
`);
    expect(r).toEqual([1, 5]);
  });

  it('for-in iterates map keys', async () => {
    const r = await run(`
d = {"x": 10, "y": 20}
ks = []
for k in d:
  ks[ks.length] = k
ks
`);
    expect(r).toEqual(['x', 'y']);
  });
});

describe('Tera index assignment', () => {
  it('overwrites and appends (grow) on lists', async () => {
    const r = await run(`
xs = [0, 0, 0]
xs[1] = 9
xs[3] = 7
xs
`);
    expect(r).toEqual([0, 9, 0, 7]);
  });

  it('compound index assignment', async () => {
    const r = await run(`
xs = [5, 5]
xs[0] += 3
xs
`);
    expect(r).toEqual([8, 5]);
  });

  it('append-via-index builds a list in O(1) steps', async () => {
    const r = await run(`
out = []
for i in range(4):
  out[out.length] = i * i
out
`);
    expect(r).toEqual([0, 1, 4, 9]);
  });
});

describe('Tera string indexing', () => {
  it('char index, negative index, slice, .length', async () => {
    const r = await run(`
s = "hello"
[s[0], s[-1], s[1:4], s.length]
`);
    expect(r).toEqual(['h', 'o', 'ell', 5]);
  });
});

describe('Tera list comprehension', () => {
  it('maps over a range', async () => {
    expect(await run('[x * 2 for x in range(4)]')).toEqual([0, 2, 4, 6]);
  });

  it('filters with if', async () => {
    expect(await run('[x for x in range(6) if x > 3]')).toEqual([4, 5]);
  });

  it('iterates a list', async () => {
    expect(await run('[w for w in "a b c".split(" ")]')).toEqual(['a', 'b', 'c']);
  });
});

describe('Tera array concat and .length', () => {
  it('concatenates lists with +', async () => {
    expect(await run('[1, 2] + [3, 4]')).toEqual([1, 2, 3, 4]);
  });

  it('.length works on list and string', async () => {
    expect(await run('[[1, 2, 3].length, "abcd".length]')).toEqual([3, 4]);
  });
});

describe('Tera existing assignment forms still work', () => {
  it('simple, compound, and destructuring assignment', async () => {
    const r = await run(`
x = 5
x += 2
a, b = [10, 20]
[x, a, b]
`);
    expect(r).toEqual([7, 10, 20]);
  });
});
