import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../../../src/backend/target.js';
import { applyAutocast, MixedPrecisionPass, PrecisionClass, DEFAULT_AUTOCAST_OPS } from '../../../../src/compiler/passes/precision/mixed_precision.js';
import { registry } from '../../../../src/compiler/ir/graph/ops.js';
import { PassResult } from '../../../../src/compiler/passes/pass.js';
import { F32 as F, F16, I32, T } from '../../../_utils/ir_fixture.js';

const BF16 = ScalarType.BF16;
const inner = (mod) => (mod.functions ? mod.functions().next().value : mod);
const ops = (func) => [...func.ops()];
const named = (func, name) => ops(func).filter((o) => o.opName === name);
const dtypesOf = (func, name) => named(func, name).map((o) => o.getResult(0).type.dtype);

function matmulChain(name = 'mm') {
  return inner(buildFunction(name, [T([4, 8]), T([8, 4])], [T([4, 4])], (b, a) => {
    b.returnOp([b.matmul(a[0], a[1]).getResult(0)]);
  }));
}

function convGraph(name = 'cv') {
  return inner(buildFunction(name, [T([1, 2, 6, 6]), T([3, 2, 3, 3])], [T([1, 3, 4, 4])], (b, a) => {
    b.returnOp([b.conv(a[0], a[1], [1, 1], [[0, 0], [0, 0]]).getResult(0)]);
  }));
}

function intGraph(name = 'ig') {
  const t = (s) => new TensorType(s, I32);
  return inner(buildFunction(name, [t([4, 8]), t([8, 4])], [t([4, 4])], (b, a) => {
    b.returnOp([b.matmul(a[0], a[1]).getResult(0)]);
  }));
}

describe('the default allow set comes from the op registry, not a hardcoded list', () => {
  it('dot and conv are registered ALWAYS and both autocast without an explicit allow set', () => {
    expect(registry.get('dot').getAttr('precisionClass')).toBe(PrecisionClass.ALWAYS);
    expect(registry.get('conv').getAttr('precisionClass')).toBe(PrecisionClass.ALWAYS);
    expect([...DEFAULT_AUTOCAST_OPS].sort()).toEqual(['conv', 'dot']);

    const mm = matmulChain();
    expect(applyAutocast(mm)).toBe(true);
    expect(dtypesOf(mm, 'dot')).toEqual([F16]);

    const cv = convGraph();
    expect(applyAutocast(cv)).toBe(true);
    expect(dtypesOf(cv, 'conv')).toEqual([F16]);
  });

  it('FOLLOW and ALWAYS are disjoint, and every classified op really exists', () => {
    const byClass = { ALWAYS: [], FOLLOW: [] };
    for (const def of registry.allOps()) {
      const cls = def.getAttr('precisionClass');
      if (cls) byClass[cls].push(def.name);
    }
    expect(byClass.ALWAYS.sort()).toEqual(['conv', 'dot']);
    expect(byClass.FOLLOW.length).toBeGreaterThan(0);
    expect(byClass.FOLLOW.filter((n) => byClass.ALWAYS.includes(n))).toEqual([]);
    for (const name of [...byClass.ALWAYS, ...byClass.FOLLOW]) {
      expect(registry.has(name), `precisionClass was set on unregistered op '${name}'`).toBe(true);
    }
  });

  it('a graph of only FOLLOW ops never autocasts on its own', () => {
    const func = inner(buildFunction('ew', [T([4, 4]), T([4, 4])], [T([4, 4])], (b, a) => {
      b.returnOp([b.mul(b.add(a[0], a[1]).getResult(0), a[1]).getResult(0)]);
    }));
    expect(registry.get('add').getAttr('precisionClass')).toBe(PrecisionClass.FOLLOW);
    expect(registry.get('mul').getAttr('precisionClass')).toBe(PrecisionClass.FOLLOW);
    expect(applyAutocast(func)).toBe(false);
    expect(named(func, 'convert')).toHaveLength(0);
  });
});

describe('autocast honours the requested low dtype', () => {
  it('casts to bf16 when asked instead of f16', () => {
    const func = matmulChain();
    expect(applyAutocast(func, { dtype: BF16 })).toBe(true);
    expect(dtypesOf(func, 'dot')).toEqual([BF16]);
    for (const c of named(func, 'convert')) {
      expect([BF16, F]).toContain(c.getResult(0).type.dtype);
    }
  });

  it('MixedPrecisionPass forwards its configured dtype and allow set', () => {
    const bf = matmulChain();
    expect(new MixedPrecisionPass({ dtype: BF16 }).run(bf)).toBe(PassResult.CHANGED);
    expect(dtypesOf(bf, 'dot')).toEqual([BF16]);

    const untouched = matmulChain();
    expect(new MixedPrecisionPass({ allow: new Set(['conv']) }).run(untouched)).toBe(PassResult.UNCHANGED);
    expect(dtypesOf(untouched, 'dot')).toEqual([F]);
  });
});

describe('autocast never touches non-float data', () => {
  it('an integer matmul keeps its dtype and gains no converts', () => {
    const func = intGraph();
    expect(applyAutocast(func)).toBe(false);
    expect(dtypesOf(func, 'dot')).toEqual([I32]);
    expect(named(func, 'convert')).toHaveLength(0);
  });

  it('an operand already at the low dtype is not re-cast', () => {
    const func = inner(buildFunction('low', [new TensorType([4, 8], F16), new TensorType([8, 4], F16)], [new TensorType([4, 4], F16)], (b, a) => {
      b.returnOp([b.matmul(a[0], a[1]).getResult(0)]);
    }));
    expect(applyAutocast(func)).toBe(false);
    expect(named(func, 'convert')).toHaveLength(0);
  });
});

describe('autocast is idempotent', () => {
  it('a second pass over an already-autocast graph inserts nothing new', () => {
    const func = matmulChain();
    expect(applyAutocast(func)).toBe(true);
    const after = named(func, 'convert').length;

    expect(applyAutocast(func)).toBe(false);
    expect(named(func, 'convert')).toHaveLength(after);
    expect(new MixedPrecisionPass().run(func)).toBe(PassResult.UNCHANGED);
  });
});

describe('FOLLOW propagation stops at ops that are not FOLLOW', () => {
  it('a reduce after the matmul stays f32 and keeps its up-convert', () => {
    const func = inner(buildFunction('mr', [T([4, 8]), T([8, 4])], [T([4])], (b, a) => {
      const mm = b.matmul(a[0], a[1]).getResult(0);
      const z = b.scalarConstant(0, F).getResult(0);
      b.returnOp([b.reduce(mm, z, [1], 'sum').getResult(0)]);
    }));

    expect(applyAutocast(func, { propagateFollow: true })).toBe(true);
    expect(dtypesOf(func, 'dot')).toEqual([F16]);
    expect(dtypesOf(func, 'reduce')).toEqual([F]);
    expect(named(func, 'convert').some((c) => c.getResult(0).type.dtype === F)).toBe(true);
  });

  it('an elementwise op between two matmuls sinks to f16 and the round-trip converts collapse', () => {
    const build = (name) => inner(buildFunction(name, [T([4, 8]), T([8, 8]), T([8, 8]), T([8, 4])], [T([4, 4])], (b, a) => {
      const first = b.matmul(a[0], a[1]).getResult(0);
      const mid = b.add(first, b.matmul(a[0], a[2]).getResult(0)).getResult(0);
      b.returnOp([b.matmul(mid, a[3]).getResult(0)]);
    }));

    const plain = build('plain');
    applyAutocast(plain);
    const withFollow = build('follow');
    applyAutocast(withFollow, { propagateFollow: true });

    expect(dtypesOf(plain, 'add')).toEqual([F]);
    expect(dtypesOf(withFollow, 'add')).toEqual([F16]);
    expect(named(withFollow, 'convert').length).toBeLessThan(named(plain, 'convert').length);
  });
});

describe('autocast graphs still compile and stay numerically close', () => {
  const A = new Float32Array(4 * 8).map((_, i) => Math.sin(i * 0.7) * 1.4);
  const B = new Float32Array(8 * 4).map((_, i) => Math.cos(i * 0.4) * 1.2);

  const run = (func) => {
    const out = new Float32Array(16);
    compileGraph(func, CPUTarget()).run(func.name, A, B, out);
    return Array.from(out);
  };

  it('an f16 matmul lands within f16 resolution of the f32 reference', () => {
    const reference = run(matmulChain('ref'));
    const cast = matmulChain('cast');
    applyAutocast(cast);
    const got = run(cast);

    let maxRel = 0;
    for (let i = 0; i < reference.length; i++) {
      maxRel = Math.max(maxRel, Math.abs(got[i] - reference[i]) / (1 + Math.abs(reference[i])));
    }
    expect(maxRel, `f16 matmul drifted too far: ${maxRel}`).toBeLessThan(5e-3);
    expect(maxRel, 'f16 matmul was bit-identical to f32, so no rounding happened').toBeGreaterThan(0);
  });

  it('FOLLOW propagation does not change the answer beyond f16 resolution', () => {
    const reference = run(matmulChain('ref2'));
    const propagated = matmulChain('prop');
    applyAutocast(propagated, { propagateFollow: true });
    const got = run(propagated);

    for (let i = 0; i < reference.length; i++) {
      expect(Math.abs(got[i] - reference[i]) / (1 + Math.abs(reference[i]))).toBeLessThan(5e-3);
    }
  });
});
