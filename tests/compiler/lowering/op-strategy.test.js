import { describe, it, expect, afterEach } from 'vitest';
import {
  OpStrategy,
  registerOpStrategy,
  unregisterOpStrategy,
  getOpStrategy,
  selectImplementation,
  hasOpStrategy,
} from '../../../src/compiler/passes/lowering/op_strategy.js';
import {
  registerLoweringRule,
  registerTargetLoweringRule,
  getLoweringRule,
  hasLoweringRule,
} from '../../../src/compiler/passes/lowering/lowering_registry.js';
import { TargetKind } from '../../../src/backend/target.js';
import '../../../src/compiler/passes/lowering/graph_to_tensor.js';

const TEST_OPS = ['__strat_test__', '__lr_generic__', '__lr_target__', '__lr_plevel__', '__lr_replace__', '__target_only__'];

afterEach(() => {
  for (const name of TEST_OPS) unregisterOpStrategy(name);
});

describe('OpStrategy core selection', () => {
  it('selects the highest-plevel implementation', () => {
    const s = new OpStrategy('x');
    const hi = () => 'hi';
    s.addImplementation('x.lo', () => 'lo', 10);
    s.addImplementation('x.hi', hi, 30);
    s.addImplementation('x.mid', () => 'mid', 20);
    expect(s.best().compute).toBe(hi);
    expect(s.best().name).toBe('x.hi');
    expect(s.best().plevel).toBe(30);
  });

  it('first registered wins on a plevel tie (deterministic)', () => {
    const s = new OpStrategy('x');
    const a = () => 'a';
    s.addImplementation('a', a, 15);
    s.addImplementation('b', () => 'b', 15);
    expect(s.best().compute).toBe(a);
  });

  it('enumerates all candidates in registration order', () => {
    const s = new OpStrategy('x');
    s.addImplementation('a', () => {}, 10);
    s.addImplementation('b', () => {}, 20);
    expect(s.candidates.length).toBe(2);
    expect(s.candidates.map((c) => c.name)).toEqual(['a', 'b']);
  });
});

describe('OpStrategy registry — per-target filtering', () => {
  it('keeps generic impls for every target and target impls only for their kind', () => {
    const gen = () => 'gen';
    const cuda = () => 'cuda';
    registerOpStrategy('__strat_test__', { name: 'g', compute: gen, plevel: 10, targetKind: null });
    registerOpStrategy('__strat_test__', { name: 'c', compute: cuda, plevel: 20, targetKind: TargetKind.CUDA });

    const cudaStrat = getOpStrategy('__strat_test__', { kind: TargetKind.CUDA });
    expect(cudaStrat.candidates.length).toBe(2);
    expect(cudaStrat.best().compute).toBe(cuda);

    const cpuStrat = getOpStrategy('__strat_test__', { kind: TargetKind.CPU });
    expect(cpuStrat.candidates.length).toBe(1);
    expect(cpuStrat.best().compute).toBe(gen);

    expect(selectImplementation('__strat_test__', null).compute).toBe(gen);
    expect(hasOpStrategy('__strat_test__')).toBe(true);
  });

  it('returns null for an unregistered op', () => {
    expect(getOpStrategy('__never_registered_op__')).toBe(null);
    expect(selectImplementation('__never_registered_op__')).toBe(null);
    expect(hasOpStrategy('__never_registered_op__')).toBe(false);
  });
});

describe('lowering_registry built on OpStrategy (behavior-preserving)', () => {
  it('registerLoweringRule registers a generic plevel-10 impl', () => {
    const r = () => 'r';
    registerLoweringRule('__lr_generic__', r);
    const s = getOpStrategy('__lr_generic__');
    expect(s.best().compute).toBe(r);
    expect(s.best().plevel).toBe(10);
    expect(getLoweringRule('__lr_generic__')).toBe(r);
  });

  it('registerTargetLoweringRule overrides via higher plevel, generic stays the fallback', () => {
    const gen = () => 'gen';
    const cuda = () => 'cuda';
    registerLoweringRule('__lr_target__', gen);
    registerTargetLoweringRule('__lr_target__', TargetKind.CUDA, cuda);
    expect(getLoweringRule('__lr_target__', { kind: TargetKind.CUDA })).toBe(cuda);
    expect(getLoweringRule('__lr_target__', { kind: TargetKind.CPU })).toBe(gen);
    expect(getLoweringRule('__lr_target__')).toBe(gen);
  });

  it('a lower-plevel candidate never displaces the default impl', () => {
    const def = () => 'def';
    registerLoweringRule('__lr_plevel__', def);
    registerOpStrategy('__lr_plevel__', { name: 't', compute: () => 'tuned', plevel: 5 });
    expect(getLoweringRule('__lr_plevel__')).toBe(def);
    expect(getOpStrategy('__lr_plevel__').candidates.length).toBe(2);
  });

  it('hasLoweringRule honors strategy presence and CONSTANT_OPS', () => {
    expect(hasLoweringRule('add')).toBe(true);
    expect(hasLoweringRule('dot')).toBe(true);
    expect(hasLoweringRule('constant')).toBe(true);
    expect(hasLoweringRule('scalar_constant')).toBe(true);
    expect(hasLoweringRule('__no_op_at_all__')).toBe(false);
  });

  it('real ops already lower through the OpStrategy path', () => {
    expect(typeof getLoweringRule('dot')).toBe('function');
    expect(typeof getLoweringRule('reduce')).toBe('function');
    expect(getOpStrategy('dot').best().name).toBe('dot.generic');
  });

  it('getLoweringRule returns undefined for an unregistered op (throw contract)', () => {
    expect(getLoweringRule('__no_rule_at_all__')).toBe(undefined);
    expect(getLoweringRule('__no_rule_at_all__', { kind: TargetKind.CUDA })).toBe(undefined);
  });

  it('target-only registration: matches its kind, undefined for others and for no-target', () => {
    const cuda = () => 'cuda';
    registerTargetLoweringRule('__target_only__', TargetKind.CUDA, cuda);
    expect(getLoweringRule('__target_only__', { kind: TargetKind.CUDA })).toBe(cuda);
    expect(getLoweringRule('__target_only__', { kind: TargetKind.CPU })).toBe(undefined);
    expect(getLoweringRule('__target_only__')).toBe(undefined);
  });

  it('re-registering the same impl name replaces (idempotent, matches old Map.set)', () => {
    const first = () => 'first';
    const second = () => 'second';
    registerLoweringRule('__lr_replace__', first);
    registerLoweringRule('__lr_replace__', second);
    expect(getLoweringRule('__lr_replace__')).toBe(second);
    expect(getOpStrategy('__lr_replace__').candidates.length).toBe(1);
  });
});
