import { describe, it, expect } from 'vitest';
import { anomaliesOf, profileOf } from '../../../tools/visualizer/src/catalog/pipeline.js';
import { diffLedgers, ledgerOf, parseLedger, summarize } from '../../../tools/visualizer/src/catalog/ledger.js';
import { provenanceOf } from '../../../tools/visualizer/src/catalog/provenance.js';

function snapshot(text, ops) {
  return { text, ops, bytes: 0, flops: 0, dags: [], nests: [] };
}

function step(index, over) {
  return {
    index,
    kind: 'pass',
    parent: null,
    unit: 'f',
    level: 'graph-module',
    phase: 'graphPasses',
    pass: 'p',
    outcome: 'unchanged',
    durationMs: 1,
    before: snapshot('a', 1),
    after: snapshot('a', 1),
    events: [],
    verify: null,
    interpretable: false,
    ...over,
  };
}

function response(steps) {
  return {
    kind: 'compile',
    id: 1,
    ok: true,
    error: null,
    errorPhase: null,
    steps,
    kernels: [],
    events: [],
    sourceLines: [],
    memoryPlans: [],
    tuningRounds: [],
    skipped: [],
    kernelReports: [],
    totalMs: 100,
    run: { ran: false },
  };
}

const OPTIONS = { target: 'cpu' };

describe('compile-time profile', () => {
  it('sums every run of a pass and sorts by cost', () => {
    const profile = profileOf(response([
      step(0, { pass: 'dce', durationMs: 2 }),
      step(1, { pass: 'cse', durationMs: 5 }),
      step(2, { pass: 'dce', durationMs: 3, outcome: 'changed', after: snapshot('b', 1) }),
      step(3, { kind: 'lowering', pass: 'graph → tir', durationMs: 40 }),
    ]));

    expect(profile.passes.map(p => [p.pass, p.ms, p.runs, p.changed]))
      .toEqual([['dce', 5, 2, 1], ['cse', 5, 1, 0]].sort((a, b) => b[1] - a[1]));
    expect(profile.measuredMs).toBe(10);
    expect(profile.totalMs).toBe(100);
  });

  it('leaves lowering and codegen out of pass time, so the gap is visible', () => {
    const profile = profileOf(response([step(0, { durationMs: 3 }), step(1, { kind: 'lowering', durationMs: 90 })]));
    expect(profile.totalMs - profile.measuredMs).toBe(97);
  });
});

describe('pipeline anomalies', () => {
  it('catches a pass that reports CHANGED without changing the text', () => {
    const found = anomaliesOf(response([step(0, { pass: 'cse', outcome: 'changed' })]));
    expect(found.map(a => a.kind)).toEqual(['lied-changed']);
    expect(found[0].pass).toBe('cse');
  });

  it('catches the dangerous direction: UNCHANGED over a rewritten IR', () => {
    const found = anomaliesOf(response([step(0, { after: snapshot('b', 1) })]));
    expect(found.map(a => a.kind)).toEqual(['lied-unchanged']);
  });

  it('catches two passes undoing each other', () => {
    const found = anomaliesOf(response([
      step(0, { pass: 'a', outcome: 'changed', before: snapshot('x', 1), after: snapshot('y', 1) }),
      step(1, { pass: 'b', outcome: 'changed', before: snapshot('y', 1), after: snapshot('z', 1) }),
      step(2, { pass: 'a', outcome: 'changed', before: snapshot('z', 1), after: snapshot('y', 1) }),
    ]));

    expect(found.some(a => a.kind === 'revisited' && a.step === 2)).toBe(true);
  });

  it('does not call a straight line a circle', () => {
    const found = anomaliesOf(response([
      step(0, { pass: 'a', outcome: 'changed', before: snapshot('x', 1), after: snapshot('y', 1) }),
      step(1, { pass: 'b', outcome: 'changed', before: snapshot('y', 1), after: snapshot('z', 1) }),
    ]));
    expect(found.filter(a => a.kind === 'revisited')).toHaveLength(0);
  });

  it('flags a pass that ran four times and never fired', () => {
    const found = anomaliesOf(response([0, 1, 2, 3].map(i => step(i, { pass: 'cse' }))));
    expect(found.map(a => a.kind)).toEqual(['churned']);
    expect(found[0].detail).toContain('ran 4 times');
  });

  it('names the pass that introduced an invariant failure', () => {
    const found = anomaliesOf(response([
      step(0, { pass: 'cse', verify: { introduced: ['return operand 0 shape incompatible'], carried: [] } }),
    ]));
    expect(found[0].kind).toBe('invalid');
    expect(found[0].detail).toContain('shape incompatible');
  });
});

describe('the pass ledger', () => {
  const before = response([
    step(0, { pass: 'dce', outcome: 'changed', before: snapshot('a', 10), after: snapshot('b', 7) }),
    step(1, { pass: 'cse', before: snapshot('b', 7), after: snapshot('b', 7) }),
    step(2, { pass: 'dce', before: snapshot('b', 7), after: snapshot('b', 7) }),
  ]);

  it('numbers repeated runs of the same pass so they line up', () => {
    const ledger = ledgerOf(before, 'src', OPTIONS);
    expect(ledger.entries.map(e => e.key)).toEqual([
      'dce·graph-module·1', 'cse·graph-module·1', 'dce·graph-module·2',
    ]);
  });

  it('survives a round trip through JSON', () => {
    const ledger = ledgerOf(before, 'src', OPTIONS);
    expect(parseLedger(JSON.stringify(ledger))).toEqual(ledger);
    expect(parseLedger('{"format":"something-else"}')).toBeNull();
    expect(parseLedger('not json')).toBeNull();
  });

  it('calls an identical run identical', () => {
    const diff = diffLedgers(ledgerOf(before, 'src', OPTIONS), ledgerOf(before, 'src', OPTIONS));
    expect(diff.changed).toHaveLength(0);
    expect(diff.matched).toBe(3);
    expect(summarize(diff)).toContain('Identical');
  });

  it('names the run whose op count moved and leaves the rest alone', () => {
    const now = response([
      step(0, { pass: 'dce', outcome: 'changed', before: snapshot('a', 10), after: snapshot('b', 7) }),
      step(1, { pass: 'cse', before: snapshot('b', 7), after: snapshot('b', 7) }),
      step(2, { pass: 'dce', outcome: 'changed', before: snapshot('b', 7), after: snapshot('c', 6) }),
    ]);

    const diff = diffLedgers(ledgerOf(before, 'src', OPTIONS), ledgerOf(now, 'src', OPTIONS));
    expect(diff.changed).toHaveLength(1);
    expect(diff.changed[0].key).toBe('dce·graph-module·2');
    expect(diff.changed[0].was.after).toBe(7);
    expect(diff.changed[0].entry.after).toBe(6);
    expect(diff.changed[0].was.outcome).toBe('unchanged');
    expect(diff.changed[0].entry.outcome).toBe('changed');
  });

  it('reports a run that stopped happening', () => {
    const now = response([step(0, { pass: 'dce', outcome: 'changed', before: snapshot('a', 10), after: snapshot('b', 7) })]);
    const diff = diffLedgers(ledgerOf(before, 'src', OPTIONS), ledgerOf(now, 'src', OPTIONS));

    expect(diff.onlyOld.map(e => e.key)).toEqual(['cse·graph-module·1', 'dce·graph-module·2']);
    expect(summarize(diff)).toContain('no longer happen');
  });
});

describe('tracking a name through the pipeline', () => {
  const run = response([
    step(0, { pass: 'a', before: snapshot('%1 = add', 1), after: snapshot('%1 = add\n%13 = dot', 2) }),
    step(1, { pass: 'b', before: snapshot('%1 = add\n%13 = dot', 2), after: snapshot('%1 = add\n%13 = dot', 2) }),
    step(2, { pass: 'c', before: snapshot('%1 = add\n%13 = dot', 2), after: snapshot('%1 = add', 1) }),
  ]);

  it('says where a value was born and where it died', () => {
    const found = provenanceOf(run, '%13');
    expect(found.bornAt).toBe(0);
    expect(found.diedAt).toBe(2);
    expect([...found.marks.values()]).toEqual(['born', 'alive', 'died']);
  });

  it('matches whole tokens, not prefixes', () => {
    const near = response([step(0, { after: snapshot('%130 = dot', 1), before: snapshot('%130 = dot', 1) })]);
    expect(provenanceOf(near, '%13').hits).toBe(0);
    expect(provenanceOf(near, '%130').hits).toBe(1);
  });

  it('separates a prefix of a longer name from a name that is simply absent', () => {
    const near = response([step(0, { after: snapshot('%130 = dot', 1), before: snapshot('%130 = dot', 1) })]);

    const prefix = provenanceOf(near, '%13');
    expect(prefix.hits).toBe(0);
    expect(prefix.insideLonger).toBe(1);

    const missing = provenanceOf(near, '%99');
    expect(missing.hits).toBe(0);
    expect(missing.insideLonger).toBe(0);
  });

  it('does not pay for the substring scan once a whole token matched', () => {
    expect(provenanceOf(run, '%13').insideLonger).toBe(0);
  });

  it('says nothing for an empty needle', () => {
    expect(provenanceOf(run, '   ')).toBeNull();
  });
});
